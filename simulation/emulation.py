#!/usr/bin/env python3
"""Emulate Neu+ experiments with conversion.py's SNN using BindsNET on CPU/GPU.

Created by Daeyoung Kim and Jongkil Park.

Loads converted neurons, synapses, weights, and thresholds without retraining
or reconversion. This is functional emulation, not a cycle-accurate Neu+ model.
Edit the defaults below or run `python simulation/emulation.py --help` for options.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import shlex
import sys
import traceback
from contextlib import nullcontext
from pathlib import Path
from itertools import islice

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from bindsnet.network import Network
from bindsnet.network.nodes import IFNodes, Input
from bindsnet.network.monitors import Monitor
from bindsnet.network.topology import Connection

if __package__:
    from .conversion import load_model, resolve_data_directory, source_run_directory
    from .utils import (EnergyMonitor, EvaluationProgress, TeeLogger, create_timestamp_directory,
                        file_sha256, print_section, seed_all, select_device)
else:
    from conversion import load_model, resolve_data_directory, source_run_directory
    from utils import (EnergyMonitor, EvaluationProgress, TeeLogger, create_timestamp_directory,
                       file_sha256, print_section, seed_all, select_device)


# Editable experiment settings. Command-line options override these values.
SCRIPT_DIR = Path(__file__).resolve().parent

# Retained conversion; select another bundle with --model-path.
DEFAULT_MODEL_PATH = (
    SCRIPT_DIR / "results/26-09-09/15-58-15/snn/26-09-09/16-05-07/MNIST_LENET5_best_snn.pth"
)
DEFAULT_ANN_MODEL_PATH = None
DEFAULT_DATA_DIR = None  # Saved MNIST roots are resolved to data/reference_mnist when needed.
DEFAULT_OUTPUT_DIR = None  # None saves under the chosen SNN folder/emulation.
# Event duration is derived from recorded bins; --timesteps overrides it.
DEFAULT_TIMESTEPS = (200,)
DEFAULT_DT = 0.5  # milliseconds per timestep; 200 steps cover 100 ms
DEFAULT_ENCODING = "event"
DEFAULT_EVENT_PATH = SCRIPT_DIR / "data/measured_retina_events/curved_2"
DEFAULT_EVENT_DT = 1.0  # milliseconds per recorded event bin
DEFAULT_EVENT_SIM_DT = 0.5  # historical curved_2 BindsNET simulation timestep
DEFAULT_N_EVENT_STEPS = 100  # event mode derives run length from this
DEFAULT_MAX_RATE = 100.0  # Hz; recorded events are unchanged (bias sources only).
DEFAULT_SAMPLES = 1000  # Test IDs 0..999; 0 selects all remaining samples.
DEFAULT_SAMPLE_OFFSET = 0
DEFAULT_BATCH_SIZE = 1  # sequential samples, matching the retained accuracy run
DEFAULT_NUM_WORKERS = 0
DEFAULT_DEVICE = "cuda"
DEFAULT_GPU_ID = 0
DEFAULT_SEED = 42
DEFAULT_DOWNLOAD = False
DEFAULT_COMPARE_ANN = True
DEFAULT_CLIP_HIDDEN = False
DEFAULT_MEM_INIT = "zero"
DEFAULT_MAX_ACCURACY_DROP = 1.0  # percentage points relative to the ANN
DEFAULT_PROGRESS = False
# Measurements describe CPU/GPU software execution, not Neu+ chip performance.
DEFAULT_ENERGY = False  # --energy enables sensors; unsupported sensors are errors.
DEFAULT_LATENCY = False  # accuracy-only evaluation unless measurement is requested
DEFAULT_ENERGY_INTERVAL = 0.1  # wall-clock seconds between sensor reads, not SNN dt
DEFAULT_ENERGY_WARMUP_BATCHES = 1  # unmeasured warmup per timestep condition
DEFAULT_ENCODING_DT = 1.0  # rate-coding input bins in ms; None uses simulation dt
EXECUTION_BACKEND = "bindsnet_dense_static"


class SoftIFNodes(IFNodes):
    """Reference IF dynamics with subtractive reset instead of a hard reset."""

    def forward(self, x):
        self.v += (self.refrac_count == 0).float() * x
        self.refrac_count = (self.refrac_count > 0).float() * (self.refrac_count - self.dt)
        self.s = self.v >= self.thresh
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v -= self.s.float() * self.thresh
        if self.lbound is not None:
            self.v.clamp_(min=self.lbound)
        super(IFNodes, self).forward(x)


def resolve_ann_path(stored_path, override=None):
    """Resolve saved repository-relative ANN paths independently of the cwd.

    Explicit --ann-model-path overrides remain relative to the caller's cwd.
    Absolute saved paths are preserved, and custom SNNs keep their own ANN.
    """
    path = Path(override if override is not None else stored_path).expanduser()
    if override is None and not path.is_absolute():
        path = SCRIPT_DIR.parent / path
    return path.resolve()


def resolve_snn_path(path):
    """Accept a converted SNN bundle or a folder containing exactly one *_snn.pth.

    Use conversion.py's bundle, not an ANN checkpoint. The default selects the
    retained model; pass --model-path to use a different file or conversion folder.
    No search for the latest conversion is performed.
    """
    path = Path(path).expanduser().resolve(strict=True)
    if path.is_dir():
        candidates = sorted(path.glob("*_snn.pth"))
        if len(candidates) != 1:
            raise ValueError("Provide an exact *_snn.pth file or its conversion folder.")
        path = candidates[0]
    return path


def load_snn_model(path):
    """Load and validate conversion.py's versioned tensor/dictionary format."""
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(bundle, dict) or bundle.get("format_version") != 1:
        raise ValueError("Expected a version-1 SNN bundle produced by conversion.py.")
    expected = dict(input_layer=784, conv1_1=4704, pool1=1176, conv2_1=1600,
                    pool2=400, conv3_1=120, linear1=84, lin_out=10)
    if bundle.get("architecture") != "LeNet5" or bundle.get("dataset") != "MNIST":
        raise ValueError("Only converted MNIST LeNet5 models are supported.")
    if bundle.get("input_shape") != [1, 1, 28, 28]:
        raise ValueError("SNN input shape must be [1, 1, 28, 28].")
    neurons = bundle.get("neurons", {})
    if {k: v for k, v in neurons.items() if not k.startswith("bias_")} != expected:
        raise ValueError("SNN neuron counts do not match LeNet5.")
    if list(k for k in neurons if not k.startswith("bias_")) != list(expected):
        raise ValueError("SNN layers are not in feed-forward order.")
    if bundle.get("readout_layer", "lin_out") != "lin_out":
        raise ValueError("The LeNet5 readout layer must be lin_out.")
    biases = bundle.get("bias_neurons", {})
    if set(biases) - set(expected.keys() - {'input_layer'}):
        raise ValueError("Unknown bias target layer.")
    if set(neurons) != set(expected) | {f"bias_{name}" for name in biases}:
        raise ValueError("Bias neuron metadata is inconsistent.")
    bases = bundle.get("layer_bases", {})
    if set(bases) != set(neurons):
        raise ValueError("Missing global neuron base addresses.")
    intervals = sorted((bases[name], bases[name] + size) for name, size in neurons.items())
    if any(not isinstance(base, int) or base < 0 for base in bases.values()):
        raise ValueError("Global base addresses must be non-negative integers.")
    if any(right > following for (_, right), (following, _) in zip(intervals, intervals[1:])):
        raise ValueError("Global neuron address ranges overlap.")
    names = list(expected)
    connections = [(names[i - 1], name, name) for i, name in enumerate(names) if i]
    for name, size in biases.items():
        if size != neurons[name] or neurons[f"bias_{name}"] != 1:
            raise ValueError(f"Invalid bias source size: {name}")
        connections.append((f"bias_{name}", name, f"bias_{name}"))
    if set(bundle.get("synapses", {})) != {key for _, _, key in connections}:
        raise ValueError("SNN synapse keys do not match the feed-forward graph.")
    for source_name, target_name, key in connections:
        arrays = bundle['synapses'][key]
        if not isinstance(arrays, (tuple, list)) or len(arrays) != 3:
            raise ValueError(f"Invalid synapse arrays: {key}")
        source, target, weights = arrays
        if any(not isinstance(a, torch.Tensor) or a.ndim != 1 for a in arrays):
            raise ValueError(f"Synapse arrays must be 1D tensors: {key}")
        if not len(source) or len(source) != len(target) or len(source) != len(weights):
            raise ValueError(f"Inconsistent synapse lengths: {key}")
        if source.dtype != torch.int64 or target.dtype != torch.int64 or not torch.isfinite(weights).all():
            raise ValueError(f"Invalid synapse indices or weights: {key}")
        for indices, name in ((source, source_name), (target, target_name)):
            if (indices < bases[name]).any() or (indices >= bases[name] + neurons[name]).any():
                raise ValueError(f"Synapse index out of bounds: {key} / {name}")
    for name in names[1:]:
        threshold = bundle.get('neuron_thresholds', {}).get(name)
        if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold <= 0:
            raise ValueError(f"Invalid firing threshold: {name}")
    return bundle


def encode_spikes(images, steps, dt, max_rate, mode="uniform", generator=None, encoding_dt=None):
    """Rate-code images on input bins and inject once at each bin's start.

    Uniform mode uses a deterministic phase accumulator; Poisson mode uses
    Bernoulli trials per bin. Uniform counts equal floor(number_of_bins * pixel
    * max_rate * encoding_dt / 1000), without forcing a first spike at time zero.

    encoding_dt=None uses simulation dt. Larger bins must be integer multiples
    of dt; intervening steps remain silent. For timing matched to event replay,
    use --encoding uniform --encoding-dt 1 --dt 0.5 --timesteps 200:
    100 input bins span 100 ms, as with --event-dt 1 --n-event-steps 100.
    """
    encoding_dt = dt if encoding_dt is None else encoding_dt
    stride = event_steps_per_bin(dt, encoding_dt)
    bins = (steps + stride - 1) // stride
    scale = max_rate * encoding_dt / 1000.0
    if not 0 < scale <= 1:
        raise ValueError("max_rate * encoding_dt / 1000 must be in (0, 1].")
    values = images.flatten(1).to(torch.float64)
    if not torch.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Input images must have finite pixel values in [0, 1].")
    rates = values * scale
    if mode == "uniform":
        time_axis = torch.arange(bins + 1, device=images.device, dtype=torch.float64)[:, None, None]
        counts = torch.floor(time_axis * rates)
        encoded = counts[1:] > counts[:-1]
    elif mode == "poisson":
        random = torch.rand((bins, *rates.shape), device=images.device, generator=generator)
        encoded = random < rates
    else:
        raise ValueError(f"Unknown encoding: {mode}")
    if stride == 1:
        return encoded
    spikes = torch.zeros((steps, *rates.shape), dtype=torch.bool, device=images.device)
    spikes[::stride] = encoded
    return spikes


def event_steps_per_bin(dt, event_dt):
    """Preserve recorded timing exactly; refuse fractional or compressed bins."""
    if not all(math.isfinite(v) and v > 0 for v in (dt, event_dt)):
        raise ValueError('dt and event_dt must be finite and positive.')
    ratio = event_dt / dt
    if not math.isfinite(ratio) or ratio < 1 or not math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-9):
        raise ValueError('event_dt / dt must be an integer >= 1 (e.g. 1 / 0.5).')
    return round(ratio)


def validate_events(events, n_neurons=784):
    """Validate numeric [event_bin, neuron] pairs without unsafe pickle loading."""
    events = np.asarray(events)
    if events.ndim != 2 or events.shape[1] != 2 or events.dtype.kind not in 'iuf':
        raise ValueError('Events must be a numeric array shaped [number_of_events, 2].')
    if not np.isfinite(events).all() or (events < 0).any() or (events != np.floor(events)).any():
        raise ValueError('Event time and neuron indices must be finite non-negative integers.')
    if (events[:, 1] >= n_neurons).any() or (events[:, 0] >= 2**63 - 1).any():
        raise ValueError('Event index out of bounds.')
    return events.astype(np.int64, copy=False)


def load_event_samples(directory, sample_offset=0, samples=0, dataset_size=10000):
    """Load consecutive MNIST test IDs, recorded events, and file fingerprints.

    Files are {MNIST_test_index}_spikes.npy arrays shaped [events, 2], with
    columns [event_bin, flattened_pixel]. samples=0 selects all remaining IDs
    from sample_offset. Missing files are errors, never silently skipped or
    relabeled by lexicographic order.

    Return (indices, event_arrays, manifest_rows). Arrays are cached once so
    each timestep sweep uses exactly the same recorded events.
    """
    directory = Path(directory).expanduser().resolve(strict=True)
    if not directory.is_dir():
        raise ValueError('Event path must be a directory.')
    files = {}
    for path in directory.glob('*_spikes.npy'):
        prefix = path.stem.removesuffix('_spikes')
        if not prefix.isascii() or not prefix.isdecimal() or prefix != str(int(prefix)):
            raise ValueError(f'Expected canonical MNIST sample ID in filename: {path.name}')
        files[int(prefix)] = path
    if not files or sample_offset < 0 or samples < 0:
        raise ValueError('No event files or invalid sample selection.')
    count = samples or max(files) + 1 - sample_offset
    if count <= 0 or sample_offset + count > dataset_size:
        raise ValueError('Event sample interval is outside the available MNIST test IDs.')
    indices = list(range(sample_offset, sample_offset + count))
    arrays, manifest = [], []
    for index in indices:
        if index not in files:
            raise FileNotFoundError(f'Missing event sample: {directory / f"{index}_spikes.npy"}')
        path = files[index]
        events = validate_events(np.load(path, allow_pickle=False))
        arrays.append(events)
        manifest.append(dict(sample_index=index, event_file=str(path), sha256=file_sha256(path),
                             events=len(events), first_event_bin=int(events[:, 0].min()) if len(events) else '',
                             last_event_bin=int(events[:, 0].max()) if len(events) else ''))
    return indices, arrays, manifest


def encode_event_batch(events, steps, dt, event_dt=1.0, n_event_steps=100, n_neurons=784):
    """Replay recorded events as binary [time, batch, neuron] spikes.

    Events enter at the start of their bins; event_dt / dt must be an integer
    at least 1. With event_dt=1 ms, dt=0.5 ms, and n_event_steps=100, a full
    recording spans 200 simulation steps. Events are not rate-coded or scaled
    by max_rate; that setting affects only optional bias sources in event mode.

    Keep bins 0..n_event_steps-1 and simulation times before `steps`. Longer
    runs append silence, never repeat or stretch the recording. Duplicate pairs
    coalesce to one spike. Invalid indices are rejected even outside the window.
    """
    stride = event_steps_per_bin(dt, event_dt)
    if steps < 1 or n_event_steps < 1 or not events:
        raise ValueError('Simulation steps, recorded bins, and batch size must be positive.')
    spikes = torch.zeros((steps, len(events), n_neurons), dtype=torch.bool)
    for batch_index, array in enumerate(events):
        array = validate_events(array, n_neurons)
        # Filter before multiplying to avoid overflow for late event indices.
        kept = array[array[:, 0] < min(n_event_steps, (steps - 1) // stride + 1)]
        spikes[torch.from_numpy(kept[:, 0] * stride), batch_index, torch.from_numpy(kept[:, 1])] = True
    return spikes


class BindsNETEmulator:
    """Build a fixed BindsNET network from the converted neurons and synapses.

    Reuses the reference simulator's dense BindsNET Connection and static
    synchronous loop. Converted weights, thresholds, soft-reset IF dynamics,
    and the one-timestep delay between layers are preserved without learning.
    """

    def __init__(self, bundle, device="cpu", dt=1.0, max_rate=1000.0,
                 clip_hidden=False, mem_init="zero"):
        self.bundle = bundle
        self.device = torch.device(device)
        self.dt, self.max_rate = dt, max_rate
        self.mem_init = mem_init
        self.output_name = bundle.get("readout_layer", "lin_out")
        self.network = Network(dt=dt, learning=False)
        neurons, bases = bundle['neurons'], bundle['layer_bases']
        self.main_names = [name for name in neurons if not name.startswith('bias_')]
        self.input_name = self.main_names[0]
        self.bias_names = [name for name in neurons if name.startswith('bias_')]
        for name in [self.input_name, *self.bias_names]:
            self.network.add_layer(Input(n=neurons[name]), name=name)
        # Output voltage stays signed for readout, even when hidden clipping is enabled.
        for name in self.main_names[1:]:
            layer = SoftIFNodes(n=neurons[name], thresh=float(bundle['neuron_thresholds'][name]),
                                reset=0.0, refrac=0,
                                lbound=0.0 if clip_hidden and name != self.output_name else None)
            self.network.add_layer(layer, name=name)
        connections = [(self.main_names[i-1], name, name)
                       for i, name in enumerate(self.main_names) if i]
        connections.extend((name, name.removeprefix('bias_'), name) for name in self.bias_names)
        for source_name, target_name, key in connections:
            source, target, weights = bundle['synapses'][key]
            matrix = torch.zeros((neurons[source_name], neurons[target_name]), dtype=torch.float32)
            matrix[source - bases[source_name], target - bases[target_name]] = weights.float()
            connection = Connection(source=self.network.layers[source_name],
                                    target=self.network.layers[target_name], w=matrix)
            self.network.add_connection(connection, source=source_name, target=target_name)
        self.monitor = Monitor(self.network.layers[self.output_name], state_vars=['s'])
        self.network.add_monitor(self.monitor, name='output_spikes')
        self.network.to(self.device).eval()

    @torch.no_grad()
    def _run_network_static(self, inputs, steps):
        """Use the reference simulator's inference-only synchronous update order.

        Compute all synaptic inputs before advancing any layer, so connections
        read the previous timestep's spikes. Unlike the general Network.run(),
        this no-learning path does not recompute and discard inputs at the end
        of each step. Batch resizing and resets are handled by predict_spikes().
        """
        layer_names = list(self.network.layers)
        connections = list(self.network.connections.items())
        target_inputs = {}
        for (_, target_name), connection in connections:
            if target_name not in target_inputs:
                target_inputs[target_name] = torch.zeros(
                    self.network.batch_size, *connection.target.shape,
                    device=connection.target.s.device)

        for step in range(steps):
            for value in target_inputs.values():
                value.zero_()
            for (_, target_name), connection in connections:
                target_inputs[target_name].add_(connection.compute(connection.source.s))
            for name in layer_names:
                value = target_inputs.get(name)
                if name in inputs:
                    value = inputs[name][step] if value is None else value + inputs[name][step]
                self.network.layers[name].forward(x=value)
            for monitor in self.network.monitors.values():
                monitor.record()

    @torch.no_grad()
    def predict(self, images, steps, encoding="uniform", generator=None, encoding_dt=None):
        """Encode images, then use the same spike-input engine as event replay."""
        images = images.to(self.device)
        spikes = encode_spikes(images, steps, self.dt, self.max_rate, encoding, generator, encoding_dt)
        return self.predict_spikes(spikes)

    @torch.no_grad()
    def predict_spikes(self, spikes):
        """Reset and run binary [time, batch, input_neuron] spikes.

        Every independent batch starts from reset. Return (scores, spike_counts),
        where scores = spike_count * threshold + final_voltage - initial_voltage.
        The residual voltage preserves signed scores even when no spike is emitted.
        """
        if spikes.ndim != 3 or min(spikes.shape[:2]) < 1 or spikes.shape[2] != self.bundle['neurons'][self.input_name]:
            raise ValueError('Expected nonempty spikes shaped [time, batch, input_neurons].')
        if not ((spikes == 0) | (spikes == 1)).all():
            raise ValueError('Input spikes must be binary.')
        spikes = spikes.to(self.device)
        steps, size, _ = spikes.shape
        if size != self.network.batch_size:
            self.network.batch_size = size
            for layer in self.network.layers.values():
                layer.set_batch_size(size)
        self.network.reset_state_variables()
        if self.mem_init == 'half':
            for name in self.main_names[1:]:
                layer = self.network.layers[name]
                layer.v.copy_(torch.zeros_like(layer.v) + layer.thresh / 2)
        output = self.network.layers[self.output_name]
        initial = output.v.clone()
        inputs = {self.input_name: spikes}
        # Bias input is continuous current, independent of recorded event spikes.
        for name in self.bias_names:
            inputs[name] = torch.full((steps, size, 1), self.max_rate * self.dt / 1000.0,
                                     device=self.device)
        # Synchronous propagation adds one timestep of delay per layer.
        self._run_network_static(inputs, steps)
        recorded = self.monitor.get('s')
        if recorded.shape[0] != steps:
            raise RuntimeError("BindsNET executed an unexpected number of timesteps.")
        counts = recorded.sum(0)
        scores = counts * output.thresh + output.v - initial
        if not torch.isfinite(scores).all():
            raise RuntimeError("Non-finite SNN output scores.")
        return scores.detach().cpu(), counts.detach().cpu()


def write_csv(path, rows):
    """Write scalar evaluation records for inspection without Python tooling."""
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    """Read command-line overrides for the editable experiment defaults.

    Run from the repository root, for example:
        python simulation/emulation.py
        python simulation/emulation.py --model-path /path/to/MODEL_snn.pth

    Event replay is the default; rate coding requires --encoding uniform or
    --encoding poisson. --timesteps explicitly selects run lengths or sweeps.
    """
    parser = argparse.ArgumentParser(description="Evaluate converted MNIST SNNs with BindsNET.")
    parser.add_argument('--model-path', type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument('--ann-model-path', type=Path, default=DEFAULT_ANN_MODEL_PATH)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--timesteps', type=int, nargs='+', default=None,
                        help='Run lengths; default: image settings or full event recording. Events are never repeated.')
    parser.add_argument('--dt', type=float, default=None, help='Simulation timestep in ms; default: 0.5.')
    parser.add_argument('--encoding', choices=('uniform', 'poisson', 'event'), default=DEFAULT_ENCODING)
    parser.add_argument('--event-path', type=Path, default=DEFAULT_EVENT_PATH)
    parser.add_argument('--event-dt', type=float, default=DEFAULT_EVENT_DT, help='Duration of one recorded event bin in ms.')
    parser.add_argument('--n-event-steps', type=int, default=DEFAULT_N_EVENT_STEPS, help='Recorded bins; default: 100 for curved_2.')
    parser.add_argument('--max-rate', type=float, default=DEFAULT_MAX_RATE, help='Maximum input rate in Hz.')
    parser.add_argument('--samples', type=int, default=DEFAULT_SAMPLES, help='0 evaluates all remaining images or consecutive event files.')
    parser.add_argument('--sample-offset', type=int, default=DEFAULT_SAMPLE_OFFSET)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--num-workers', type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default=DEFAULT_DEVICE)
    parser.add_argument('--gpu-id', type=int, default=DEFAULT_GPU_ID)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--mem-init', choices=('zero', 'half'), default=DEFAULT_MEM_INIT)
    parser.add_argument('--clip-hidden', action=argparse.BooleanOptionalAction, default=DEFAULT_CLIP_HIDDEN)
    parser.add_argument('--download', action=argparse.BooleanOptionalAction, default=DEFAULT_DOWNLOAD)
    parser.add_argument('--compare-ann', action=argparse.BooleanOptionalAction, default=DEFAULT_COMPARE_ANN)
    parser.add_argument('--progress', action=argparse.BooleanOptionalAction, default=DEFAULT_PROGRESS)
    parser.add_argument('--energy', action=argparse.BooleanOptionalAction, default=DEFAULT_ENERGY,
                        help='Measure NVML GPU or RAPL CPU energy over the evaluation window.')
    parser.add_argument('--latency', action=argparse.BooleanOptionalAction, default=None,
                        help='Measure evaluation latency and throughput; off by default, implied by --energy.')
    parser.add_argument('--energy-interval', type=float, default=DEFAULT_ENERGY_INTERVAL,
                        help='Wall-clock power polling interval in seconds, not SNN dt.')
    parser.add_argument('--energy-warmup-batches', type=int, default=DEFAULT_ENERGY_WARMUP_BATCHES,
                        help='Unmeasured warmup batches per timestep condition when energy is enabled.')
    parser.add_argument('--encoding-dt', type=float, default=None,
                        help='Rate-coding input bin in ms; default: 1. Events use --event-dt.')
    parser.add_argument('--max-accuracy-drop', type=float, default=DEFAULT_MAX_ACCURACY_DROP,
                        help='Allowed SNN accuracy drop from ANN, in percentage points.')
    args = parser.parse_args()
    if args.latency is None:
        args.latency = args.energy or DEFAULT_LATENCY
    elif args.energy and not args.latency:
        parser.error('--energy requires latency measurement; omit --no-latency or disable energy.')
    if args.dt is None:
        args.dt = DEFAULT_EVENT_SIM_DT if args.encoding == 'event' else DEFAULT_DT
    if args.encoding != 'event' and args.encoding_dt is None:
        args.encoding_dt = DEFAULT_ENCODING_DT
    try:
        if args.encoding == 'event':
            event_stride = event_steps_per_bin(args.dt, args.event_dt)
            if args.n_event_steps < 1:
                raise ValueError('n_event_steps must be positive.')
            args.event_path = args.event_path.expanduser().resolve(strict=True)
        if args.timesteps is None:
            args.timesteps = [args.n_event_steps * event_stride] if args.encoding == 'event' else DEFAULT_TIMESTEPS
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if not math.isfinite(args.energy_interval) or args.energy_interval <= 0 or args.energy_warmup_batches < 0:
        parser.error('Energy interval must be positive and finite; warmup batches must be non-negative.')
    if args.encoding_dt is not None:
        try:
            if args.encoding == 'event':
                raise ValueError('Event input uses --event-dt, not --encoding-dt.')
            event_steps_per_bin(args.dt, args.encoding_dt)
            if args.max_rate * args.encoding_dt / 1000 > 1:
                raise ValueError('max_rate * encoding_dt / 1000 must not exceed 1.')
        except ValueError as error:
            parser.error(str(error))
    if args.model_path is None:
        parser.error('Set DEFAULT_MODEL_PATH or pass --model-path.')
    if args.batch_size < 1 or min(args.timesteps) < 1:
        parser.error('Batch size and all timestep counts must be positive.')
    if min(args.samples, args.sample_offset, args.num_workers, args.gpu_id) < 0:
        parser.error('Sample counts, offsets, worker count, and GPU index must be non-negative.')
    if not all(math.isfinite(v) for v in (args.dt, args.max_rate, args.max_accuracy_drop)):
        parser.error('Time, rate, and accuracy tolerance must be finite.')
    if args.dt <= 0 or args.max_rate <= 0 or args.max_rate * args.dt / 1000 > 1:
        parser.error('Require dt > 0 and 0 < max_rate * dt / 1000 <= 1.')
    if args.max_accuracy_drop < 0:
        parser.error('Accuracy tolerance must be non-negative.')
    args.timesteps = sorted(set(args.timesteps))
    try:
        args.model_path = resolve_snn_path(args.model_path)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return args


def main():
    """Compare ANN/SNN predictions and save timestamped evaluation reports.

    By default, outputs go to <SNN folder>/emulation/YY-MM-DD/HH-MM-SS/:
    emulation_output.txt, emulation_summary.json, metrics.csv, and prediction
    and confusion-matrix CSVs for each run length. No figures are generated.
    Energy and timing fields are included only when measurement is enabled.
    """
    args = parse_args()
    base = args.output_dir.expanduser().resolve() if args.output_dir else args.model_path.parent / 'emulation'
    output = create_timestamp_directory(base)
    logger = TeeLogger(output / 'emulation_output.txt')
    energy_monitor = None
    try:
        print_section('Emulation Conditions', [(key, value) for key, value in vars(args).items()], logger=logger)
        seed_all(args.seed)
        device = select_device(args.device, args.gpu_id)
        bundle = load_snn_model(args.model_path)
        source_path = resolve_ann_path(bundle['source_weights'], args.ann_model_path)
        model = None
        if args.compare_ann:
            if not source_path.is_file():
                raise FileNotFoundError('Original ANN missing: supply --ann-model-path or --no-compare-ann.')
            if file_sha256(source_path) != bundle['source_sha256']:
                raise ValueError('ANN fingerprint does not match the converted SNN model.')
            model, _, _, _ = load_model(source_path)
            model.to(device).eval()
        conversion_summary = args.model_path.parent / 'conversion_summary.json'
        config = {}
        if conversion_summary.is_file():
            with conversion_summary.open() as file:
                config['data_dir'] = json.load(file)['data_dir']
        data_dir = resolve_data_directory(args, source_run_directory(source_path), config)
        dataset = datasets.MNIST(data_dir, train=False, download=args.download, transform=transforms.ToTensor())
        event_arrays, event_manifest, files = None, None, {}
        input_info = dict(mode=args.encoding, ann_input='original MNIST test images')
        if args.encoding != 'event':
            input_info['encoding_bin_ms'] = args.encoding_dt or args.dt
        if args.encoding == 'event':
            indices, event_arrays, event_manifest = load_event_samples(
                args.event_path, args.sample_offset, args.samples, len(dataset))
            count = len(indices)
            for row in event_manifest:
                row['label'] = int(dataset.targets[row['sample_index']])
            manifest_path = output / 'event_manifest.csv'
            write_csv(manifest_path, event_manifest)
            files['event_manifest'] = str(manifest_path)
            input_info.update(event_path=str(args.event_path), event_dt_ms=args.event_dt,
                              recorded_bins=args.n_event_steps, recording_duration_ms=args.n_event_steps * args.event_dt,
                              simulation_steps_per_event_bin=event_steps_per_bin(args.dt, args.event_dt),
                              raw_events=sum(len(a) for a in event_arrays),
                              label_mapping='file ID = original MNIST test index',
                              duplicates='coalesced to one binary spike', padding='silent; never repeat events',
                              max_rate_applies_to='bias sources only; recorded events are unchanged')
        else:
            count = args.samples or len(dataset) - args.sample_offset
            if count <= 0 or args.sample_offset + count > len(dataset):
                raise ValueError('Requested sample interval is outside the MNIST test set.')
            indices = list(range(args.sample_offset, args.sample_offset + count))
        loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
        labels = dataset.targets[indices].cpu()
        # Normalization may calibrate on the first N test images. --sample-offset N
        # excludes them; otherwise any overlap is recorded rather than hidden.
        normalization = bundle.get('normalization', {})
        calibration_count = normalization.get('samples', 0) if normalization.get('enabled') else 0
        overlap = max(0, min(args.sample_offset + count, calibration_count) - args.sample_offset)
        print_section('Resolved Evaluation', [
            ('SNN file', args.model_path), ('SNN SHA256', file_sha256(args.model_path)),
            ('ANN file', source_path if model is not None else 'comparison disabled'),
            ('Data directory', data_dir), ('Sample interval', f'{indices[0]}..{indices[-1]}'),
            ('Calibration overlap', overlap), ('Device', device), ('BindsNET', importlib.metadata.version('bindsnet')),
            ('Execution backend', EXECUTION_BACKEND), ('Connection', 'BindsNET dense [source, target]'),
            ('Torch CPU threads', torch.get_num_threads()), ('Torch interop threads', torch.get_num_interop_threads()),
            ('PyTorch', str(torch.__version__)), ('Neuron', 'soft-reset IF, threshold from bundle'),
            ('Propagation', 'synchronous, one-layer delay'), ('Readout', 'spikes * threshold + final V - initial V'),
            ('Input encoding', args.encoding), ('Output directory', output),
        ], logger=logger)
        print_section('Input Conditions', list(input_info.items()), logger=logger)
        ann_predictions = None
        ann_accuracy = None
        # Even in event mode, the ANN baseline sees matching MNIST images, not events.
        if model is not None:
            print('Evaluating the original ANN on the same sample interval...')
            predictions = []
            with torch.no_grad():
                for images, _ in loader:
                    predictions.append(model(images.to(device)).argmax(1).cpu())
            ann_predictions = torch.cat(predictions)
            ann_accuracy = 100.0 * (ann_predictions == labels).sum().item() / count
            print(f'ANN accuracy: {ann_accuracy:.4f}% ({count} samples)')
            model.cpu()
        emulator = BindsNETEmulator(bundle, device, args.dt, args.max_rate,
                                    args.clip_hidden, args.mem_init)
        # --energy also enables synchronized latency for energy = average power *
        # latency and EDP = energy * latency. NVML includes idle/other GPU processes;
        # CPU power requires RAPL. No sensor or timer is started in accuracy-only mode.
        if args.latency:
            energy_monitor = EnergyMonitor(device, args.energy_interval, enabled=args.energy)
        if args.energy:
            print_section('Energy Measurement Conditions', [
                *energy_monitor.metadata.items(), ('warmup_batches', args.energy_warmup_batches),
                ('Energy formula', 'average power [W] * measured latency [s]'),
                ('EDP formula', 'energy [J] * measured latency [s]'),
            ], logger=logger)
            if energy_monitor.metadata.get('other_compute_pids') or energy_monitor.metadata.get('multiple_compute_processes'):
                print('WARNING: Other compute processes share this GPU; measured power is not exclusive to this evaluation.')
        metrics = []
        confusion = None
        for steps in args.timesteps:
            generator = torch.Generator(device=device).manual_seed(args.seed)
            all_predictions, records = [], []
            correct = completed = 0
            if args.energy and args.energy_warmup_batches:
                warm_offset = 0
                warm_generator = torch.Generator(device=device).manual_seed(args.seed)
                for images, targets in islice(loader, args.energy_warmup_batches):
                    if event_arrays is not None:
                        spikes = encode_event_batch(event_arrays[warm_offset:warm_offset + len(targets)],
                                                    steps, args.dt, args.event_dt, args.n_event_steps)
                        emulator.predict_spikes(spikes)
                    else:
                        emulator.predict(images, steps, args.encoding, warm_generator, args.encoding_dt)
                    warm_offset += len(targets)
            # Measure encoding/transfers, SNN execution, and prediction processing
            # together. Setup, ANN evaluation, warmup, and CSV/JSON writes are excluded.
            with energy_monitor if energy_monitor is not None else nullcontext():
                with EvaluationProgress(logger, count, f'{steps} steps', enabled=args.progress,
                                        measure_time=args.latency) as progress:
                    for images, targets in loader:
                        input_counts = None
                        if event_arrays is not None:
                            batch_events = event_arrays[completed:completed + len(targets)]
                            spikes = encode_event_batch(batch_events, steps, args.dt, args.event_dt, args.n_event_steps)
                            input_counts = spikes.sum((0, 2))
                            scores, spike_counts = emulator.predict_spikes(spikes)
                        else:
                            scores, spike_counts = emulator.predict(images, steps, args.encoding, generator, args.encoding_dt)
                        predictions = scores.argmax(1)
                        all_predictions.append(predictions)
                        correct += (predictions == targets).sum().item()
                        for offset, target in enumerate(targets):
                            row = dict(sample_index=indices[completed + offset], label=int(target),
                                       snn_prediction=int(predictions[offset]),
                                       ann_prediction=int(ann_predictions[completed + offset]) if ann_predictions is not None else '',
                                       output_spikes=int(spike_counts[offset].sum()))
                            if input_counts is not None:
                                event_index = completed + offset
                                row.update(event_file=event_manifest[event_index]['event_file'],
                                           raw_events=len(event_arrays[event_index]), input_spikes=int(input_counts[offset]))
                            row.update({f'score_{i}': float(scores[offset, i]) for i in range(10)})
                            records.append(row)
                        completed += len(targets)
                        progress.update(completed, 100.0 * correct / completed, ann_accuracy)
            predictions = torch.cat(all_predictions)
            accuracy = 100.0 * correct / count
            gap = ann_accuracy - accuracy if ann_accuracy is not None else None
            agreement = 100.0 * (predictions == ann_predictions).sum().item() / count if ann_predictions is not None else None
            row = dict(timesteps=steps, duration_ms=steps * args.dt, samples=count,
                       ann_accuracy=ann_accuracy, snn_accuracy=accuracy, accuracy_drop_pp=gap,
                       agreement_percent=agreement,
                       within_accuracy_tolerance=gap <= args.max_accuracy_drop if gap is not None else None)
            if event_arrays is not None:
                stride = event_steps_per_bin(args.dt, args.event_dt)
                bin_limit = min(args.n_event_steps, (steps - 1) // stride + 1)
                retained = sum(int((a[:, 0] < bin_limit).sum()) for a in event_arrays)
                input_count = sum(r['input_spikes'] for r in records)
                row.update(event_rows_outside_window=input_info['raw_events'] - retained,
                           duplicate_event_rows=retained - input_count, input_spikes=input_count,
                           silent_tail_steps=max(0, steps - args.n_event_steps * stride))
            if args.latency:
                elapsed = energy_monitor.latency_seconds
                row.update(elapsed_seconds=elapsed, latency_seconds=elapsed,
                           samples_per_second=count / elapsed, amortized_latency_per_sample_s=elapsed / count)
            if args.energy:
                row.update(energy_monitor.result)
                row['energy_per_sample_j'] = row['total_energy_j'] / count
                energy_path = output / f'energy_{steps}steps.csv'
                energy_monitor.save(energy_path)
                files[f'energy_{steps}steps'] = str(energy_path)
            metrics.append(row)
            prediction_path = output / f'predictions_{steps}steps.csv'
            write_csv(prediction_path, records)
            files[f'predictions_{steps}steps'] = str(prediction_path)
            confusion = torch.bincount(labels * 10 + predictions, minlength=100).reshape(10, 10)
            confusion_path = output / f'confusion_{steps}steps.csv'
            write_csv(confusion_path, [dict(label=i, **{f'pred_{j}': int(confusion[i, j]) for j in range(10)}) for i in range(10)])
            files[f'confusion_{steps}steps'] = str(confusion_path)
            print_section(f'{steps}-Step Result', list(row.items()), logger=logger)
        metrics_path = output / 'metrics.csv'
        write_csv(metrics_path, metrics)
        files['metrics'] = str(metrics_path)
        if args.energy:
            energy_path = output / 'energy_summary.csv'
            fields = ('timesteps', 'duration_ms', 'samples', 'snn_accuracy', 'avg_power_w',
                      'latency_seconds', 'total_energy_j', 'edp_js', 'energy_per_sample_j',
                      'amortized_latency_per_sample_s', 'energy_samples', 'energy_sampling_warning')
            write_csv(energy_path, [{key: row[key] for key in fields} for row in metrics])
            files['energy_summary'] = str(energy_path)
            print_section('Energy Summary', [
                *[(f'{r["timesteps"]} steps',
                   f'{r["avg_power_w"]:.3f} W | {r["latency_seconds"]:.6f} s | '
                   f'{r["total_energy_j"]:.3f} J | {r["edp_js"]:.3f} J*s') for r in metrics],
                ('Columns', 'average power | total latency | energy | EDP'),
                ('Energy data', energy_path),
            ], logger=logger)
        files['log'] = str(output / 'emulation_output.txt')
        summary = dict(status='complete', snn_model=str(args.model_path), snn_sha256=file_sha256(args.model_path),
                       ann_model=str(source_path) if model is not None else None, ann_accuracy=ann_accuracy,
                       dataset='MNIST', split='test', sample_offset=args.sample_offset, samples=count,
                       calibration_overlap=overlap, normalization=normalization, device=str(device),
                       bindsnet_version=importlib.metadata.version('bindsnet'), torch_version=str(torch.__version__),
                       python_version=sys.version.split()[0], command=shlex.join(sys.argv),
                       config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                       data_dir=str(data_dir), input=input_info, readout='spikes * threshold + final V - initial V',
                       propagation='synchronous', execution_backend=EXECUTION_BACKEND,
                       torch_num_threads=torch.get_num_threads(),
                       torch_num_interop_threads=torch.get_num_interop_threads(), metrics=metrics, files=files)
        if args.energy:
            summary['energy_measurement'] = energy_monitor.metadata
        elif args.latency:
            summary['latency_measurement'] = {key: energy_monitor.metadata[key] for key in ('timebase', 'window')}
        summary_path = output / 'emulation_summary.json'
        with summary_path.open('w', encoding='utf-8') as file:
            json.dump(summary, file, indent=2)
        print_section('Emulation Summary', [
            ('ANN accuracy', f'{ann_accuracy:.4f}%' if ann_accuracy is not None else 'not evaluated'),
            *[(f'{r["timesteps"]} steps', f'SNN {r["snn_accuracy"]:.4f}%') for r in metrics],
            ('Summary', summary_path), ('Output log', output / 'emulation_output.txt'),
        ], logger=logger)
    except BaseException:
        print('Emulation failed. Traceback follows.')
        traceback.print_exc()
        raise
    finally:
        try:
            if energy_monitor is not None:
                energy_monitor.close()
        finally:
            logger.close()


if __name__ == '__main__':
    main()
