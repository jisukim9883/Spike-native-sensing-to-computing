#!/usr/bin/env python3
"""Convert a trained MNIST LeNet-5 ANN for use with Neu+ hardware.

Created by Daeyoung Kim and Jongkil Park.

The purpose of this conversion is to prepare the trained model for Neu+
hardware experiments. ANN layers are parsed, optionally normalized, and
compiled into explicit neurons, synaptic connections, weights, and thresholds
for the Neu+ experiment workflow. The same converted SNN is saved as a
*_snn.pth bundle that emulation.py can load directly for GPU emulation before
hardware experiments. This script does not program or run the Neu+ hardware.

Quick start from the repository root::

    python3 simulation/conversion.py
    python3 simulation/conversion.py --model-path /path/to/model/MNIST_LENET5_best.pth

You may supply a plain best/last state_dict, a training checkpoint, or a run
folder containing exactly one best model. Bias is inferred from the weights.
Without options, DEFAULT_MODEL_PATH selects the retained ANN from
results/26-09-09/15-58-15. It does not search for the latest training run.
To convert a newly trained model, pass its printed best-weight path using
--model-path or edit DEFAULT_MODEL_PATH. CLI options override the defaults.

Parser and NetworkCompiler are defined in this file, below the editable
defaults. The conversion pipeline is:
load -> validate ANN -> parse -> calibrate/normalize -> compile -> save.
Defaults match the retained conversion: compile the original weights without
normalization and evaluate all 10,000 MNIST test images on CUDA GPU 0, using
batch size 64 and local data. Optional --normalize uses the first 1,000 test
images (ToTensor) for calibration, but changes the retained experiment.

Results go to the source run's snn/YY-MM-DD/HH-MM-SS directory. Every conversion
gets its own directory. Input files are never modified. The output includes:

* <weight_stem>_snn.pth: versioned tensor/dictionary bundle, loadable with
  torch.load(path, map_location="cpu", weights_only=True).
* <weight_stem>_neurons.pkl, _synapses.pkl, _neuron_thresholds.pkl: the existing
  simulator's sparse connection format; bias files are added when necessary.
* <weight_stem>_parsed.pth: the parsed/normalized ANN state_dict (bias=True).
* conversion_summary.json and conversion_output.txt: source fingerprint,
  settings, ANN accuracy checks, normalization scales, and generated paths.

The SNN bundle stores structure and parameters, not a pickled Python model.
It does not run a spiking simulation or report SNN accuracy. Timestep, input
encoding, and neuron dynamics are selected by the subsequent simulator.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shlex
import sys
import time
import traceback
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.modules.utils import _pair
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

if __package__:
    from .training import LeNet5
    from .utils import (TeeLogger, create_timestamp_directory, file_sha256,
                        print_section, resolve_mnist_directory, save_pickle, seed_all, select_device)
else:
    from training import LeNet5
    from utils import (TeeLogger, create_timestamp_directory, file_sha256,
                       print_section, resolve_mnist_directory, save_pickle, seed_all, select_device)


SCRIPT_DIR = Path(__file__).resolve().parent

# Experiment defaults match the retained run; --model-path selects another ANN.
DEFAULT_MODEL_PATH = SCRIPT_DIR / "results/26-09-09/15-58-15/model/MNIST_LENET5_best.pth"
DEFAULT_OUTPUT_DIR = None  # None creates a dated folder inside the source run/snn.
DEFAULT_DATA_DIR = SCRIPT_DIR / "data/reference_mnist"  # Training metadata is preferred when present.
DEFAULT_NORMALIZE = False
DEFAULT_CALIBRATION_SAMPLES = 1000
DEFAULT_NORM_PERCENTILE = 99.9
DEFAULT_EVAL_SAMPLES = 0  # 0 evaluates the full MNIST test set.
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 0
DEFAULT_DEVICE = "cuda"
DEFAULT_GPU_ID = 0
DEFAULT_DOWNLOAD = False
DEFAULT_SEED = 42
INPUT_SHAPE = (1, 1, 28, 28)


class Parser:
    """Parse a private model copy, preserving the original training weights.

    Only the sequential Conv2d, ReLU, AvgPool2d, Flatten and Linear layers used
    by training.LeNet5 are supported. Call parse() before normalization.
    """

    def __init__(self, input_model, input_shape=(1, 1, 28, 28)):
        self.input_model = copy.deepcopy(input_model).eval()
        self.input_shape = tuple(input_shape)
        self.parsed_layers = []
        self.layer_thresholds = {}
        self.layer_cumul = {}
        self.normalization_factors = {}
        self._normalized = False

    def parse(self):
        """Capture execution order and inject zero biases for uniform handling."""
        ordered, handles = [], []
        names = {id(module): name for name, module in self.input_model.named_modules()}
        supported = (nn.Conv2d, nn.ReLU, nn.AvgPool2d, nn.Flatten, nn.Linear)
        try:
            for module in self.input_model.modules():
                if not list(module.children()):
                    if not isinstance(module, supported):
                        raise ValueError(f"Unsupported layer: {names[id(module)]} ({type(module).__name__})")
                    handles.append(module.register_forward_hook(
                        lambda layer, inputs, outputs: ordered.append((names[id(layer)], layer))
                    ))
            parameter = next(self.input_model.parameters())
            with torch.no_grad():
                self.input_model(torch.zeros(self.input_shape, device=parameter.device, dtype=parameter.dtype))
        finally:
            for handle in handles:
                handle.remove()
        if len({name for name, _ in ordered}) != len(ordered):
            raise ValueError("Shared/repeated layers are not supported.")
        self.parsed_layers = ordered
        for name, module in ordered:
            if isinstance(module, (nn.Conv2d, nn.Linear)) and module.bias is None:
                size = module.out_channels if isinstance(module, nn.Conv2d) else module.out_features
                module.bias = nn.Parameter(module.weight.new_zeros(size))
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.AvgPool2d)):
                self.layer_thresholds[name] = 1.0
        print(f"Parsed {len(ordered)} layers in execution order.")
        return ordered

    @torch.no_grad()
    def evaluate(self, loader, device="cpu"):
        """Return ANN classification accuracy in percent, not SNN accuracy."""
        self.input_model.to(device).eval()
        correct = total = 0
        for data, labels in loader:
            outputs = self.input_model(data.to(device))
            correct += (outputs.argmax(1).cpu() == labels.cpu()).sum().item()
            total += labels.numel()
        if total == 0:
            raise ValueError("Evaluation requires at least one sample.")
        return 100.0 * correct / total

    @torch.no_grad()
    def normalize_weights(self, loader, percentile=99.9, device="cpu"):
        """Normalize using every sample in the supplied calibration loader.

        Scales are measured after upstream layers have been normalized. The
        final signed logits share one positive scale, preserving their ranking
        up to rounding. A percentile below 100 is a robust estimate, not a
        guarantee that every normalized activation is at most one.
        """
        if not self.parsed_layers:
            raise RuntimeError("Call parse() before normalize_weights().")
        if self._normalized:
            raise RuntimeError("This model has already been normalized.")
        if not np.isfinite(percentile) or not 0 < percentile <= 100:
            raise ValueError("Normalization percentile must be in (0, 100].")
        self.input_model.to(device).eval()
        cumulative = 1.0
        for index, (name, module) in enumerate(self.parsed_layers):
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if module.bias is not None:
                    module.bias.div_(cumulative)
                has_relu = index + 1 < len(self.parsed_layers) and isinstance(
                    self.parsed_layers[index + 1][1], nn.ReLU
                )
                scale = 1.0
                if has_relu:
                    maxima = []
                    for data, _ in loader:
                        activation = data.to(device)
                        for _, layer in self.parsed_layers[:index + 2]:
                            activation = layer(activation)
                        maxima.append(activation.flatten(1).amax(1).cpu().numpy())
                    if not maxima:
                        raise ValueError("Normalization requires calibration samples.")
                    scale = float(np.percentile(np.concatenate(maxima), percentile))
                    if not np.isfinite(scale):
                        raise ValueError(f"Non-finite normalization scale in {name}.")
                    if scale <= 0:
                        scale = 1.0
                    module.weight.div_(scale)
                    if module.bias is not None:
                        module.bias.div_(scale)
                    cumulative *= scale
                self.normalization_factors[name] = scale
                print(f"  {name:<12} scale={scale:.6g}  cumulative={cumulative:.6g}")
            self.layer_cumul[name] = cumulative
        self._normalized = True
        return self.layer_thresholds

class NetworkCompiler:
    """Build and save Conv, Pool, and Linear connections for training.LeNet5.

    Call setup_layers() to create the arrays, then build() to sort and save.
    Both no-bias and bias-trained models are handled automatically. Residual
    networks, pooling absorption, and partial conversion are not supported.
    """

    CORE_SIZE = 1024

    def __init__(self, parsed_layers, input_shape, save_path, model_name,
                 layer_thresholds=None):
        self.parsed_layers = parsed_layers
        self.input_shape = tuple(input_shape)
        self.save_path = Path(save_path)
        self.model_name = model_name
        self.neurons, self.synapses = {}, {}
        self.bias_neurons, self.bias_synapses = {}, {}
        self.neuron_thresholds = {}
        self.layer_bases, self.layer_shapes = {}, {}
        self.layer_thresholds = layer_thresholds or {}
        self.files = {}

    def setup_layers(self):
        """Expand operations into point connections with aligned global indices."""
        if self.neurons:
            raise RuntimeError("This compiler has already built its layers.")
        shape = self.input_shape[1:]
        source_base = 0
        self.neurons['input_layer'] = int(np.prod(shape))
        self.layer_bases['input_layer'] = 0
        self.layer_shapes['input_layer'] = list(shape)
        for name, module in self.parsed_layers:
            if isinstance(module, nn.ReLU):
                continue
            if isinstance(module, nn.Flatten):
                shape = (int(np.prod(shape)),)
                continue
            if not isinstance(module, (nn.Conv2d, nn.AvgPool2d, nn.Linear)):
                raise ValueError(f"Unsupported compiler layer: {name}")
            if isinstance(module, nn.Conv2d) and (
                module.groups != 1 or module.dilation != (1, 1) or
                module.padding_mode != "zeros" or isinstance(module.padding, str)
            ):
                raise ValueError("Only standard zero-padded LeNet-5 convolutions are supported.")
            if isinstance(module, nn.AvgPool2d) and (
                _pair(module.padding) != (0, 0) or module.ceil_mode or module.divisor_override is not None
            ):
                raise ValueError("Only unpadded LeNet-5 average pooling is supported.")
            parameter = next(module.parameters(), None)
            device = parameter.device if parameter is not None else torch.device('cpu')
            with torch.no_grad():
                output_shape = tuple(module(torch.zeros((1, *shape), device=device)).shape[1:])
            source, target, weights = self._connections(module, shape, output_shape)
            target_base = ((source_base + int(np.prod(shape)) - 1) // self.CORE_SIZE + 1) * self.CORE_SIZE
            self.neurons[name] = int(np.prod(output_shape))
            self.layer_bases[name] = target_base
            self.layer_shapes[name] = list(output_shape)
            self.neuron_thresholds[name] = float(self.layer_thresholds.get(name, 1.0))
            self.synapses[name] = [source + source_base, target + target_base, weights]
            if isinstance(module, (nn.Conv2d, nn.Linear)) and module.bias is not None:
                bias = module.bias.detach().cpu().numpy()
                if isinstance(module, nn.Conv2d):
                    bias = np.repeat(bias, int(np.prod(output_shape[1:])))
                if np.any(bias != 0):
                    self.bias_neurons[name] = len(bias)
                    self.bias_synapses[name] = bias.copy()
            print(f"  {name:<12} neurons={self.neurons[name]:>5,}  synapses={len(source):>7,}")
            shape, source_base = output_shape, target_base

        bias_base = ((source_base + int(np.prod(shape)) - 1) // self.CORE_SIZE + 1) * self.CORE_SIZE
        for index, (name, weights) in enumerate(self.bias_synapses.items()):
            bias_name = f"bias_{name}"
            size = len(weights)
            self.neurons[bias_name] = 1
            self.layer_bases[bias_name] = bias_base + index
            self.layer_shapes[bias_name] = [1]
            self.synapses[bias_name] = [
                np.full(size, bias_base + index, dtype=np.int64),
                np.arange(size, dtype=np.int64) + self.layer_bases[name],
                weights.astype(np.float64),
            ]

    @staticmethod
    def _connections(module, shape, output_shape):
        """Vectorize the source/target loops of the reference compiler."""
        if isinstance(module, nn.Linear):
            source = np.repeat(np.arange(module.in_features, dtype=np.int64), module.out_features)
            target = np.tile(np.arange(module.out_features, dtype=np.int64), module.in_features)
            weights = module.weight.detach().cpu().numpy().T.reshape(-1).astype(np.float64)
            return source, target, weights
        channels, height, width = shape
        _, out_h, out_w = output_shape
        kh, kw = _pair(module.kernel_size)
        sh, sw = _pair(module.stride or module.kernel_size)
        if isinstance(module, nn.Conv2d):
            oh, ow, oc, ky, kx, ic = np.indices(
                (out_h, out_w, module.out_channels, kh, kw, channels), dtype=np.int64
            ).reshape(6, -1)
            ph, pw = _pair(module.padding)
            ih, iw = oh * sh + ky - ph, ow * sw + kx - pw
            valid = (ih >= 0) & (ih < height) & (iw >= 0) & (iw < width)
            source = ic * height * width + ih * width + iw
            target = oc * out_h * out_w + oh * out_w + ow
            weights = module.weight.detach().cpu().numpy()[oc, ic, ky, kx]
            return source[valid], target[valid], weights[valid].astype(np.float64)
        oh, ow, ky, kx, channel = np.indices(
            (out_h, out_w, kh, kw, channels), dtype=np.int64
        ).reshape(5, -1)
        source = channel * height * width + (oh * sh + ky) * width + ow * sw + kx
        target = channel * out_h * out_w + oh * out_w + ow
        return source, target, np.full(len(source), 1.0 / (kh * kw), dtype=np.float64)

    def build(self):
        """Save compatible neuron, synapse, threshold, and optional bias pickles."""
        if not self.neurons:
            raise RuntimeError("Call setup_layers() before build().")
        self.save_path.mkdir(parents=True, exist_ok=True)
        for name, (source, target, weights) in self.synapses.items():
            order = np.argsort(source)
            self.synapses[name] = [source[order], target[order], weights[order]]
        artifacts = {'neurons': self.neurons, 'synapses': self.synapses,
                     'neuron_thresholds': self.neuron_thresholds}
        if self.bias_neurons:
            artifacts.update(bias_neurons=self.bias_neurons, bias_synapses=self.bias_synapses)
        for suffix, value in artifacts.items():
            self.files[suffix] = save_pickle(self.save_path / f"{self.model_name}_{suffix}.pkl", value)
        return self.neurons, self.synapses, self.bias_neurons, self.bias_synapses

    def state_dict(self):
        """Return a portable bundle that torch.load accepts with weights_only=True."""
        return {
            'format_version': 1, 'architecture': 'LeNet5', 'dataset': 'MNIST',
            'input_shape': list(self.input_shape), 'core_size': self.CORE_SIZE,
            'neurons': self.neurons,
            'synapses': {name: [torch.from_numpy(a.copy()) for a in arrays]
                         for name, arrays in self.synapses.items()},
            'neuron_thresholds': self.neuron_thresholds,
            'bias_neurons': self.bias_neurons,
            'bias_synapses': {name: torch.from_numpy(a.copy()) for name, a in self.bias_synapses.items()},
            'layer_bases': self.layer_bases, 'layer_shapes': self.layer_shapes,
        }


def resolve_model_path(path: Path) -> Path:
    """Resolve a weight file or select the unique best weight in a run folder.

    Ambiguous directories fail with candidate names rather than guessing which
    model the experiment intended. File paths always select that exact file.
    """
    path = path.expanduser().resolve(strict=True)
    if path.is_file():
        return path
    model_dir = path / "model" if (path / "model").is_dir() else path
    candidates = sorted(model_dir.glob("*_best.pth"))
    if not candidates:
        candidates = sorted([*model_dir.glob("*.pth"), *model_dir.glob("*.pt")])
    if len(candidates) != 1:
        names = ", ".join(p.name for p in candidates) or "none"
        raise ValueError(f"Provide an exact model file; candidates in {model_dir}: {names}")
    return candidates[0].resolve()


def source_run_directory(path: Path) -> Path:
    """Find the experiment directory for model/ or checkpoints/ weight files."""
    return path.parent.parent if path.parent.name in ("model", "checkpoints") else path.parent


def load_model(path: Path):
    """Load plain weights or a training checkpoint and infer the bias setting.

    Only tensor state_dict data are accepted. Strict LeNet5 loading rejects
    incomplete weights, incompatible architectures, and mismatched dimensions.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("Expected a state_dict or a training checkpoint dictionary.")
    checkpoint = "model_state_dict" in payload
    weights = payload["model_state_dict"] if checkpoint else payload
    if not isinstance(weights, Mapping) or not weights:
        raise ValueError("The model state_dict is empty or invalid.")
    for name, value in weights.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError("The input is not an ANN tensor state_dict.")
        if not torch.isfinite(value).all():
            raise ValueError(f"Non-finite trained weights: {name}")
    bias = any(name.endswith(".bias") for name in weights)
    model = LeNet5(bias=bias)
    model.load_state_dict(weights, strict=True)
    config = payload.get("args", {}) if checkpoint else {}
    return model.eval(), bias, dict(config), "checkpoint" if checkpoint else "state_dict"


def resolve_data_directory(args, run_dir, checkpoint_config):
    """Prefer CLI, checkpoint, then summary paths; relocate the old MNIST root."""
    if args.data_dir is not None:
        return resolve_mnist_directory(args.data_dir)
    config = checkpoint_config
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        with summary_path.open() as file:
            config = {**json.load(file).get("config", {}), **config}
    path = Path(config.get("data_dir", DEFAULT_DATA_DIR)).expanduser()
    if not path.is_absolute():
        path = SCRIPT_DIR.parent / path
    return resolve_mnist_directory(path)


def build_loaders(args, data_dir):
    """Use deterministic, exact-sized test subsets for calibration and ANN checks."""
    dataset = datasets.MNIST(data_dir, train=False, download=args.download,
                             transform=transforms.ToTensor())
    eval_count = args.eval_samples or len(dataset)
    calibration_count = args.calibration_samples if args.normalize else 0
    if eval_count > len(dataset) or calibration_count > len(dataset):
        raise ValueError(f"Requested more samples than the MNIST test set ({len(dataset)}).")
    options = dict(batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    evaluation = DataLoader(Subset(dataset, range(eval_count)), **options)
    calibration = DataLoader(Subset(dataset, range(calibration_count)), **options)
    return evaluation, calibration


def parse_args():
    """Read overrides for the top-of-file defaults and check basic conditions."""
    parser = argparse.ArgumentParser(description="Convert a trained MNIST LeNet-5 model into SNN files.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH,
                        help="Best/last ANN weights, a checkpoint, or a run/model directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Output base directory; default: the source run's snn directory.")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="MNIST root; default: training metadata, then simulation/data/reference_mnist.")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=DEFAULT_NORMALIZE)
    parser.add_argument("--calibration-samples", "--data-size", type=int, default=DEFAULT_CALIBRATION_SAMPLES)
    parser.add_argument("--norm-percentile", type=float, default=DEFAULT_NORM_PERCENTILE)
    parser.add_argument("--eval-samples", type=int, default=DEFAULT_EVAL_SAMPLES,
                        help="ANN validation sample count; 0 uses all 10,000 test images.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=DEFAULT_DEVICE)
    parser.add_argument("--gpu-id", type=int, default=DEFAULT_GPU_ID)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=DEFAULT_DOWNLOAD)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.model_path is None:
        parser.error("Set DEFAULT_MODEL_PATH or pass --model-path.")
    if args.batch_size < 1 or args.calibration_samples < 1:
        parser.error("Batch size and calibration sample count must be positive.")
    if min(args.num_workers, args.eval_samples, args.gpu_id) < 0:
        parser.error("Worker count, evaluation sample count, and GPU index must be non-negative.")
    if not math.isfinite(args.norm_percentile) or not 0 < args.norm_percentile <= 100:
        parser.error("--norm-percentile must be in (0, 100].")
    try:
        args.model_path = resolve_model_path(Path(args.model_path))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return args


def main():
    """Convert one selected ANN and save the complete SNN artifact and audit log."""
    args = parse_args()
    source_path = args.model_path
    run_dir = source_run_directory(source_path)
    output_base = Path(args.output_dir).expanduser().resolve() if args.output_dir else run_dir / "snn"
    output = create_timestamp_directory(output_base)
    logger = TeeLogger(output / "conversion_output.txt")
    started = time.monotonic()
    try:
        print_section("Conversion Conditions", [
            ("Source model", source_path), ("Source run", run_dir),
            ("Output directory", output), ("Command", shlex.join(sys.argv)),
            ("Architecture", "LeNet5 / MNIST"), ("Input shape", INPUT_SHAPE),
            ("Normalization", args.normalize), ("Percentile", args.norm_percentile),
            ("Calibration samples", args.calibration_samples if args.normalize else 0),
            ("Evaluation samples", args.eval_samples or "full test set"),
            ("Batch size", args.batch_size), ("Workers", args.num_workers),
            ("Download", args.download), ("Device requested", args.device),
            ("GPU index", args.gpu_id), ("Seed", args.seed),
        ], logger=logger)
        seed_all(args.seed)
        device = select_device(args.device, args.gpu_id)
        fingerprint = file_sha256(source_path)
        print("[1/5] Loading trained weights and MNIST...")
        model, bias, checkpoint_config, source_kind = load_model(source_path)
        model.to(device)
        data_dir = resolve_data_directory(args, run_dir, checkpoint_config)
        print_section("Resolved Model And Data", [
            ("Source format", source_kind), ("Source SHA256", fingerprint),
            ("Bias", bias), ("Device", device), ("Data directory", data_dir),
            ("Input transform", "ToTensor, [0, 1]"), ("Calibration split", "MNIST test, first N samples"),
        ], logger=logger)
        evaluation, calibration = build_loaders(args, data_dir)
        print("[2/5] Parsing and validating the ANN...")
        original_accuracy = Parser(model).evaluate(evaluation, device)
        parser = Parser(model, INPUT_SHAPE)
        parser.parse()
        parsed_accuracy = parser.evaluate(evaluation, device)
        if parsed_accuracy != original_accuracy:
            raise RuntimeError("Parsing changed ANN accuracy; conversion was stopped.")
        normalized_accuracy = None
        print("[3/5] Normalizing weights..." if args.normalize else "[3/5] Normalization disabled.")
        if args.normalize:
            parser.normalize_weights(calibration, args.norm_percentile, device)
            normalized_accuracy = parser.evaluate(evaluation, device)
        print_section("ANN Validation", [
            ("Original accuracy", f"{original_accuracy:.2f}%"),
            ("Parsed accuracy", f"{parsed_accuracy:.2f}%"),
            ("Normalized accuracy", f"{normalized_accuracy:.2f}%" if normalized_accuracy is not None else "disabled"),
        ], logger=logger)
        print("[4/5] Compiling SNN neurons and sparse synapses...")
        parser.input_model.cpu()
        compiler = NetworkCompiler(parser.parsed_layers, INPUT_SHAPE, output, source_path.stem,
                                   layer_thresholds=parser.layer_thresholds)
        compiler.setup_layers()
        compiler.build()
        print("[5/5] Saving SNN model and conversion metadata...")
        normalization = {
            "enabled": args.normalize, "percentile": args.norm_percentile,
            "samples": len(calibration.dataset), "split": "MNIST test",
            "factors": parser.normalization_factors, "cumulative_factors": parser.layer_cumul,
        }
        snn_path = output / f"{source_path.stem}_snn.pth"
        bundle = compiler.state_dict()
        bundle.update(source_weights=str(source_path), source_sha256=fingerprint,
                      normalization=normalization, source_bias=bias, readout_layer="lin_out")
        torch.save(bundle, snn_path)
        parsed_path = output / f"{source_path.stem}_parsed.pth"
        torch.save(parser.input_model.state_dict(), parsed_path)
        files = {key: str(path) for key, path in compiler.files.items()}
        files.update(snn_model=str(snn_path), parsed_ann=str(parsed_path),
                     log=str(output / "conversion_output.txt"))
        summary = {
            "status": "complete", "format_version": 1,
            "source_weights": str(source_path), "source_sha256": fingerprint,
            "source_format": source_kind, "architecture": "LeNet5", "dataset": "MNIST",
            "bias": bias, "parsed_ann_bias": True, "input_shape": list(INPUT_SHAPE),
            "data_dir": str(data_dir), "device": str(device), "seed": args.seed,
            "batch_size": args.batch_size, "num_workers": args.num_workers, "download": args.download,
            "evaluation_samples": len(evaluation.dataset), "normalization": normalization,
            "ann_accuracy": {"original": original_accuracy, "parsed": parsed_accuracy,
                             "normalized": normalized_accuracy},
            "snn_accuracy": None, "neurons": compiler.neurons,
            "synapse_counts": {name: len(arrays[0]) for name, arrays in compiler.synapses.items()},
            "neuron_thresholds": compiler.neuron_thresholds,
            "elapsed_seconds": time.monotonic() - started, "files": files,
        }
        summary_path = output / "conversion_summary.json"
        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)
        print_section("Conversion Summary", [
            ("SNN model", snn_path), ("Neuron count", sum(compiler.neurons.values())),
            ("Synapse count", sum(summary["synapse_counts"].values())),
            ("Bias sources", len(compiler.bias_neurons)), ("Neuron thresholds", "1.0"),
            ("Elapsed", f"{summary['elapsed_seconds']:.2f} s"),
            ("Summary", summary_path), ("Output log", files["log"]),
        ], logger=logger)
    except BaseException:
        print("Conversion failed. Traceback follows.")
        traceback.print_exc()
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    main()
