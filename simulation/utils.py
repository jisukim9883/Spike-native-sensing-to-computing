"""Shared helpers for scripts in the simulation directory.

Created by Daeyoung Kim and Jongkil Park.

Keep general-purpose logging, experiment-directory, and reproducibility helpers
here so training or inference scripts can reuse the same behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import random
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import monotonic, sleep

import torch
from torch import nn

try:
    from rich import box
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn
    from rich.table import Table
    from rich.text import Text
except ImportError:
    RICH_AVAILABLE = False
else:
    RICH_AVAILABLE = True


@dataclass(frozen=True)
class RunDirectories:
    """Directories and timestamp assigned to one simulation run."""

    run_dir: Path
    model_dir: Path
    checkpoint_dir: Path
    date: str
    time: str


class TeeLogger:
    """Mirror stdout and stderr to a terminal and a UTF-8 text log file.

    Use this as a short-lived object around an experiment. Calling ``close()``
    restores the original streams even after a training error. Rich displays
    use ``terminal`` directly and send permanent records to ``write_log``;
    this keeps animation frames and terminal control codes out of the log.
    """

    class _Tee:
        def __init__(self, stream, log_file) -> None:
            self.stream = stream
            self.log_file = log_file

        def write(self, data: str) -> int:
            written = self.stream.write(data)
            self.log_file.write(data)
            self.log_file.flush()
            return written

        def flush(self) -> None:
            self.stream.flush()
            self.log_file.flush()

        def __getattr__(self, name: str):
            return getattr(self.stream, name)

    def __init__(self, log_path: Path) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = log_path.open("w", encoding="utf-8")
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = self._Tee(self._stdout, self._log_file)
        sys.stderr = self._Tee(self._stderr, self._log_file)

    @property
    def terminal(self):
        """Return the original stdout for transient terminal-only rendering."""
        return self._stdout

    def write_log(self, text: str) -> None:
        """Append a permanent plain-text record without redrawing the terminal."""
        self._log_file.write(text)
        self._log_file.flush()

    def close(self) -> None:
        """Restore stdout/stderr and close the associated log file."""
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        self._log_file.close()


def serializable_args(args: argparse.Namespace) -> dict[str, object]:
    """Return command-line arguments in a JSON-compatible representation."""
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def resolve_mnist_directory(path: Path) -> Path:
    """Resolve an MNIST root, including the former project-local data path.

    Historical commands and model metadata used simulation/data. Only that
    exact directory is redirected; custom dataset roots are never replaced.
    torchvision still stores its files under MNIST/raw within the new root.
    """
    resolved = path.expanduser().resolve()
    legacy_root = Path(__file__).resolve().parent / "data"
    if resolved == legacy_root:
        return legacy_root / "reference_mnist"
    return resolved


def seed_all(seed: int) -> None:
    """Set CPU and CUDA random seeds for a reproducible training run."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(device_arg: str, gpu_id: int) -> torch.device:
    """Resolve ``auto``, ``cpu``, or ``cuda`` to the PyTorch device to use."""
    if device_arg == "cpu":
        return torch.device("cpu")

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device(f"cuda:{gpu_id}")

    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def create_run_directories(output_dir: Path) -> RunDirectories:
    """Create ``YY-MM-DD/HH-MM-SS`` result folders for one run."""
    now = datetime.now()
    date = now.strftime("%y-%m-%d")
    time = now.strftime("%H-%M-%S")
    run_dir = output_dir / date / time
    model_dir = run_dir / "model"
    checkpoint_dir = run_dir / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return RunDirectories(
        run_dir=run_dir,
        model_dir=model_dir,
        checkpoint_dir=checkpoint_dir,
        date=date,
        time=time,
    )


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return the total and trainable parameter counts of a model."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def print_section(
    title: str,
    rows: list[tuple[str, object]],
    logger: TeeLogger | None = None,
) -> None:
    """Show a styled terminal section and preserve unwrapped values in the log.

    Piped output and environments without Rich receive plain text. Values are
    rendered literally, including paths or run names containing brackets.
    """
    plain = "\n".join([title, "-" * 72, *(f"{key:<24}: {value}" for key, value in rows), ""]) + "\n"
    console = Console(file=logger.terminal, highlight=False) if logger and RICH_AVAILABLE else None
    if console is None or not console.is_terminal:
        print(plain, end="")
        return

    logger.write_log(plain)
    table = Table.grid(padding=(0, 2), expand=True)
    table.add_column(style="dim", no_wrap=True)
    table.add_column(ratio=1, overflow="fold")
    for key, value in rows:
        table.add_row(Text(key), Text(str(value)))
    console.print(Panel(table, title=Text(title, style="bold"), border_style="cyan", box=box.SQUARE))
    console.print()


class TrainingDisplay:
    """Display batch progress and recent epochs while logging every epoch once.

    Use as a context manager around the training loop. ``start_epoch`` and
    ``start_phase`` reset the counters, ``update_batch`` advances the active
    train/test phase, and ``finish_epoch`` records the completed result.
    The Live screen writes directly to the terminal; all completed metrics
    go to the text log even when they scroll out of the recent-epoch table.
    Non-interactive output, disabled progress, or missing Rich use plain lines.
    """

    def __init__(
        self,
        logger: TeeLogger,
        epochs: int,
        enabled: bool = True,
        history_rows: int = 10,
        refresh_per_second: float = 4,
    ) -> None:
        self.logger = logger
        self.epochs = epochs
        self.rows: deque = deque(maxlen=history_rows)
        self.best_accuracy = 0.0
        self.best_epoch = 0
        self.epoch = 0
        self.phase = "Preparing"
        self.last_epoch_seconds = None
        self.started_at = monotonic()
        self.console = Console(file=logger.terminal, highlight=False) if RICH_AVAILABLE else None
        self.live = None
        if enabled and self.console is not None and self.console.is_terminal:
            self.progress = Progress(
                TextColumn("[bold cyan]{task.description}"),
                BarColumn(bar_width=None, complete_style="cyan", finished_style="green"),
                MofNCompleteColumn(),
                TextColumn("[dim]ETA"),
                TimeRemainingColumn(),
                console=self.console,
                auto_refresh=False,
                expand=True,
            )
            self.epoch_task = self.progress.add_task("Epochs", total=epochs)
            self.batch_task = self.progress.add_task("Batch", total=1, start=False)
            self.live = Live(
                console=self.console,
                get_renderable=self._render,
                refresh_per_second=refresh_per_second,
                transient=True,
                redirect_stdout=False,
                redirect_stderr=False,
            )

    def __enter__(self) -> TrainingDisplay:
        self.started_at = monotonic()
        if self.live is not None:
            self.live.start(refresh=True)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.live is not None:
            self.live.stop()
            if self.rows:
                self.console.print(self._results_table())
            if exc_type is not None:
                self.logger.write_log("Training interrupted.\n")
                self.console.print(Text("Training interrupted.", style="bold yellow"))

    def start_epoch(self, epoch: int) -> None:
        """Set the epoch number before the training and evaluation phases."""
        self.epoch = epoch

    def start_phase(self, phase: str, total_batches: int) -> None:
        """Reset the batch counter for either the train or test loader."""
        self.phase = phase
        if self.live is not None:
            self.progress.reset(self.batch_task, total=total_batches, description=phase)
            self.live.refresh()

    def update_batch(self, completed: int) -> None:
        """Update a cheap counter; Rich redraws at the configured refresh rate."""
        if self.live is not None:
            self.progress.update(self.batch_task, completed=completed)

    def finish_epoch(
        self,
        epoch: int,
        lr: float,
        train_loss: float,
        test_loss: float,
        accuracy: float,
        seconds: float,
        is_best: bool,
    ) -> None:
        """Keep the latest table row and persist a complete epoch summary."""
        if is_best:
            self.best_accuracy = accuracy
            self.best_epoch = epoch
        self.last_epoch_seconds = seconds
        self.rows.append((epoch, lr, train_loss, test_loss, accuracy, seconds, is_best))
        message = (
            f"Epoch {epoch:03d}/{self.epochs:03d} | lr {lr:.6f} | "
            f"train_loss {train_loss:.4f} | test_loss {test_loss:.4f} | "
            f"test_acc {accuracy:.2f}% | time {seconds:.2f}s | "
            f"best {self.best_accuracy:.2f}% @ {self.best_epoch}"
            + (" *best" if is_best else "")
        )
        if self.live is None:
            print(message)
        else:
            self.logger.write_log(message + "\n")
            self.progress.update(self.epoch_task, completed=epoch)
            self.phase = "Complete" if epoch == self.epochs else "Epoch complete"
            self.live.refresh()

    def _results_table(self) -> Table:
        compact = self.console.width < 80
        table = Table(box=box.SIMPLE_HEAVY, header_style="bold cyan", expand=True)
        table.add_column("Epoch", justify="right")
        if not compact:
            table.add_column("LR", justify="right", style="dim")
        for label in ("Train loss", "Test loss", "Test acc"):
            table.add_column(label, justify="right")
        if not compact:
            table.add_column("Time", justify="right")
        table.add_column("Best", justify="center")
        for epoch, lr, train_loss, test_loss, accuracy, seconds, is_best in list(self.rows):
            values = [str(epoch)]
            if not compact:
                values.append(f"{lr:.2e}")
            values.extend([f"{train_loss:.4f}", f"{test_loss:.4f}", f"{accuracy:.2f}%"])
            if not compact:
                values.append(f"{seconds:.1f}s")
            values.append("*" if is_best else "")
            table.add_row(*values, style="bold green" if is_best else "")
        return table

    def _render(self) -> Panel:
        elapsed = monotonic() - self.started_at
        best = f"{self.best_accuracy:.2f}% (epoch {self.best_epoch})" if self.best_epoch else "--"
        speed = f"{self.last_epoch_seconds:.1f}s/epoch" if self.last_epoch_seconds is not None else "--"
        status = Text(f"{self.phase} | epoch {self.epoch}/{self.epochs}", style="yellow")
        summary = Text.assemble(
            ("Best ", "dim"), (best, "bold green"),
            ("   Elapsed ", "dim"), (f"{elapsed:.0f}s", "cyan"),
            ("   Speed ", "dim"), (speed, "cyan"),
        )
        return Panel(
            Group(self.progress, status, Text(""), self._results_table(), summary),
            title=Text("Training Progress / MNIST / LeNet-5", style="bold"),
            border_style="cyan",
            box=box.SQUARE,
        )


def create_timestamp_directory(base_dir: Path) -> Path:
    """Reserve a fresh YY-MM-DD/HH-MM-SS directory without replacing a run."""
    for _ in range(30):
        now = datetime.now()
        path = base_dir / now.strftime("%y-%m-%d") / now.strftime("%H-%M-%S")
        try:
            path.mkdir(parents=True, exist_ok=False)
            return path
        except FileExistsError:
            sleep(0.05)
    raise FileExistsError(f"Could not reserve a timestamp directory under {base_dir}")


def file_sha256(path: Path) -> str:
    """Fingerprint the exact source weights used by a saved conversion."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_pickle(path: Path, value) -> Path:
    """Save an array/dictionary artifact compatible with the existing simulator."""
    with path.open("wb") as file:
        pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)
    return path


class EvaluationProgress:
    """Log evaluation accuracy, optionally showing measured speed and ETA."""

    def __init__(self, logger, total, description, enabled=True, measure_time=True):
        self.logger = logger
        self.total = total
        self.description = description
        self.measure_time = measure_time
        self.started = None
        self.progress = None
        if enabled and RICH_AVAILABLE:
            console = Console(file=logger.terminal, highlight=False)
            if console.is_terminal:
                columns = [TextColumn("[bold cyan]{task.description}"),
                           BarColumn(bar_width=None), MofNCompleteColumn(),
                           TextColumn("[green]{task.fields[accuracy]}")]
                if measure_time:
                    columns.extend([TextColumn("[dim]ETA"), TimeRemainingColumn()])
                self.progress = Progress(
                    *columns, console=console, transient=True, expand=True,
                    get_time=monotonic if measure_time else lambda: 0.0,
                    redirect_stdout=False, redirect_stderr=False,
                )
                self.task = self.progress.add_task(description, total=total, accuracy="--")

    def __enter__(self):
        if self.measure_time:
            self.started = monotonic()
        if self.progress is not None:
            self.progress.start()
        return self

    def update(self, completed, snn_accuracy, ann_accuracy=None):
        ann = f"{ann_accuracy:.2f}%" if ann_accuracy is not None else "--"
        message = (f"{self.description} | {completed:5d}/{self.total} | "
                   f"SNN {snn_accuracy:.2f}% | ANN {ann}")
        if self.measure_time:
            elapsed = monotonic() - self.started
            message += f" | {completed / max(elapsed, 1e-9):.1f} samples/s"
        if self.progress is None:
            print(message)
        else:
            self.logger.write_log(message + "\n")
            self.progress.update(self.task, completed=completed, accuracy=f"SNN {snn_accuracy:.2f}%")

    def __exit__(self, exc_type, exc_value, traceback):
        if self.progress is not None:
            self.progress.stop()


def summarize_energy_samples(samples, kind="power"):
    """Return a time-weighted energy summary and cumulative trace.

    Samples are (elapsed_wall_seconds, watts) or (elapsed_wall_seconds,
    unwrapped_joules) for RAPL counters. The first/last samples MUST coincide
    with the latency boundaries. GPU power is integrated by the trapezoidal
    rule; CPU energy comes directly from counter differences. No TDP estimates
    or process-level attribution are applied.
    """
    import math

    if kind not in ("power", "counter") or len(samples) < 2 or samples[0][0] != 0:
        raise ValueError("Energy samples require a zero-time start and an end sample.")
    if any(not math.isfinite(t) or not math.isfinite(v) or t < 0 or v < 0 for t, v in samples):
        raise ValueError("Invalid power/energy sensor reading.")
    if any(b[0] <= a[0] for a, b in zip(samples, samples[1:])):
        raise ValueError("Energy sample times must strictly increase.")
    energy = 0.0
    powers = [samples[0][1]] if kind == "power" else []
    trace = [dict(timestamp_s=0.0, power_w=samples[0][1] if kind == "power" else None,
                  cumulative_energy_j=0.0, cumulative_edp_js=0.0)]
    for (t0, v0), (t1, v1) in zip(samples, samples[1:]):
        if kind == "power":
            power = v1
            energy += (v0 + v1) * 0.5 * (t1 - t0)
        else:
            if v1 < v0:
                raise ValueError("Energy counter decreased after rollover correction.")
            power = (v1 - v0) / (t1 - t0)
            energy = v1 - samples[0][1]
        powers.append(power)
        trace.append(dict(timestamp_s=t1, power_w=power, cumulative_energy_j=energy,
                          cumulative_edp_js=energy * t1))
    latency = samples[-1][0]
    average = energy / latency
    total_energy = average * latency
    summary = dict(latency_seconds=latency, avg_power_w=average,
                   max_sampled_power_w=max(powers), total_energy_j=total_energy,
                   edp_js=total_energy * latency, energy_samples=len(samples))
    return summary, trace


class EnergyMonitor:
    """Measure one evaluation window, optionally sampling GPU/CPU energy.

    GPU: NVML whole-device power, matched by CUDA device UUID (also works with
    CUDA_VISIBLE_DEVICES remapping). CPU: top-level RAPL package counters only,
    without adding overlapping core/DRAM subdomains. Unsupported sensors fail
    explicitly; disabled monitoring still measures synchronized wall latency.

    Power polling uses actual monotonic timestamps, not nominal intervals or
    SNN timesteps. Boundary readings bracket the evaluated code; sensor-read
    and sampling-thread shutdown overhead after the end is excluded. Sensor
    resolution/averaging limits remain even with a shorter polling interval.
    """

    def __init__(self, device="cpu", interval=0.1, enabled=True):
        import math
        import threading

        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("Energy sampling interval must be finite and positive.")
        self.device = torch.device(device)
        self.interval = interval
        self.enabled = enabled
        self.kind = "power"
        self.metadata = dict(enabled=enabled, sampling_interval_s=interval,
                             timebase="wall clock, not simulation time",
                             window="SNN evaluation including input encoding, transfers, and result processing; excludes ANN, warmup, setup, and report files")
        self._stop = threading.Event()
        self._thread = None
        self._close_sensor = lambda: None
        self._read_value = None
        if enabled:
            try:
                self._configure_sensor()
                self._read_checked()
            except Exception:
                self._close_sensor()
                raise

    def _configure_sensor(self):
        import os

        if self.device.type == "cuda":
            try:
                import pynvml
            except ImportError as error:
                raise RuntimeError("GPU energy requires: python3 -m pip install nvidia-ml-py") from error
            pynvml.nvmlInit()
            self._close_sensor = pynvml.nvmlShutdown
            uuid = str(torch.cuda.get_device_properties(self.device).uuid)
            if not uuid.startswith(("GPU-", "MIG-")):
                uuid = "GPU-" + uuid
            handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)
            self._read_value = lambda: pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            name = pynvml.nvmlDeviceGetName(handle)
            self.metadata.update(method="nvml", scope="whole GPU device; includes idle and other processes, excludes CPU/system power",
                                 device_uuid=uuid, device_name=name.decode() if isinstance(name, bytes) else name,
                                 idle_subtracted=False)
            try:
                pids = [p.pid for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)]
                self.metadata['registered_compute_pids'] = pids
                self.metadata['multiple_compute_processes'] = len(pids) > 1
                self.metadata['other_compute_pids'] = [pid for pid in pids if pid != os.getpid()] if os.getpid() in pids else None
                if os.getpid() not in pids:
                    self.metadata['pid_note'] = 'NVML host PIDs may differ from container PIDs; direct process attribution is unavailable.'
            except pynvml.NVMLError:
                self.metadata['registered_compute_pids'] = None
                self.metadata['multiple_compute_processes'] = None
                self.metadata['other_compute_pids'] = None
        elif self.device.type == "cpu":
            domains = []
            for path in sorted(Path('/sys/class/powercap').glob('*-rapl:*')):
                if path.name.count(':') == 1 and (path / 'name').is_file() and (path / 'name').read_text().strip().startswith('package-'):
                    maximum = int((path / 'max_energy_range_uj').read_text())
                    if maximum <= 0:
                        raise RuntimeError(f"Invalid RAPL counter range: {path}")
                    domains.append((path / 'energy_uj', maximum))
            if not domains:
                raise RuntimeError("CPU package RAPL energy counters unavailable. Use --no-energy or a supported GPU; no TDP estimate will be substituted.")
            self.kind = 'counter'
            previous = [int(path.read_text()) for path, _ in domains]
            accumulated = 0

            def read_counter():
                nonlocal previous, accumulated
                current = [int(path.read_text()) for path, _ in domains]
                for old, new, (_, maximum) in zip(previous, current, domains):
                    if not 0 <= new <= maximum or not 0 <= old <= maximum:
                        raise RuntimeError("Invalid RAPL counter value.")
                    accumulated += new - old if new >= old else new - old + maximum
                previous = current
                return accumulated / 1e6

            self._read_value = read_counter
            self.metadata.update(method="rapl", scope="CPU package energy; excludes overlapping subdomains and non-package system components",
                                 domains=[str(path) for path, _ in domains], idle_subtracted=False)
        else:
            raise RuntimeError(f"Energy measurement is unsupported for {self.device}.")

    def _read_checked(self):
        import math

        value = float(self._read_value())
        if not math.isfinite(value) or value < 0:
            raise RuntimeError("Power/energy sensor returned a non-finite or negative value.")
        return value

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def __enter__(self):
        import threading

        self._synchronize()
        self._samples = []
        self._error = None
        self._stop.clear()
        self.result = None
        self.trace = []
        initial = self._read_checked() if self.enabled else None
        self.started = monotonic()
        if self.enabled:
            self._samples.append((0.0, initial))
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def _poll(self):
        while not self._stop.wait(self.interval):
            try:
                value = self._read_checked()
                self._samples.append((monotonic() - self.started, value))
            except Exception as error:
                self._error = error
                self._stop.set()

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self._synchronize()
        finally:
            self.latency_seconds = monotonic() - self.started
            self._stop.set()
            if self._thread is not None:
                self._thread.join()
                self._thread = None
        if not self.enabled or exc_type is not None:
            return False
        if self._error is not None:
            raise RuntimeError("Energy sampling failed; no energy result is reported.") from self._error
        final = self._read_checked()
        samples = [s for s in self._samples if s[0] < self.latency_seconds]
        samples.append((self.latency_seconds, final))
        self.result, self.trace = summarize_energy_samples(samples, self.kind)
        self.result['energy_sampling_warning'] = (
            'Only boundary samples; use a longer evaluation for a reliable average.' if len(samples) < 3 else '')
        return False

    def save(self, path):
        """Save the raw power / cumulative energy / cumulative EDP time series."""
        import csv

        if self.result is None:
            raise RuntimeError("No completed energy measurement to save.")
        with Path(path).open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=list(self.trace[0]))
            writer.writeheader()
            writer.writerows(self.trace)

    def close(self):
        """Always release the polling thread and sensor, including after errors."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self._close_sensor()
        self._close_sensor = lambda: None
