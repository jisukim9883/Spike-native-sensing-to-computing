#!/usr/bin/env python3
"""Train a LeNet-5 ANN on MNIST for later SNN conversion experiments.

Created by Daeyoung Kim and Jongkil Park.

This script trains a compact LeNet-5 model with MNIST and saves plain PyTorch
``state_dict`` weights. The saved ``*_best.pth`` file can therefore be loaded
by a later conversion/simulation script without depending on this trainer.

The original experiment uses 50 epochs, batch size 128, seed 42, Adam with
lr=0.001 and no weight decay, and CosineAnnealingLR(T_max=50). DataLoader
workers and pinned memory are disabled. One shuffled batch is inspected
before model initialization to preserve the original RNG consumption order.
Defaults match the retained run in results/26-09-09/15-58-15, using CUDA GPU 0
and existing local MNIST files. Each execution trains a new model; it does
not load the retained weights. Exact numerical reproduction also depends on
the dataset, PyTorch/CUDA versions, and GPU matching the recorded environment.

Quick start
-----------
Run from the repository root::

    python3 simulation/training.py

The editable defaults are collected in the "Experiment defaults" section
below. For a one-off experiment, command-line options override those defaults,
for example::

    python3 simulation/training.py --epochs 20 --batch-size 256 --device cuda

Each run is stored under ``simulation/results/YY-MM-DD/HH-MM-SS/``. The folder
contains model weights, full checkpoints, per-epoch metrics, a JSON summary,
and ``training_output.txt``. The text log begins with the complete experiment
conditions so the result can be reproduced or compared later.

Text output is the default, matching training_output.txt in the retained run.
With Rich installed, --progress in an interactive terminal shows batch progress,
recent epoch metrics, the best accuracy, and timing. ``--no-progress`` switches
to epoch-by-epoch text output. Piped output uses text automatically. Every
completed epoch is retained in the log regardless of the visible table length.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    from .utils import (
        TeeLogger,
        TrainingDisplay,
        count_parameters,
        create_run_directories,
        print_section,
        resolve_mnist_directory,
        seed_all,
        select_device,
        serializable_args,
    )
except ImportError:
    from utils import (
        TeeLogger,
        TrainingDisplay,
        count_parameters,
        create_run_directories,
        print_section,
        resolve_mnist_directory,
        seed_all,
        select_device,
        serializable_args,
    )


SCRIPT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Experiment defaults
# Change these values for the usual experiment setting. Command-line options
# take priority, so individual runs can still be adjusted without editing here.
# ---------------------------------------------------------------------------
DEFAULT_RUN_NAME = "MNIST_LENET5"
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 128
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_OPTIMIZER = "adam"
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_MOMENTUM = 0.9  # SGD only; Adam uses betas=(0.9, 0.999), not momentum.
DEFAULT_USE_BIAS = False
DEFAULT_SEED = 42
DEFAULT_TRUE_RANDOM = False
DEFAULT_DEVICE = "cuda"
DEFAULT_GPU_ID = 0
DEFAULT_NUM_WORKERS = 0
DEFAULT_DATA_DIR = SCRIPT_DIR / "data/reference_mnist"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"
DEFAULT_DOWNLOAD = False

# Terminal display: recent rows only affect the screen, never the saved log.
DEFAULT_PROGRESS = False
DEFAULT_HISTORY_ROWS = 10
DISPLAY_REFRESH_PER_SECOND = 4


class LeNet5(nn.Module):
    """LeNet-5 variant for 28x28 grayscale MNIST images.

    The layer names are kept explicit to make weight mapping to a subsequent
    SNN conversion workflow straightforward. ``bias=False`` is the default,
    matching the original simulation-oriented training setup.
    """

    def __init__(self, num_classes: int = 10, bias: bool = DEFAULT_USE_BIAS) -> None:
        super().__init__()
        self.conv1_1 = nn.Conv2d(1, 6, kernel_size=5, padding=2, stride=1, bias=bias)
        self.relu1_1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(6, 16, kernel_size=5, padding=0, stride=1, bias=bias)
        self.relu2_1 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(16, 120, kernel_size=5, padding=0, stride=1, bias=bias)
        self.relu3_1 = nn.ReLU()

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(120, 84, bias=bias)
        self.relu4 = nn.ReLU()
        self.lin_out = nn.Linear(84, num_classes, bias=bias)

        self._init_weights()

    def _init_weights(self) -> None:
        """Apply the initialization used by the existing training workflow."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the ten MNIST class logits for an input batch."""
        x = self.conv1_1(x)
        x = self.relu1_1(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.relu2_1(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.relu3_1(x)

        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu4(x)
        return self.lin_out(x)


@dataclass
class EpochMetrics:
    """Metrics recorded after one completed training epoch."""

    epoch: int
    lr: float
    train_loss: float
    test_loss: float
    test_accuracy: float
    epoch_time_s: float


def parse_args() -> argparse.Namespace:
    """Read command-line overrides and validate the requested experiment."""
    parser = argparse.ArgumentParser(
        description="Train a LeNet-5 ANN on MNIST and save conversion-ready weights."
    )
    parser.add_argument("--name", type=str, default=DEFAULT_RUN_NAME, help="Run name used in output filenames.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Training and test batch size.")
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE, help="Initial learning rate.")
    parser.add_argument(
        "--optimizer",
        type=str,
        default=DEFAULT_OPTIMIZER,
        choices=("adam", "sgd"),
        help="Optimizer to use.",
    )
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="Weight decay used with SGD.")
    parser.add_argument("--momentum", type=float, default=DEFAULT_MOMENTUM, help="Momentum used with SGD.")
    parser.add_argument(
        "--bias",
        dest="bias",
        action="store_true",
        default=DEFAULT_USE_BIAS,
        help="Enable biases in convolution and linear layers.",
    )
    parser.add_argument("--no-bias", dest="bias", action="store_false", help="Disable biases in convolution and linear layers.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducible runs.")
    parser.add_argument(
        "--true-random",
        action="store_true",
        default=DEFAULT_TRUE_RANDOM,
        help="Do not set a fixed random seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        choices=("auto", "cpu", "cuda"),
        help="Training device; auto selects CUDA when available.",
    )
    parser.add_argument("--gpu-id", type=int, default=DEFAULT_GPU_ID, help="CUDA device index when using a GPU.")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="DataLoader worker count.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="MNIST root containing MNIST/raw (default: simulation/data/reference_mnist).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Base directory for results.")
    parser.add_argument(
        "--download",
        dest="download",
        action="store_true",
        default=DEFAULT_DOWNLOAD,
        help="Download MNIST when it is missing.",
    )
    parser.add_argument(
        "--no-download",
        dest="download",
        action="store_false",
        help="Use only an existing MNIST download.",
    )
    parser.add_argument(
        "--progress", action=argparse.BooleanOptionalAction, default=DEFAULT_PROGRESS,
        help="Show a live training dashboard in an interactive terminal.",
    )
    parser.add_argument(
        "--history-rows", type=int, default=DEFAULT_HISTORY_ROWS,
        help="Number of recent epochs visible in the live table (all epochs are logged).",
    )
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1.")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")
    if args.lr <= 0:
        parser.error("--lr must be positive.")
    if args.gpu_id < 0:
        parser.error("--gpu-id must be non-negative.")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative.")
    if args.history_rows < 1:
        parser.error("--history-rows must be at least 1.")
    args.data_dir = resolve_mnist_directory(args.data_dir)
    return args


def build_dataloaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    download: bool,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    """Create MNIST training and test loaders using the same ToTensor input path."""
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(
        root=str(data_dir),
        train=True,
        download=download,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=str(data_dir),
        train=False,
        download=download,
        transform=transform,
    )
    pin_memory = False  # Match the original experiment's DataLoader behavior.

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def initialize_model(train_loader: DataLoader, bias: bool = DEFAULT_USE_BIAS) -> LeNet5:
    """Inspect one shuffled batch before initializing the original LeNet-5.

    Creating the iterator and reading its first batch consumes PyTorch RNG
    state. The original training experiment did this before constructing the
    model, so omitting it changes the initial weights even with the same seed.
    Training itself starts a fresh iterator and still uses every sample.
    """
    images, _ = next(iter(train_loader))
    print(f"Initialization preview: shape={tuple(images.shape)}, "
          f"pixel range=[{images.min().item()}, {images.max().item()}]")
    return LeNet5(bias=bias)


def build_optimizer(
    model: nn.Module,
    optimizer_name: str,
    lr: float,
    momentum: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Build the selected optimizer for the LeNet-5 parameters."""
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)


def print_experiment_conditions(
    args: argparse.Namespace,
    device: torch.device,
    run_dir: Path,
    model_dir: Path,
    checkpoint_dir: Path,
    log_path: Path,
    train_loader: DataLoader,
    test_loader: DataLoader,
    model: nn.Module,
    date: str,
    run_time: str,
    logger: TeeLogger,
) -> None:
    """Print the complete initial conditions captured by ``training_output.txt``."""
    total_params, trainable_params = count_parameters(model)
    optimizer_detail = args.optimizer
    if args.optimizer == "sgd":
        optimizer_detail = f"sgd, momentum={args.momentum}, weight_decay={args.weight_decay}"

    print_section(
        "Experiment Conditions",
        [
            ("Run name", args.name),
            ("Date", date),
            ("Time", run_time),
            ("Command", " ".join(sys.argv)),
            ("Run directory", run_dir),
            ("Model directory", model_dir),
            ("Checkpoint directory", checkpoint_dir),
            ("Output log", log_path),
        ],
        logger=logger,
    )
    print_section(
        "Dataset",
        [
            ("Dataset", "MNIST"),
            ("Data directory", args.data_dir),
            ("Download enabled", args.download),
            ("Input transform", "ToTensor"),
            ("Train samples", len(train_loader.dataset)),
            ("Test samples", len(test_loader.dataset)),
            ("Batch size", args.batch_size),
            ("DataLoader workers", args.num_workers),
        ],
        logger=logger,
    )
    print_section(
        "Model And Training",
        [
            ("Architecture", "LeNet5"),
            ("Layer sequence", "Conv5x5-6, AvgPool, Conv5x5-16, AvgPool, Conv5x5-120, FC84, FC10"),
            ("Bias", args.bias),
            ("Total parameters", total_params),
            ("Trainable parameters", trainable_params),
            ("Epochs", args.epochs),
            ("Optimizer", optimizer_detail),
            ("Learning rate", args.lr),
            ("Scheduler", "CosineAnnealingLR"),
            ("Loss", "CrossEntropyLoss"),
            ("Seed", "true-random" if args.true_random else args.seed),
            ("Device", device),
        ],
        logger=logger,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    on_batch: Callable[[int], None] | None = None,
) -> float:
    """Train for one epoch and return the mean batch loss."""
    model.train()
    running_loss = 0.0

    for batch_index, (data, targets) in enumerate(loader, start=1):
        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if on_batch is not None:
            on_batch(batch_index)

    return running_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    on_batch: Callable[[int], None] | None = None,
) -> tuple[float, float]:
    """Evaluate the model and return mean loss plus classification accuracy in percent."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_index, (data, targets) in enumerate(loader, start=1):
        data = data.to(device)
        targets = targets.to(device)
        outputs = model(data)
        loss = criterion(outputs, targets)

        running_loss += loss.item()
        predictions = outputs.argmax(dim=1)
        total += targets.size(0)
        correct += (predictions == targets).sum().item()
        if on_batch is not None:
            on_batch(batch_index)

    accuracy = 100.0 * correct / total
    return running_loss / len(loader), accuracy


def save_weights(path: Path, model: nn.Module) -> None:
    """Save only ``state_dict`` weights for a later SNN conversion step."""
    torch.save(model.state_dict(), path)


def save_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    accuracy: float,
    args: argparse.Namespace,
) -> None:
    """Save full training state for inspection or possible training resumption."""
    torch.save(
        {
            "epoch": epoch,
            "accuracy": accuracy,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": serializable_args(args),
            "architecture": "LeNet5",
            "dataset": "MNIST",
        },
        path,
    )


def write_metrics(path: Path, history: list[EpochMetrics]) -> None:
    """Write one row of scalar metrics per completed epoch to CSV."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(history[0]).keys()))
        writer.writeheader()
        for row in history:
            writer.writerow(asdict(row))


def main() -> None:
    """Run training, persist artifacts, and always restore terminal output streams."""
    args = parse_args()

    if not args.true_random:
        seed_all(args.seed)

    device = select_device(args.device, args.gpu_id)
    run_paths = create_run_directories(args.output_dir)
    log_path = run_paths.run_dir / "training_output.txt"
    logger = TeeLogger(log_path)

    try:
        print(f"Preparing run: {args.name}")
        print(f"Run directory: {run_paths.run_dir}")
        print(f"Log file: {log_path}")
        print()

        train_loader, test_loader = build_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            download=args.download,
            device=device,
        )

        model = initialize_model(train_loader, bias=args.bias).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = build_optimizer(model, args.optimizer, args.lr, args.momentum, args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        print_experiment_conditions(
            args=args,
            device=device,
            run_dir=run_paths.run_dir,
            model_dir=run_paths.model_dir,
            checkpoint_dir=run_paths.checkpoint_dir,
            log_path=log_path,
            train_loader=train_loader,
            test_loader=test_loader,
            model=model,
            date=run_paths.date,
            run_time=run_paths.time,
            logger=logger,
        )

        best_accuracy = 0.0
        best_epoch = 0
        best_weight_path = run_paths.model_dir / f"{args.name}_best.pth"
        last_weight_path = run_paths.model_dir / f"{args.name}_last.pth"
        best_checkpoint_path = run_paths.checkpoint_dir / f"{args.name}_best_checkpoint.pth"
        last_checkpoint_path = run_paths.checkpoint_dir / f"{args.name}_last_checkpoint.pth"
        history: list[EpochMetrics] = []
        training_started_at = time.time()

        print("Training started.")
        with TrainingDisplay(
            logger, args.epochs, enabled=args.progress,
            history_rows=args.history_rows, refresh_per_second=DISPLAY_REFRESH_PER_SECOND,
        ) as display:
            for epoch in range(1, args.epochs + 1):
                epoch_started_at = time.time()
                current_lr = optimizer.param_groups[0]["lr"]
                display.start_epoch(epoch)
                display.start_phase("Training", len(train_loader))
                train_loss = train_one_epoch(
                    model, train_loader, criterion, optimizer, device, on_batch=display.update_batch,
                )
                display.start_phase("Evaluating", len(test_loader))
                test_loss, test_accuracy = evaluate(
                    model, test_loader, criterion, device, on_batch=display.update_batch,
                )
                scheduler.step()

                epoch_time_s = time.time() - epoch_started_at
                history.append(
                    EpochMetrics(
                        epoch=epoch,
                        lr=current_lr,
                        train_loss=train_loss,
                        test_loss=test_loss,
                        test_accuracy=test_accuracy,
                        epoch_time_s=epoch_time_s,
                    )
                )

                is_best = test_accuracy > best_accuracy
                if is_best:
                    best_accuracy = test_accuracy
                    best_epoch = epoch
                    save_weights(best_weight_path, model)
                    save_training_checkpoint(
                        best_checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        accuracy=test_accuracy,
                        args=args,
                    )

                display.finish_epoch(
                    epoch=epoch, lr=current_lr, train_loss=train_loss,
                    test_loss=test_loss, accuracy=test_accuracy,
                    seconds=epoch_time_s, is_best=is_best,
                )

        save_weights(last_weight_path, model)
        save_training_checkpoint(
            last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=args.epochs,
            accuracy=history[-1].test_accuracy,
            args=args,
        )

        metrics_path = run_paths.run_dir / "metrics.csv"
        write_metrics(metrics_path, history)

        total_time_s = time.time() - training_started_at
        summary = {
            "timestamp": {"date": run_paths.date, "time": run_paths.time},
            "run_name": args.name,
            "dataset": "MNIST",
            "architecture": "LeNet5",
            "best_accuracy": round(best_accuracy, 4),
            "best_epoch": best_epoch,
            "final_accuracy": round(history[-1].test_accuracy, 4),
            "total_time_s": round(total_time_s, 6),
            "avg_epoch_time_s": round(total_time_s / args.epochs, 6),
            "config": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "optimizer": args.optimizer,
                "weight_decay": args.weight_decay if args.optimizer == "sgd" else 0.0,
                "momentum": args.momentum if args.optimizer == "sgd" else 0.0,
                "bias": args.bias,
                "seed": None if args.true_random else args.seed,
                "device": str(device),
                "data_dir": str(args.data_dir),
                "output_dir": str(args.output_dir),
                "download": args.download,
                "num_workers": args.num_workers,
            },
            "files": {
                "best_weights": str(best_weight_path),
                "last_weights": str(last_weight_path),
                "best_checkpoint": str(best_checkpoint_path),
                "last_checkpoint": str(last_checkpoint_path),
                "metrics": str(metrics_path),
                "log": str(log_path),
            },
        }

        summary_path = run_paths.run_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)

        print()
        print_section(
            "Training Summary",
            [
                ("Best accuracy", f"{best_accuracy:.2f}%"),
                ("Best epoch", best_epoch),
                ("Final accuracy", f"{history[-1].test_accuracy:.2f}%"),
                ("Total time", f"{total_time_s:.2f} s"),
                ("Average epoch time", f"{total_time_s / args.epochs:.2f} s"),
                ("Best weights", best_weight_path),
                ("Last weights", last_weight_path),
                ("Metrics", metrics_path),
                ("Summary", summary_path),
                ("Output log", log_path),
            ],
            logger=logger,
        )
    except Exception:
        print("Training failed. Traceback follows.")
        traceback.print_exc()
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    os.chdir(SCRIPT_DIR.parent)
    main()
