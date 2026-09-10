# Spike-native sensing-to-computing

Code and data for "Spike-native Sensing-to-Computing Integrated System Enabled by Stochastic Operation of a Silicon Nanomembrane Artificial Retina".

| Script | Reproduces |
|---|---|
| `optimization/vd_optimization.py` | Fig. 3, Supplementary Table 2 |
| `optimization/seed_validation.py` | Fig. 1g, Supplementary Fig. 7 |
| `optimization/adaptive_threshold.py` | Fig. 4, Supplementary Table 3 |
| `optimization/boundary_check.py` | Methods (search-range verification) |
| `optimization/verify_results.py` | Verifies saved optimization results |
| `simulation/training.py` | Methods (MNIST LeNet-5 ANN training) |
| `simulation/conversion.py` | Methods (ANN-to-SNN conversion) |
| `simulation/emulation.py` | Methods (BindsNET CPU/GPU inference using measured retina spikes) |
| `simulation/utils.py` | Shared experiment setup, logging, and optional energy profiling |

## Installation

From the repository root, with Anaconda or Miniconda installed:

```bash
conda create --prefix ./.venv --channel conda-forge python=3.11.14 pip
conda activate ./.venv
python install.py
```

This installs [requirements.txt](requirements.txt) for all scripts and applies
BindsNET compatibility fixes without changing SNN dynamics.
With an existing Python 3.11 environment, run `python install.py` directly.
The GPU setup requires Linux x86-64 and an NVIDIA driver compatible with CUDA 12.8.

<details>
<summary>Verify the installation and run tests</summary>

```bash
python - <<'CHECK'
import numpy, pandas, torch, torchvision, bindsnet, rich, pynvml
from simulation import training, conversion, emulation
print("Python imports: OK")
print("NumPy:", numpy.__version__, "pandas:", pandas.__version__)
print("PyTorch:", torch.__version__, "torchvision:", torchvision.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
CHECK
```

For GPU execution, `CUDA available` should report `True`. Run CPU-only tests with:

```bash
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s simulation/tests -v
```

</details>

## Demo

From the repository root:

### Optimization

Optimize the bias-voltage distribution and threshold using measured photocurrent
lookup tables in `data/`.

```bash
python optimization/vd_optimization.py      # 2-5 min
python optimization/seed_validation.py      # 1-2 min
python optimization/adaptive_threshold.py   # 5-10 min
python optimization/boundary_check.py       # 15-30 min
```

Verifies that the bounded search grids do not limit the optimization result; results are written to `results/boundary_check.csv`.

Check the generated optimization results:

```bash
python optimization/verify_results.py
```

### ANN training, conversion, and inference

Train a LeNet-5 ANN on MNIST, convert it for Neu+, and run SNN inference with
BindsNET using the measured retina recordings.

**Train the ANN**

```bash
PYTHONHASHSEED=42 python simulation/training.py --download
```

**Convert the trained ANN**

Use the ANN model path printed by training:

```bash
python simulation/conversion.py --model-path /path/to/MNIST_LENET5_best.pth
```

**Run SNN inference**

Use the SNN model path printed by conversion:

```bash
python simulation/emulation.py --model-path /path/to/MNIST_LENET5_best_snn.pth
```

When `--model-path` is omitted, conversion and emulation use the retained models,
not the latest training or conversion run.

## Results

### Optimization

**Voltage distribution**

Expected output:

| Distribution | Final R2 | theta | p1 | p2 |
|---|---:|---:|---:|---:|
| Beta | 0.96310 | 4925.0 | 1.0000 | 3.2105 |
| Logistic | 0.96141 | 4574.4 | 0.1400 | 0.1400 |
| Uniform | 0.94655 | 4854.1 | 0.1737 | 0.2395 |

**Seed validation**

Expected output:

| Statistic | Value |
|---|---:|
| R2 (mean +/- sd) | 0.9629 +/- 0.0011 |
| seed 42 | 0.96141 |

**Adaptive threshold**

Expected output (Supplementary Table 3):

| Condition | Mode | mu (V) | sigma (V) | theta | R2 |
|---|---|---:|---:|---:|---:|
| dim | Full | 0.21 | 0.14 | 2816.3 | 0.9567 |
| dim | theta-only | 0.14 | 0.14 | 2750.6 | 0.9371 |
| dim | Fixed | 0.14 | 0.14 | 4574.4 | 0.8465 |
| normal | Full | 0.14 | 0.14 | 4574.4 | 0.9619 |
| bright | Full | 0.15 | 0.11 | 5452.7 | 0.9751 |
| bright | theta-only | 0.14 | 0.14 | 5459.4 | 0.9717 |
| bright | Fixed | 0.14 | 0.14 | 4574.4 | 0.9208 |

### ANN and SNN inference

Recorded results ([original experiment](simulation/results/26-09-09/15-58-15/)):

| Evaluation | Samples | Accuracy |
|---|---:|---:|
| Best ANN, MNIST test images | 10,000 | 99.20% |
| ANN, matching test images | 1,000 | 98.70% |
| SNN, measured retina events | 1,000 | 96.00% |

## Experimental hardware

| Component | Specification |
|---|---|
| CPU | 2 x Intel Xeon Gold 6326 @ 2.90 GHz; 32 physical cores, 64 threads total |
| GPU | NVIDIA RTX A6000 |
| NVIDIA driver | 570.86.16 |

## Instructions for use

### Optimization

The scripts read the lookup tables in `data/`, measured under dim, normal and bright illumination: the first column `intensity` lists the input levels, and the remaining columns (named by bias voltage: `0.00V`, `0.01V`, ...) contain the array-averaged photocurrent. Any lookup table in this format can be used. Raw per-pixel measurement data are available from the corresponding authors upon reasonable request.

Scripts for SNN training and the CPU/GPU/chip inference benchmark are described in Methods; chip interface code requires custom hardware and is available upon reasonable request.

### Simulation data and outputs

Use `--download` to fetch missing MNIST files into
`simulation/data/reference_mnist/MNIST/raw`; existing files are reused.
`--no-download` (the default) uses only the local cache.

Measured retina recordings are stored as
`simulation/data/measured_retina_events/curved_2/{index}_spikes.npy`, with columns
`[event_bin, flattened_pixel]` and indices matching MNIST test labels.

`--data-dir` selects the MNIST cache root, not its `MNIST/raw` subfolder;
`--event-path` selects the recordings directory.

Settings can be changed at the top of each script or overridden with command-line
options. `--device cpu` selects CPU execution; `--gpu-id` selects a CUDA device.
`--progress` enables the interactive display.

By default, training saves runs to `simulation/results/YY-MM-DD/HH-MM-SS/`.
Conversion and emulation save dated results under the selected run's `snn/`
and the selected SNN folder's `emulation/`, respectively. Outputs include models,
PKL network data, TXT logs, CSV results, and JSON summaries; no figures are generated.

SNN timing and energy measurements are disabled by default. Use `--latency`
for timing or `--energy` for power, energy, and EDP.
These are CPU/GPU measurements, not Neu+ chip measurements.

GPU energy measurement requires NVML power-query support; CPU energy
measurement requires readable RAPL counters. Inference can work even when
energy measurement is unavailable.

## License

MIT License. See `LICENSE`.
