"""Install project dependencies and BindsNET compatibility fixes.

Run ``python install.py`` in a Python 3.11 environment. Packages are installed
from the repository's requirements.txt using the active Python interpreter.
Use ``--patch-only`` when dependencies are already installed.

Only BindsNET imports are patched; network equations, weights, and thresholds
are unchanged. Unknown source layouts are rejected before any patch is written.
"""

import argparse
import importlib.metadata
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

PATCHES = (
    (
        "datasets/collate.py",
        "from torch._six import container_abcs, string_classes, int_classes\n",
        "import collections.abc as container_abcs\nstring_classes = (str,)\nint_classes = (int,)\n",
    ),
    (
        "pipeline/base_pipeline.py",
        "from torch._six import container_abcs, string_classes\n",
        "import collections.abc as container_abcs\nstring_classes = (str,)\n",
    ),
    (
        "environment/environment.py",
        "\nimport gym\n",
        "\ntry:\n    import gymnasium as gym\nexcept ImportError:\n    import gym\n",
    ),
)


def apply_compatibility(package_dir):
    """Validate all three edits before writing, and return the changed filenames."""
    pending = []
    for relative, old, new in PATCHES:
        path = Path(package_dir) / relative
        content = path.read_bytes()
        newline = "\r\n" if b"\r\n" in content else "\n"
        old = old.replace("\n", newline).encode()
        new = new.replace("\n", newline).encode()
        if content.count(old) == 1 and new not in content:
            pending.append((path, content.replace(old, new, 1)))
        elif old not in content and content.count(new) == 1:
            continue
        else:
            raise ValueError(f"Unrecognized BindsNET source; no patches applied: {path}")
    for path, content in pending:
        path.write_bytes(content)
    return [path for path, _ in pending]


def patch_installed_bindsnet():
    """Apply import fixes to BindsNET 0.2.7 in the active environment."""
    distribution = importlib.metadata.distribution("bindsnet")
    if distribution.version != "0.2.7":
        raise ValueError(f"Expected bindsnet==0.2.7, found {distribution.version}.")
    package_dir = Path(distribution.locate_file("bindsnet"))
    changed = apply_compatibility(package_dir)
    for path in changed:
        print(f"Patched: {path.relative_to(package_dir)}")
    print("BindsNET compatibility ready." if changed else "BindsNET compatibility already applied.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--patch-only", action="store_true",
        help="Apply BindsNET compatibility fixes without installing packages.",
    )
    args = parser.parse_args(argv)
    if sys.version_info[:2] != (3, 11):
        parser.error("Activate a Python 3.11 environment before running this installer.")
    if not args.patch_only:
        print("Installing project dependencies...", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")],
            cwd=ROOT, check=True,
        )
    patch_installed_bindsnet()


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError,
            importlib.metadata.PackageNotFoundError) as error:
        raise SystemExit(str(error))
