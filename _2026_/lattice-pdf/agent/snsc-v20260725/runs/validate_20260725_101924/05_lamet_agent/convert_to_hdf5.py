#!/usr/bin/env python3
"""
Convert 2pt correlator data from .npy format to HDF5 format for lamet-agent.

lamet-agent expects HDF5 structure:
    /{source_operator}/{sink_operator}/{momentum}
    e.g., /Cg5g4/Cg5g4/PX0PY0PZ2

    Shape: (Lt, n_cfg) complex

Usage:
    python convert_to_hdf5.py --run-dir /path/to/run_dir
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np


def convert_2pt_to_hdf5(
    data_dir: Path,
    output_path: Path,
    Nconf: int = 3,
    Nt: int = 72,
    Px: int = 0,
    Py: int = 0,
    Pz: int = 2,
    conf_start: int = 6250,
    conf_step: int = 200,
    source_op: str = "Cg5g4",
    sink_op: str = "Cg5g4",
    element: str = "_Cg5g4",
    mom_smear: int = -2,
) -> Path:
    """Convert 2pt .npy files to lamet-agent HDF5 format.

    Args:
        data_dir: Directory containing conf_XXXX/ subdirectories.
        output_path: Path for the output HDF5 file.
        Nconf: Number of configurations.
        Nt: Temporal extent.
        Px, Py, Pz: Momentum components.
        conf_start: Starting config ID.
        conf_step: Config ID step.
        source_op: Source operator name (for HDF5 group).
        sink_op: Sink operator name (for HDF5 group).
        element: File naming element (e.g., "_Cg5g4").
        mom_smear: Momentum smearing parameter.

    Returns:
        Path to the created HDF5 file.
    """
    momentum_str = f"PX{Px}PY{Py}PZ{Pz}"
    group_path = f"{source_op}/{sink_op}/{momentum_str}"

    # Collect 2pt data from all configs
    # lamet-agent expects shape (Lt, n_cfg) complex
    all_corr = np.zeros((Nt, Nconf), dtype=complex)

    for i in range(Nconf):
        conf_id = conf_start + i * conf_step
        conf_dir = data_dir / f"conf_{conf_id}"
        fname = (
            f"twopt_slice_pp_Px{Px}Py{Py}Pz{Pz}"
            f"_eginphase{abs(mom_smear)}"
            f"{element}_nopol_ss_conf{conf_id}.npy"
        )
        path = conf_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"2pt file not found: {path}")

        corr = np.load(path)
        assert corr.shape == (Nt, Nt), f"Expected ({Nt},{Nt}), got {corr.shape}"

        # Take diagonal: C(t, t) for effective mass analysis
        # Or use all time slices
        diag = np.array([corr[t, t] for t in range(Nt)])
        all_corr[:, i] = diag

    # Write HDF5
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        grp = f.create_group(group_path)
        dset = grp.create_dataset(
            "correlator",
            data=all_corr,
            dtype=complex,
        )
        # Add metadata attributes
        dset.attrs["Nt"] = Nt
        dset.attrs["Nconf"] = Nconf
        dset.attrs["momentum"] = momentum_str
        dset.attrs["source_operator"] = source_op
        dset.attrs["sink_operator"] = sink_op
        dset.attrs["ensemble"] = "L24x72"
        dset.attrs["description"] = (
            f"Proton 2pt correlator, diagonal (t,t), {Nconf} configs"
        )

    print(f"[convert_to_hdf5] Wrote {group_path} to {output_path}")
    print(f"[convert_to_hdf5] Shape: {all_corr.shape}, dtype: {all_corr.dtype}")
    print(f"[convert_to_hdf5] Value range: [{np.abs(all_corr).min():.2e}, {np.abs(all_corr).max():.2e}]")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert 2pt to lamet-agent HDF5")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory")
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory")
    parser.add_argument("--output", type=str, default=None, help="Output HDF5 path")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    data_dir = Path(args.data_dir) if args.data_dir else run_dir / "02_sample_data"
    output_path = (
        Path(args.output) if args.output
        else run_dir / "05_lamet_agent" / "artifacts" / "proton_2pt.h5"
    )

    # Load config
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        params = config["parameters"]
    else:
        params = {}

    convert_2pt_to_hdf5(
        data_dir=data_dir,
        output_path=output_path,
        Nconf=params.get("Nconf", 3),
        Nt=params.get("Nt", 72),
        Px=params.get("Px", 0),
        Py=params.get("Py", 0),
        Pz=params.get("Pz", 2),
        conf_start=params.get("conf_start", 6250),
        conf_step=params.get("conf_step", 200),
        element=params.get("element", "_Cg5g4"),
        mom_smear=params.get("mom_smear", -2),
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
