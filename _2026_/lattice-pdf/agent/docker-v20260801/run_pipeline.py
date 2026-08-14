#!/usr/bin/env python3
"""Master pipeline — docker-v20260801 GPU: auto-plateau effective mass + Pz=0 calibration."""
from __future__ import annotations
import argparse, json, logging, os, sys, time, traceback, shutil
from datetime import datetime
from pathlib import Path
import numpy as np

_SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPT_DIR))
from utils import (Timer, print_banner, setup_logging,
    dump_config_snapshot, get_output_tree, get_gpu_device_info,
    set_compute_dtype, get_compute_dtype, save_intermediate)

def check_environment(config, logger):
    print_banner("Step 00: Environment Check", logger)
    results = {"status": "ok", "checks": {}}
    for mod_name in ["numpy", "scipy", "matplotlib", "cupy"]:
        try:
            m = __import__(mod_name)
            results["checks"][mod_name] = {"ok": True, "version": getattr(m, "__version__", "?")}
        except ImportError:
            results["checks"][mod_name] = {"ok": False}
    gpu = get_gpu_device_info()
    results["gpu"] = gpu
    logger.info(f"GPU: {gpu.get('device_name','?')} ({gpu.get('free_memory_gb','?')} GB free)")
    # Check data paths
    paths, cids = config["data_paths"], config["parameters"]["conf_ids"]
    for cid in cids:
        for tag, p in [
            ("eigvec", f"{paths['eigenvector_base']}/{cid}/eigvecs_t000_{cid}"),
            ("peram",  f"{paths['perambulator_base']}/light/{cid}/perams.{cid}.0.0"),
            ("gauge",  f"{paths['gauge_config_base']}/{paths['gauge_config_pattern'].format(conf_id=cid)}"),
        ]:
            if os.path.exists(p):
                logger.info(f"  OK {tag} conf={cid}")
            else:
                logger.error(f"  MISS {tag} conf={cid}")
                results["all_ok"] = False
    results.setdefault("all_ok", True)
    logger.info("All checks passed!" if results.get("all_ok") else "WARNING: some paths missing")
    return results

def generate_report(config, pz0, output_dir, results, logger):
    print_banner("Step 04: Final Report", logger)
    params, ens = config["parameters"], config["ensemble"]
    m0 = pz0.get("m0_gev", "?")

    # Compute expected boosted energy
    p_phys = abs(params["Pz"]) * 2 * np.pi / (ens["Nx"] * ens["alttc"]) * 0.1973
    E_exp = np.sqrt(float(m0)**2 + p_phys**2) if isinstance(m0, (int, float)) else "?"

    lines = [
        f"# Gluon PDF Pipeline — docker-v20260801 GPU Report",
        f"**{datetime.now():%Y-%m-%d %H:%M:%S}** | Precision: {config['precision']['compute_dtype']}",
        f"",
        f"## Pz=0 Calibration",
        f"| Quantity | Value |",
        f"|----------|-------|",
        f"| m0 (rest mass) | {m0:.4f} GeV |" if isinstance(m0,float) else f"| m0 | {m0} |",
        f"| p (Pz={params['Pz']}) | {p_phys:.4f} GeV |",
        f"| E_expected | {E_exp:.4f} GeV |" if isinstance(E_exp,float) else f"| E_expected | {E_exp} |",
        f"",
        f"## 2pt Results (auto-plateau)",
    ]
    tp = output_dir / "data" / "compute_2pt_summary.json"
    if tp.exists():
        with open(tp) as f:
            td = json.load(f)
        for cid_str, cd in td.items():
            if cd.get("status") == "ok":
                for pz_str, pd in cd.get("results", {}).items():
                    lines.append(f"- conf={cid_str} Pz={pz_str}: "
                        f"meff={pd.get('meff_plateau_gev','?'):.4f} +/- "
                        f"{pd.get('meff_plateau_err_gev',0):.4f} GeV "
                        f"(plateau t in {pd.get('plateau_t','?')})")
    lines.append("")
    lines.append("## Output")
    lines.append("```")
    lines.append(get_output_tree(output_dir))
    lines.append("```")
    (output_dir / "final_report.md").write_text("\n".join(lines))
    logger.info("Report saved.")

def main():
    p = argparse.ArgumentParser(description="docker-v20260801 GPU pipeline")
    p.add_argument("--conf-id", type=int, default=None)
    p.add_argument("--skip-2pt", action="store_true")
    p.add_argument("--skip-ope", action="store_true")
    p.add_argument("--skip-analysis", action="store_true")
    p.add_argument("--precision", type=str, default="complex64",
                   choices=["complex64", "complex128"])
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    config_path = _SCRIPT_DIR / "run_config.json"
    with open(config_path) as f:
        config = json.load(f)
    set_compute_dtype(args.precision)
    if args.conf_id is not None:
        config["parameters"]["conf_ids"] = [args.conf_id]
        config["parameters"]["Nconf"] = 1

    ts = datetime.now().strftime(config["output"]["timestamp_format"])
    output_dir = (_SCRIPT_DIR / ts).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(config["output"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir / "run.log", "pipeline_v20260801",
                           console_level=logging.DEBUG if args.verbose else logging.INFO)
    logger.info(f"{'='*70}")
    logger.info(f"  docker-v20260801 GPU Pipeline | Precision: {args.precision}")
    logger.info(f"  Configs: {config['parameters']['conf_ids']}")
    logger.info(f"  Output:  {output_dir}")
    logger.info(f"{'='*70}")
    dump_config_snapshot(config, output_dir, logger)

    pipeline_start = time.perf_counter()
    all_results = {"run_id": config["run_id"], "steps": {}}
    data_dir, plots_dir = output_dir / "data", output_dir / "plots"

    # Step 0: Env check
    env = check_environment(config, logger)
    all_results["steps"]["00_env"] = env
    save_intermediate(env, output_dir, "env_check.json", logger)

    # Step 1: 2pt
    if not args.skip_2pt:
        try:
            from compute_2pt_gpu import run_2pt_computation_gpu
            tp = run_2pt_computation_gpu(config, data_dir, logger)
            all_results["steps"]["01_2pt"] = {"status": "ok"}
            save_intermediate(tp, data_dir, "compute_2pt_summary.json", logger)
        except Exception as e:
            logger.error(f"2pt failed: {e}")
            traceback.print_exc()
    else:
        logger.info("2pt SKIPPED")

    # Step 2: OPE
    if not args.skip_ope:
        try:
            from compute_ope_gpu import compute_ope_all_configs_gpu
            ope = compute_ope_all_configs_gpu(config, data_dir, logger)
            all_results["steps"]["02_ope"] = {"status": "ok"}
            save_intermediate(ope, data_dir, "compute_ope_summary.json", logger)
        except Exception as e:
            logger.error(f"OPE failed: {e}")
            traceback.print_exc()
    else:
        logger.info("OPE SKIPPED")

    # Step 3: Analysis
    if not args.skip_analysis:
        try:
            from analyze_ratio import run_analysis
            ana = run_analysis(config, data_dir, plots_dir, logger)
            all_results["steps"]["03_analysis"] = {"status": ana.get("status","?")}
            save_intermediate(ana, plots_dir, "analysis_summary.json", logger)
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            traceback.print_exc()
    else:
        logger.info("Analysis SKIPPED")

    # Report
    pz0 = {}
    try:
        generate_report(config, pz0, output_dir, all_results, logger)
    except Exception as e:
        logger.error(f"Report: {e}")

    elapsed = time.perf_counter() - pipeline_start
    all_results["total_elapsed_s"] = elapsed
    with open(output_dir / "run_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    shutil.copy(output_dir / "run.log", log_dir / f"docker-v20260801_{ts}.log")
    logger.info(f"\n{'='*70}")
    logger.info(f"  Done: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"  Log:  {log_dir / f'docker-v20260801_{ts}.log'}")
    logger.info(f"{'='*70}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
