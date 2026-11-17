"""Standalone script to compile completed cell summaries into summary_ablation.csv.

Usage:
    python compile_ablation_summary.py [run_directory]

Example:
    python compile_ablation_summary.py
    python compile_ablation_summary.py results/ablation/run_20261110_141341
"""

import csv
import glob
import json
import sys
from pathlib import Path

CANONICAL_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
NUM_CLASSES = 6


def find_target_run_dir(arg_dir=None):
    if arg_dir:
        p = Path(arg_dir)
        if p.exists():
            return p
        print(f"[Error] Specified directory not found: {arg_dir}")
        sys.exit(1)

    # Search for latest run_* folder under results/ablation
    patterns = [
        "results/ablation/run_*",
        "./results/ablation/run_*",
        "../results/ablation/run_*",
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(pat))

    if not candidates:
        print("[Error] No ablation run folders found under results/ablation/")
        sys.exit(1)

    candidates = sorted(candidates, key=lambda x: Path(x).stat().st_mtime)
    return Path(candidates[-1])


def main():
    target_dir = find_target_run_dir(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"\nScanning run directory: {target_dir.resolve()}\n")

    # Find all cell_summary.json files
    summary_files = list(target_dir.glob("*/cell_summary.json"))
    if not summary_files:
        print(f"[Warning] No cell_summary.json files found in {target_dir}")
        return

    summaries = []
    for sf in summary_files:
        try:
            with open(sf, "r", encoding="utf-8") as f:
                data = json.load(f)
                summaries.append(data)
        except Exception as e:
            print(f"Error reading {sf}: {e}")

    # Sort order: none first, then by attack_type, then by percentage
    order = {"none": 0, "random": 1, "zero": 2, "fdla": 3, "pcfdla": 4}
    summaries.sort(key=lambda s: (order.get(s.get("attack_type", ""), 99), s.get("attacker_pct", 0)))

    # Find baseline
    baseline_acc = None
    for s in summaries:
        if s.get("attack_type") == "none" and s.get("attacker_pct") == 0:
            baseline_acc = s.get("mean_final_accuracy_honest")
            break

    # 1. Write summary_ablation.csv
    summary_fields = [
        "attack_type", "attacker_pct", "num_malicious", "malicious_client_ids",
        "mean_round1_accuracy_honest",
        "mean_final_accuracy_honest", "mean_final_accuracy_all",
        "accuracy_drop_vs_baseline_pct",
    ]
    summary_csv_path = target_dir / "summary_ablation.csv"
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields, extrasaction="ignore")
        writer.writeheader()
        for s in summaries:
            row = dict(s)
            fh = row.get("mean_final_accuracy_honest")
            if baseline_acc is not None and fh is not None:
                row["accuracy_drop_vs_baseline_pct"] = round((baseline_acc - fh) * 100, 3)
            else:
                row["accuracy_drop_vs_baseline_pct"] = ""
            for k in ("mean_round1_accuracy_honest", "mean_final_accuracy_honest", "mean_final_accuracy_all"):
                if isinstance(row.get(k), float):
                    row[k] = round(row[k], 4)
            writer.writerow(row)

    print(f"-> Generated: {summary_csv_path}")

    # 2. Write per_class_ablation.csv
    pc_fields = ["attack_type", "attacker_pct"] + [f"final_acc_{lbl}" for lbl in CANONICAL_LABELS]
    pc_csv_path = target_dir / "per_class_ablation.csv"
    with open(pc_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=pc_fields, extrasaction="ignore")
        writer.writeheader()
        for s in summaries:
            row = {"attack_type": s.get("attack_type", ""), "attacker_pct": s.get("attacker_pct", "")}
            for lbl in CANONICAL_LABELS:
                val = s.get(f"final_acc_{lbl}")
                row[f"final_acc_{lbl}"] = round(val, 4) if isinstance(val, float) else val
            writer.writerow(row)

    print(f"-> Generated: {pc_csv_path}")

    # 3. Print Console Table
    print("\n" + "=" * 90)
    print("  CURRENT ABLATION SUMMARY")
    print("=" * 90)
    hdr = (
        f"{'Attack':<10} | {'Pct':>5} | {'Mal.':>5} | "
        f"{'R1 Honest':>10} | {'Final Honest':>12} | "
        f"{'Final All':>10} | {'Drop vs Base':>13}"
    )
    print(hdr)
    print("-" * len(hdr))
    for s in summaries:
        at = s.get("attack_type", "?")
        pct = s.get("attacker_pct", "?")
        nm = s.get("num_malicious", "?")
        r1 = s.get("mean_round1_accuracy_honest", None)
        fh = s.get("mean_final_accuracy_honest", None)
        fa = s.get("mean_final_accuracy_all", None)
        if baseline_acc is not None and fh is not None:
            drop = (baseline_acc - fh) * 100
            drop_str = f"{drop:+.2f}%"
        else:
            drop_str = "---"

        r1_str = f"{r1*100:.2f}%" if isinstance(r1, float) else "---"
        fh_str = f"{fh*100:.2f}%" if isinstance(fh, float) else "---"
        fa_str = f"{fa*100:.2f}%" if isinstance(fa, float) else "---"

        print(
            f"{at:<10} | {str(pct):>5} | {str(nm):>5} | "
            f"{r1_str:>10} | {fh_str:>12} | {fa_str:>10} | {drop_str:>13}"
        )
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
