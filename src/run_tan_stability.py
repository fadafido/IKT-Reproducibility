"""
TAN Network Stability Extension (Part C)
==========================================
Runs IKT pipeline on ASS-09 with 20+ random seeds, extracting
the TAN Bayesian Network structure from each WEKA output.

Research question: Does the TAN topology remain stable across
different k-means random initializations?

Usage:
  python run_tan_stability.py                    # default 20 seeds
  python run_tan_stability.py --seeds 30         # custom seed count
  python run_tan_stability.py --dataset algebra   # other datasets

Results saved to: results_tan_stability/<dataset>_tan_stability.txt

Author: Fadi Alazayem (25002207) - BUiD MSc AI
"""

import subprocess
import sys
import os
import re
import argparse
import time
import shutil
import numpy as np
from collections import Counter

# Project root (one level up from src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# CONFIGURATION
# ============================================================
# 20 seeds: first 5 match multi-seed runs, then 15 new ones
ALL_SEEDS = [42, 123, 456, 789, 1011,
             2024, 3333, 5555, 7777, 9999,
             1234, 2468, 3690, 4812, 6543,
             8765, 1357, 2469, 3580, 4691]

DATASET_CONFIG = {
    "ass09": {
        "name": "ASSISTments 2009",
        "fe_script": "FeatureEngineering_v2.py",
        "work_dir": "data/ass09",
        "our_mean_auc": 0.8056,
    },
    "algebra": {
        "name": "Algebra 2005-2006",
        "fe_script": "FeatureEngineering_v2.py",
        "work_dir": "data/algebra",
        "our_mean_auc": 0.7692,
    },
    "ass12": {
        "name": "ASSISTments 2012 (5K cap)",
        "fe_script": "FeatureEngineering_v2.py",
        "work_dir": "data/ass12",
        "our_mean_auc": 0.7786,
    },
}

TEMP_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_temp_tan_fe.py")


# ============================================================
# UTILITIES
# ============================================================
def detect_weka():
    """Find weka.jar."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(PROJECT_ROOT, "weka.jar"),
        os.path.join(PROJECT_ROOT, "weka-mac", "weka.jar"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    print("ERROR: Cannot find weka.jar")
    sys.exit(1)


def detect_java_version():
    """Check if Java needs --add-opens flag."""
    try:
        result = subprocess.run(["java", "-version"], capture_output=True, text=True)
        return "1.8" not in result.stderr
    except Exception:
        return False


def run_pipeline_with_seed(seed, work_dir, fe_script):
    """Create temp FE script with modified seed, run it."""
    src_dir = os.path.dirname(os.path.abspath(__file__))
    fe_path = os.path.join(src_dir, fe_script)

    with open(fe_path, "r") as f:
        content = f.read()

    content = re.sub(r"RANDOM_SEED = \d+", f"RANDOM_SEED = {seed}", content)

    with open(TEMP_SCRIPT, "w") as f:
        f.write(content)

    result = subprocess.run(
        [sys.executable, TEMP_SCRIPT],
        capture_output=True, text=True,
    )

    # Clean up temp file
    if os.path.exists(TEMP_SCRIPT):
        os.remove(TEMP_SCRIPT)

    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:300]}")
        return False
    return True


def run_weka_tan_full(train_arff, test_arff, weka_jar, needs_add_opens):
    """Run WEKA TAN and return full output (for structure extraction)."""
    cmd = ["java", "-Xmx2g"]
    if needs_add_opens:
        cmd += ["--add-opens", "java.base/java.lang=ALL-UNNAMED"]
    cmd += [
        "-cp", weka_jar,
        "weka.classifiers.bayes.BayesNet",
        "-t", train_arff,
        "-T", test_arff,
        "-Q", "weka.classifiers.bayes.net.search.local.TAN",
        "--", "-S", "BAYES",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout


def extract_auc(weka_output):
    """Extract AUC from WEKA output."""
    in_test = False
    for line in weka_output.split("\n"):
        if "Error on test data" in line:
            in_test = True
        if in_test and "Weighted Avg." in line:
            parts = line.split()
            try:
                return float(parts[-2])
            except (ValueError, IndexError):
                pass
    return None


def extract_tan_structure(weka_output):
    """
    Extract TAN network parent structure from WEKA BayesNet output.

    WEKA BayesNet output contains a network definition section like:
        LogScore Bayes: -xxxxxxx
        LogScore BDeu: -xxxxxxx
        LogScore MDL: -xxxxxxx
        LogScore ENTROPY: -xxxxxxx
        LogScore AIC: -xxxxxxx
        
        skill_ID(2): correctness ability_profile
        skill_mastery(2): correctness skill_ID
        ability_profile(1): correctness
        problem_difficulty(2): correctness skill_ID
        correctness(2):

    The node lines follow the pattern: name(N): parent1 parent2 ...
    where N is the number of values (NOT number of parents).
    The class variable (correctness) has no parents listed after the colon.

    KNOWN FEATURES of the IKT model:
      skill_ID, skill_mastery, ability_profile, problem_difficulty, correctness
    """
    structure = {}
    
    # Known IKT feature names — only parse lines matching these
    KNOWN_NODES = {'skill_ID', 'skill_mastery', 'ability_profile', 
                   'problem_difficulty', 'correctness'}
    
    # Find all lines matching: word(number): optional_parents
    # This regex captures: node_name, num_values, parents_string
    pattern = re.compile(r"^(\w+)\((\d+)\):\s*(.*)$", re.MULTILINE)
    matches = pattern.findall(weka_output)
    
    for attr_name, num_values, parents_str in matches:
        # CRITICAL: Only accept known IKT feature nodes
        # This filters out LogScore lines and any other non-node matches
        if attr_name not in KNOWN_NODES:
            continue
        
        # Parse parent list (space-separated), filter to known nodes only
        if parents_str.strip():
            raw_parents = parents_str.strip().split()
            # Only keep valid parent names (filters out stray numbers/text)
            parents = sorted([p for p in raw_parents if p in KNOWN_NODES])
        else:
            parents = []
        
        structure[attr_name] = parents
    
    return structure


def structure_to_string(structure):
    """Convert structure dict to a canonical string for comparison."""
    if not structure:
        return "EMPTY"
    parts = []
    for node in sorted(structure.keys()):
        parents = structure[node]
        parts.append(f"{node}<-[{','.join(parents)}]")
    return " | ".join(parts)


def compute_edge_agreement(structures):
    """
    Compute pairwise agreement rate of TAN edges across all seeds.
    Returns the fraction of seed pairs that have identical structure.
    """
    n = len(structures)
    if n < 2:
        return 1.0, 0

    canonical = [structure_to_string(s) for s in structures]
    agreements = 0
    total_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            total_pairs += 1
            if canonical[i] == canonical[j]:
                agreements += 1

    return agreements / total_pairs if total_pairs > 0 else 0, total_pairs


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="TAN Network Stability Extension")
    parser.add_argument("--dataset", default="ass09", choices=["ass09", "algebra", "ass12"])
    parser.add_argument("--seeds", type=int, default=20, help="Number of seeds (max 20)")
    args = parser.parse_args()

    ds = args.dataset
    cfg = DATASET_CONFIG[ds]
    num_seeds = min(args.seeds, len(ALL_SEEDS))
    seeds = ALL_SEEDS[:num_seeds]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    work_dir = os.path.join(PROJECT_ROOT, cfg["work_dir"])

    weka_jar = detect_weka()
    needs_add_opens = detect_java_version()

    results_dir = os.path.join(PROJECT_ROOT, "results/tan_stability")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 65)
    print(f"TAN NETWORK STABILITY EXTENSION - {cfg['name']}")
    print(f"Seeds: {num_seeds} ({seeds[:5]}...)" if num_seeds > 5 else f"Seeds: {seeds}")
    print(f"WEKA: {weka_jar}")
    print(f"Working dir: {work_dir}")
    print("=" * 65)

    all_results = []   # (seed, auc, structure_dict, structure_string)
    start_total = time.time()

    for i, seed in enumerate(seeds):
        print(f"\n--- Run {i+1}/{num_seeds}: Seed {seed} ---")
        start = time.time()

        # Step 1: Run feature engineering with this seed
        print("  Feature engineering...")
        success = run_pipeline_with_seed(seed, work_dir, cfg["fe_script"])
        if not success:
            print(f"  SKIPPED (FE failed)")
            continue

        # Step 2: Run WEKA TAN
        train_arff = os.path.join(work_dir, "train_data.arff")
        test_arff = os.path.join(work_dir, "test_data.arff")

        print("  WEKA TAN...")
        weka_output = run_weka_tan_full(train_arff, test_arff, weka_jar, needs_add_opens)

        # Save WEKA output
        with open(os.path.join(results_dir, f"{ds}_weka_seed{seed}.txt"), "w") as f:
            f.write(weka_output)

        # Step 3: Extract AUC and TAN structure
        auc = extract_auc(weka_output)
        structure = extract_tan_structure(weka_output)
        struct_str = structure_to_string(structure)

        elapsed = time.time() - start

        if auc:
            print(f"  AUC = {auc:.4f}  |  TAN: {struct_str}  |  {elapsed:.0f}s")
            all_results.append((seed, auc, structure, struct_str))
        else:
            print(f"  FAILED (no AUC)")

        # Save per-seed ARFF files
        data_dir = work_dir
        for fname in ["train_data.arff", "test_data.arff", "train_data_out.csv", "test_data_out.csv"]:
            src = os.path.join(data_dir, fname)
            if os.path.exists(src):
                base, ext = os.path.splitext(fname)
                shutil.copy(src, os.path.join(results_dir, f"{ds}_{base}_seed{seed}{ext}"))

    total_time = time.time() - start_total

    # ============================================================
    # ANALYSIS
    # ============================================================
    report = []
    report.append("=" * 75)
    report.append(f"TAN NETWORK STABILITY ANALYSIS - {cfg['name']}")
    report.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Seeds tested: {len(all_results)} / {num_seeds}")
    report.append(f"Total runtime: {total_time:.0f}s ({total_time/60:.1f} min)")
    report.append("=" * 75)
    report.append("")

    if not all_results:
        report.append("NO SUCCESSFUL RUNS")
        report_text = "\n".join(report)
        print(report_text)
        return

    # ---- AUC STATISTICS ----
    aucs = [r[1] for r in all_results]
    report.append("AUC STATISTICS (across all seeds)")
    report.append("-" * 40)
    report.append(f"  N:      {len(aucs)}")
    report.append(f"  Mean:   {np.mean(aucs):.4f}")
    report.append(f"  Std:    {np.std(aucs):.4f}")
    report.append(f"  Min:    {np.min(aucs):.4f}")
    report.append(f"  Max:    {np.max(aucs):.4f}")
    report.append(f"  Range:  {np.max(aucs) - np.min(aucs):.4f}")
    report.append(f"  Median: {np.median(aucs):.4f}")
    report.append("")

    # Per-seed detail
    report.append("PER-SEED RESULTS")
    report.append("-" * 55)
    report.append(f"  {'Seed':<8} {'AUC':<8} {'TAN Structure'}")
    report.append("  " + "-" * 52)
    for seed, auc, structure, struct_str in all_results:
        report.append(f"  {seed:<8} {auc:.4f}   {struct_str}")
    report.append("")

    # ---- STRUCTURE STABILITY ----
    structures = [r[2] for r in all_results]
    struct_strings = [r[3] for r in all_results]

    unique_structures = list(set(struct_strings))
    struct_counts = Counter(struct_strings)

    report.append("TAN STRUCTURE STABILITY")
    report.append("-" * 40)
    report.append(f"  Unique topologies found: {len(unique_structures)}")
    report.append("")

    for idx, (struct, count) in enumerate(struct_counts.most_common()):
        pct = count / len(all_results) * 100
        report.append(f"  Topology {idx+1} ({count}/{len(all_results)} = {pct:.0f}%):")
        report.append(f"    {struct}")
        # Which seeds produced this?
        matching_seeds = [r[0] for r in all_results if r[3] == struct]
        matching_aucs = [r[1] for r in all_results if r[3] == struct]
        report.append(f"    Seeds: {matching_seeds}")
        report.append(f"    AUC range: {min(matching_aucs):.4f} - {max(matching_aucs):.4f}")
        report.append("")

    # Pairwise agreement
    agreement_rate, total_pairs = compute_edge_agreement(structures)
    report.append(f"  Pairwise structural agreement: {agreement_rate:.1%} ({int(agreement_rate * total_pairs)}/{total_pairs} pairs)")
    report.append("")

    # ---- EDGE-LEVEL ANALYSIS ----
    if structures and structures[0]:
        report.append("EDGE-LEVEL STABILITY (per parent-child relationship)")
        report.append("-" * 55)

        # For each node, count how often each parent set appears
        all_nodes = set()
        for s in structures:
            all_nodes.update(s.keys())

        for node in sorted(all_nodes):
            parent_sets = []
            for s in structures:
                if node in s:
                    parent_sets.append(tuple(sorted(s[node])))
            if parent_sets:
                parent_counts = Counter(parent_sets)
                most_common_parents, mc_count = parent_counts.most_common(1)[0]
                stability = mc_count / len(parent_sets) * 100
                report.append(f"  {node}:")
                for parents, count in parent_counts.most_common():
                    pct = count / len(parent_sets) * 100
                    report.append(f"    parents=[{', '.join(parents)}]: {count}/{len(parent_sets)} ({pct:.0f}%)")
        report.append("")

    # ---- KEY FINDINGS ----
    report.append("=" * 75)
    report.append("KEY FINDINGS")
    report.append("=" * 75)
    report.append("")

    if len(unique_structures) == 1:
        report.append("1. TAN TOPOLOGY IS PERFECTLY STABLE across all seeds.")
        report.append(f"   All {len(all_results)} seeds produced identical network structure.")
        report.append("   This means k-means initialization does NOT affect the learned")
        report.append("   Bayesian network topology, supporting IKT's robustness.")
    else:
        report.append(f"1. TAN TOPOLOGY SHOWS {len(unique_structures)} VARIANTS across {len(all_results)} seeds.")
        dominant = struct_counts.most_common(1)[0]
        report.append(f"   Dominant topology appears in {dominant[1]}/{len(all_results)} runs ({dominant[1]/len(all_results):.0%}).")

    report.append("")
    report.append(f"2. AUC VARIANCE IS {'MINIMAL' if np.std(aucs) < 0.005 else 'MODERATE'}: ")
    report.append(f"   std={np.std(aucs):.4f}, range={np.max(aucs)-np.min(aucs):.4f}")
    report.append(f"   This confirms that random seed choice has {'negligible' if np.std(aucs) < 0.005 else 'some'} impact on performance.")
    report.append("")
    report.append(f"3. COMPARISON WITH 5-SEED RUN:")
    report.append(f"   5-seed mean: {cfg['our_mean_auc']:.4f}")
    report.append(f"   {len(all_results)}-seed mean: {np.mean(aucs):.4f}")
    report.append(f"   Difference: {abs(np.mean(aucs) - cfg['our_mean_auc']):.4f}")
    report.append("")

    report_text = "\n".join(report)
    print("\n" + report_text)

    # Save report
    outfile = os.path.join(results_dir, f"{ds}_tan_stability.txt")
    with open(outfile, "w") as f:
        f.write(report_text)
    print(f"\nFull report saved to: {outfile}")


if __name__ == "__main__":
    main()
