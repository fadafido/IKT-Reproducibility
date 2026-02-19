"""
Multi-Seed IKT Runner
=====================
Runs the IKT pipeline with different random seeds to measure
variance in the ability_profile clustering and final AUC.

Usage: python run_multi_seed.py [num_seeds]
Default: 5 seeds (42, 123, 456, 789, 1011)

This produces results for Part A of the assignment.
"""

from random import seed
import subprocess
import sys
import os
import re

# Project root (one level up from src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


SEEDS = [42, 123, 456, 789, 1011]


def run_weka_tan(train_arff, test_arff, weka_jar, seed=0):
    """Run WEKA TAN and extract AUC from output."""
    cmd = [
        "java",
        "-Xmx2g",
        "-cp",
        weka_jar,
        "weka.classifiers.bayes.BayesNet",
        "-t",
        train_arff,
        "-T",
        test_arff,
        "-Q",
        "weka.classifiers.bayes.net.search.local.TAN",
        "--",
        "-S",
        "BAYES",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout

    # Save full WEKA output for this run
    os.makedirs(os.path.join(PROJECT_ROOT, "results", "multi_seed"), exist_ok=True)
    with open(os.path.join(PROJECT_ROOT, "results", "multi_seed", f"weka_output_seed{seed}.txt"), "w") as f:
        f.write(output)

    # Find ROC Area in Weighted Avg line
    # Find ROC Area in Weighted Avg line (test data section only)
    in_test_section = False
    for line in output.split("\n"):
        if "Error on test data" in line:
            in_test_section = True
        if in_test_section and "Weighted Avg." in line:
            parts = line.split()
            # ROC Area is the 8th numeric column (index 8 in parts list)
            # Parts: Weighted Avg. TP FP Prec Recall F MCC ROC PRC
            try:
                roc_area = float(parts[-2])  # second to last = ROC Area
                return roc_area
            except (ValueError, IndexError):
                pass
    return None


def modify_seed_and_run(seed):
    """Modify FeatureEngineering_v2.py seed and run pipeline."""
    import shutil

    # Read current file
    with open(os.path.join(os.path.dirname(__file__), "FeatureEngineering_v2.py"), "r") as f:
        content = f.read()

    # Replace seed
    content = re.sub(r"RANDOM_SEED = \d+", f"RANDOM_SEED = {seed}", content)

    temp_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_temp_multi_seed_fe.py")
    with open(temp_script, "w") as f:
        f.write(content)

    # Run pipeline
    result = subprocess.run(
        [sys.executable, temp_script], capture_output=True, text=True
    )

    # Clean up temp file
    if os.path.exists(temp_script):
        os.remove(temp_script)

    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return None

    # Save copies per seed
    results_dir = os.path.join(PROJECT_ROOT, "results", "multi_seed")
    data_dir = os.path.join(PROJECT_ROOT, "data", "ass09")
    os.makedirs(results_dir, exist_ok=True)
    for fname in ["train_data.arff", "test_data.arff", "train_data_out.csv", "test_data_out.csv"]:
        src_file = os.path.join(data_dir, fname)
        if os.path.exists(src_file):
            name, ext = os.path.splitext(fname)
            shutil.copy(src_file, os.path.join(results_dir, f"{name}_seed{seed}{ext}"))
    print(f"  Saved ARFF/CSV files to results_seed/  (seed {seed})")

    return True


def main():
    num_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else len(SEEDS)
    seeds = SEEDS[:num_seeds]
    weka_jar = os.path.join(PROJECT_ROOT, "weka.jar")
    # Adjust path as needed

    print("=" * 60)
    print("IKT Multi-Seed Reproducibility Test")
    print(f"Seeds: {seeds}")
    print("=" * 60)

    results = []

    for i, seed in enumerate(seeds):
        print(f"\n--- Run {i+1}/{len(seeds)}: Seed {seed} ---")

        print("  Running feature engineering...")
        success = modify_seed_and_run(seed)
        if not success:
            continue

        print("  Running WEKA TAN...")
        auc = run_weka_tan(os.path.join(PROJECT_ROOT, "data", "ass09", "train_data.arff"), os.path.join(PROJECT_ROOT, "data", "ass09", "test_data.arff"), weka_jar, seed)

        if auc:
            results.append((seed, auc))
            print(f"  ✅ Seed {seed}: AUC = {auc:.4f}")
        else:
            print(f"  ❌ Seed {seed}: Failed to extract AUC")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Seed':<10} {'AUC':<10}")
    print("-" * 20)
    for seed, auc in results:
        print(f"{seed:<10} {auc:<10.4f}")

    if results:
        aucs = [r[1] for r in results]
        import numpy as np


        print("-" * 20)
        print(f"{'Mean':<10} {np.mean(aucs):<10.4f}")
        print(f"{'Std':<10} {np.std(aucs):<10.4f}")
        print(f"{'Min':<10} {np.min(aucs):<10.4f}")
        print(f"{'Max':<10} {np.max(aucs):<10.4f}")
        print(f"\nPaper reported: 0.797 (5-fold CV)")
        print(f"Difference: {abs(np.mean(aucs) - 0.797):.4f}")


if __name__ == "__main__":
    main()
