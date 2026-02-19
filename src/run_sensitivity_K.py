"""
Sensitivity Analysis: K Clusters
=================================
Tests IKT pipeline with different numbers of ability clusters.
Paper uses K=7. We test K = 4, 5, 6, 7, 8, 10.

Does NOT modify FeatureEngineering_v2.py.
Creates a temporary copy for each run.

Usage: python run_sensitivity_K.py
"""

import subprocess
import sys
import os
import re
import shutil
import platform

# Project root (one level up from src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


K_VALUES = [4, 5, 6, 7, 8, 10]
FIXED_SEED = 42  # Keep seed constant to isolate K's effect

TEMP_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_temp_feature_eng.py")


def detect_weka():
    """Find weka.jar and determine if --add-opens is needed."""
    # Check common locations
    candidates = [
        os.path.join(PROJECT_ROOT, "weka.jar"),
        os.path.join(PROJECT_ROOT, "weka-mac", "weka.jar"),
    ]
    
    weka_jar = None
    for path in candidates:
        if os.path.exists(path):
            weka_jar = path
            break
    
    if not weka_jar:
        print("ERROR: Cannot find weka.jar")
        print("Place weka.jar in the project folder or in weka-mac/")
        sys.exit(1)
    
    # Check Java version for --add-opens flag
    try:
        result = subprocess.run(["java", "-version"], capture_output=True, text=True)
        version_str = result.stderr  # Java prints version to stderr
        needs_add_opens = "1.8" not in version_str
    except Exception:
        needs_add_opens = False
    
    return weka_jar, needs_add_opens


def run_weka_tan(train_arff, test_arff, weka_jar, needs_add_opens, label=""):
    """Run WEKA TAN and extract test AUC."""
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
    output = result.stdout

    # Save full WEKA output
    os.makedirs(os.path.join(PROJECT_ROOT, "results", "sensitivity_K"), exist_ok=True)
    with open(os.path.join(PROJECT_ROOT, "results", "sensitivity_K", f"weka_output_{label}.txt"), "w") as f:
        f.write(output)

    # Find ROC Area in test data section only
    in_test_section = False
    for line in output.split("\n"):
        if "Error on test data" in line:
            in_test_section = True
        if in_test_section and "Weighted Avg." in line:
            parts = line.split()
            try:
                roc_area = float(parts[-2])
                return roc_area
            except (ValueError, IndexError):
                pass
    return None


def run_pipeline_with_K(k_value, seed):
    """Create temp script with modified K, run it, return success."""
    # Read original
    with open(os.path.join(os.path.dirname(__file__), "FeatureEngineering_v2.py"), "r") as f:
        content = f.read()

    # Modify K and seed
    content = re.sub(r"CLUSTER_NUM = \d+", f"CLUSTER_NUM = {k_value}", content)
    content = re.sub(r"RANDOM_SEED = \d+", f"RANDOM_SEED = {seed}", content)

    # Write temp file
    with open(TEMP_SCRIPT, "w") as f:
        f.write(content)

    # Run temp script
    result = subprocess.run(
        [sys.executable, TEMP_SCRIPT], capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return False

    # Save ARFF files per K value
    results_dir = os.path.join(PROJECT_ROOT, "results", "sensitivity_K")
    data_dir = os.path.join(PROJECT_ROOT, "data", "ass09")
    os.makedirs(results_dir, exist_ok=True)
    for fname in ["train_data.arff", "test_data.arff", "train_data_out.csv", "test_data_out.csv"]:
        src_file = os.path.join(data_dir, fname)
        if os.path.exists(src_file):
            name, ext = os.path.splitext(fname)
            shutil.copy(src_file, os.path.join(results_dir, f"{name}_K{k_value}{ext}"))
    print(f"  Saved ARFF/CSV files to results/sensitivity_K/ (K={k_value})")

    return True


def main():
    weka_jar, needs_add_opens = detect_weka()

    print("=" * 60)
    print("IKT Sensitivity Analysis: K Clusters")
    print(f"K values: {K_VALUES}")
    print(f"Fixed seed: {FIXED_SEED}")
    print(f"WEKA: {weka_jar}")
    print(f"Java add-opens: {needs_add_opens}")
    print("=" * 60)

    results = []

    for i, k in enumerate(K_VALUES):
        print(f"\n--- Run {i+1}/{len(K_VALUES)}: K={k} ---")

        print(f"  Running feature engineering (K={k}, seed={FIXED_SEED})...")
        success = run_pipeline_with_K(k, FIXED_SEED)
        if not success:
            continue

        print("  Running WEKA TAN...")
        auc = run_weka_tan(os.path.join(PROJECT_ROOT, "data", "ass09", "train_data.arff"), os.path.join(PROJECT_ROOT, "data", "ass09", "test_data.arff"), weka_jar, needs_add_opens, f"K{k}")

        if auc:
            results.append((k, auc))
            print(f"  -> K={k}: AUC = {auc:.4f}")
        else:
            print(f"  -> K={k}: Failed to extract AUC")

    # Cleanup temp file
    if os.path.exists(TEMP_SCRIPT):
        os.remove(TEMP_SCRIPT)

    # Summary
    print("\n" + "=" * 60)
    print("SENSITIVITY RESULTS: K CLUSTERS")
    print(f"(Fixed seed={FIXED_SEED}, interval=20)")
    print("=" * 60)
    print(f"{'K':<10} {'AUC':<10}")
    print("-" * 20)
    for k, auc in results:
        marker = " <-- paper default" if k == 7 else ""
        print(f"{k:<10} {auc:<10.4f}{marker}")

    if results:
        import numpy as np

        aucs = [r[1] for r in results]
        print("-" * 20)
        print(f"{'Best K':<10} {results[max(range(len(results)), key=lambda i: results[i][1])][0]} (AUC={max(aucs):.4f})")
        print(f"{'Worst K':<10} {results[min(range(len(results)), key=lambda i: results[i][1])][0]} (AUC={min(aucs):.4f})")
        print(f"{'Range':<10} {max(aucs) - min(aucs):.4f}")


if __name__ == "__main__":
    main()
