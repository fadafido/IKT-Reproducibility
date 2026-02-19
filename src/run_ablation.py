"""
IKT Ablation Study (Paper Table 5 Reproduction)
=================================================
Runs IKT-1, IKT-2, IKT-3 variants by subsetting features
from existing CSV files (no re-running BKT/k-means needed).

  IKT-1: skill_ID + skill_mastery               (2 features)
  IKT-2: skill_ID + skill_mastery + ability      (3 features)
  IKT-3: all 4 features                          (already done)

Usage:
  python run_ablation.py --dataset ass09
  python run_ablation.py --dataset algebra
  python run_ablation.py --dataset ass12

Results saved to: results_ablation/<dataset>_ablation_results.txt

Author: Fadi Alazayem (25002207) - BUiD MSc AI
"""

import subprocess
import sys
import os
import argparse
import time
import pandas as pd

# Project root (one level up from src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# DATASET CONFIGURATION
# ============================================================
DATASET_CONFIG = {
    "ass09": {
        "name": "ASSISTments 2009",
        "csv_dir": "data/ass09",                # root: train_data.csv, test_data.csv
        "relation": "ASS2009",
        "paper_ikt1": 0.705,
        "paper_ikt2": 0.715,
        "paper_ikt3": 0.797,
        "paper_rmse1": 0.443,
        "paper_rmse2": 0.441,
        "paper_rmse3": 0.411,
        "our_ikt3": 0.8056,
    },
    "algebra": {
        "name": "Algebra 2005-2006",
        "csv_dir": "data/algebra",
        "relation": "Algebra05",
        "paper_ikt1": 0.731,
        "paper_ikt2": 0.734,
        "paper_ikt3": 0.846,      # Table 5 value (Table 3 says 0.851)
        "paper_rmse1": 0.395,
        "paper_rmse2": 0.394,
        "paper_rmse3": 0.354,
        "our_ikt3": 0.7692,
    },
    "ass12": {
        "name": "ASSISTments 2012 (5K cap)",
        "csv_dir": "data/ass12",
        "relation": "ASS2012",
        "paper_ikt1": 0.690,
        "paper_ikt2": 0.696,
        "paper_ikt3": 0.767,
        "paper_rmse1": 0.437,
        "paper_rmse2": 0.435,
        "paper_rmse3": 0.413,
        "our_ikt3": 0.7786,
    },
}

# Feature subsets for each IKT variant
IKT_VARIANTS = {
    "IKT-1": ["skill_ID", "skill_mastery", "correctness"],
    "IKT-2": ["skill_ID", "skill_mastery", "ability_profile", "correctness"],
    "IKT-3": ["skill_ID", "skill_mastery", "ability_profile", "problem_difficulty", "correctness"],
}


# ============================================================
# UTILITIES
# ============================================================
def detect_weka():
    """Find weka.jar - check multiple locations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(PROJECT_ROOT, "weka.jar"),
        os.path.join(PROJECT_ROOT, "weka-mac", "weka.jar"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    print("ERROR: Cannot find weka.jar")
    print("Place weka.jar in this folder or weka-mac/ subfolder")
    sys.exit(1)


def detect_java_version():
    """Check if Java needs --add-opens flag (Java 9+)."""
    try:
        result = subprocess.run(["java", "-version"], capture_output=True, text=True)
        return "1.8" not in result.stderr
    except Exception:
        return False


def csv_to_arff_subset(csv_file, arff_file, columns, relation_name):
    """Read CSV, keep only specified columns, write ARFF."""
    df = pd.read_csv(csv_file)
    df = df[columns]

    # Build ARFF header dynamically
    header_lines = [f"@relation {relation_name}", ""]
    for col in columns:
        if col == "correctness":
            header_lines.append("@attribute correctness {1,0}")
        else:
            header_lines.append(f"@attribute {col} numeric")
    header_lines.append("")
    header_lines.append("@data")
    header = "\n".join(header_lines) + "\n"

    with open(arff_file, "w") as f:
        f.write(header)
        for _, row in df.iterrows():
            vals = []
            for col in columns:
                if col == "skill_mastery":
                    vals.append(str(row[col]))
                else:
                    vals.append(str(int(row[col])))
            f.write(",".join(vals) + "\n")

    return len(df)


def run_weka_tan(train_arff, test_arff, weka_jar, needs_add_opens):
    """Run WEKA TAN and extract test AUC + RMSE."""
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

    auc = None
    rmse = None

    # Extract AUC from test section
    in_test = False
    for line in output.split("\n"):
        if "Error on test data" in line:
            in_test = True
        if in_test and "Root mean squared error" in line:
            parts = line.split()
            try:
                rmse = float(parts[-1])
            except (ValueError, IndexError):
                pass
        if in_test and "Weighted Avg." in line:
            parts = line.split()
            try:
                auc = float(parts[-2])
            except (ValueError, IndexError):
                pass

    return auc, rmse, output


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="IKT Ablation Study")
    parser.add_argument("--dataset", required=True, choices=["ass09", "algebra", "ass12"],
                        help="Dataset to run ablation on")
    args = parser.parse_args()

    ds = args.dataset
    cfg = DATASET_CONFIG[ds]
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Locate source CSVs
    csv_dir = os.path.join(PROJECT_ROOT, cfg["csv_dir"])
    train_csv = os.path.join(csv_dir, "train_data_out.csv")
    test_csv = os.path.join(csv_dir, "test_data_out.csv")

    if not os.path.exists(train_csv):
        # Fallback: try without _out suffix
        train_csv = os.path.join(csv_dir, "train_data.csv")
        test_csv = os.path.join(csv_dir, "test_data.csv")

    if not os.path.exists(train_csv):
        # Try results from multi-seed run
        train_csv = os.path.join(PROJECT_ROOT, "results/multi_seed", "train_data_out_seed42.csv")
        test_csv = os.path.join(PROJECT_ROOT, "results/multi_seed", "test_data_out_seed42.csv")

    if not os.path.exists(train_csv):
        print(f"ERROR: Cannot find train/test CSV in {csv_dir}/")
        print("Run the main pipeline first (FeatureEngineering_v2.py)")
        sys.exit(1)

    print(f"Using CSVs from: {os.path.dirname(train_csv)}")

    weka_jar = detect_weka()
    needs_add_opens = detect_java_version()

    # Output directory
    results_dir = os.path.join(PROJECT_ROOT, "results/ablation")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 65)
    print(f"IKT ABLATION STUDY - {cfg['name']}")
    print(f"WEKA: {weka_jar}")
    print(f"Java add-opens: {needs_add_opens}")
    print("=" * 65)

    results = []
    start_total = time.time()

    for variant_name, columns in IKT_VARIANTS.items():
        print(f"\n--- {variant_name}: {[c for c in columns if c != 'correctness']} ---")
        start = time.time()

        # Create subset ARFFs in results_ablation/
        train_arff = os.path.join(results_dir, f"{ds}_{variant_name}_train.arff")
        test_arff = os.path.join(results_dir, f"{ds}_{variant_name}_test.arff")

        n_train = csv_to_arff_subset(train_csv, train_arff, columns,
                                      f"{cfg['relation']}_{variant_name}")
        n_test = csv_to_arff_subset(test_csv, test_arff, columns,
                                     f"{cfg['relation']}_{variant_name}")
        print(f"  Created ARFFs: {n_train} train, {n_test} test rows")

        # Run WEKA
        print(f"  Running WEKA TAN...")
        auc, rmse, weka_output = run_weka_tan(train_arff, test_arff, weka_jar, needs_add_opens)

        elapsed = time.time() - start

        # Save WEKA output
        with open(os.path.join(results_dir, f"{ds}_{variant_name}_weka.txt"), "w") as f:
            f.write(weka_output)

        if auc:
            print(f"  AUC = {auc:.4f}  |  RMSE = {rmse:.4f}  |  Time: {elapsed:.1f}s")
            results.append((variant_name, columns, auc, rmse, elapsed))
        else:
            print(f"  FAILED to extract AUC")
            results.append((variant_name, columns, None, None, elapsed))

    total_time = time.time() - start_total

    # ============================================================
    # SUMMARY
    # ============================================================
    summary_lines = []
    summary_lines.append("=" * 75)
    summary_lines.append(f"ABLATION STUDY RESULTS - {cfg['name']}")
    summary_lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append(f"Total runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
    summary_lines.append("=" * 75)
    summary_lines.append("")
    summary_lines.append(f"{'Variant':<10} {'Features':<45} {'Our AUC':<10} {'Paper AUC':<10} {'Delta':<10} {'Our RMSE':<10} {'Paper RMSE':<10}")
    summary_lines.append("-" * 75)

    paper_aucs = [cfg["paper_ikt1"], cfg["paper_ikt2"], cfg["paper_ikt3"]]
    paper_rmses = [cfg["paper_rmse1"], cfg["paper_rmse2"], cfg["paper_rmse3"]]

    for i, (vname, cols, auc, rmse, elapsed) in enumerate(results):
        feat_str = ", ".join([c for c in cols if c != "correctness"])
        if auc:
            delta = auc - paper_aucs[i]
            delta_str = f"{delta:+.4f}"
            rmse_str = f"{rmse:.4f}" if rmse else "N/A"
        else:
            delta_str = "FAILED"
            rmse_str = "FAILED"
        summary_lines.append(
            f"{vname:<10} {feat_str:<45} {auc if auc else 'N/A':<10} "
            f"{paper_aucs[i]:<10} {delta_str:<10} {rmse_str:<10} {paper_rmses[i]:<10}"
        )

    summary_lines.append("-" * 75)
    summary_lines.append("")

    # Key analysis
    if all(r[2] for r in results):
        auc1, auc2, auc3 = results[0][2], results[1][2], results[2][2]
        summary_lines.append("FEATURE CONTRIBUTION ANALYSIS:")
        summary_lines.append(f"  skill_mastery (BKT) alone:          AUC = {auc1:.4f}")
        summary_lines.append(f"  + ability_profile (k-means):        AUC = {auc2:.4f}  (+{auc2-auc1:.4f})")
        summary_lines.append(f"  + problem_difficulty (IRT):          AUC = {auc3:.4f}  (+{auc3-auc2:.4f})")
        summary_lines.append(f"  Total improvement (IKT-1 -> IKT-3): +{auc3-auc1:.4f}")
        summary_lines.append("")

        # Compare with paper's feature contributions
        p1, p2, p3 = paper_aucs
        summary_lines.append("PAPER FEATURE CONTRIBUTION:")
        summary_lines.append(f"  skill_mastery (BKT) alone:          AUC = {p1:.3f}")
        summary_lines.append(f"  + ability_profile (k-means):        AUC = {p2:.3f}  (+{p2-p1:.3f})")
        summary_lines.append(f"  + problem_difficulty (IRT):          AUC = {p3:.3f}  (+{p3-p2:.3f})")
        summary_lines.append(f"  Total improvement (IKT-1 -> IKT-3): +{p3-p1:.3f}")

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    # Save to file
    outfile = os.path.join(results_dir, f"{ds}_ablation_results.txt")
    with open(outfile, "w") as f:
        f.write(summary_text)
    print(f"\nResults saved to: {outfile}")


if __name__ == "__main__":
    main()
