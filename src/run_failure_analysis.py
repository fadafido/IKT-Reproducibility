"""
IKT Failure Case Analysis
===========================
Runs WEKA TAN with -p 0 to get per-instance predictions,
then analyzes high-confidence misclassifications.

This feeds Part B (Critical Analysis) of the assignment.

Usage:
  python run_failure_analysis.py --dataset ass09
  python run_failure_analysis.py --dataset algebra
  python run_failure_analysis.py --dataset ass12

Results saved to: results_failure/<dataset>_failure_analysis.txt

Author: Fadi Alazayem (25002207) - BUiD MSc AI
"""

import subprocess
import sys
import os
import argparse
import time
import re
import pandas as pd
import numpy as np

# Project root (one level up from src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# DATASET CONFIGURATION
# ============================================================
DATASET_CONFIG = {
    "ass09": {
        "name": "ASSISTments 2009",
        "arff_dir": "data/ass09",
        "csv_dir": "data/ass09",
    },
    "algebra": {
        "name": "Algebra 2005-2006",
        "arff_dir": "data/algebra",
        "csv_dir": "data/algebra",
    },
    "ass12": {
        "name": "ASSISTments 2012 (5K cap)",
        "arff_dir": "data/ass12",
        "csv_dir": "data/ass12",
    },
}


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


def run_weka_predictions(train_arff, test_arff, weka_jar, needs_add_opens):
    """Run WEKA TAN with -p 0 to get per-instance predictions."""
    cmd = ["java", "-Xmx2g"]
    if needs_add_opens:
        cmd += ["--add-opens", "java.base/java.lang=ALL-UNNAMED"]
    cmd += [
        "-cp", weka_jar,
        "weka.classifiers.bayes.BayesNet",
        "-t", train_arff,
        "-T", test_arff,
        "-p", "0",       # Output predictions for test instances
        "-Q", "weka.classifiers.bayes.net.search.local.TAN",
        "--", "-S", "BAYES",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout, result.stderr


def parse_weka_predictions(prediction_text):
    """
    Parse WEKA -p 0 output into a DataFrame.

    WEKA -p 0 output format (typical):
      inst#     actual  predicted error prediction
          1     1:1         1:1       0.952
          2     2:0         1:1   +   0.631
          ...

    The prediction column is the confidence for the predicted class.
    """
    lines = prediction_text.strip().split("\n")
    records = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith("=") or line.startswith("inst"):
            continue

        # Try to parse prediction lines
        # Format: inst# actual predicted [+] prediction
        parts = line.split()
        if len(parts) < 4:
            continue

        try:
            inst = int(parts[0])
        except ValueError:
            continue

        actual_str = parts[1]    # e.g. "1:1" or "2:0"
        pred_str = parts[2]      # e.g. "1:1" or "2:0"

        # Check if there's an error marker (+)
        has_error = "+" in parts
        if has_error:
            confidence = float(parts[-1])
        else:
            confidence = float(parts[-1])

        # Parse actual and predicted class
        # WEKA format: classIndex:classLabel
        actual_label = actual_str.split(":")[-1]
        pred_label = pred_str.split(":")[-1]

        records.append({
            "inst": inst,
            "actual": int(actual_label),
            "predicted": int(pred_label),
            "correct": actual_label == pred_label,
            "confidence": confidence,
        })

    return pd.DataFrame(records)


def analyze_failures(df, test_csv_path, results_dir, dataset_name):
    """Comprehensive failure case analysis."""
    lines = []

    total = len(df)
    correct_count = df["correct"].sum()
    wrong_count = total - correct_count
    accuracy = correct_count / total * 100

    lines.append(f"Total test instances:    {total}")
    lines.append(f"Correct predictions:     {correct_count} ({accuracy:.2f}%)")
    lines.append(f"Wrong predictions:       {wrong_count} ({100-accuracy:.2f}%)")
    lines.append("")

    # ---- HIGH-CONFIDENCE ERRORS ----
    lines.append("=" * 65)
    lines.append("HIGH-CONFIDENCE ERRORS (predicted with >80% confidence, but WRONG)")
    lines.append("=" * 65)

    high_conf_errors = df[(~df["correct"]) & (df["confidence"] > 0.80)]
    lines.append(f"Count: {len(high_conf_errors)} ({len(high_conf_errors)/total*100:.2f}% of all test)")
    lines.append("")

    if len(high_conf_errors) > 0:
        lines.append("  Confidence distribution of high-confidence errors:")
        for threshold in [0.80, 0.85, 0.90, 0.95]:
            ct = len(high_conf_errors[high_conf_errors["confidence"] >= threshold])
            lines.append(f"    >= {threshold:.0%} confidence: {ct}")
        lines.append("")

        # Break down by error type
        fp = high_conf_errors[(high_conf_errors["actual"] == 0) & (high_conf_errors["predicted"] == 1)]
        fn = high_conf_errors[(high_conf_errors["actual"] == 1) & (high_conf_errors["predicted"] == 0)]
        lines.append(f"  False Positives (predicted correct, actually wrong): {len(fp)}")
        lines.append(f"  False Negatives (predicted wrong, actually correct): {len(fn)}")
        lines.append("")

        # Show top 10 worst errors
        worst = high_conf_errors.nlargest(10, "confidence")
        lines.append("  Top 10 highest-confidence errors:")
        lines.append(f"  {'Inst#':<8} {'Actual':<8} {'Predicted':<10} {'Confidence':<12}")
        lines.append("  " + "-" * 38)
        for _, row in worst.iterrows():
            lines.append(f"  {int(row['inst']):<8} {int(row['actual']):<8} {int(row['predicted']):<10} {row['confidence']:.4f}")

    lines.append("")

    # ---- LOW-CONFIDENCE CORRECT ----
    lines.append("=" * 65)
    lines.append("LOW-CONFIDENCE CORRECT (correct but <60% confidence = 'lucky')")
    lines.append("=" * 65)
    low_conf_correct = df[(df["correct"]) & (df["confidence"] < 0.60)]
    lines.append(f"Count: {len(low_conf_correct)} ({len(low_conf_correct)/total*100:.2f}% of all test)")
    lines.append("")

    # ---- CONFUSION MATRIX ----
    lines.append("=" * 65)
    lines.append("CONFUSION MATRIX")
    lines.append("=" * 65)
    tp = len(df[(df["actual"] == 1) & (df["predicted"] == 1)])
    tn = len(df[(df["actual"] == 0) & (df["predicted"] == 0)])
    fp_all = len(df[(df["actual"] == 0) & (df["predicted"] == 1)])
    fn_all = len(df[(df["actual"] == 1) & (df["predicted"] == 0)])

    lines.append(f"                    Predicted=1(correct)  Predicted=0(wrong)")
    lines.append(f"  Actual=1(correct)    TP={tp:<8}           FN={fn_all:<8}")
    lines.append(f"  Actual=0(wrong)      FP={fp_all:<8}           TN={tn:<8}")
    lines.append("")

    precision = tp / (tp + fp_all) if (tp + fp_all) > 0 else 0
    recall = tp / (tp + fn_all) if (tp + fn_all) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    lines.append(f"  Precision: {precision:.4f}")
    lines.append(f"  Recall:    {recall:.4f}")
    lines.append(f"  F1-Score:  {f1:.4f}")
    lines.append("")

    # ---- CONFIDENCE DISTRIBUTION ----
    lines.append("=" * 65)
    lines.append("CONFIDENCE DISTRIBUTION")
    lines.append("=" * 65)
    bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    lines.append(f"  {'Bin':<15} {'Total':<8} {'Correct':<10} {'Wrong':<8} {'Accuracy':<10}")
    lines.append("  " + "-" * 51)
    for i in range(len(bins) - 1):
        mask = (df["confidence"] >= bins[i]) & (df["confidence"] < bins[i+1])
        if i == len(bins) - 2:  # last bin includes 1.0
            mask = (df["confidence"] >= bins[i]) & (df["confidence"] <= bins[i+1])
        bin_df = df[mask]
        if len(bin_df) > 0:
            bin_correct = bin_df["correct"].sum()
            bin_wrong = len(bin_df) - bin_correct
            bin_acc = bin_correct / len(bin_df) * 100
            lines.append(f"  {bins[i]:.1f}-{bins[i+1]:.1f}        {len(bin_df):<8} {bin_correct:<10} {bin_wrong:<8} {bin_acc:.1f}%")
    lines.append("")

    # ---- FEATURE ANALYSIS OF ERRORS (if CSV available) ----
    if os.path.exists(test_csv_path):
        lines.append("=" * 65)
        lines.append("FEATURE PATTERNS IN ERRORS")
        lines.append("=" * 65)
        try:
            test_features = pd.read_csv(test_csv_path)
            # Align by index (WEKA inst# is 1-based)
            if len(test_features) >= len(df):
                # Add prediction results to feature data
                feature_df = test_features.iloc[:len(df)].copy()
                feature_df["pred_correct"] = df["correct"].values
                feature_df["confidence"] = df["confidence"].values

                # Analyze error patterns by skill mastery level
                lines.append("\n  Error rate by skill_mastery quartile:")
                feature_df["mastery_bin"] = pd.qcut(feature_df["skill_mastery"], 4,
                                                      labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"],
                                                      duplicates="drop")
                for grp_name, grp in feature_df.groupby("mastery_bin", observed=True):
                    err_rate = (1 - grp["pred_correct"].mean()) * 100
                    lines.append(f"    {grp_name}: {len(grp)} instances, error rate = {err_rate:.1f}%")

                lines.append("")

                # Error rate by difficulty
                lines.append("  Error rate by problem_difficulty:")
                for diff_val in sorted(feature_df["problem_difficulty"].unique()):
                    sub = feature_df[feature_df["problem_difficulty"] == diff_val]
                    if len(sub) >= 10:  # Only show for meaningful sample sizes
                        err_rate = (1 - sub["pred_correct"].mean()) * 100
                        lines.append(f"    difficulty={int(diff_val)}: {len(sub)} instances, error rate = {err_rate:.1f}%")

                lines.append("")

                # Error rate by ability profile
                lines.append("  Error rate by ability_profile (cluster):")
                for cl in sorted(feature_df["ability_profile"].unique()):
                    sub = feature_df[feature_df["ability_profile"] == cl]
                    if len(sub) >= 10:
                        err_rate = (1 - sub["pred_correct"].mean()) * 100
                        lines.append(f"    cluster={int(cl)}: {len(sub)} instances, error rate = {err_rate:.1f}%")

        except Exception as e:
            lines.append(f"  (Could not load feature CSV: {e})")

    lines.append("")

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="IKT Failure Case Analysis")
    parser.add_argument("--dataset", required=True, choices=["ass09", "algebra", "ass12"])
    args = parser.parse_args()

    ds = args.dataset
    cfg = DATASET_CONFIG[ds]
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Locate ARFF files
    arff_dir = os.path.join(PROJECT_ROOT, cfg["arff_dir"])
    train_arff = os.path.join(arff_dir, "train_data.arff")
    test_arff = os.path.join(arff_dir, "test_data.arff")
    test_csv = os.path.join(arff_dir, "test_data_out.csv")

    if not os.path.exists(train_arff):
        print(f"ERROR: Cannot find train_data.arff in {arff_dir}/")
        print("Run the main pipeline first.")
        sys.exit(1)

    weka_jar = detect_weka()
    needs_add_opens = detect_java_version()

    results_dir = os.path.join(PROJECT_ROOT, "results/failure")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 65)
    print(f"IKT FAILURE CASE ANALYSIS - {cfg['name']}")
    print(f"WEKA: {weka_jar}")
    print("=" * 65)

    start = time.time()

    # Run WEKA with predictions
    print("\nRunning WEKA TAN with per-instance predictions (-p 0)...")
    pred_output, stderr = run_weka_predictions(train_arff, test_arff, weka_jar, needs_add_opens)

    # Save raw WEKA prediction output
    raw_file = os.path.join(results_dir, f"{ds}_weka_predictions_raw.txt")
    with open(raw_file, "w") as f:
        f.write(pred_output)
    print(f"  Raw predictions saved: {raw_file}")

    if stderr and "Exception" in stderr:
        print(f"WEKA ERROR: {stderr[:500]}")
        sys.exit(1)

    # Parse predictions
    print("Parsing predictions...")
    df = parse_weka_predictions(pred_output)
    print(f"  Parsed {len(df)} predictions")

    if len(df) == 0:
        print("ERROR: No predictions parsed. Check WEKA output.")
        sys.exit(1)

    # Save predictions CSV
    pred_csv = os.path.join(results_dir, f"{ds}_predictions.csv")
    df.to_csv(pred_csv, index=False)
    print(f"  Predictions CSV: {pred_csv}")

    # Analyze
    print("\nAnalyzing failure cases...\n")
    analysis = analyze_failures(df, test_csv, results_dir, ds)

    elapsed = time.time() - start

    # Build full report
    report_lines = []
    report_lines.append("=" * 65)
    report_lines.append(f"IKT FAILURE CASE ANALYSIS - {cfg['name']}")
    report_lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Runtime: {elapsed:.1f}s")
    report_lines.append(f"Train ARFF: {train_arff}")
    report_lines.append(f"Test ARFF: {test_arff}")
    report_lines.append("=" * 65)
    report_lines.append("")
    report_lines.append(analysis)

    report = "\n".join(report_lines)
    print(report)

    # Save report
    outfile = os.path.join(results_dir, f"{ds}_failure_analysis.txt")
    with open(outfile, "w") as f:
        f.write(report)
    print(f"\nFull report saved to: {outfile}")


if __name__ == "__main__":
    main()
