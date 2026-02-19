"""
5-Fold Cross-Validation using Author's Preprocessed Data
=========================================================
Runs WEKA TAN classifier on the original authors' 5-fold CV splits
for all 3 datasets (ASS-09, ASS-12, KDD/Algebra).

This reproduces the paper's exact evaluation protocol:
  - 5-fold student-level cross-validation
  - TAN (Tree-Augmented Naive Bayes) with Bayes scoring
  - Reports AUC and RMSE per fold + mean ± std

Data source: Authors' shared Google Drive
  https://drive.google.com/drive/folders/1Wuilcb_ash1r5MT3tgMc78PDPUh0n3uT

Usage:
    python run_5fold_cv.py

Requirements:
    - Java installed (for WEKA)
    - weka.jar in project root
"""

import subprocess
import re
import os
import sys
import time
import numpy as np

# Project root (one level up from src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# CONFIGURATION
# ============================================================
WEKA_JAR = os.path.join(PROJECT_ROOT, "weka.jar")
DATA_DIR = os.path.join(PROJECT_ROOT, "data_5fold")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "5fold_cv")

DATASETS = {
    "ASS09": {
        "folder": "ASS09",
        "paper_auc": 0.797,
        "paper_rmse": 0.421,
    },
    "ASS12": {
        "folder": "ASS12",
        "paper_auc": 0.767,
        "paper_rmse": 0.432,
    },
    "KDD": {
        "folder": "KDD",
        "paper_auc": 0.851,
        "paper_rmse": 0.354,
    },
}

NUM_FOLDS = 5


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def run_weka_tan(train_arff, test_arff):
    """
    Run WEKA TAN classifier and return parsed results.
    Returns dict with: accuracy, auc, rmse, kappa, precision, recall, f1,
                        confusion_matrix, network_structure, full_output
    """
    cmd = [
        "java", "-Xmx8g", "-cp", WEKA_JAR,
        "weka.classifiers.bayes.BayesNet",
        "-t", train_arff,
        "-T", test_arff,
        "-Q", "weka.classifiers.bayes.net.search.local.TAN",
        "--", "-S", "BAYES",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

    if result.returncode != 0:
        log(f"  WEKA ERROR (rc={result.returncode}): {result.stderr[:500]}")
        if result.stdout:
            log(f"  WEKA STDOUT (last 300 chars): ...{result.stdout[-300:]}")
        return None

    output = result.stdout
    if not output.strip():
        log(f"  WEKA produced empty output. stderr: {result.stderr[:500]}")
        return None

    results = {"full_output": output}

    # IMPORTANT: Parse from TEST DATA section only (after "Error on test data")
    # The output contains both training and test sections - we need test only
    test_section_match = re.search(r"=== Error on test data ===(.+)", output, re.DOTALL)
    if not test_section_match:
        log(f"  WARNING: Could not find test data section in WEKA output")
        return results

    test_section = test_section_match.group(1)

    # Parse AUC (ROC Area) from Weighted Avg line in TEST section
    # Format: Weighted Avg.    TP    FP    Prec   Recall  F     MCC    ROC    PRC
    auc_match = re.search(
        r"Weighted Avg\.\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.-]+\s+([\d.]+)",
        test_section
    )
    if auc_match:
        results["auc"] = float(auc_match.group(1))

    # Parse RMSE from test section
    rmse_match = re.search(r"Root mean squared error\s+([\d.]+)", test_section)
    if rmse_match:
        results["rmse"] = float(rmse_match.group(1))

    # Parse accuracy from test section
    acc_match = re.search(r"Correctly Classified Instances\s+\d+\s+([\d.]+)\s*%", test_section)
    if acc_match:
        results["accuracy"] = float(acc_match.group(1))

    # Parse Kappa from test section
    kappa_match = re.search(r"Kappa statistic\s+([\d.-]+)", test_section)
    if kappa_match:
        results["kappa"] = float(kappa_match.group(1))

    # Parse network structure (from full output, before test section)
    struct_match = re.search(r"Bayes Network Classifier.*?LogScore", output, re.DOTALL)
    if struct_match:
        results["network_structure"] = struct_match.group(0)

    return results


def run_dataset(name, config):
    """Run 5-fold CV for one dataset."""
    log(f"\n{'='*60}")
    log(f"Dataset: {name}")
    log(f"Paper AUC: {config['paper_auc']}, Paper RMSE: {config['paper_rmse']}")
    log(f"{'='*60}")

    folder = os.path.join(DATA_DIR, config["folder"])
    aucs = []
    rmses = []
    accuracies = []
    fold_results = []

    for fold in range(1, NUM_FOLDS + 1):
        train_arff = os.path.join(folder, f"train_fold{fold}.arff")
        test_arff = os.path.join(folder, f"test_fold{fold}.arff")

        if not os.path.exists(train_arff) or not os.path.exists(test_arff):
            log(f"  Fold {fold}: MISSING FILES - skipping")
            continue

        log(f"  Fold {fold}: Running WEKA TAN...")
        start = time.time()
        results = run_weka_tan(train_arff, test_arff)
        elapsed = time.time() - start

        if results is None:
            log(f"  Fold {fold}: FAILED")
            continue

        auc = results.get("auc", 0)
        rmse = results.get("rmse", 0)
        acc = results.get("accuracy", 0)

        aucs.append(auc)
        rmses.append(rmse)
        accuracies.append(acc)
        fold_results.append(results)

        log(f"  Fold {fold}: AUC={auc:.4f}, RMSE={rmse:.4f}, Acc={acc:.2f}% ({elapsed:.1f}s)")

        # Save individual fold output
        out_file = os.path.join(RESULTS_DIR, f"{name}_fold{fold}_weka.txt")
        with open(out_file, "w") as f:
            f.write(results["full_output"])

    if not aucs:
        log(f"  NO RESULTS for {name}")
        return None

    # Compute statistics
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    mean_rmse = np.mean(rmses)
    std_rmse = np.std(rmses)
    mean_acc = np.mean(accuracies)

    delta_auc = mean_auc - config["paper_auc"]
    delta_rmse = mean_rmse - config["paper_rmse"]

    log(f"\n  --- {name} Summary (5-Fold CV) ---")
    log(f"  AUC:  {mean_auc:.4f} ± {std_auc:.4f}  (Paper: {config['paper_auc']}, Δ={delta_auc:+.4f})")
    log(f"  RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}  (Paper: {config['paper_rmse']}, Δ={delta_rmse:+.4f})")
    log(f"  Acc:  {mean_acc:.2f}%")
    log(f"  Per-fold AUCs: {[f'{a:.4f}' for a in aucs]}")
    log(f"  Per-fold RMSEs: {[f'{r:.4f}' for r in rmses]}")

    return {
        "name": name,
        "aucs": aucs,
        "rmses": rmses,
        "accuracies": accuracies,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "mean_rmse": mean_rmse,
        "std_rmse": std_rmse,
        "paper_auc": config["paper_auc"],
        "paper_rmse": config["paper_rmse"],
        "delta_auc": delta_auc,
        "delta_rmse": delta_rmse,
        "fold_results": fold_results,
    }


def main():
    log("=" * 60)
    log("IKT 5-Fold CV Reproduction (Author's Preprocessed Data)")
    log("=" * 60)

    # Verify WEKA
    if not os.path.exists(WEKA_JAR):
        log(f"ERROR: weka.jar not found at {WEKA_JAR}")
        sys.exit(1)

    # Verify Java
    try:
        subprocess.run(["java", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        log("ERROR: Java not found. Install Java to run WEKA.")
        sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = {}
    for name, config in DATASETS.items():
        result = run_dataset(name, config)
        if result:
            all_results[name] = result

    # Final summary table
    log(f"\n{'='*60}")
    log("FINAL SUMMARY: 5-Fold CV Results vs Paper (Table 3 & 4)")
    log(f"{'='*60}")
    log(f"{'Dataset':<10} {'Our AUC':>10} {'Paper AUC':>10} {'ΔAUC':>8} {'Our RMSE':>10} {'Paper RMSE':>11} {'ΔRMSE':>8}")
    log("-" * 70)

    for name in ["ASS09", "ASS12", "KDD"]:
        if name in all_results:
            r = all_results[name]
            log(
                f"{name:<10} "
                f"{r['mean_auc']:>10.4f} "
                f"{r['paper_auc']:>10.3f} "
                f"{r['delta_auc']:>+8.4f} "
                f"{r['mean_rmse']:>10.4f} "
                f"{r['paper_rmse']:>11.3f} "
                f"{r['delta_rmse']:>+8.4f}"
            )

    # Save summary
    summary_file = os.path.join(RESULTS_DIR, "5fold_cv_summary.txt")
    with open(summary_file, "w") as f:
        f.write("IKT 5-Fold CV Reproduction Results\n")
        f.write("=" * 60 + "\n")
        f.write("Data source: Authors' preprocessed 5-fold CV splits\n")
        f.write("Classifier: WEKA BayesNet with TAN search, Bayes scoring\n\n")

        f.write(f"{'Dataset':<10} {'Our AUC':>10} {'±Std':>8} {'Paper AUC':>10} {'ΔAUC':>8} "
                f"{'Our RMSE':>10} {'±Std':>8} {'Paper RMSE':>11} {'ΔRMSE':>8}\n")
        f.write("-" * 85 + "\n")

        for name in ["ASS09", "ASS12", "KDD"]:
            if name in all_results:
                r = all_results[name]
                f.write(
                    f"{name:<10} "
                    f"{r['mean_auc']:>10.4f} "
                    f"{r['std_auc']:>8.4f} "
                    f"{r['paper_auc']:>10.3f} "
                    f"{r['delta_auc']:>+8.4f} "
                    f"{r['mean_rmse']:>10.4f} "
                    f"{r['std_rmse']:>8.4f} "
                    f"{r['paper_rmse']:>11.3f} "
                    f"{r['delta_rmse']:>+8.4f}\n"
                )

        f.write("\n\nPer-Fold Details\n")
        f.write("=" * 60 + "\n")
        for name in ["ASS09", "ASS12", "KDD"]:
            if name in all_results:
                r = all_results[name]
                f.write(f"\n{name}:\n")
                for i, (auc, rmse) in enumerate(zip(r["aucs"], r["rmses"]), 1):
                    f.write(f"  Fold {i}: AUC={auc:.4f}, RMSE={rmse:.4f}\n")
                f.write(f"  Mean:   AUC={r['mean_auc']:.4f}±{r['std_auc']:.4f}, "
                        f"RMSE={r['mean_rmse']:.4f}±{r['std_rmse']:.4f}\n")

    log(f"\nSummary saved to: {summary_file}")
    log("Individual fold outputs saved to: results_5fold/")
    log("DONE!")


if __name__ == "__main__":
    main()
