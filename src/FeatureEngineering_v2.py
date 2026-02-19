"""
Modernized IKT Feature Engineering Pipeline
============================================
Original: Minn et al. (AAAI 2022) - https://github.com/Simon-tan/IKT
Modified: Fadi Alazayem (25002207) - BUiD MSc AI, Feb 2026

Changes from original:
  - Replaced TensorFlow 1.x k-means with scikit-learn KMeans
  - Removed all tf.Session() / tf.app.run() dependencies
  - Added logging and progress indicators
  - Added seed control for reproducibility
  - No changes to BKT logic or feature extraction logic

This ensures faithful reproduction while running on modern Python.
"""

import os
import numpy as np
import time
import csv
import math
import pandas as pd
from sklearn.cluster import KMeans
from BKT import BKT

# Project root (one level up from src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# CONFIGURATION - Change these as needed
# ============================================================
DATA_NAME = "4_Ass_09"
BATCH_SIZE = 32
CLUSTER_NUM = 7  # Number of ability profile clusters (paper: 7)
PROBLEM_LEN = 20  # Time interval length in attempts (paper: 20)
RANDOM_SEED = 1011  # For reproducibility - change for multi-seed runs


def log(msg):
    """Simple timestamped logging."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


# ============================================================
# K-MEANS CLUSTERING (REPLACED TensorFlow → scikit-learn)
# ============================================================
def k_means_clust(
    train_students,
    test_students,
    max_stu,
    max_seg,
    num_clust,
    num_skills,
    num_iter,
    seed=42,
):
    """
    Assign ability profiles using k-means clustering.

    ORIGINAL: Used tf.Session() with manual k-means in TensorFlow 1.x
    MODERNIZED: Uses sklearn.cluster.KMeans (same algorithm, same result)

    Each student's performance vector (success rates across all skills)
    is clustered to detect their ability profile at each time interval.

    Args:
        train_students: list of performance vectors (features + 3 identifiers)
        test_students: same format for test students
        max_stu: maximum student ID + 1
        max_seg: maximum segment ID + 1
        num_clust: number of clusters (K=7 in paper)
        num_skills: total number of skills
        num_iter: max iterations for k-means (40 in paper)
        seed: random seed for reproducibility

    Returns:
        cluster: 2D array [student_id, segment_id] → cluster assignment
    """
    identifiers = 3  # Last 3 values are metadata, not features
    max_stu = int(max_stu)
    max_seg = int(max_seg)
    cluster = np.zeros((max_stu, max_seg))

    # Extract feature vectors (remove last 3 identifier columns)
    train_data = np.array([s[:-identifiers] for s in train_students])

    log(
        f"  K-Means: {len(train_data)} training vectors, K={num_clust}, "
        f"dim={train_data.shape[1]}, seed={seed}"
    )

    # ---- THIS IS THE KEY CHANGE ----
    # Original used tf.Session() with manual centroid updates
    # sklearn.KMeans uses the same Lloyd's algorithm
    kmeans = KMeans(
        n_clusters=num_clust,
        max_iter=num_iter,
        n_init=1,  # Original code did 1 random init
        random_state=seed,
        algorithm="lloyd",  # Same as TF implementation
    )
    kmeans.fit(train_data)
    centroids_val = kmeans.cluster_centers_
    # ---- END OF KEY CHANGE ----

    # Assign train students to nearest cluster
    for s in train_students:
        inst = s[:-identifiers]
        student_id = int(s[-2])
        seg_id = int(s[-1])
        min_dist = float("inf")
        closest_clust = 0
        for j in range(num_clust):
            cur_dist = euclideanDistance(inst, centroids_val[j])
            if cur_dist < min_dist:
                min_dist = cur_dist
                closest_clust = j
        cluster[student_id, seg_id] = closest_clust

    # Assign test students to nearest cluster (using SAME centroids)
    for s in test_students:
        inst = s[:-identifiers]
        student_id = int(s[-2])
        seg_id = int(s[-1])
        min_dist = float("inf")
        closest_clust = 0
        for j in range(num_clust):
            cur_dist = euclideanDistance(inst, centroids_val[j])
            if cur_dist < min_dist:
                min_dist = cur_dist
                closest_clust = j
        cluster[student_id, seg_id] = closest_clust

    del train_students, test_students
    return cluster


# ============================================================
# PROBLEM DIFFICULTY (unchanged from original)
# ============================================================
def difficulty_data(students, max_items):
    """
    Calculate problem difficulty on scale 1-10.
    Based on average success rate across all students' first attempts.
    Problems with < 4 attempts get default difficulty of 5.

    Paper reference: Equation 8-9
    """
    limit = 3  # minimum 4 students (>3) to calculate difficulty
    xtotal = np.zeros(max_items + 1)
    x1 = np.zeros(max_items + 1)
    items = []
    Allitems = []
    item_diff = {}

    for student in students:
        item_ids = student[3]
        correctness = student[2]
        for j in range(len(item_ids)):
            key = item_ids[j]
            xtotal[key] += 1
            if int(correctness[j]) == 0:
                x1[key] += 1
            if xtotal[key] > limit and key > 0 and key not in items:
                items.append(key)
            if xtotal[key] > 0 and key not in Allitems:
                Allitems.append(key)

    for i in items:
        # Map error rate to difficulty level 0-10
        diff = (np.around(float(x1[i]) / float(xtotal[i]), decimals=1) * 10).astype(int)
        item_diff[i] = diff

    log(
        f"  Problem difficulty: {len(item_diff)} problems rated, "
        f"{len(Allitems)} total unique problems"
    )
    return item_diff


def euclideanDistance(instance1, instance2):
    """Euclidean distance between two vectors."""
    distance = 0
    for x in range(len(instance1)):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


# ============================================================
# DATA READING (unchanged from original)
# ============================================================
def read_data_from_csv_file(trainfile, testfile):
    """
    Read IKT-format CSV files (4-line format per student):
      Line 1: num_problems, student_id
      Line 2: skill_id sequence (comma-separated)
      Line 3: problem_id sequence (comma-separated)
      Line 4: response sequence (0/1, comma-separated)

    Also runs BKT to compute skill mastery for each student.
    """
    rows = []
    max_skills = 0
    max_steps = 0
    max_items = 0
    studentids = []
    train_ids = []
    test_ids = []
    problem_len = PROBLEM_LEN

    log(f"  Reading training data: {trainfile}")
    with open(trainfile, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            rows.append(row)

    skill_rows = []
    correct_rows = []
    stu_rows = []
    opp_rows = []
    index = 0

    while index < len(rows):
        if int(rows[index][0]) > problem_len:
            problems = int(rows[index][0])
            student_id = int(rows[index][1])
            train_ids.append(student_id)

            tmp_max_skills = max(map(int, rows[index + 1]))
            if tmp_max_skills > max_skills:
                max_skills = tmp_max_skills

            tmp_max_items = max(map(int, rows[index + 2]))
            if tmp_max_items > max_items:
                max_items = tmp_max_items

            skill_rows = np.append(skill_rows, rows[index + 1])
            correct_rows = np.append(correct_rows, rows[index + 3])
            stu_rows = np.append(stu_rows, [student_id] * len(rows[index + 1]))
            opp_rows = np.append(opp_rows, list(range(len(rows[index + 1]))))
        index += 4

    log(f"  Reading test data: {testfile}")
    with open(testfile, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            rows.append(row)

    while index < len(rows):
        if int(rows[index][0]) > problem_len:
            problems = int(rows[index][0])
            student_id = int(rows[index][1])
            test_ids.append(student_id)

            tmp_max_skills = max(map(int, rows[index + 1]))
            if tmp_max_skills > max_skills:
                max_skills = tmp_max_skills

            tmp_max_items = max(map(int, rows[index + 2]))
            if tmp_max_items > max_items:
                max_items = tmp_max_items

            skill_rows = np.append(skill_rows, rows[index + 1])
            correct_rows = np.append(correct_rows, rows[index + 3])
            stu_rows = np.append(stu_rows, [student_id] * len(rows[index + 1]))
            opp_rows = np.append(opp_rows, list(range(len(rows[index + 1]))))
        index += 4

    max_skills = max_skills + 1
    max_items = max_items + 1

    log(f"  Train students: {len(train_ids)}, Test students: {len(test_ids)}")
    log(f"  Max skills: {max_skills}, Max items: {max_items}")

    data = pd.DataFrame(
        {
            "stus": stu_rows,
            "skills": skill_rows,
            "corrects": correct_rows,
            "opp": opp_rows,
        }
    ).astype(int)

    log("  Running BKT assessment (brute-force fitting)...")
    bkt_ass = BKTAssessment(data, train_ids, max_skills)
    log("  BKT assessment complete.")

    del skill_rows, correct_rows, stu_rows, opp_rows, data

    # Second pass: build structured student records with BKT mastery
    index = 0
    tuple_rows = []
    while index < len(rows):
        if int(rows[index][0]) > problem_len:
            problems = int(rows[index][0])
            student_id = int(rows[index][1])
            studentids.append(student_id)

            if problems > problem_len:
                tmp_max_steps = int(rows[index][0])
                if tmp_max_steps > max_steps:
                    max_steps = tmp_max_steps

                asses = bkt_ass[student_id]

                len_problems = int(int(problems) / problem_len) * problem_len
                rest_problems = problems - len_problems

                ele_p = [int(e) for e in rows[index + 1]]
                ele_c = [int(e) for e in rows[index + 3]]
                ele_d = [int(e) for e in rows[index + 2]]
                ele_a = [float(e) for e in asses]

                if rest_problems > 0:
                    rest = problem_len - rest_problems
                    for i in range(rest):
                        ele_p.append(-1)
                        ele_c.append(-1)
                        ele_d.append(-1)
                        ele_a.append(-1)

                ele_p_array = np.reshape(np.asarray(ele_p), (-1, problem_len))
                ele_c_array = np.reshape(np.asarray(ele_c), (-1, problem_len))
                ele_d_array = np.reshape(np.asarray(ele_d), (-1, problem_len))
                ele_a_array = np.reshape(np.asarray(ele_a), (-1, problem_len))

                n_pieces = ele_p_array.shape[0]

                for j in range(n_pieces):
                    s1 = [student_id, j, problems]
                    if (j > -1) & (j < (n_pieces - 1)):
                        s1.append(1)
                        s2 = np.append(
                            ele_p_array[j, :], ele_p_array[j + 1, 0]
                        ).tolist()
                        s3 = np.append(
                            ele_c_array[j, :], ele_c_array[j + 1, 0]
                        ).tolist()
                        s4 = np.append(
                            ele_d_array[j, :], ele_d_array[j + 1, 0]
                        ).tolist()
                        s5 = np.append(
                            ele_a_array[j, :], ele_a_array[j + 1, 0]
                        ).tolist()
                    else:
                        s1.append(-1)
                        s2 = ele_p_array[j, :].tolist()
                        s3 = ele_c_array[j, :].tolist()
                        s4 = ele_d_array[j, :].tolist()
                        s5 = ele_a_array[j, :].tolist()
                    tup = (s1, s2, s3, s4, s5)
                    tuple_rows.append(tup)
        index += 4

    max_steps = max_steps + 1

    # Split back into train/test
    train_students = []
    test_students = []
    for t in tuple_rows:
        if int(t[0][0]) in train_ids:
            train_students.append(t)
        if int(t[0][0]) in test_ids:
            test_students.append(t)

    return (
        train_students,
        test_students,
        studentids,
        max_skills,
        max_items,
        train_ids,
        test_ids,
    )


# ============================================================
# BKT ASSESSMENT (unchanged from original)
# ============================================================
def get_bktdata(df):
    BKT_dict = {}
    DKT_skill_dict = {}
    DKT_res_dict = {}

    for kc in list(df["skills"].unique()):
        kc_df = df[df["skills"] == kc].sort_values(["stus"], ascending=True)
        stu_cfa_dict = {}
        for stu in list(kc_df["stus"].unique()):
            df_final = (
                kc_df[kc_df["stus"] == int(stu)]
                .reset_index()
                .sort_values(["opp"], ascending=True)
            )
            stu_cfa_dict[int(stu)] = list(df_final["corrects"])
        BKT_dict[int(kc)] = stu_cfa_dict

    for stu in list(df["stus"].unique()):
        stu_df = df[df["stus"] == int(stu)].sort_values(["opp"], ascending=True)
        DKT_skill_dict[int(stu)] = list(stu_df["skills"])
        DKT_res_dict[int(stu)] = list(stu_df["corrects"])

    return BKT_dict, DKT_skill_dict, DKT_res_dict


def BKTAssessment(data, train_ids, max_skills):
    bkt_data, dkt_skill, dkt_res = get_bktdata(data)
    DL, DT, DG, DS = {}, {}, {}, {}

    skills_fitted = 0
    total_skills = len(bkt_data.keys())

    for i in bkt_data.keys():
        skill_data = bkt_data[i]
        train_data = []
        for j in skill_data.keys():
            if int(j) in train_ids:
                train_data.append(list(map(int, skill_data[j])))

        bkt = BKT(step=0.1, bounded=False, best_k0=True)
        if len(train_data) > 2:
            DL[i], DT[i], DG[i], DS[i] = bkt.fit(train_data)
        else:
            DL[i], DT[i], DG[i], DS[i] = 0.5, 0.2, 0.1, 0.1

        skills_fitted += 1
        if skills_fitted % 20 == 0:
            log(f"    BKT fitted {skills_fitted}/{total_skills} skills...")

    del bkt_data

    mastery = bkt.inter_predict(dkt_skill, dkt_res, DL, DT, DG, DS, max_skills)
    del dkt_skill, dkt_res

    return mastery


# ============================================================
# CLUSTER DATA (unchanged from original)
# ============================================================
def cluster_data(students, max_stu, num_skills, datatype):
    """
    Build performance vectors for k-means clustering.
    Each vector = success rate on each skill up to current time interval.
    """
    success = []
    max_seg = 0
    xtotal = np.zeros((max_stu, num_skills))
    x1 = np.zeros((max_stu, num_skills))
    x0 = np.zeros((max_stu, num_skills))

    index = 0
    while index + BATCH_SIZE < len(students):
        for i in range(BATCH_SIZE):
            student = students[index + i]
            student_id = int(student[0][0])
            seg_id = int(student[0][1])

            if int(student[0][3]) == 1:
                tmp_seg = seg_id
                if tmp_seg > max_seg:
                    max_seg = tmp_seg
                problem_ids = student[1]
                correctness = student[2]
                for j in range(len(problem_ids)):
                    key = problem_ids[j]
                    xtotal[student_id, key] += 1
                    if int(correctness[j]) == 1:
                        x1[student_id, key] += 1
                    else:
                        x0[student_id, key] += 1

                xsr = [
                    (x + 1.4) / (y + 2)
                    for x, y in zip(x1[student_id], xtotal[student_id])
                ]
                x = np.nan_to_num(xsr)
                x = np.append(x, student_id)
                x = np.append(x, seg_id)
                success.append(x)

        index += BATCH_SIZE

    return success, max_seg


# ============================================================
# FEATURE EXTRACTION (unchanged from original)
# ============================================================
def get_features(students, item_diff, max_stu, cluster, num_skills, datatype):
    """
    Extract the 4 features for each student interaction:
      1. skill_ID: which skill is being practiced
      2. skill_mastery: BKT-estimated P(learned) for that skill
      3. ability_profile: cluster ID from k-means (learning transfer)
      4. problem_difficulty: difficulty level 0-10

    Plus target: correctness (0 or 1)

    Output saved as CSV file ready for ARFF conversion.
    """
    index = 0
    stu_list, p0_list, p1_list, p2_list, p3_list, p4_list = [], [], [], [], [], []

    while index + BATCH_SIZE < len(students):
        for i in range(BATCH_SIZE):
            student = students[index + i]
            student_id = student[0][0]
            seg_id = int(student[0][1])

            # Ability profile from previous segment (paper: Equation 7)
            if seg_id > 0:
                cluster_id = cluster[student_id, (seg_id - 1)] + 1
            else:
                cluster_id = 0  # Initial profile for all students

            skill_ids = student[1]
            correctness = student[2]
            items = student[3]
            bkt = student[4]

            for j in range(len(skill_ids) - 1):
                target_indx = j + 1
                skill_id = int(skill_ids[target_indx])
                item = int(items[target_indx])
                kcass = np.round(float(bkt[target_indx]), 6)
                correct = int(correctness[target_indx])

                if skill_id > -1:
                    # Problem difficulty lookup
                    df = int(item_diff[item]) if item in item_diff.keys() else 5

                    stu_list.append(student_id)
                    p0_list.append(int(skill_id))
                    p1_list.append(float(kcass))
                    p2_list.append(int(cluster_id))
                    p3_list.append(int(df))
                    p4_list.append(int(correct))

        index += BATCH_SIZE

    data = pd.DataFrame(
        {
            "skill_ID": p0_list,
            "skill_mastery": p1_list,
            "ability_profile": p2_list,
            "problem_difficulty": p3_list,
            "correctness": p4_list,
        }
    )

    outfile = os.path.join(PROJECT_ROOT, "data", "ass09", f"{datatype}_data_out.csv")
    data.to_csv(outfile, index=None, header=True)
    log(f"  Saved {datatype} features: {len(data)} rows -> {outfile}")
    return


# ============================================================
# CSV → ARFF CONVERTER (NEW - saves manual conversion step)
# ============================================================
def csv_to_arff(csv_file, arff_file, relation_name="ASS2009"):
    """
    Convert IKT feature CSV to WEKA ARFF format.
    Adds the required header that WEKA needs.
    """
    df = pd.read_csv(csv_file)

    header = f"""@relation {relation_name}
@attribute skill_ID numeric
@attribute skill_mastery numeric
@attribute ability_profile numeric
@attribute problem_difficulty numeric
@attribute correctness {{1,0}}
@data
"""
    with open(arff_file, "w") as f:
        f.write(header)
        for _, row in df.iterrows():
            f.write(
                f"{int(row['skill_ID'])},{row['skill_mastery']},"
                f"{int(row['ability_profile'])},{int(row['problem_difficulty'])},"
                f"{int(row['correctness'])}\n"
            )

    log(f"  Converted {csv_file} -> {arff_file} ({len(df)} rows)")


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    log("=" * 60)
    log("IKT Feature Engineering Pipeline (Modernized)")
    log(f"Dataset: {DATA_NAME}")
    log(f"Random seed: {RANDOM_SEED}")
    log(f"Clusters: {CLUSTER_NUM}, Interval: {PROBLEM_LEN}")
    log("=" * 60)

    start_time = time.time()

    train_data = os.path.join(PROJECT_ROOT, "data", "ass09", f"{DATA_NAME}_train.csv")
    test_data = os.path.join(PROJECT_ROOT, "data", "ass09", f"{DATA_NAME}_test.csv")

    # Step 1: Read data + BKT assessment
    log("\nSTEP 1: Reading data and fitting BKT models...")
    (
        train_students,
        test_students,
        student_ids,
        max_skills,
        max_items,
        train_ids,
        test_ids,
    ) = read_data_from_csv_file(train_data, test_data)
    num_skills = max_skills

    # Step 2: Calculate problem difficulty
    log("\nSTEP 2: Calculating problem difficulty...")
    item_diff = difficulty_data(train_students + test_students, max_items)

    # Step 3: Build performance vectors for clustering
    log("\nSTEP 3: Building cluster data (performance vectors)...")
    train_cluster_data, train_max_seg = cluster_data(
        train_students, max(train_ids) + 1, max_skills, "train"
    )
    test_cluster_data, test_max_seg = cluster_data(
        test_students, max(test_ids) + 1, max_skills, "test"
    )

    max_stu = max(student_ids) + 1
    max_seg = max(int(train_max_seg), int(test_max_seg)) + 1

    # Step 4: K-means clustering (THIS IS WHERE TF WAS REPLACED)
    log("\nSTEP 4: Running k-means clustering (sklearn)...")
    cluster = k_means_clust(
        train_cluster_data,
        test_cluster_data,
        max_stu,
        max_seg,
        CLUSTER_NUM,
        max_skills,
        num_iter=40,
        seed=RANDOM_SEED,
    )

    # Step 5: Extract features
    log("\nSTEP 5: Extracting features...")
    get_features(train_students, item_diff, max_stu, cluster, max_skills, "train")
    get_features(test_students, item_diff, max_stu, cluster, max_skills, "test")

    # Step 6: Convert to ARFF (bonus - saves manual step)
    log("\nSTEP 6: Converting to ARFF format...")
    csv_to_arff(os.path.join(PROJECT_ROOT, "data", "ass09", "train_data_out.csv"), os.path.join(PROJECT_ROOT, "data", "ass09", "train_data.arff"))
    csv_to_arff(os.path.join(PROJECT_ROOT, "data", "ass09", "test_data_out.csv"), os.path.join(PROJECT_ROOT, "data", "ass09", "test_data.arff"))

    elapsed = time.time() - start_time
    log(f"\nDONE! Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    log("Next step: Load train_data.arff and test_data.arff into WEKA")
    log("  -> Classify -> BayesNet  searchAlgorithm: TAN")


if __name__ == "__main__":
    main()
