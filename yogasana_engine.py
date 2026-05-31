import json
import os
import numpy as np

# Load knowledgebase for templates and configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(BASE_DIR, "knowledgebase.json")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

def load_knowledgebase():
    if not os.path.exists(KB_PATH):
        raise FileNotFoundError(f"Knowledgebase file not found at {KB_PATH}")
    with open(KB_PATH, "r") as f:
        return json.load(f)

KB = load_knowledgebase()
JOINTS_MAP = KB["joints_map"]
TEMPLATES = KB["poses"]

def calculate_angle(p_a, p_b, p_c):
    """
    Calculates the angle ABC in degrees where B is the vertex joint.
    Coordinates can be 2D [x, y].
    """
    a = np.array(p_a)
    b = np.array(p_b)
    c = np.array(p_c)
    
    ba = a - b
    bc = c - b
    
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    
    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return 180.0
        
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return float(np.degrees(angle))

def extract_angles(landmarks):
    """
    Given a list of 33 MediaPipe landmarks, extracts the 8 key joint angles.
    landmarks can be a list of dicts with 'x', 'y', 'visibility' or MediaPipe NormalizedLandmarkList.
    """
    # Convert landmarks to a structure we can index easily
    coords = {}
    for idx, lm in enumerate(landmarks):
        # MediaPipe landmarks have x, y, visibility
        coords[idx] = (lm.x, lm.y)
        
    angles = {}
    for joint_name, mapping in JOINTS_MAP.items():
        idx_a = mapping["a"]
        idx_b = mapping["b"]
        idx_c = mapping["c"]
        
        # Calculate 2D angle
        angle = calculate_angle(coords[idx_a], coords[idx_b], coords[idx_c])
        angles[joint_name] = round(angle, 1)
        
    return angles

def evaluate_posture(detected_pose, actual_angles):
    """
    Compares the actual joint angles against the template for the detected pose.
    Returns:
        score: posture score out of 100
        corrections: list of text feedback comments
        details: dict of {joint: (template_angle, actual_angle, error)}
    """
    if detected_pose not in TEMPLATES:
        return 100, [], {}
        
    pose_template = TEMPLATES[detected_pose]["angles"]
    pose_feedback = TEMPLATES[detected_pose]["feedback"]
    
    errors = []
    corrections = []
    details = {}
    
    # We evaluate all 8 joints
    for joint_name, template_angle in pose_template.items():
        actual_angle = actual_angles.get(joint_name, 180.0)
        error = actual_angle - template_angle
        abs_error = abs(error)
        errors.append(abs_error)
        
        details[joint_name] = {
            "template": template_angle,
            "actual": actual_angle,
            "error": error
        }
        
        # Check if error exceeds tolerance threshold (e.g. 15 degrees)
        threshold = KB["algorithm"]["posture_evaluation"].get("incorrect_threshold_degrees", 15.0)
        if abs_error > threshold:
            feedback_data = pose_feedback.get(joint_name, {})
            if error < 0: # Actual angle is too small
                msg = feedback_data.get("too_low", "")
            else: # Actual angle is too large
                msg = feedback_data.get("too_high", "")
                
            if msg:
                corrections.append(msg)
                
    # Calculate score
    mean_error = np.mean(errors) if errors else 0.0
    score = max(0.0, 100.0 - mean_error)
    
    return int(round(score)), corrections, details

def generate_synthetic_samples(pose_name, ideal_angles, num_samples=30, std_dev=3.0):
    """
    Generates synthetic samples by adding Gaussian noise to ideal template angles.
    Enforces physically plausible limits [0, 180].
    """
    import csv
    samples = []
    headers = [
        "left_elbow", "right_elbow", 
        "left_shoulder", "right_shoulder", 
        "left_hip", "right_hip", 
        "left_knee", "right_knee", 
        "pose"
    ]
    
    np.random.seed(42 + hash(pose_name) % 1000)
    
    for _ in range(num_samples):
        sample = {}
        for joint, ideal in ideal_angles.items():
            noise = np.random.normal(0, std_dev)
            val = ideal + noise
            val = np.clip(val, 0.0, 180.0)
            sample[joint] = round(val, 1)
        sample["pose"] = pose_name
        samples.append(sample)
        
    return headers, samples

def save_to_csv(pose_name, headers, samples):
    import csv
    filename = f"{pose_name.lower().replace(' ', '_')}.csv"
    filepath = os.path.join(DATA_DIR, filename)
    
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(samples)
    print(f"Saved {len(samples)} samples to {filepath}")
    return filepath

def train_and_save_model():
    import csv
    import pickle
    from sklearn.ensemble import RandomForestClassifier

    # Find all CSV files in DATA_DIR
    if not os.path.exists(DATA_DIR):
        print("Data directory does not exist.")
        return False
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if not csv_files:
        print("No CSV files found. Cannot train model.")
        return False
        
    all_features = []
    all_labels = []
    
    headers = [
        "left_elbow", "right_elbow", 
        "left_shoulder", "right_shoulder", 
        "left_hip", "right_hip", 
        "left_knee", "right_knee"
    ]
    
    for f_name in csv_files:
        filepath = os.path.join(DATA_DIR, f_name)
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                features = [float(row[h]) for h in headers]
                label = row["pose"]
                all_features.append(features)
                all_labels.append(label)
                
    X = np.array(all_features)
    y = np.array(all_labels)
    
    print(f"Training RandomForestClassifier on {len(X)} total samples from {len(csv_files)} poses...")
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X, y)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "yoga_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
        
    print(f"Model saved successfully to {model_path}")
    return True

