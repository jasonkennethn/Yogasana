import os
os.environ["TK_SILENCE_DEPRECATION"] = "1"

# Silence protobuf and other warnings to keep terminal logs clean
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys
import subprocess

def check_and_install_requirements():
    """
    Checks if required libraries are installed. If any are missing,
    automatically installs them from requirements.txt or directly.
    """
    required_packages = {
        "numpy": "numpy",
        "pandas": "pandas",
        "cv2": "opencv-python",
        "mediapipe": "mediapipe<=0.10.14",
        "sklearn": "scikit-learn"
    }
    
    missing = []
    for module_name, pip_name in required_packages.items():
        try:
            __import__(module_name)
        except ImportError:
            missing.append(pip_name)
            
    if missing:
        print("Missing required libraries. Installing them now...")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        req_file = os.path.join(base_dir, "requirements.txt")
        try:
            if os.path.exists(req_file):
                print(f"Running: {sys.executable} -m pip install -r {req_file}")
                result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", req_file], check=False)
                if result.returncode != 0:
                    print("Standard installation failed. Retrying with --user...")
                    subprocess.run([sys.executable, "-m", "pip", "install", "--user", "-r", req_file], check=True)
            else:
                print(f"Running: {sys.executable} -m pip install " + " ".join(missing))
                result = subprocess.run([sys.executable, "-m", "pip", "install"] + missing, check=False)
                if result.returncode != 0:
                    print("Standard installation failed. Retrying with --user...")
                    subprocess.run([sys.executable, "-m", "pip", "install", "--user"] + missing, check=True)
            print("Successfully installed all missing requirements.")
        except Exception as e:
            print(f"Error during automatic requirement installation: {e}", file=sys.stderr)
            
    # Check for tkinter as well (cannot be installed via pip, needs system package on some Linux distros)
    try:
        import tkinter
    except ImportError:
        print("Error: 'tkinter' is missing. On Linux (Ubuntu/Debian), please run 'sudo apt-get install python3-tk' to install it.", file=sys.stderr)
        sys.exit(1)

check_and_install_requirements()

# Import the third-party dependencies now that they are guaranteed to be present
import json
import csv
import time
import pickle
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk, scrolledtext
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp

# Import core engine
import yogasana_engine as engine


class YogasanaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Yogasana - AI Yoga Pose Evaluator")
        self.root.geometry("500x560")
        
        # Initialize MediaPipe Pose solutions
        self.mp_pose = mp.solutions.pose
        
        # Build UI layout immediately so the window is drawn
        self.build_ui()
        
        # Schedule heavy initialization and focus-forcing asynchronously
        self.root.after(50, self.deferred_init)
        
    def deferred_init(self):
        # Force focus to front
        self.force_focus()
        # Setup directories and templates (and auto-train model if missing)
        self.check_and_seed_data()
        # Update dashboard status
        self.update_model_status()
        
    def force_focus(self):
        # Force window to front safely
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after(100, lambda: self.root.attributes("-topmost", False))
        self.root.focus_force()
        
    def check_and_seed_data(self):
        """
        Creates data/ and models/ directories, seeds synthetic data if none exists,
        and trains the model automatically so that it is ready to use.
        """
        os.makedirs(engine.DATA_DIR, exist_ok=True)
        os.makedirs(engine.MODELS_DIR, exist_ok=True)
        
        # Seed CSV dataset files if the data directory is empty
        csv_files = [f for f in os.listdir(engine.DATA_DIR) if f.endswith(".csv")]
        if not csv_files:
            print("Seeding pre-trained pose CSV datasets...")
            templates = engine.TEMPLATES
            for pose_name, details in templates.items():
                ideal_angles = details["angles"]
                headers, samples = engine.generate_synthetic_samples(pose_name, ideal_angles, num_samples=30)
                engine.save_to_csv(pose_name, headers, samples)
            print("Pose CSV datasets seeded.")
            
        # Train model if missing
        model_path = os.path.join(engine.MODELS_DIR, "yoga_model.pkl")
        if not os.path.exists(model_path):
            print("Training model automatically...")
            self._run_retrain_model_pipeline()
            print("Model trained.")

    def build_ui(self):
        # Center the main window on the user's active monitor screen
        width = 500
        height = 560
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        
        main_container = ttk.Frame(self.root, padding=20)
        main_container.pack(fill="both", expand=True)
        
        # Title Frame
        title_frame = ttk.Frame(main_container)
        title_frame.pack(fill="x", side="top", pady=(0, 15))
        
        lbl_title = ttk.Label(
            title_frame, 
            text="YOGASANA", 
            font=("Helvetica", 24, "bold"),
            anchor="center"
        )
        lbl_title.pack(fill="x", pady=(0, 2))
        
        lbl_subtitle = ttk.Label(
            title_frame, 
            text="ML-BASED YOGA POSTURE EVALUATION SYSTEM", 
            font=("Helvetica", 9, "bold"),
            foreground="gray",
            anchor="center"
        )
        lbl_subtitle.pack(fill="x")
        
        # Model Status Frame
        status_frame = ttk.LabelFrame(main_container, text=" Model Status ", padding=15)
        status_frame.pack(fill="x", pady=(0, 20))
        
        self.lbl_status_val = ttk.Label(
            status_frame, 
            text="CHECKING STATUS...", 
            font=("Helvetica", 14, "bold"),
            anchor="center"
        )
        self.lbl_status_val.pack(fill="x")
        
        # Action Buttons Menu Frame
        btn_frame = ttk.LabelFrame(main_container, text=" Action Menu ", padding=15)
        btn_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        self.btn_identify = ttk.Button(btn_frame, text="Identify Pose", command=self.identify_pose)
        self.btn_identify.pack(fill="x", pady=5)
        
        btn_create = ttk.Button(btn_frame, text="Create & Train New Pose", command=self.create_new_pose)
        btn_create.pack(fill="x", pady=5)
        
        btn_train = ttk.Button(btn_frame, text="Train Existing Pose", command=self.train_existing_pose)
        btn_train.pack(fill="x", pady=5)
        
        btn_view = ttk.Button(btn_frame, text="View Available Poses", command=self.view_poses)
        btn_view.pack(fill="x", pady=5)
        
        btn_exit = ttk.Button(btn_frame, text="Exit", command=self.exit_app)
        btn_exit.pack(fill="x", pady=5)
        
        # Footer
        lbl_footer = ttk.Label(
            main_container,
            text="Powered by MediaPipe & Random Forest Classifier",
            font=("Helvetica", 8),
            foreground="gray",
            anchor="center"
        )
        lbl_footer.pack(side="bottom", fill="x", pady=(10, 0))

    def update_model_status(self):
        model_path = os.path.join(engine.MODELS_DIR, "yoga_model.pkl")
        if os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    clf = pickle.load(f)
                num_classes = len(clf.classes_)
                self.lbl_status_val.config(text=f"Trained ({num_classes} Poses)", foreground="green")
            except Exception as e:
                self.lbl_status_val.config(text="Corrupted Model", foreground="orange")
        else:
            self.lbl_status_val.config(text="Not Trained", foreground="red")

    def draw_skeleton_hud(self, frame, landmarks, prediction=None, confidence=0.0, score=100, corrections=None, details=None):
        """
        Draws skeleton lines (light-blue), joints (green if correct, red if incorrect),
        joint angle values, and the HUD boxes.
        """
        h, w, _ = frame.shape
        coords = {}
        for idx, lm in enumerate(landmarks.landmark):
            # Convert normal landmarks to screen coordinates
            coords[idx] = (int(lm.x * w), int(lm.y * h))
            
        # 1. Draw Connections (Cyan skeleton line)
        connections = [
            (11, 12), (11, 23), (12, 24), (23, 24), # Torso box
            (11, 13), (13, 15),                    # Left arm
            (12, 14), (14, 16),                    # Right arm
            (23, 25), (25, 27),                    # Left leg
            (24, 26), (26, 28)                     # Right leg
        ]
        
        for start_idx, end_idx in connections:
            if start_idx in coords and end_idx in coords:
                cv2.line(frame, coords[start_idx], coords[end_idx], (235, 206, 135), 2, cv2.LINE_AA) # Cyan-blue hue
                
        # 2. Draw active joints feedback
        # Key joint indices mapping in the engine
        joint_landmarks = {
            "left_elbow": 13,
            "right_elbow": 14,
            "left_shoulder": 11,
            "right_shoulder": 12,
            "left_hip": 23,
            "right_hip": 24,
            "left_knee": 25,
            "right_knee": 26
        }
        
        threshold = engine.KB["algorithm"]["posture_evaluation"].get("incorrect_threshold_degrees", 15.0)
        
        for joint_name, idx in joint_landmarks.items():
            if idx not in coords:
                continue
                
            center = coords[idx]
            is_correct = True
            err_text = ""
            
            if details and joint_name in details:
                joint_detail = details[joint_name]
                abs_err = abs(joint_detail["error"])
                if abs_err > threshold:
                    is_correct = False
                    err_text = f"{joint_detail['actual']:.0f}/{joint_detail['template']:.0f}"
                    
            color = (46, 204, 113) if is_correct else (60, 76, 231) # Green or Red (BGR format: 231, 76, 60)
            
            # Draw outer glow circle
            cv2.circle(frame, center, 11, color, 1, cv2.LINE_AA)
            # Draw filled center
            cv2.circle(frame, center, 6, color, -1, cv2.LINE_AA)
            
            # If incorrect, print angle comparison next to joint
            if err_text:
                cx, cy = center
                # Add background for text
                (tw, th), tb = cv2.getTextSize(err_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(frame, (cx + 10, cy - th - 5), (cx + 10 + tw + 5, cy + tb), (0, 0, 0), -1)
                cv2.putText(frame, err_text, (cx + 12, cy - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # 3. Draw Translucent HUD Overlays
        # Semi-transparent top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 85), (20, 20, 20), -1)
        # Semi-transparent bottom bar
        cv2.rectangle(overlay, (0, h - 85), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # HUD Text - Title
        cv2.putText(frame, "YOGASANA AI", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (46, 204, 113), 2, cv2.LINE_AA)
        
        # HUD Text - Left Info (Detection)
        if prediction:
            pose_str = f"Pose: {prediction}"
            conf_str = f"Confidence: {confidence:.0f}%"
            cv2.putText(frame, pose_str, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, conf_str, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Searching pose...", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
            
        # HUD Text - Right Info (Evaluation Score)
        if prediction:
            score_str = f"Alignment: {score}%"
            cv2.putText(frame, score_str, (w - 200, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Status Text
            status_text = "POSE CORRECT" if score >= 85 else "ADJUST POSTURE"
            status_color = (46, 204, 113) if score >= 85 else (60, 76, 231)
            cv2.putText(frame, status_text, (w - 200, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1, cv2.LINE_AA)
            
        # HUD Text - Bottom Corrections Feed
        if corrections and len(corrections) > 0:
            cv2.putText(frame, "CORRECTION FEEDBACK:", (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 76, 231), 1, cv2.LINE_AA)
            # Display up to 2 corrections
            y_offset = h - 40
            for correction in corrections[:2]:
                cv2.putText(frame, f"- {correction}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                y_offset += 20
        else:
            if prediction:
                cv2.putText(frame, "Perfect pose structure! Keep holding.", (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (46, 204, 113), 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Align your full skeleton inside the camera view.", (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    def identify_pose(self):
        """
        Runs real-time camera tracking, feeds skeleton details to random forest classifier,
        evaluates performance score, overlays graphics, and displays suggestions.
        """
        model_path = os.path.join(engine.MODELS_DIR, "yoga_model.pkl")
        if not os.path.exists(model_path):
            success = self._run_retrain_model_pipeline()
            if not success:
                messagebox.showerror("Model Not Found", "The yoga model is not trained and automatic training failed. Please ensure dataset CSV files are present in the 'data/' folder.")
                return
            
        try:
            with open(model_path, "rb") as f:
                clf = pickle.load(f)
        except Exception as e:
            messagebox.showerror("Model Error", f"Error loading model: {e}")
            return
            
        # Open Camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Camera Error", "Could not access the webcam.")
            return
            
        # Hide Main Menu during webcam capture to focus view
        self.root.withdraw()
        
        # Load MediaPipe Pose
        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Mirror view, shape dimensions
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Perform MediaPipe extraction
                results = pose.process(rgb_frame)
                
                # Evaluate results
                prediction = None
                confidence = 0.0
                score = 100
                corrections = []
                details = {}
                
                if results.pose_landmarks:
                    try:
                        # Extract angle values
                        actual_angles = engine.extract_angles(results.pose_landmarks.landmark)
                        
                        # Prepare input array for Random Forest Classifier
                        features = [
                            actual_angles["left_elbow"], actual_angles["right_elbow"],
                            actual_angles["left_shoulder"], actual_angles["right_shoulder"],
                            actual_angles["left_hip"], actual_angles["right_hip"],
                            actual_angles["left_knee"], actual_angles["right_knee"]
                        ]
                        features_arr = np.array([features])
                        
                        # Predict pose class
                        pred_pose = clf.predict(features_arr)[0]
                        probabilities = clf.predict_proba(features_arr)[0]
                        max_idx = np.argmax(probabilities)
                        prob_val = probabilities[max_idx] * 100
                        
                        # Display classification if confidence is above threshold (e.g. 50%)
                        if prob_val > 50.0:
                            prediction = pred_pose
                            confidence = prob_val
                            
                            # Evaluate alignment correctness
                            score, corrections, details = engine.evaluate_posture(prediction, actual_angles)
                        else:
                            prediction = "Searching Pose..."
                            confidence = prob_val
                    except Exception as e:
                        print(f"Prediction Pipeline error: {e}")
                        
                    # Draw graphics overlays
                    self.draw_skeleton_hud(frame, results.pose_landmarks, prediction, confidence, score, corrections, details)
                else:
                    # Draw visual screen prompts when no skeleton is seen
                    h, w, _ = frame.shape
                    cv2.rectangle(frame, (0, 0), (w, 80), (20, 20, 20), -1)
                    cv2.rectangle(frame, (0, h - 80), (w, h), (20, 20, 20), -1)
                    cv2.putText(frame, "STAND BACK - ALIGN FULL BODY IN CAMERA", (50, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 76, 231), 2, cv2.LINE_AA)
                    cv2.putText(frame, "Press 'q' or click window close to Exit", (50, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                    
                # Render window
                window_name = "Yogasana Real-Time Pose Identification"
                cv2.imshow(window_name, frame)
                
                # Check escape keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27: # 'q' or Escape
                    break
                    
                # Support window close button click
                try:
                    if cv2.getWindowProperty(window_name, 4) < 1:
                        break
                except Exception:
                    break
                    
        # Release hardware bindings
        cap.release()
        cv2.destroyAllWindows()
        # Restore GUI control
        self.root.deiconify()

    def capture_pose_data(self, pose_name, is_new=True):
        """
        Runs OpenCV feed to record coordinates, shows countdown, saves samples,
        and returns list of sample data dicts.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Camera Error", "Could not access the webcam.")
            return None
            
        self.root.withdraw()
        
        samples_recorded = []
        target_samples = 30
        frame_interval_trigger = 6 # Record 1 sample every 6 valid frames (~200ms at 30fps)
        valid_frames_counter = 0
        
        countdown_secs = 5
        start_time = time.time()
        
        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                results = pose.process(rgb_frame)
                
                # Overlay dark header/footer overlays for prompt text
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 20), -1)
                cv2.rectangle(overlay, (0, h - 80), (w, h), (20, 20, 20), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                
                elapsed = time.time() - start_time
                
                # Draw visual lines for body skeleton
                if results.pose_landmarks:
                    coords = {}
                    for idx, lm in enumerate(results.pose_landmarks.landmark):
                        coords[idx] = (int(lm.x * w), int(lm.y * h))
                        
                    connections = [
                        (11, 12), (11, 23), (12, 24), (23, 24),
                        (11, 13), (13, 15), (12, 14), (14, 16),
                        (23, 25), (25, 27), (24, 26), (26, 28)
                    ]
                    for s, e in connections:
                        if s in coords and e in coords:
                            cv2.line(frame, coords[s], coords[e], (255, 191, 0), 2, cv2.LINE_AA) # Blue-Cyan neutral
                    for idx in [11, 12, 13, 14, 23, 24, 25, 26]:
                        if idx in coords:
                            cv2.circle(frame, coords[idx], 8, (255, 191, 0), -1, cv2.LINE_AA)
                            
                # Stage 1: Initial Preparation Countdown
                if elapsed < countdown_secs:
                    count_val = int(countdown_secs - elapsed) + 1
                    cv2.putText(frame, f"POSE: {pose_name.upper()}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"GET IN POSITION IN: {count_val}s", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (46, 204, 113), 2, cv2.LINE_AA)
                    cv2.putText(frame, "Align your complete body in frame and hold STILL.", (30, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                # Stage 2: Capture Data Samples
                else:
                    cv2.putText(frame, "RECORDING MOVEMENT DATA", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 76, 231), 2, cv2.LINE_AA)
                    progress_text = f"Samples: {len(samples_recorded)} / {target_samples}"
                    cv2.putText(frame, progress_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(frame, "HOLD STILL! Recording posture geometry.", (30, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (46, 204, 113), 1, cv2.LINE_AA)
                    
                    if results.pose_landmarks:
                        valid_frames_counter += 1
                        if valid_frames_counter % frame_interval_trigger == 0:
                            try:
                                actual_angles = engine.extract_angles(results.pose_landmarks.landmark)
                                samples_recorded.append(actual_angles)
                            except Exception as e:
                                print(f"Sample extraction error: {e}")
                                
                    if len(samples_recorded) >= target_samples:
                        break
                        
                window_name = f"Recording: {pose_name}"
                cv2.imshow(window_name, frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                try:
                    if cv2.getWindowProperty(window_name, 4) < 1:
                        break
                except Exception:
                    break
                    
        cap.release()
        cv2.destroyAllWindows()
        self.root.deiconify()
        
        if len(samples_recorded) < target_samples:
            messagebox.showwarning("Cancelled", "Recording was cancelled. Not enough samples were gathered.")
            return None
            
        return samples_recorded

    def create_new_pose(self):
        pose_name = simpledialog.askstring("Create Pose", "Enter Pose Name:", parent=self.root)
        if not pose_name:
            return
            
        pose_name = pose_name.strip()
        if len(pose_name) < 2:
            messagebox.showerror("Input Error", "Pose name must be at least 2 characters long.")
            return
            
        # Check duplicate
        if pose_name in engine.TEMPLATES:
            resp = messagebox.askyesno("Pose Exists", f"Pose '{pose_name}' already exists in templates.\nWould you like to overwrite it?")
            if not resp:
                return
                
        # Start capture
        samples = self.capture_pose_data(pose_name, is_new=True)
        if not samples:
            return
            
        # 1. Save data samples to CSV
        headers = [
            "left_elbow", "right_elbow", 
            "left_shoulder", "right_shoulder", 
            "left_hip", "right_hip", 
            "left_knee", "right_knee", 
            "pose"
        ]
        
        file_safe_name = pose_name.lower().replace(' ', '_')
        csv_path = os.path.join(engine.DATA_DIR, f"{file_safe_name}.csv")
        
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for sample in samples:
                row = {k: v for k, v in sample.items()}
                row["pose"] = pose_name
                writer.writerow(row)
                
        # 2. Update Template in knowledgebase
        # Calculate mean averages for the template reference
        means = {}
        for joint in headers[:-1]:
            joint_vals = [s[joint] for s in samples]
            means[joint] = round(float(np.mean(joint_vals)), 1)
            
        # Load KB
        with open(engine.KB_PATH, "r") as f:
            kb = json.load(f)
            
        # Add or update pose template
        kb["poses"][pose_name] = {
            "angles": means,
            "feedback": {
                joint: {
                    "too_low": f"Extend your {joint.replace('_', ' ')}.",
                    "too_high": f"Bend your {joint.replace('_', ' ')}."
                } for joint in headers[:-1]
            }
        }
        
        # Save KB
        with open(engine.KB_PATH, "w") as f:
            json.dump(kb, f, indent=2)
            
        # Hot-reload template engine configurations
        engine.TEMPLATES[pose_name] = kb["poses"][pose_name]
        engine.KB = kb
        
        # 3. Retrain model automatically
        retrain_success = self._run_retrain_model_pipeline()
        
        if retrain_success:
            messagebox.showinfo("Success", f"Pose '{pose_name}' created successfully!\nTemplate generated & Model retrained with the new pose.")
        else:
            messagebox.showwarning("Pose Created", f"Pose '{pose_name}' created and template saved,\nbut model training encountered an issue.")
            
        self.update_model_status()

    def train_existing_pose(self):
        # Choose pose
        pose_name = self.choose_existing_pose()
        if not pose_name:
            return
            
        # Capture more samples
        samples = self.capture_pose_data(pose_name, is_new=False)
        if not samples:
            return
            
        # 1. Append samples to CSV
        file_safe_name = pose_name.lower().replace(' ', '_')
        csv_path = os.path.join(engine.DATA_DIR, f"{file_safe_name}.csv")
        
        # Check if CSV exists, if not write header
        write_header = not os.path.exists(csv_path)
        headers = [
            "left_elbow", "right_elbow", 
            "left_shoulder", "right_shoulder", 
            "left_hip", "right_hip", 
            "left_knee", "right_knee", 
            "pose"
        ]
        
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if write_header:
                writer.writeheader()
            for sample in samples:
                row = {k: v for k, v in sample.items()}
                row["pose"] = pose_name
                writer.writerow(row)
                
        # 2. Re-compute template averages based on full CSV content
        df = pd.read_csv(csv_path)
        means = {}
        for joint in headers[:-1]:
            means[joint] = round(float(df[joint].mean()), 1)
            
        # Load KB
        with open(engine.KB_PATH, "r") as f:
            kb = json.load(f)
            
        # Update template angles
        if pose_name in kb["poses"]:
            kb["poses"][pose_name]["angles"] = means
        else:
            # Fallback if somehow template was missing
            kb["poses"][pose_name] = {
                "angles": means,
                "feedback": {
                    joint: {
                        "too_low": f"Extend your {joint.replace('_', ' ')}.",
                        "too_high": f"Bend your {joint.replace('_', ' ')}."
                    } for joint in headers[:-1]
                }
            }
            
        # Save KB
        with open(engine.KB_PATH, "w") as f:
            json.dump(kb, f, indent=2)
            
        # Reload templates
        engine.TEMPLATES[pose_name] = kb["poses"][pose_name]
        engine.KB = kb
        
        # 3. Retrain model
        retrain_success = self._run_retrain_model_pipeline()
        
        if retrain_success:
            messagebox.showinfo("Success", f"Added {len(samples)} new samples to '{pose_name}'.\nTemplate updated & Model retrained successfully!")
        else:
            messagebox.showwarning("Data Added", f"Added samples to '{pose_name}' and template updated,\nbut model training encountered an issue.")
            
        self.update_model_status()

    def choose_existing_pose(self):
        poses = list(engine.TEMPLATES.keys())
        if not poses:
            messagebox.showinfo("No Poses", "No poses available in the templates.")
            return None
            
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Pose")
        dialog.geometry("320x350")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog on the screen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - (320 // 2)
        y = (screen_height // 2) - (350 // 2)
        dialog.geometry(f"320x350+{x}+{y}")
        
        dialog.lift()
        dialog.focus_force()
        
        main_frame = ttk.Frame(dialog, padding=15)
        main_frame.pack(fill="both", expand=True)
        
        label = ttk.Label(main_frame, text="Choose Pose to Improve:", font=("Helvetica", 11, "bold"), anchor="center")
        label.pack(fill="x", pady=(0, 10))
        
        frame = ttk.Frame(main_frame)
        frame.pack(fill="both", expand=True)
        
        scrollbar = ttk.Scrollbar(frame)
        scrollbar.pack(side="right", fill="y")
        
        listbox = tk.Listbox(
            frame, 
            yscrollcommand=scrollbar.set, 
            font=("Helvetica", 10), 
            relief="solid", 
            bd=1
        )
        for pose in poses:
            listbox.insert("end", pose)
        listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=listbox.yview)
        
        selected_pose = [None]
        
        def on_select():
            sel = listbox.curselection()
            if sel:
                selected_pose[0] = listbox.get(sel[0])
                dialog.destroy()
            else:
                messagebox.showwarning("Selection", "Please select a pose first.", parent=dialog)
                
        btn_select = ttk.Button(
            main_frame, 
            text="Select Pose", 
            command=on_select
        )
        btn_select.pack(pady=(15, 0))
        
        self.root.wait_window(dialog)
        return selected_pose[0]

    def _run_retrain_model_pipeline(self):
        """
        Internal pipeline runner to compile models from files.
        """
        try:
            success = engine.train_and_save_model()
            return success
        except Exception as e:
            print(f"Retraining error: {e}")
            return False

    def view_poses(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Available Yoga Poses")
        dialog.geometry("480x520")
        dialog.transient(self.root)
        
        # Center the dialog on the screen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - (480 // 2)
        y = (screen_height // 2) - (520 // 2)
        dialog.geometry(f"480x520+{x}+{y}")
        
        dialog.lift()
        dialog.focus_force()
        
        main_frame = ttk.Frame(dialog, padding=15)
        main_frame.pack(fill="both", expand=True)
        
        title = ttk.Label(main_frame, text="Available Yoga Poses & Targets", font=("Helvetica", 14, "bold"), anchor="center")
        title.pack(fill="x", pady=(0, 10))
        
        # ScrolledText widget
        txt_area = scrolledtext.ScrolledText(main_frame, font=("Courier", 11), relief="solid", bd=1)
        txt_area.pack(fill="both", expand=True)
        
        # Build text content
        content = ""
        for pose_name, details in engine.TEMPLATES.items():
            content += f"=========================================\n"
            content += f"  {pose_name.upper()}\n"
            content += f"=========================================\n"
            content += "Ideal Joint Angles:\n"
            for joint, angle in details["angles"].items():
                friendly_name = joint.replace("_", " ").title()
                content += f"  • {friendly_name:<16} : {angle}°\n"
            content += "\n"
            
        txt_area.insert("insert", content)
        # Prevent keyboard editing to keep text readable without disabled gray-out
        txt_area.bind("<Key>", lambda e: "break")
        
        btn_close = ttk.Button(
            main_frame, 
            text="Close", 
            command=dialog.destroy
        )
        btn_close.pack(pady=(15, 0))

    def exit_app(self):
        self.root.quit()


def main():
    root = tk.Tk()
    app = YogasanaApp(root)
    
    # Handle clean window manager close
    def on_closing():
        app.exit_app()
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
