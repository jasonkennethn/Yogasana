# Yogasana: AI Yoga Pose Evaluator

Yogasana is a cross-platform, machine learning-powered yoga pose evaluation system. Using computer vision via MediaPipe landmark tracking and a Random Forest Classifier, the system identifies user yoga postures in real time, evaluates joint alignment against standard pose templates, and displays dynamic, real-time correction feedback.

---

## Key Features

- **Real-Time Landmark Tracking**: Extracts 33 skeleton coordinates from a live webcam feed using Google MediaPipe Pose.
- **Posture Classification**: Classifies postures using a Scikit-Learn Random Forest model.
- **Dynamic Alignment Scoring**: Calculates a score out of 100 based on joint angle deviations:
  $$\text{Score} = 100 - \text{Average Angle Error}$$
- **Real-Time Corrections Feed**: Displays visual overlays highlighting joints as green (correct) or red (requires adjustment) with direct angle metrics and correction comments on the camera HUD.
- **Native Cross-Platform UI**: Built using standard themed `ttk` widgets that render natively according to OS configurations (respecting Light/Dark modes) on Windows, macOS, and Linux, preventing color overriding or rendering hangs.
- **Auto-Installation**: Performs a startup check on required packages and installs them automatically from `requirements.txt` if missing.
- **Auto-Training on Startup**: Automatically seeds synthetic pose datasets and trains the Random Forest model on first boot if missing, ensuring the application is immediately ready to run.
- **Interactive Pose Customization**: Supports creating and improving yoga poses dynamically by recording new posture frames directly inside the app.

---

## 6 Pre-trained Poses Target Angles

The system comes pre-seeded with ideal templates for 6 key poses:

| Pose Name | L Elbow | R Elbow | L Shoulder | R Shoulder | L Hip | R Hip | L Knee | R Knee |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mountain Pose** | 180° | 180° | 15° | 15° | 180° | 180° | 180° | 180° |
| **Tree Pose** | 90° | 90° | 35° | 35° | 120° | 180° | 45° | 180° |
| **Warrior II** | 180° | 180° | 90° | 90° | 110° | 150° | 90° | 180° |
| **Chair Pose** | 180° | 180° | 160° | 160° | 115° | 115° | 110° | 110° |
| **Cobra Pose** | 140° | 140° | 55° | 55° | 160° | 160° | 180° | 180° |
| **Triangle Pose**| 180° | 180° | 90° | 90° | 100° | 130° | 180° | 180° |

---

## Directory Structure

```
Yogasana/
├── app.py                 # Main native GUI application
├── yogasana_engine.py     # Core mathematical calculations & ML model training pipelines
├── knowledgebase.json     # Joint maps, pose template angles, and feedback advice text
├── requirements.txt       # Core pinned third-party dependencies
├── LICENSE                # MIT License
├── README.md              # Project documentation
├── data/                  # Folder containing CSV datasets for each pose
└── models/                # Folder containing the compiled RandomForest Classifier model (pkl)
```

---

## Requirements

The project uses the following dependencies:
- `numpy`
- `pandas`
- `opencv-python`
- `mediapipe` (version `<=0.10.14`)
- `scikit-learn`
- `tkinter` (Standard Python GUI library)

---

## Installation & Running

1. **Clone or navigate** to the project directory:
   ```bash
   cd Yogasana
   ```
2. **Execute the application**:
   ```bash
   python app.py
   ```
   *Note: If any dependencies are missing in your active environment, the application will automatically install them before booting the main dashboard.*

---

## How to Use

1. **Identify Pose**: Starts the webcam. Align your full body in the frame. The system will detect your pose, highlight incorrect joint angles in red, and display tips on how to improve your alignment on screen. Press `q` or `Esc` to exit.
2. **Create & Train New Pose**: Prompts you to name a new pose. Position yourself in front of the camera. The system will run a 5-second preparation countdown, record 30 sample frames of your posture, update `knowledgebase.json`, and automatically retrain the classifier.
3. **Train Existing Pose**: Allows you to select one of the current poses and record 30 more samples of it to improve classification robustness and refine ideal template angles.
4. **View Available Poses**: Displays the list of preconfigured poses and their target joint angle specifications.
