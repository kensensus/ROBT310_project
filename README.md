# Face Attendance System

A real-time face recognition-based attendance tracking system using OpenCV and LBPH (Local Binary Patterns Histograms). The system automatically marks entry and exit timestamps with optional Telegram notifications.

## Features

- Real-time face detection and recognition using Haar Cascades and LBPH
- Automatic entry/exit tracking with cooldown periods
- CSV-based attendance logging by date
- Optional Telegram bot integration for instant notifications
- Fullscreen interface with status indicators
- Stable detection algorithm to prevent false positives

## Requirements

- Python 3.7+
- Webcam
- Operating System: Linux, macOS, or Windows

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/face_attendance.git
cd face_attendance
```

### 2. Create Virtual Environment

**Linux/macOS:**
```bash
python3 -m venv face_attendance_env
source face_attendance_env/bin/activate
```

**Windows:**
```cmd
python -m venv face_attendance_env
face_attendance_env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install opencv-contrib-python
pip install requests
```

### 4. Download Haar Cascade

Download `haarcascade_frontalface_default.xml` from the OpenCV repository:
```bash
wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
```

Or manually download and place it in the project root.

## Project Structure

```
face_attendance/
├── collect_faces.py          # Collect training images
├── train_lbph.py             # Train the LBPH model
├── recognize_attendance.py    # Main attendance system
├── telegram_bot.py           # Telegram notification module
├── setup_telegram.py         # Configure Telegram bot
├── haarcascade_frontalface_default.xml
├── dataset/                  # Training images (created automatically)
│   └── [person_name]/
├── attendance/               # Daily CSV files (created automatically)
│   └── YYYY-MM-DD.csv
├── trainer.yml              # Trained LBPH model (generated)
├── labels.json              # Label mapping (generated)
└── telegram_config.json     # Telegram credentials (optional)
```

## Usage

### Step 1: Collect Face Data

Collect face images for each person (minimum 100 images recommended):

```bash
python collect_faces.py
```

- Enter the person's name when prompted
- Press 's' to start capturing
- Press 'q' to stop
- Repeat for each person

### Step 2: Train the Model

Train the LBPH recognizer on collected images:

```bash
python train_lbph.py
```

This generates `trainer.yml` and `labels.json`.

### Step 3: Configure Telegram (Optional)

Set up Telegram notifications:

```bash
python setup_telegram.py
```

Follow the prompts to:
1. Create a bot via [@BotFather](https://t.me/botfather)
2. Get your chat ID via [@userinfobot](https://t.me/userinfobot)
3. Enter credentials when prompted

### Step 4: Run Attendance System

Start the attendance recognition system:

```bash
python recognize_attendance.py
```

- The system runs in fullscreen mode
- Automatically detects faces and marks entry/exit
- Press 'q' to quit
- Attendance is saved in `attendance/YYYY-MM-DD.csv`

## Configuration

### Adjusting Detection Parameters

Edit `recognize_attendance.py`:

```python
# Cooldown between markings (minutes)
tracker = AttendanceTracker(cooldown_minutes=0.5)

# Confidence threshold (lower = stricter)
threshold = 60

# Stable frames required before marking
required_stable_frames = 10
```

### Face Detection Settings

```python
faces = face_cascade.detectMultiScale(
    gray_eq,
    scaleFactor=1.1,    # Detection sensitivity
    minNeighbors=5,     # Minimum detections required
    minSize=(120, 120)  # Minimum face size in pixels
)
```

## Attendance File Format

CSV files are saved in `attendance/` with the following format:

```csv
date,time,name,action,confidence
2025-01-15,09:30:45,John Doe,Entry,45.32
2025-01-15,17:15:22,John Doe,Exit,42.18
```

## Troubleshooting

### Camera Not Opening
- Check if camera is being used by another application
- Verify camera permissions
- Try different camera index: `cv2.VideoCapture(1)`

### Poor Recognition Accuracy
- Collect more training images (200+ recommended)
- Ensure good lighting during collection and recognition
- Adjust `threshold` value in `recognize_attendance.py`
- Retrain model with `train_lbph.py`

### Telegram Not Working
- Verify bot token and chat ID in `telegram_config.json`
- Check internet connection
- Ensure bot is started (send /start to your bot)

## Linux Automation (Optional)

Use the provided shell script for automated execution:

```bash
chmod +x run_with_venv.sh
./run_with_venv.sh recognize_attendance.py
```

## Technical Details

- **Face Detection**: Haar Cascade Classifier (frontal face)
- **Recognition Algorithm**: LBPH (Local Binary Patterns Histograms)
- **Image Preprocessing**: Histogram equalization for better contrast
- **Stability Algorithm**: Requires 10 consecutive stable frames before marking
- **Auto-reset**: Resets detection after 30 frames without face

## Credits

Built with OpenCV and Python for ROBT 310 - Image Processing course project.
