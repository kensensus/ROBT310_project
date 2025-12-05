import os
# Prefer X11 backend for Qt to avoid missing Wayland plugin in some OpenCV builds
# This must be set before importing cv2 so the Qt plugin selection happens correctly.
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
import cv2
import json
from datetime import datetime, timedelta
from telegram_bot import TelegramNotifier, format_attendance_message, load_config, TELEGRAM_AVAILABLE

# Load label mapping
def load_labels(path="labels.json"):
    if not os.path.exists(path):
        print("[ERROR] labels.json not found. Run train_lbph.py first.")
        return None
    with open(path, "r") as f:
        id_to_label = json.load(f)
    return id_to_label

class AttendanceTracker:
    def __init__(self, enable_telegram=True):
        self.user_status = {}  # {name: "Entry"/"Exit"}
        self.currently_visible = {}  # {name: frame_count} - track who is currently in view
        self.exit_grace_frames = 60  # ~2 seconds at 30fps before marking exit
        self.load_today_status()
        
        # Initialize Telegram
        self.telegram = None
        if enable_telegram and TELEGRAM_AVAILABLE:
            config = load_config()
            if config and config.get("enabled", False):
                try:
                    # Support both 'bot_token' and 'token' key names
                    token = config.get("bot_token") or config.get("token")
                    chat_id = config.get("chat_id")
                    
                    if token and chat_id:
                        self.telegram = TelegramNotifier(token, chat_id)
                        print("[OK] Telegram notifications enabled")
                    else:
                        print("[WARNING] Telegram config missing token or chat_id")
                except Exception as e:
                    print(f"[WARNING] Failed to initialize Telegram: {e}")
            else:
                print("[WARNING] Telegram not enabled in config or config not found")
        elif enable_telegram and not TELEGRAM_AVAILABLE:
            print("[WARNING] python-telegram-bot not installed. Telegram notifications disabled.")
    
    def load_today_status(self):
        """Load today's attendance to restore status"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        csv_path = os.path.join("attendance", f"{date_str}.csv")
        
        if os.path.exists(csv_path):
            with open(csv_path, "r") as f:
                lines = f.readlines()[1:]  # Skip header
                for line in lines:
                    parts = line.strip().split(",")
                    if len(parts) >= 4:
                        name, action = parts[2], parts[3]
                        self.user_status[name] = action
    
    def update_visibility(self, name, frame_count):
        """Update that person is currently visible"""
        self.currently_visible[name] = frame_count
    
    def check_exits(self, frame_count):
        """Remove people who haven't been seen recently from visibility tracking.

        IMPORTANT: Do NOT auto-mark exits here. Disappearance from camera does
        not imply an Exit event. Instead, we remove them from the `currently_visible`
        map and return the list of removed names so the caller can clear any
        per-session state (like `marked_this_session`) if desired.
        """
        removed = []
        for name, last_seen in list(self.currently_visible.items()):
            if frame_count - last_seen > self.exit_grace_frames:
                removed.append(name)
                del self.currently_visible[name]
        return removed
    
    def mark_attendance(self, name, confidence):
        """Mark entry or exit based on current status"""
        os.makedirs("attendance", exist_ok=True)
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        
        # Determine action - toggle between Entry and Exit
        current_status = self.user_status.get(name)
        if current_status is None:
            action = "Entry"
        else:
            action = "Exit" if current_status == "Entry" else "Entry"
        
        print(f"[DEBUG] {name}: Current status={current_status}, Next action={action}")
        
        csv_path = os.path.join("attendance", f"{date_str}.csv")
        exists = os.path.exists(csv_path)
        
        with open(csv_path, "a") as f:
            if not exists:
                f.write("date,time,name,action,confidence\n")
            f.write(f"{date_str},{time_str},{name},{action},{confidence:.2f}\n")
        
        # Update status BEFORE sending telegram
        self.user_status[name] = action
        
        # Send Telegram notification with better error handling
        print(f"[TELEGRAM] Attempting to send notification for {name} - {action}")
        if self.telegram:
            try:
                message = format_attendance_message(name, action, time_str, confidence)
                self.telegram.send_sync(message)
                print(f"[TELEGRAM] Notification sent successfully for {name} - {action}")
            except Exception as e:
                print(f"[TELEGRAM] Failed to send notification: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"[TELEGRAM] Telegram notifier not initialized")
        
        print(f"[OK] Marked {action}: {name} at {time_str} (conf={confidence:.2f})")
        return action
    
    def mark_exit(self, name):
        """Mark exit for a person who left camera view"""
        if self.user_status.get(name) != "Entry":
            return None
        
        os.makedirs("attendance", exist_ok=True)
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        
        action = "Exit"
        
        csv_path = os.path.join("attendance", f"{date_str}.csv")
        exists = os.path.exists(csv_path)
        
        with open(csv_path, "a") as f:
            if not exists:
                f.write("date,time,name,action,confidence\n")
            f.write(f"{date_str},{time_str},{name},{action},0.00\n")
        
        # Update status
        self.user_status[name] = action
        
        # Send Telegram notification
        if self.telegram:
            message = format_attendance_message(name, action, time_str, 0)
            self.telegram.send_sync(message)
            print(f"[TELEGRAM] Notification sent for {name}")
        
        print(f"[OK] Marked {action}: {name} at {time_str} (auto-exit)")
        return action

    def get_status(self, name):
        """Return current status for a user (or None)."""
        return self.user_status.get(name)
    
    def mark_all_exit_on_close(self):
        """Mark all users with Entry status as Exit when system closes"""
        try:
            date_str = datetime.now().strftime("%Y-%m-%d")
            time_str = datetime.now().strftime("%H:%M:%S")
            csv_path = os.path.join("attendance", f"{date_str}.csv")
            
            if not os.path.exists(csv_path):
                return
            
            # Mark all Entry users as Exit
            with open(csv_path, "a") as f:
                for name, status in self.user_status.items():
                    if status == "Entry":
                        f.write(f"{date_str},{time_str},{name},Exit,0.00\n")
                        print(f"[AUTO-EXIT] Marked {name} as Exit on system close")
                        
                        # Send Telegram notification
                        if self.telegram:
                            message = format_attendance_message(name, "Exit", time_str, 0)
                            self.telegram.send_sync(message)
            
            print("[OK] All users marked as Exit")
        except Exception as e:
            print(f"[WARNING] Failed to mark users as Exit: {e}")
    
def main():
    # Load trained LBPH model
    if not os.path.exists("trainer.yml"):
        print("[ERROR] trainer.yml not found. Run train_lbph.py first.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")
    print("[OK] Loaded LBPH model.")

    id_to_label = load_labels("labels.json")
    if id_to_label is None:
        return

    cascade_path = "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return
    
    # Improve camera settings for better image quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    print("[OK] Attendance system running. Press 'q' to quit.")
    
    tracker = AttendanceTracker(enable_telegram=True)

    window_name = "Attendance"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    threshold = 65  #  Increased threshold for more lenient matching
    current_person = {"name": "Unknown", "confidence": 0, "stable_count": 0}
    required_stable_frames = 8  # Reduced from 10 for faster detection
    frame_count = 0
    frames_without_face = 0
    reset_threshold = 30
    marked_this_session = {}
    faces = []  # Initialize faces list
    
    # Notification system
    notification = {"text": "", "time": None, "duration": 3}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame")
            break

        frame_count += 1
        
        # OPTIMIZED PREPROCESSING
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for better contrast (fast operation)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_eq = clahe.apply(gray)
        
        # Simple bilateral filter (faster than NlMeans, still reduces noise)
        gray_eq = cv2.bilateralFilter(gray_eq, 5, 50, 50)

        # SMART DETECTION - Only run heavy detection every N frames
        detect_this_frame = (frame_count % 3 == 0)
        
        if detect_this_frame:
            # First attempt: Standard detection
            faces = face_cascade.detectMultiScale(
                gray_eq,
                scaleFactor=1.05,
                minNeighbors=4,
                minSize=(100, 100),
                maxSize=(400, 400)
            )
            
            # If no faces found, try with relaxed parameters
            if len(faces) == 0:
                faces = face_cascade.detectMultiScale(
                    gray_eq,
                    scaleFactor=1.1,
                    minNeighbors=3,
                    minSize=(80, 80),
                    maxSize=(500, 500)
                )
        # else: reuse faces from previous frame (faces list persists)

        # Reset if no faces detected for a while
        if len(faces) == 0:
            frames_without_face += 1
            if frames_without_face >= reset_threshold:
                current_person = {"name": "Unknown", "confidence": 0, "stable_count": 0}
        else:
            frames_without_face = 0

        name_display = current_person["name"]
        color = (0, 255, 0) if name_display != "Unknown" else (0, 0, 255)
        confidence_display = f"{current_person['confidence']:.1f}" if current_person['confidence'] > 0 else "-"

        # Only recognize every other frame for better performance
        recognize_this_frame = (frame_count % 2 == 0)
        
        if recognize_this_frame and len(faces) > 0:
            # Take the largest face (closest to camera)
            faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, w, h = faces_sorted[0]
            
            face_roi = gray_eq[y:y+h, x:x+w]
            
            # Skip blur detection on every frame for performance
            if frame_count % 5 == 0:  # Only check blur every 5th frame
                laplacian_var = cv2.Laplacian(face_roi, cv2.CV_64F).var()
                if laplacian_var < 50:
                    cv2.putText(frame, "Face too blurry", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    # Don't continue - still try to recognize
            
            face_roi_resized = cv2.resize(face_roi, (150, 150))
            label_id, confidence = recognizer.predict(face_roi_resized)

            if confidence < threshold:
                name = id_to_label.get(str(label_id), "Unknown")
                
                if name == current_person["name"]:
                    current_person["stable_count"] += 1
                    current_person["confidence"] = confidence
                else:
                    current_person = {"name": name, "confidence": confidence, "stable_count": 1}
                
                if name != "Unknown":
                    tracker.update_visibility(name, frame_count)
                
                if current_person["stable_count"] >= required_stable_frames:
                    if name not in marked_this_session:
                        print(f"[DEBUG] Marking {name} - marked_this_session: {list(marked_this_session.keys())}")
                        action = tracker.mark_attendance(name, confidence)
                        if action:
                            notification["text"] = f"{action} Marked: {name}"
                            notification["time"] = datetime.now()
                            marked_this_session[name] = action
                            print(f"[DEBUG] Added {name} to marked_this_session with action {action}")
                    else:
                        print(f"[DEBUG] {name} already in marked_this_session with action {marked_this_session[name]}")
                    current_person["stable_count"] = required_stable_frames
            else:
                if current_person["stable_count"] > 0:
                    current_person["stable_count"] -= 1
                if current_person["stable_count"] == 0:
                    current_person = {"name": "Unknown", "confidence": 0, "stable_count": 0}
        
        # Check for exits
        exited_people = tracker.check_exits(frame_count)
        for name in exited_people:
            if name in marked_this_session:
                print(f"[DEBUG] Removing {name} from marked_this_session (was {marked_this_session[name]})")
                marked_this_session.pop(name, None)
            if current_person["name"] == name:
                current_person = {"name": "Unknown", "confidence": 0, "stable_count": 0}
                print(f"[DEBUG] Reset current_person for {name}")
            print(f"[INFO] {name} left camera view - ready for status toggle on return")

        # Simplified drawing
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

        # Status display
        if name_display != "Unknown":
            status = tracker.get_status(name_display)
            status_text = f"Status: {status if status else 'Not Present'}"
            if current_person["stable_count"] < required_stable_frames:
                status_text = f"Detecting... ({current_person['stable_count']}/{required_stable_frames})"
        else:
            status_text = "Status: Waiting for face..."

        # Simplified guidance (only when no faces)
        if len(faces) == 0 and frame_count % 30 == 0:  # Update message every 30 frames
            cv2.putText(frame, "No face detected", 
                       (50, frame.shape[0]//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Top banner
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
        cv2.putText(frame, f"Person: {name_display}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        # Bottom status
        cv2.rectangle(frame, (0, frame.shape[0]-80), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.putText(frame, f"{status_text}   Confidence: {confidence_display}", 
                    (20, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # Show notification
        if notification["time"]:
            elapsed = (datetime.now() - notification["time"]).total_seconds()
            if elapsed < notification["duration"]:
                cv2.rectangle(frame, (frame.shape[1]//4, 100), 
                            (3*frame.shape[1]//4, 200), (0, 200, 0), -1)
                cv2.putText(frame, notification["text"], 
                           (frame.shape[1]//4 + 20, 160),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            else:
                notification["time"] = None

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Mark all users as Exit before closing
    print("Closing system...")
    tracker.mark_all_exit_on_close()
    
    cap.release()
    cv2.destroyAllWindows()
    print("Exiting attendance system.")

if __name__ == "__main__":
    main()
