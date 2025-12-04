import cv2
import json
import os
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
        """Check if anyone has left the camera view"""
        exited = []
        for name, last_seen in list(self.currently_visible.items()):
            if frame_count - last_seen > self.exit_grace_frames:
                # Person hasn't been seen for grace period
                if self.user_status.get(name) == "Entry":
                    exited.append(name)
                del self.currently_visible[name]
        return exited
    
    def mark_attendance(self, name, confidence):
        """Mark entry or exit based on current status"""
        os.makedirs("attendance", exist_ok=True)
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        
        # Determine action - toggle between Entry and Exit
        if name not in self.user_status:
            action = "Entry"
        else:
            action = "Exit" if self.user_status[name] == "Entry" else "Entry"
        
        csv_path = os.path.join("attendance", f"{date_str}.csv")
        exists = os.path.exists(csv_path)
        
        with open(csv_path, "a") as f:
            if not exists:
                f.write("date,time,name,action,confidence\n")
            f.write(f"{date_str},{time_str},{name},{action},{confidence:.2f}\n")
        
        # Update status
        self.user_status[name] = action
        
        # Send Telegram notification on every status change
        if self.telegram:
            message = format_attendance_message(name, action, time_str, confidence)
            self.telegram.send_sync(message)
            print(f"[TELEGRAM] Notification sent for {name}")
        
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

    print("[OK] Attendance system running. Press 'q' to quit.")
    
    tracker = AttendanceTracker(enable_telegram=True)

    window_name = "Attendance"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    threshold = 60
    current_person = {"name": "Unknown", "confidence": 0, "stable_count": 0}
    required_stable_frames = 10
    frame_count = 0
    frames_without_face = 0
    reset_threshold = 30
    marked_this_session = {}  # Track who has been marked in this detection session
    
    # Notification system
    notification = {"text": "", "time": None, "duration": 3}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame")
            break

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray_eq,
            scaleFactor=1.1,  # Improved detection
            minNeighbors=5,
            minSize=(120, 120)
        )

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

        if frame_count % 2 == 0 and len(faces) > 0:
            for (x, y, w, h) in faces:
                face_roi = gray_eq[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (150, 150))  # Match training size

                label_id, confidence = recognizer.predict(face_roi)

                if confidence < threshold:
                    name = id_to_label.get(str(label_id), "Unknown")
                    
                    if name == current_person["name"]:
                        current_person["stable_count"] += 1
                        current_person["confidence"] = confidence
                    else:
                        current_person = {"name": name, "confidence": confidence, "stable_count": 1}
                    
                    # Mark person as visible
                    if name != "Unknown":
                        tracker.update_visibility(name, frame_count)
                    
                    if current_person["stable_count"] == required_stable_frames:
                        # Only mark if not already marked in this session
                        if name not in marked_this_session:
                            # Only mark Entry if person is not already "Entry" status
                            if tracker.user_status.get(name) != "Entry":
                                action = tracker.mark_attendance(name, confidence)
                                if action:
                                    notification["text"] = f"{action} Marked: {name}"
                                    notification["time"] = datetime.now()
                                    marked_this_session[name] = True
                else:
                    if current_person["stable_count"] > 0:
                        current_person["stable_count"] -= 1
                    if current_person["stable_count"] == 0:
                        current_person = {"name": "Unknown", "confidence": 0, "stable_count": 0}
        
        # Check for exits (people who left camera view)
        exited_people = tracker.check_exits(frame_count)
        for name in exited_people:
            action = tracker.mark_exit(name)
            if action:
                notification["text"] = f"{action} Marked: {name}"
                notification["time"] = datetime.now()
                # Remove from marked session so they can be marked Entry again
                marked_this_session.pop(name, None)
        
        # Draw rectangles
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

        # Status - only show if person is detected
        if name_display != "Unknown":
            status = tracker.get_status(name_display)
            status_text = f"Status: {status if status else 'Not Present'}"
            if current_person["stable_count"] < required_stable_frames:
                status_text = f"Detecting... ({current_person['stable_count']}/{required_stable_frames})"
        else:
            status_text = "Status: Waiting for face..."

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

    cap.release()
    cv2.destroyAllWindows()
    print("Exiting attendance system.")

if __name__ == "__main__":
    main()
