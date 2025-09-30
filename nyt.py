import cv2
import time
import winsound
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO(r'C:\Users\trina\OneDrive\Desktop\YAWN\best (1).pt')

cap = cv2.VideoCapture(0)

yawn_count = 0
start_time = time.time()
last_yawn_time = 0
cooldown = 3  # seconds - cooldown between yawn detections
beeped = False  # To prevent multiple beeps

def get_alert_level(count):
    if count > 3:
        return "RED"
    elif count < 2:
        return "GREEN"
    else:
        return "YELLOW"

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Webcam error")
        break

    # Run YOLO detection (smaller image → faster speed)
    results = model.predict(frame, conf=0.7, verbose=False)
    names = results[0].names
    boxes = results[0].boxes

    detected_yawn = False

    for box in boxes:
        cls = int(box.cls[0])
        label = names[cls]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Draw bounding box
        color = (0, 255, 0) if label.lower() != "yawn" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if label.lower() == "yawn":
            current_time = time.time()
            if current_time - last_yawn_time > cooldown:
                yawn_count += 1
                last_yawn_time = current_time
                print(f"✅ Yawn Detected. Count: {yawn_count}")
            detected_yawn = True

    # Reset count every 60s
    elapsed_time = time.time() - start_time
    if elapsed_time > 60:
        yawn_count = 0
        start_time = time.time()
        beeped = False

    # Get alert level
    alert_level = get_alert_level(yawn_count)

    # Show zone & count
    zone_color = (0, 255, 0) if alert_level == "GREEN" else (0, 255, 255) if alert_level == "YELLOW" else (0, 0, 255)
    cv2.putText(frame, f'ALERT: {alert_level}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, zone_color, 3)
    cv2.putText(frame, f'Yawn Count: {yawn_count}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, zone_color, 2)

    # Beep only once per red zone
    if alert_level == "RED" and not beeped:
        winsound.Beep(1000, 500)
        beeped = True

    cv2.imshow("Yawn Detection - 3 Level Alert", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
