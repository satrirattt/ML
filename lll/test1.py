import cv2
import os
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog

# โหลดโมเดล YOLOv8
model = YOLO("yolov8n.pt")

# เลือกไฟล์วิดีโอ
root = tk.Tk()
root.withdraw()
video_path = filedialog.askopenfilename(
    title="เลือกไฟล์วิดีโอ",
    filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
)

if not video_path:
    print("ไม่ได้เลือกไฟล์")
    exit()

# สร้างโฟลเดอร์เก็บผลลัพธ์
output_folder = "frames_with_motorcycle4"
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps)  # 1 วินาทีต่อเฟรม

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ตรวจทุก 1 วินาที
    if frame_count % frame_interval == 0:

        results = model(frame)
        found_motorcycle = False

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                conf = float(box.conf[0])

                if label == "motorcycle" and conf > 0.5:
                    found_motorcycle = True
                    break

        # ถ้ามีมอไซในเฟรม → บันทึกทั้งภาพ
        if found_motorcycle:
            filename = os.path.join(
                output_folder,
                f"frame_{saved_count:04d}.jpg"
            )
            cv2.imwrite(filename, frame)
            saved_count += 1

    frame_count += 1

cap.release()

print("เสร็จแล้ว!")
print("จำนวนภาพที่มีมอไซ:", saved_count)
