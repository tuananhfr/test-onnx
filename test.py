import cv2
from ultralytics import YOLO
import time

# 1. Load model ONNX
# Nếu dùng CPU Intel, hãy cân nhắc export sang format='openvino' sẽ nhanh hơn nhiều
model = YOLO("license_plates1.onnx", task="detect")

# 2. Cấu hình Camera IP
rtsp_url = "rtsp://tuananh:tuananh123@192.168.0.60:554/stream1"

# 3. Chạy inference với các tham số tối ưu cho CPU
# vid_stride: Cứ N khung hình mới xử lý 1 (giảm tải CPU)
# imgsz=1280: Detect xa (10-20m)
# conf=0.3: Tăng confidence để giảm false positives
# device='cpu': Force CPU (tránh overhead GPU)
# verbose=False: Giảm log spam
results = model.predict(
    source=rtsp_url,
    imgsz=1280,
    conf=0.3,  # Tăng từ 0.25 -> 0.3 để giảm false positive
    iou=0.5,   # NMS threshold
    stream=True,
    vid_stride=5,  # Giảm từ 10 -> 5 để responsive hơn (vẫn tiết kiệm CPU)
    device='cpu',  # Force CPU
    verbose=False,  # Tắt log spam
    max_det=10  # Giới hạn max 10 detections/frame (tránh overhead)
)

print("=== YOLOv8 License Plate Detection ===")
print(f"Model: 1280x1280 ONNX")
print(f"Stream: {rtsp_url}")
print(f"Vid stride: 5 (process every 5th frame)")
print(f"Confidence: 0.3")
print("Press 'q' to quit\n")

# FPS tracking
fps_start_time = time.time()
frame_count = 0
fps = 0

for r in results:
    # r.plot() trả về ảnh đã vẽ khung
    frame_annotated = r.plot()

    # Calculate FPS
    frame_count += 1
    if frame_count % 10 == 0:  # Update FPS every 10 frames
        fps = 10 / (time.time() - fps_start_time)
        fps_start_time = time.time()

    # Tối ưu hiển thị: Resize nhỏ lại
    display_frame = cv2.resize(frame_annotated, (960, 540))

    # Show FPS and detection count on frame
    num_detections = len(r.boxes)
    cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Detections: {num_detections}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Hiển thị
    cv2.imshow("YOLOv8 License Plate Detection - 1280px", display_frame)

    # Thoát khi nhấn q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("\nStopped.")
