# from ultralytics import YOLO
# import cv2
# import os

# # ===== CONFIG =====
# image_path = 'test/img4.jpg'   # đường dẫn ảnh đầu vào
# output_dir = 'cropped_bowls'            # thư mục lưu ảnh đã crop
# target_class_id = 45                    # bowl có ID = 45 trong COCO
# model_path = 'model/yolov8m.pt'             
# # ===================


# os.makedirs(output_dir, exist_ok=True)

# # Load model 
# model = YOLO(model_path)

# # Load ảnh
# image = cv2.imread(image_path)
# height, width, _ = image.shape

# # Chạy detect
# results = model(image_path)[0]

# count = 0
# for box in results.boxes:
#     cls_id = int(box.cls[0].item())
#     if cls_id == target_class_id:
#         # Lấy tọa độ bounding box
#         x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
#         # Crop hình
#         cropped = image[y1:y2, x1:x2]
#         # Lưu ảnh đã crop
#         crop_path = os.path.join(output_dir, f'bowl_{count}.jpg')
#         cv2.imwrite(crop_path, cropped)
#         print(f'Cropped bowl saved to: {crop_path}')
#         count += 1

# if count == 0:
#     print("No bowl detected")
# else:
#     print(f'Total bowls cropped: {count}')
from ultralytics import YOLO
import cv2
import os
import sys

# ===== CONFIG =====
output_dir = 'cropped_bowls'            # thư mục lưu ảnh đã crop
target_class_id = 45                    # bowl có ID = 45 trong COCO
model_path = 'model/yolov8m.pt'             
# ===================

# Lấy đường dẫn ảnh từ dòng lệnh
if len(sys.argv) < 2:
    print("chưa nhập đường dẫn ảnh!")
    sys.exit(1)

image_path = sys.argv[1]

os.makedirs(output_dir, exist_ok=True)

# Load model 
model = YOLO(model_path)

# Load ảnh
image = cv2.imread(image_path)
height, width, _ = image.shape

# Chạy detect
results = model(image_path)[0]

count = 0
for box in results.boxes:
    cls_id = int(box.cls[0].item())
    if cls_id == target_class_id:
        # Lấy tọa độ bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        # Crop hình
        cropped = image[y1:y2, x1:x2]
        # Lưu ảnh đã crop
        crop_path = os.path.join(output_dir, f'bowl_{count}.jpg')
        cv2.imwrite(crop_path, cropped)
        print(f'Cropped bowl saved to: {crop_path}')
        count += 1

if count == 0:
    print("No bowl detected.")
else:
    print(f'Total bowls cropped: {count}')
