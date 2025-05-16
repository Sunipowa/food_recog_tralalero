import subprocess

# ===== Nhập ảnh =====
image_path = "cache/cropped_img.jpg"

# ===== Gọi predict.py =====
subprocess.run(['python', 'predict.py', image_path])

# ===== Gọi regconize.py =====
subprocess.run(['python', 'regconize.py'])
