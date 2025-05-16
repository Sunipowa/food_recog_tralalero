import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import subprocess
import threading

root = tk.Tk()
root.title("TRALALAGUI")
root.geometry("1000x600")
root.configure(bg="white")

# ==== Load icon ====
def load_icon(path):
    img = Image.open(path)
    img = img.resize((50, 50))
    return ImageTk.PhotoImage(img)

icon_img = load_icon("source/icon.jpg")

# ==== Left panel ====
left_frame = tk.Frame(root, width=200, bg="white")
left_frame.pack(side="left", fill="y")

tk.Label(left_frame, image=icon_img, bg="white").pack(pady=(10, 5))
tk.Label(left_frame, text="TRALALAGUI", font=("Arial", 14, "bold"), bg="white").pack()
tk.Label(left_frame, text="FOOD_RECOG_MACHINE", font=("Arial", 9), bg="white", fg="gray").pack()
tk.Entry(left_frame, font=("Arial", 12), justify="center").pack(pady=20, ipadx=10, ipady=5)
tk.Button(left_frame, text="TOTAL BILL :", bg="#2f80ed", fg="white", font=("Arial", 12, "bold")).pack(pady=10)

# ==== Output box to show script output ====
output_text = tk.Text(left_frame, height=10, width=25, font=("Arial", 10))
output_text.pack(pady=10)

# ==== Right panel ====
right_frame = tk.Frame(root, bg="white")
right_frame.pack(side="right", expand=True, fill="both")

# ==== Giao diện chính ====
main_view = tk.Frame(right_frame, bg="white")
main_view.pack(fill="both", expand=True)

tk.Label(main_view, text="RECOGNITION GOGOGO", font=("Arial", 16, "bold"), bg="white").pack(pady=10)
tk.Label(main_view, text="Món ăn khách hàng order :", font=("Arial", 12), bg="white").pack()

media_frame = tk.Frame(main_view, width=640, height=480, bg="black")
media_frame.pack(pady=10)
media_label = tk.Label(media_frame)
media_label.pack()

btn_frame = tk.Frame(main_view, bg="white")
btn_frame.pack()
tk.Button(btn_frame, text="Camera", font=("Arial", 11), command=lambda: show_camera()).pack(side="left", padx=10)
tk.Button(btn_frame, text="+ Ảnh", font=("Arial", 11), command=lambda: open_image()).pack(side="left", padx=10)
tk.Button(btn_frame, text="Tính tiền", font=("Arial", 11), command=lambda: crop_and_save()).pack(side="left", padx=10)

# ==== Giao diện croped_bowl ====
cropped_view = tk.Frame(right_frame, bg="white")

# ==== Biến toàn cục ====
cap = None
current_image = None

# ==== Show camera ====
def show_camera():
    global cap, current_image

    if os.path.exists("cropped_bowls"):
        for f in os.listdir("cropped_bowls"):
            os.remove(os.path.join("cropped_bowls", f))

    if cap is None:
        cap = cv2.VideoCapture(1)

    def update_frame():
        global current_image
        if cap:
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame).resize((640, 480))
                current_image = img.copy()
                img_tk = ImageTk.PhotoImage(img)
                media_label.config(image=img_tk)
                media_label.image = img_tk
            media_label.after(15, update_frame)

    update_frame()


# ==== Mở ảnh từ file ====

def open_image():
    global cap, current_image

    if os.path.exists("cropped_bowls"):
        for f in os.listdir("cropped_bowls"):
            os.remove(os.path.join("cropped_bowls", f))

    if cap:
        cap.release()
        cap = None

    path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    if not path:
        return
    img = Image.open(path).resize((640, 480))
    current_image = img.copy()
    img_tk = ImageTk.PhotoImage(img)
    media_label.config(image=img_tk)
    media_label.image = img_tk



# ==== Cắt ảnh, lưu, chạy script và hiển thị output ====
def crop_and_save():
    global current_image
    if current_image is None:
        print("Không có ảnh để crop!")
        return
    width, height = current_image.size
    cropped_img = current_image.crop((0, 0, width, height))

    os.makedirs("cache", exist_ok=True)
    os.makedirs("cropped_bowls", exist_ok=True)
    save_path = os.path.join("cropped_bowls", "cropped_img.jpg")

    cropped_img.save(save_path)

    output_text.delete("1.0", tk.END)

    # Hàm chạy script trong thread riêng để không làm đứng GUI
    def run_script():
        process = subprocess.Popen(
            ['python', 'main.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        for line in process.stdout:
            output_text.insert(tk.END, line)
            output_text.see(tk.END)  # Scroll xuống cuối
        process.stdout.close()
        process.wait()
        show_cropped_images()

    threading.Thread(target=run_script).start()

# ==== Hiển thị ảnh từ folder croped_bowl ====
def show_cropped_images():
    main_view.pack_forget()
    for widget in cropped_view.winfo_children():
        widget.destroy()

    cropped_view.pack(fill="both", expand=True)

    tk.Label(cropped_view, text="Món ăn của bạn:", font=("Arial", 14, "bold"), bg="white").pack(pady=10)
    img_container = tk.Frame(cropped_view, bg="white")
    img_container.pack(pady=10)

    folder = "cropped_bowls"
    files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))] if os.path.exists(folder) else []
    if not files:
        tk.Label(cropped_view, text="Không có ảnh", bg="white", fg="gray").pack()
    else:
        for i, filename in enumerate(files):
            path = os.path.join(folder, filename)
            img = Image.open(path).resize((150, 150))
            img_tk = ImageTk.PhotoImage(img)
            panel = tk.Label(img_container, image=img_tk, bg="white")
            panel.image = img_tk
            panel.grid(row=i // 4, column=i % 4, padx=10, pady=10)

    tk.Button(cropped_view, text="⬅Back", font=("Arial", 11), command=show_main_view).pack(pady=10)

# ==== Quay về giao diện chính ====
def show_main_view():
    cropped_view.pack_forget()
    main_view.pack(fill="both", expand=True)

# ==== Run GUI ====
root.mainloop()
