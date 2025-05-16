# food_recog_tralalero
# 🍱 FOOD_RECOG_MACHINE – Hệ thống Nhận Diện Món Ăn và Tính Tiền Tự Động

Ứng dụng sử dụng Python kết hợp với giao diện `Tkinter` để:
- Nhận diện món ăn từ ảnh hoặc camera
- Cắt và xử lý hình ảnh món ăn
- Nhận diện món ăn bằng mô hình CNN ba đầu (multi-head CNN)
- Tính tiền tự động theo bảng giá (`gia_tien.json`)
- Hiển thị kết quả trên giao diện người dùng

---

## 🧰 Tính năng chính

✅ Giao diện trực quan với ảnh và nút bấm  
✅ Hỗ trợ ảnh tĩnh và camera trực tiếp (kể cả từ điện thoại)  
✅ Dự đoán món ăn bằng mô hình học sâu (PyTorch CNN)  
✅ In kết quả và tổng tiền theo bảng giá định trước  
✅ Có thể dễ dàng mở rộng thêm các món mới hoặc thay mô hình

Chú thích file :

main.py : code tổng hợp các file .py ở dưới và chạy chỉ với 1 lần duy nhất.
recognize.py : code để nhận diện thức ăn qua 
predict.py :  code nhận diện món ăn từ dữ liệu input băng tạo bouncing box để khoanh vùng món ăn (công cụ là Yolov8)
gui.py : code giao diện bằng thư viện tkinter => cũng như dây chính là code mở phần mềm nên khi chạy chỉ cần 
cnn.py : code train mô hình cnn để nhận diện món ăn
requirement : file để tải các thư viện về máy đảm bảo máy chạy ổn định ( tạo môi trường cho mô hình)

Trước khi sử dụng file cần đảm bảo các bạn đã tải đầy đủ các file và chạy
