# Tạo dictionary
sinh_vien = {
    "ten": "Nguyen Van A",
    "tuoi": 20,
    "diem": 8.5
}

# Truy xuất giá trị
print(sinh_vien["ten"])  # Nguyen Van A

# Thêm/sửa phần tử
sinh_vien["lop"] = "CNTT1"
sinh_vien["diem"] = 9.0

# Xóa phần tử
del sinh_vien["tuoi"]

# Các phương thức quan trọng
sinh_vien.keys()    # Lấy tất cả key
sinh_vien.values()  # Lấy tất cả value
sinh_vien.items()   # Lấy cặp (key, value)
print(sinh_vien)