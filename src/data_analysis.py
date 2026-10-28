import os
import yaml
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict


# ----------------------------
# Khởi tạo cấu hình
# ----------------------------

plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Tạo thư mục đầu ra
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_analysis.out")
os.makedirs(output_dir, exist_ok=True)

# Khởi tạo bộ đếm thống kê
stats = {
    "total_images": 0,          # Tổng số ảnh
    "missing_labels": 0,        # Số file nhãn bị thiếu
    "corrupted_images": 0,      # Số ảnh hỏng/không đọc được
    "invalid_annotations": 0    # Số dòng nhãn không hợp lệ
}

# ----------------------------
# Nạp cấu hình dataset
# ----------------------------
# Xác định vị trí file data.yaml
script_dir = os.path.dirname(os.path.abspath(__file__))
data_yaml_path = os.path.join(script_dir, "dataset", "data.yaml")

if not os.path.exists(data_yaml_path):
    print(f"Lỗi: không tìm thấy file data.yaml - {data_yaml_path}")
    exit()

# Đọc file cấu hình
with open(data_yaml_path, "r") as file:
    data = yaml.safe_load(file)

# Dựng đường dẫn động
base_dir = os.path.dirname(data_yaml_path)  # Thư mục gốc của dataset
class_names = data["names"]

# Dựng đường dẫn các tập dữ liệu
train_dir = os.path.join(base_dir, data["train"])
val_dir = os.path.join(base_dir, data["val"])
test_dir = os.path.join(base_dir, data["test"])

# Dựng đường dẫn thư mục nhãn
train_label_dir = os.path.join(base_dir, "train/labels")
val_label_dir = os.path.join(base_dir, "valid/labels")
test_label_dir = os.path.join(base_dir, "test/labels")

# Kiểm tra các thư mục quan trọng có tồn tại không
for path in [train_dir, val_dir, test_dir]:
    if not os.path.exists(path):
        print(f"Lỗi: thư mục không tồn tại - {path}")
        exit()

# ----------------------------
# 1. Thống kê phân bố lớp
# ----------------------------
def count_class_distribution(image_dir, label_dir):
    image_dir = os.path.join(image_dir, "images")
    class_counts = defaultdict(int)
    
    # Duyệt các file ảnh để kiểm tra nhãn
    image_files = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png"))]
    stats["total_images"] += len(image_files)  # Cộng dồn tổng số ảnh
    
    for img_file in image_files:
        label_file = img_file.rsplit(".", 1)[0] + ".txt"
        label_path = os.path.join(label_dir, label_file)
        
        # Kiểm tra file nhãn có tồn tại không
        if not os.path.exists(label_path):
            stats["missing_labels"] += 1
            continue  # Bỏ qua ảnh không có nhãn

    # Thống kê các nhãn hợp lệ
    label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
    for label_file in label_files:
        label_path = os.path.join(label_dir, label_file)
        try:
            with open(label_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    try:
                        class_id = int(line.strip().split()[0])
                        class_counts[class_id] += 1
                    except:
                        stats["invalid_annotations"] += 1
        except Exception as e:
            print(f"Đọc file nhãn thất bại: {label_path} - {str(e)}")
    
    return class_counts

# Thống kê phân bố của từng tập dữ liệu
train_counts = count_class_distribution(train_dir, train_label_dir)
val_counts = count_class_distribution(val_dir, val_label_dir)
test_counts = count_class_distribution(test_dir, test_label_dir)

# Vẽ biểu đồ phân bố lớp
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(class_names))
width = 0.25

ax.bar(x - width, [train_counts[i] for i in range(len(class_names))], width, label="Tập train")
ax.bar(x, [val_counts[i] for i in range(len(class_names))], width, label="Tập validation")
ax.bar(x + width, [test_counts[i] for i in range(len(class_names))], width, label="Tập test")

ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=45)
ax.set_ylabel("Số mẫu")
ax.set_title("Phân bố lớp của các tập dữ liệu")
ax.legend()

plt.savefig(os.path.join(output_dir, "class_distribution.png"), bbox_inches="tight")
plt.close()


# ----------------------------
# 2. Phân tích bounding box
# ----------------------------
def analyze_bbox(label_dir):
    widths, heights, centers_x, centers_y = [], [], [], []
    
    for label_file in os.listdir(label_dir):
        if not label_file.endswith(".txt"):
            continue
            
        with open(os.path.join(label_dir, label_file), "r") as f:
            for line in f.readlines():
                try:
                    _, x, y, w, h = map(float, line.strip().split())
                    widths.append(w)
                    heights.append(h)
                    centers_x.append(x)
                    centers_y.append(y)
                except:
                    stats["invalid_annotations"] += 1
    
    return widths, heights, centers_x, centers_y

# ----------------------------
# 3. Vẽ các biểu đồ phân tích
# ----------------------------

widths, heights, centers_x, centers_y = analyze_bbox(train_label_dir)

plt.figure(figsize=(15, 6))

# Phân bố kích thước
plt.subplot(1, 2, 1)
plt.hist([widths, heights], bins=50, label=["Chiều rộng", "Chiều cao"], alpha=0.7)
plt.xlabel("Kích thước đã chuẩn hoá")
plt.ylabel("Tần suất")
plt.title("Phân bố kích thước bounding box")
plt.legend()

# Phân bố tâm box
plt.subplot(1, 2, 2)
plt.scatter(centers_x, centers_y, s=5, alpha=0.3)
plt.xlabel("Toạ độ X đã chuẩn hoá")
plt.ylabel("Toạ độ Y đã chuẩn hoá")
plt.title("Phân bố tâm bounding box")

plt.savefig(os.path.join(output_dir, "bbox_analysis.png"), bbox_inches="tight")
plt.close()

# Biểu đồ tròn: phân bố lớp
plt.figure(figsize=(8, 8))
plt.pie(
    [train_counts[i] for i in range(len(class_names))],
    labels=class_names,
    autopct='%1.1f%%',
    startangle=90
)
plt.title("Phân bố lớp của tập train")
plt.savefig(os.path.join(output_dir, "class_distribution_pie.png"), bbox_inches="tight")
plt.close()

# Heatmap: phân bố tâm bounding box
plt.figure(figsize=(10, 8))
sns.kdeplot(x=centers_x, y=centers_y, cmap="Reds", shade=True)
plt.xlabel("Toạ độ X đã chuẩn hoá")
plt.ylabel("Toạ độ Y đã chuẩn hoá")
plt.title("Heatmap tâm bounding box")
plt.savefig(os.path.join(output_dir, "bbox_heatmap.png"), bbox_inches="tight")
plt.close()

# Boxplot: phân bố kích thước bounding box
plt.figure(figsize=(10, 6))
plt.boxplot([widths, heights], labels=["Chiều rộng", "Chiều cao"])
plt.xlabel("Kích thước bounding box")
plt.ylabel("Giá trị đã chuẩn hoá")
plt.title("Boxplot kích thước bounding box")
plt.savefig(os.path.join(output_dir, "bbox_boxplot.png"), bbox_inches="tight")
plt.close()

# Biểu đồ cột: phân tích mất cân bằng lớp
plt.figure(figsize=(10, 6))
plt.bar(class_names, [train_counts[i] for i in range(len(class_names))])
plt.xlabel("Lớp")
plt.ylabel("Số mẫu")
plt.title("Số mẫu theo lớp của tập train")
plt.savefig(os.path.join(output_dir, "class_imbalance.png"), bbox_inches="tight")
plt.close()

# ----------------------------
# Tạo báo cáo chất lượng dữ liệu
# ----------------------------
if stats['total_images'] > 0:
    missing_percent = stats['missing_labels'] / stats['total_images']
    corrupted_percent = stats['corrupted_images'] / stats['total_images']
else:
    missing_percent = 0.0
    corrupted_percent = 0.0

report = f"""
{'='*20} BÁO CÁO CHẤT LƯỢNG DỮ LIỆU {'='*20}
Tổng số ảnh: {stats['total_images']}
Ảnh có nhãn hợp lệ: {stats['total_images'] - stats['missing_labels'] - stats['corrupted_images']}
File nhãn bị thiếu: {stats['missing_labels']} ({missing_percent:.1%})
Ảnh hỏng/không đọc được: {stats['corrupted_images']} ({corrupted_percent:.1%})
Số dòng nhãn không hợp lệ: {stats['invalid_annotations']}
{'='*40}
"""

print(report)

# Lưu báo cáo ra file
report_file_path = os.path.join(output_dir, "data_quality_report.txt")
with open(report_file_path, "w", encoding="utf-8") as report_file:
    report_file.write(report)