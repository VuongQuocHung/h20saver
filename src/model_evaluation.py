from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm
from ultralytics import YOLO

# ----------------------------
# Tham số cấu hình
# ----------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "h20saver" / "best.pt"
DATA_YAML_PATH = PROJECT_DIR / "dataset" / "data.yaml"

CONF_THRESH = 0.25  # Ngưỡng độ tin cậy
IOU_THRESH = 0.5    # Ngưỡng IoU
NUM_IMAGES = None   # Số ảnh được chọn (None = toàn bộ tập test)
SEED = 0            # Seed lấy mẫu ngẫu nhiên, để kết quả tái lập được

# ----------------------------
# Khởi tạo model và dataset
# ----------------------------
model = YOLO(MODEL_PATH)

with open(DATA_YAML_PATH, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)


def resolve_split_dir(split):
    # Giống quy tắc phân giải của Ultralytics: tương đối theo data["path"] (mặc định là thư mục chứa yaml),
    # nếu không tồn tại và bắt đầu bằng "../" (định dạng xuất của Roboflow) thì bỏ "../" rồi thử lại
    base_dir = Path(data.get("path") or DATA_YAML_PATH.parent)
    if not base_dir.is_absolute():
        base_dir = DATA_YAML_PATH.parent / base_dir
    split_dir = (base_dir / data[split]).resolve()
    if not split_dir.exists() and data[split].startswith("../"):
        split_dir = (base_dir / data[split][3:]).resolve()
    return split_dir


test_dir = resolve_split_dir("test")
test_label_dir = test_dir.parent / "labels" if test_dir.name == "images" else test_dir / "labels"
names = data["names"]
class_names = names if isinstance(names, dict) else dict(enumerate(names))

# ----------------------------
# Lấy danh sách ảnh test
# ----------------------------
test_images = sorted(
    p.name for p in test_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")
)
if NUM_IMAGES is None or NUM_IMAGES >= len(test_images):
    selected_images = test_images
else:
    rng = np.random.default_rng(SEED)
    selected_images = list(rng.choice(test_images, NUM_IMAGES, replace=False))


# ----------------------------
# Nơi chứa các chỉ số đánh giá
# ----------------------------
metrics = {
    "total": defaultdict(int),
    "classes": defaultdict(lambda: {
        "TP": 0, "FP": 0, "FN": 0,
        "precision": 0, "recall": 0, "f1": 0
    })
}

# ----------------------------
# Các hàm xử lý
# ----------------------------
def parse_label(label_path, img_width, img_height):
    boxes = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            class_id, xc, yc, w, h = map(float, parts[:5])
            x1 = (xc - w/2) * img_width
            y1 = (yc - h/2) * img_height
            x2 = (xc + w/2) * img_width
            y2 = (yc + h/2) * img_height
            boxes.append([int(class_id), x1, y1, x2, y2])
    return boxes

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2]-box1[0])*(box1[3]-box1[1])
    box2_area = (box2[2]-box2[0])*(box2[3]-box2[1])

    return inter_area / (box1_area + box2_area - inter_area + 1e-6)

# ----------------------------
# Luồng xử lý chính
# ----------------------------
for img_name in tqdm(selected_images, desc="Processing Images"):
    # Đọc ảnh
    img = cv2.imread(str(test_dir / img_name))
    if img is None:
        continue
    h, w = img.shape[:2]

    # Lấy nhãn tương ứng
    label_path = test_label_dir / (Path(img_name).stem + ".txt")
    gt_boxes = parse_label(label_path, w, h) if label_path.exists() else []

    # Model dự đoán
    results = model.predict(img, conf=CONF_THRESH, verbose=False)
    detections = []
    for result in results:
        for box in result.boxes:
            xyxy = box.xyxy.cpu().numpy()[0]
            class_id = int(box.cls)
            conf = float(box.conf)
            detections.append([class_id, conf, *xyxy])

    # Ghép kết quả phát hiện (theo độ tin cậy từ cao xuống thấp, mỗi box thật chỉ ghép tối đa một lần)
    matched_gt = set()
    for det in sorted(detections, key=lambda x: -x[1]):
        max_iou = 0
        match_idx = -1
        for i, gt in enumerate(gt_boxes):
            if i in matched_gt or det[0] != gt[0]:
                continue
            iou = calculate_iou(det[2:], gt[1:])
            if iou > max_iou:
                max_iou = iou
                match_idx = i

        if max_iou >= IOU_THRESH:
            metrics["classes"][det[0]]["TP"] += 1
            metrics["total"]["TP"] += 1
            matched_gt.add(match_idx)
        else:
            metrics["classes"][det[0]]["FP"] += 1
            metrics["total"]["FP"] += 1

    # Đếm bỏ sót: mọi box thật chưa được ghép đều tính là FN
    for i, gt in enumerate(gt_boxes):
        if i not in matched_gt:
            metrics["classes"][gt[0]]["FN"] += 1
            metrics["total"]["FN"] += 1

# ----------------------------
# Tính các chỉ số
# ----------------------------
def calculate_metrics(stats):
    precision = stats["TP"] / (stats["TP"] + stats["FP"] + 1e-6)
    recall = stats["TP"] / (stats["TP"] + stats["FN"] + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    return precision, recall, f1

# Chỉ số tổng thể
total_precision, total_recall, total_f1 = calculate_metrics(metrics["total"])
metrics["total"].update({
    "precision": total_precision,
    "recall": total_recall,
    "f1": total_f1
})

# Chỉ số theo từng lớp
for class_id in metrics["classes"]:
    stats = metrics["classes"][class_id]
    precision, recall, f1 = calculate_metrics(stats)
    stats.update({
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

# ----------------------------
# Tạo báo cáo
# ----------------------------
print("\n\nBáo cáo đánh giá: ")
print(f"Model: {MODEL_PATH}")
print(f"Số ảnh test: {len(selected_images)}")
print(f"Tổng số phát hiện: {metrics['total']['TP'] + metrics['total']['FP']}")
print(f"Tổng số nhãn thật: {metrics['total']['TP'] + metrics['total']['FN']}")
print(f"Precision tổng thể: {metrics['total']['precision']:.2%}")
print(f"Recall tổng thể: {metrics['total']['recall']:.2%}")
print(f"F1 tổng thể: {metrics['total']['f1']:.2%}")

print("\nThống kê theo lớp:")
print(f"{'Lớp':<15} {'Precision':<10} {'Recall':<10} {'F1':<10} {'TP':<5} {'FP':<5} {'FN':<5}")
for class_id in sorted(metrics["classes"]):
    cls = metrics["classes"][class_id]
    name = class_names[class_id]
    print(f"{name:<15} {cls['precision']:<10.2%} {cls['recall']:<10.2%} {cls['f1']:<10.2%} "
          f"{cls['TP']:<5} {cls['FP']:<5} {cls['FN']:<5}")

# ----------------------------
# Chỉ số chính thức của Ultralytics (mAP trên tập test)
# ----------------------------
print("\n[Tập test] Đánh giá chính thức của Ultralytics:")
test_metrics = model.val(data=str(DATA_YAML_PATH), split="test")
print(
    f"Kết quả test: mAP@0.5={test_metrics.box.map50:.3f}, "
    f"mAP@0.5:0.95={test_metrics.box.map:.3f}"
)
