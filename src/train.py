import yaml
import os
import numpy as np
from ultralytics import YOLO
from sklearn.utils.class_weight import compute_class_weight

# ----------------------------
# Tham số cấu hình
# ----------------------------
class CustomConfig:
    # Tham số dữ liệu
    data_yaml = "dataset/data.yaml"
    img_size = 640
    batch_size = 23
    epochs = 100
    patience = 10

    # Tham số tăng cường dữ liệu
    augment = {
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "translate": 0.1,
        "scale": 0.9,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.2,
    }

    # Tham số tối ưu
    lr0 = 0.001
    lrf = 0.01
    momentum = 0.937
    weight_decay = 0.0005

    # Trọng số hàm loss (tham số gốc của Ultralytics: box / cls / dfl)
    # Tăng trọng số cls từ mặc định 0.5 lên 1.0 để xử lý mất cân bằng lớp
    loss_box_gain: float = 7.5
    loss_cls_gain: float = 1.0
    loss_dfl_gain: float = 1.5

    # Tham số model
    pretrained = "yolo11x.pt"
    freeze_layers = None       # list[int] | None, ví dụ [0, 1, 2] là đóng băng 3 lớp đầu
    multi_scale: bool = True

    # Đầu ra của quá trình train
    project: str = "runs/detect"
    experiment_name: str = "h20saver"
    resume: bool = False


# ----------------------------
# Chuẩn bị dữ liệu: phân tích phân bố lớp
# ----------------------------
def prepare_dataset(config):
    with open(config.data_yaml, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    train_labels = []
    train_label_dir = os.path.join(
        os.path.dirname(config.data_yaml),
        data["train"].replace("images", "labels"),
    )
    for label_file in os.listdir(train_label_dir):
        label_path = os.path.join(train_label_dir, label_file)
        if not os.path.isfile(label_path):
            continue
        with open(label_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    class_id = int(line.split()[0])
                    train_labels.append(class_id)
                except (ValueError, IndexError):
                    continue

    classes = np.unique(train_labels)
    weights = compute_class_weight("balanced", classes=classes, y=train_labels)
    class_weights = {int(k): round(float(v), 3) for k, v in zip(classes, weights)}

    print(f"[Dữ liệu] Trọng số lớp (chỉ để tham khảo): {class_weights}")
    print("[Dữ liệu] Lưu ý: Ultralytics YOLO không hỗ trợ truyền trực tiếp trọng số cho từng lớp.")
    print("[Dữ liệu] Nếu mất cân bằng lớp nghiêm trọng, hãy bổ sung lớp thiểu số bằng tăng cường dữ liệu hoặc oversampling.")
    return class_weights


# ----------------------------
# Train model
# ----------------------------
def train_yolo(config):
    model = YOLO(config.pretrained)

    train_kwargs = {
        "data": config.data_yaml,
        "epochs": config.epochs,
        "imgsz": config.img_size,
        "batch": config.batch_size,
        "lr0": config.lr0,
        "lrf": config.lrf,
        "momentum": config.momentum,
        "weight_decay": config.weight_decay,
        "patience": config.patience,
        "optimizer": "AdamW",
        "cos_lr": True,
        "freeze": config.freeze_layers,
        "multi_scale": config.multi_scale,
        "project": config.project,
        "name": config.experiment_name,
        "resume": config.resume,
        # Tăng cường dữ liệu
        "hsv_h": config.augment["hsv_h"],
        "hsv_s": config.augment["hsv_s"],
        "hsv_v": config.augment["hsv_v"],
        "translate": config.augment["translate"],
        "scale": config.augment["scale"],
        "fliplr": config.augment["fliplr"],
        "mosaic": config.augment["mosaic"],
        "mixup": config.augment["mixup"],
        # Hệ số loss — YOLO11 có sẵn DFL, tăng hệ số cls để giảm mất cân bằng lớp
        "box": config.loss_box_gain,
        "cls": config.loss_cls_gain,
        "dfl": config.loss_dfl_gain,
    }

    print("=" * 40)
    print("  Bắt đầu train model...")
    print("=" * 40)

    results = model.train(**train_kwargs)

    final_weights = f"{config.experiment_name}_final.pt"
    model.save(final_weights)
    print(f"[Trọng số] Đã lưu tại: {final_weights}")

    return model


# ----------------------------
# Chương trình chính
# ----------------------------
if __name__ == "__main__":
    config = CustomConfig()
    class_weights = prepare_dataset(config)
    model = train_yolo(config)

    # Xuất ONNX (FP16 nửa độ chính xác, đơn giản hoá để suy luận)
    print("\n[Xuất] Đang xuất model ONNX...")
    model.export(format="onnx", imgsz=config.img_size, half=True, simplify=True)

    # Kiểm định
    print("\n[Kiểm định] Đang đánh giá model...")
    metrics = model.val()
    print(
        f"Kết quả kiểm định: mAP@0.5={metrics.box.map50:.3f}, "
        f"mAP@0.5:0.95={metrics.box.map:.3f}"
    )