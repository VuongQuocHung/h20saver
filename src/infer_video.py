import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_DIR / "models" / "h20saver" / "best.pt"
DEFAULT_OUTPUT = PROJECT_DIR / "outputs" / "result.mp4"
WINDOW_NAME = "H20Saver - Drowning Detection"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the H20Saver YOLO model on a video and display the result."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input video path.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"YOLO checkpoint path (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Annotated output video (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum detection confidence (default: 0.25).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (default: 640).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Inference device (default: auto).",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=2.0,
        help="Seconds of history kept per tracked person (default: 2.0).",
    )
    parser.add_argument(
        "--alarm-ratio",
        type=float,
        default=0.7,
        help="Raise the alarm when this fraction of the window is 'drowning' (default: 0.7).",
    )
    parser.add_argument(
        "--clear-ratio",
        type=float,
        default=0.3,
        help="Clear the alarm when the fraction drops below this (default: 0.3).",
    )
    parser.add_argument(
        "--drowning-conf",
        type=float,
        default=0.5,
        help="A 'drowning' box only counts toward the alarm at this confidence (default: 0.5).",
    )
    parser.add_argument(
        "--labels",
        choices=("alarm", "all"),
        default="alarm",
        help="Write text only on alarmed people, or on every box (default: alarm).",
    )
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="Ultralytics tracker config (default: bytetrack.yaml).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the live preview window.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save the annotated output video.",
    )
    return parser.parse_args()


def select_device(requested):
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but PyTorch cannot access the GPU.")
        return 0
    return 0 if torch.cuda.is_available() else "cpu"


def open_writer(output_path, fps, width, height):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV cannot create output video: {output_path}")
    return writer


def find_class_id(names, class_name):
    for class_id, name in names.items():
        if name == class_name:
            return class_id
    raise ValueError(f"Model has no '{class_name}' class: {names}")


class DrowningMonitor:
    """Chỉ bật cảnh báo cho từng người được theo dõi khi họ bị nhận là 'drowning' liên tục.

    Mỗi track giữ `window_frames` phiếu gần nhất. Cảnh báo bật khi ít nhất
    `alarm_ratio` của cả cửa sổ là phiếu 'drowning', và chỉ tắt khi tỉ lệ này
    xuống dưới `clear_ratio`, nên một khung hình bị nhận nhầm không thể tự
    bật hay tự tắt cảnh báo.
    """

    def __init__(self, window_frames, alarm_ratio, clear_ratio):
        self.window_frames = window_frames
        self.alarm_ratio = alarm_ratio
        self.clear_ratio = clear_ratio
        self.history = {}
        self.last_seen = {}
        self.alarmed = set()

    def update(self, frame_index, track_ids, drowning_votes):
        for track_id, vote in zip(track_ids, drowning_votes):
            history = self.history.setdefault(track_id, deque(maxlen=self.window_frames))
            history.append(vote)
            self.last_seen[track_id] = frame_index

            ratio = sum(history) / self.window_frames
            if ratio >= self.alarm_ratio:
                self.alarmed.add(track_id)
            elif ratio < self.clear_ratio:
                self.alarmed.discard(track_id)

        # Người biến mất khỏi khung hình (ví dụ chìm xuống) vẫn giữ cảnh báo thêm một cửa sổ.
        for track_id, seen in list(self.last_seen.items()):
            if frame_index - seen > self.window_frames:
                del self.history[track_id], self.last_seen[track_id]
                self.alarmed.discard(track_id)
        return self.alarmed


def read_tracks(result):
    boxes = result.boxes
    if boxes is None or boxes.id is None:
        return [], [], [], []
    return (
        boxes.id.int().tolist(),
        boxes.cls.int().tolist(),
        boxes.conf.tolist(),
        boxes.xyxy.int().tolist(),
    )


ALARM_COLOR = (0, 0, 255)
CLASS_COLORS = {
    "drowning": (0, 165, 255),
    "swimming": (80, 200, 80),
    "out of water": (170, 170, 170),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def banner_height(frame):
    return round(40 * frame.shape[0] / 720)


def draw_label(frame, text, x, y, color, scale, thickness):
    (text_w, text_h), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    label_h = text_h + baseline + 2
    # Không còn chỗ phía trên box (sát mép trên hoặc nằm dưới banner cảnh báo): vẽ nhãn vào trong box.
    if y - label_h < banner_height(frame):
        y = max(y, banner_height(frame)) + label_h
    cv2.rectangle(frame, (x, y - text_h - baseline - 2), (x + text_w + 4, y), color, -1)
    cv2.putText(
        frame, text, (x + 2, y - baseline), FONT, scale, (255, 255, 255), thickness, cv2.LINE_AA
    )


def draw_tracks(frame, names, alarmed, tracks, label_mode):
    unit = frame.shape[0] / 720
    line = max(1, round(unit))
    scale = max(0.4, 0.45 * unit)
    text_thickness = max(1, round(unit))

    # Vẽ người bình thường trước để box cảnh báo không bao giờ bị che.
    ordered = sorted(zip(*tracks), key=lambda track: track[0] in alarmed)
    for track_id, class_id, score, (x1, y1, x2, y2) in ordered:
        name = names[class_id]
        if track_id in alarmed:
            cv2.rectangle(frame, (x1, y1), (x2, y2), ALARM_COLOR, 3 * line)
            draw_label(frame, f"#{track_id} DROWNING", x1, y1, ALARM_COLOR, scale, text_thickness)
            continue

        color = CLASS_COLORS.get(name, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, line)
        if label_mode == "all":
            draw_label(frame, f"{name} {score:.2f}", x1, y1, color, scale * 0.8, text_thickness)

    draw_legend(frame, scale, text_thickness)


def draw_legend(frame, scale, thickness):
    entries = list(CLASS_COLORS.items()) + [("ALARM", ALARM_COLOR)]
    (_, text_h), _ = cv2.getTextSize("Ag", FONT, scale, thickness)
    x, y = 10, frame.shape[0] - 10
    for name, color in entries:
        cv2.rectangle(frame, (x, y - text_h), (x + text_h, y), color, -1)
        for text_color, outline in (((0, 0, 0), 2), ((255, 255, 255), 0)):
            cv2.putText(
                frame, name, (x + text_h + 6, y), FONT, scale, text_color,
                thickness + outline, cv2.LINE_AA,
            )
        (text_w, _), _ = cv2.getTextSize(name, FONT, scale, thickness)
        x += text_h + text_w + 20


def draw_drowning_warning(frame, alarmed):
    if not alarmed:
        return

    unit = frame.shape[0] / 720
    bar_height = banner_height(frame)
    if len(alarmed) <= 3:
        who = ", ".join(f"#{track_id}" for track_id in sorted(alarmed))
    else:
        who = f"{len(alarmed)} people"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], bar_height), (0, 0, 200), -1)
    cv2.putText(
        frame,
        f"WARNING: DROWNING DETECTED ({who})",
        (12, round(28 * unit)),
        FONT,
        0.8 * unit,
        (255, 255, 255),
        max(1, round(2 * unit)),
        cv2.LINE_AA,
    )


def main():
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError("--confidence must be between 0 and 1.")
    if args.window <= 0:
        raise ValueError("--window must be positive.")
    if not args.confidence <= args.drowning_conf <= 1.0:
        raise ValueError("--drowning-conf must be between --confidence and 1.")
    if not 0.0 <= args.clear_ratio <= args.alarm_ratio <= 1.0:
        raise ValueError("Require 0 <= --clear-ratio <= --alarm-ratio <= 1.")

    device = select_device(args.device)
    model = YOLO(model_path)
    drowning_class = find_class_id(model.names, "drowning")
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV cannot open input video: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None if args.no_save else open_writer(output_path, fps, width, height)
    window_frames = max(1, round(args.window * fps))
    monitor = DrowningMonitor(window_frames, args.alarm_ratio, args.clear_ratio)

    device_name = torch.cuda.get_device_name(0) if device == 0 else "CPU"
    print(f"Model: {model_path}")
    print(f"Device: {device_name}")
    print(f"Video: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")
    print(
        f"Alarm: >= {args.alarm_ratio:.0%} of {window_frames} frames "
        f"({args.window:.1f}s) per tracked person, "
        f"drowning votes need conf >= {args.drowning_conf:.2f}"
    )
    if not args.no_show:
        print("Press Q or Esc in the video window to stop.")
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    frame_index = 0
    started_at = time.perf_counter()

    try:
        while True:
            frame_started_at = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                break

            result = model.track(
                frame,
                persist=True,
                tracker=args.tracker,
                conf=args.confidence,
                imgsz=args.imgsz,
                device=device,
                verbose=False,
            )[0]
            tracks = read_tracks(result)
            track_ids, class_ids, scores, _ = tracks
            votes = [
                class_id == drowning_class and score >= args.drowning_conf
                for class_id, score in zip(class_ids, scores)
            ]
            alarmed = monitor.update(frame_index, track_ids, votes)
            annotated_frame = frame.copy()
            draw_tracks(annotated_frame, model.names, alarmed, tracks, args.labels)
            draw_drowning_warning(annotated_frame, alarmed)

            if writer is not None:
                writer.write(annotated_frame)

            frame_index += 1
            if not args.no_show:
                cv2.imshow(WINDOW_NAME, annotated_frame)
                processing_ms = (time.perf_counter() - frame_started_at) * 1000
                wait_ms = max(1, round(1000 / fps - processing_ms))
                key = cv2.waitKey(wait_ms) & 0xFF
                if key in (ord("q"), 27):
                    break

                try:
                    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started_at
    print(f"Finished: {frame_index} frames in {elapsed:.1f} seconds")
    if writer is not None:
        print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
