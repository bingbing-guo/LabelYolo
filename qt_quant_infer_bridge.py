import argparse
import json
import os
import sys


def emit(event, **payload):
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


def load_yolo(root):
    if root:
        sys.path.insert(0, root)
    from ultralytics import YOLO
    return YOLO


def export_model(args, YOLO):
    if args.half and args.int8:
        raise ValueError("FP16 与 INT8 不能同时启用")
    if args.int8 and not args.data:
        raise ValueError("INT8 导出需要 data.yaml 校准数据")
    model = YOLO(args.weights)
    kwargs = {"format": args.format, "half": args.half, "int8": args.int8, "device": args.device}
    if args.int8:
        kwargs["data"] = args.data
    emit("started", operation="export", format=args.format)
    exported = model.export(**kwargs)
    emit("export_completed", path=os.path.abspath(str(exported)), format=args.format,
         precision="int8" if args.int8 else ("fp16" if args.half else "fp32"))


def infer(args, YOLO):
    model = YOLO(args.weights)
    emit("started", operation="infer", source=args.source)
    results = model.predict(source=args.source, save=True, task="detect", project=args.project,
                            name="infer_batch", exist_ok=False, conf=args.conf, iou=args.iou)
    for result in results:
        image_name = os.path.basename(result.path)
        for box in result.boxes:
            cls_id = int(box.cls[0])
            xyxy = [float(value) for value in box.xyxy[0].tolist()]
            emit("detection", image=image_name, class_id=cls_id,
                 class_name=str(model.names[cls_id]), confidence=float(box.conf[0]), xyxy=xyxy)
    save_dir = os.path.abspath(str(results[0].save_dir)) if results else ""
    emit("infer_completed", result_dir=save_dir, images=len(results))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["export", "infer"], required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--format", default="onnx")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--data", default="")
    parser.add_argument("--source", default="")
    parser.add_argument("--project", default="gui_runs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--ultralytics-root", default="")
    args = parser.parse_args()
    try:
        YOLO = load_yolo(args.ultralytics_root)
        export_model(args, YOLO) if args.mode == "export" else infer(args, YOLO)
        return 0
    except Exception as exc:
        emit("error", message=str(exc), exception=type(exc).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
