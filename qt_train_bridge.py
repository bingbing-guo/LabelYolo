import argparse
import json
import os
import sys
from pathlib import Path


def emit(event, **payload):
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", default="train_exp")
    parser.add_argument("--ultralytics-root", default="")
    args = parser.parse_args()

    if args.ultralytics_root:
        sys.path.insert(0, args.ultralytics_root)
    try:
        from ultralytics import YOLO
    except Exception as exc:
        emit("error", message=f"无法导入 ultralytics: {exc}")
        return 2

    emit("started", data=args.data, model=args.model, epochs=args.epochs)
    try:
        model = YOLO(args.model)

        def on_epoch_end(trainer):
            metrics = {}
            for key, value in dict(getattr(trainer, "metrics", {}) or {}).items():
                try:
                    metrics[str(key)] = float(value)
                except (TypeError, ValueError):
                    pass
            training_loss = getattr(trainer, "tloss", None)
            if training_loss is not None:
                try:
                    labelled_losses = trainer.label_loss_items(training_loss, prefix="train")
                    for key, value in dict(labelled_losses or {}).items():
                        metrics[str(key)] = float(value)
                except Exception:
                    pass
            emit("epoch", epoch=int(getattr(trainer, "epoch", 0)) + 1, metrics=metrics)

        model.add_callback("on_fit_epoch_end", on_epoch_end)
        result = model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device or None,
            lr0=args.lr0,
            optimizer=args.optimizer,
            project=args.project,
            name=args.name,
            workers=0,
            exist_ok=False,
        )
        save_dir = Path(getattr(getattr(model, "trainer", None), "save_dir", Path(args.project) / args.name)).resolve()
        best = save_dir / "weights" / "best.pt"
        last = save_dir / "weights" / "last.pt"
        emit("completed", save_dir=str(save_dir), best=str(best) if best.exists() else "",
             last=str(last) if last.exists() else "")
        return 0
    except Exception as exc:
        emit("error", message=str(exc), exception=type(exc).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
