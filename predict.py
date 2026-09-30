"""Run the supplied checkpoint on an image, video, camera or stream."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Image/video path, URL, or camera index')
    parser.add_argument('--model', default=str(Path(__file__).resolve().parents[1] / 'models/best.pt'))
    parser.add_argument('--imgsz', type=int, default=768)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', default='outputs/predict')
    args = parser.parse_args()
    if not Path(args.model).is_file():
        parser.error(f'Model not found: {args.model}')
    from ultralytics import YOLO
    source = int(args.source) if args.source.isdecimal() else args.source
    output = Path(args.output).resolve()
    model = YOLO(args.model)
    for _ in model.predict(source=source, imgsz=args.imgsz, conf=args.conf,
                           device=args.device, save=True, stream=True,
                           project=str(output.parent), name=output.name, exist_ok=False):
        pass


if __name__ == '__main__':
    main()
