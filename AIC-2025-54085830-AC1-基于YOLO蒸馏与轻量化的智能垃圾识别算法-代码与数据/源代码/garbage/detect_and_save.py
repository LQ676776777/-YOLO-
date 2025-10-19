

from pathlib import Path
import argparse
import sys
import time
import shutil
import cv2
import numpy as np

from ultralytics import YOLO
from ultralytics.utils import LOGGER

#工程根目录（按需修改）
ROOT = Path("/root/autodl-tmp/SWS3009Assg")


def is_video_source(src: str) -> bool:
    """判断是否为视频/摄像头源：文件扩展名常见视频或纯数字（摄像头）"""
    p = str(src).strip()
    if p.isdigit():
        return True
    ext = Path(p).suffix.lower()
    return ext in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def load_model(model_path: Path):
    """加载 YOLO 模型（.pt 或 .onnx）。若 .onnx 遇到 ORT 不支持的量化算子，抛出友好提示。"""
    try:
        model = YOLO(str(model_path))
        return model
    except Exception as e:
        msg = str(e)
        # 针对 INT8 ONNX 常见报错，给出指引
        if "ConvInteger" in msg or "QLinearConv" in msg or "NOT_IMPLEMENTED" in msg.upper():
            LOGGER.error(
                f"\n[ERROR] ONNX Runtime 不支持该量化模型的某些算子：\n{msg}\n"
                "👉 解决方式：\n"
                "  A) 使用 FP32 ONNX 再试；或\n"
                "  B) 做“安全量化”（仅量化 MatMul），避免 ConvInteger/QLinearConv；或\n"
                "  C) 在支持 INT8 的 ORT build/平台上运行（oneDNN/DNNL INT8）。\n"
            )
        raise


def ensure_outdir(project: Path, name: str) -> Path:
    out_dir = project / name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_mp4_init(out_path: Path, w: int, h: int, fps: float):
    # mp4v 编码器在大多数环境可用；如需 H.264 可尝试 'avc1' 或在 ffmpeg 转码
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, max(1.0, fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"无法打开 MP4 写入器：{out_path}")
    return writer


def annotate_and_write_video(model: YOLO, source: str, out_dir: Path, imgsz=704, conf=0.35, device="cpu",
                             classes=None, save_sample=True, name="demo"):
    cap = cv2.VideoCapture(0 if source.strip().isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频/摄像头：{source}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out_mp4 = out_dir / "output.mp4"
    writer = write_mp4_init(out_mp4, w, h, fps)

    LOGGER.info(f"[INFO] Writing MP4 to: {out_mp4}  ({w}x{h} @ {fps:.1f}fps)")
    t0 = time.time()
    n_frames, avg_fps, sample_saved = 0, 0.0, False

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Ultralytics 支持 numpy 数组直接推理
        results = model.predict(
            source=frame,
            imgsz=imgsz,
            conf=conf,
            device=device,
            classes=classes,
            verbose=False
        )

        # 绘制可视化（每帧只有一个 result）
        plotted = results[0].plot()  # np.ndarray, BGR
        writer.write(plotted)
        n_frames += 1

        # 计算/显示速度
        avg_fps = 0.9 * avg_fps + 0.1 * (results[0].speed['inference'] and 1000.0 / results[0].speed['inference'] or 0.0)

        # 另存一个示例帧（供报告插图）
        if save_sample and not sample_saved:
            sample_path = ROOT / "detection_sample.png"
            cv2.imwrite(str(sample_path), plotted)
            sample_saved = True

    cap.release()
    writer.release()
    LOGGER.info(f"[OK] Done. Frames: {n_frames}, time: {time.time() - t0:.1f}s, saved: {out_mp4}")
    return out_mp4


def annotate_images(model: YOLO, src: Path, out_dir: Path, imgsz=704, conf=0.35, device="cpu",
                    classes=None, stitch=False, fps=10):
    """
    处理单图或文件夹：保存标注图片；可选将多张图合成为 MP4（stitch=True）
    """
    src = Path(src)
    img_paths = []
    if src.is_dir():
        for p in sorted(src.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                img_paths.append(p)
    else:
        if src.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            img_paths = [src]

    if not img_paths:
        raise RuntimeError(f"在 {src} 未找到图片文件")

    out_imgs = []
    for p in img_paths:
        results = model.predict(
            source=str(p),
            imgsz=imgsz,
            conf=conf,
            device=device,
            classes=classes,
            save=False,
            verbose=False
        )
        plotted = results[0].plot()
        save_to = out_dir / p.name
        cv2.imwrite(str(save_to), plotted)
        out_imgs.append(save_to)

    # 可选：把图片合成为 MP4
    if stitch and out_imgs:
        sample = cv2.imread(str(out_imgs[0]))
        h, w = sample.shape[:2]
        out_mp4 = out_dir / "stitched.mp4"
        writer = write_mp4_init(out_mp4, w, h, fps)
        for p in out_imgs:
            im = cv2.imread(str(p))
            if im is None:
                continue
            if im.shape[1] != w or im.shape[0] != h:
                im = cv2.resize(im, (w, h))
            writer.write(im)
        writer.release()
        LOGGER.info(f"[OK] Stitched MP4 saved to: {out_mp4}")

    # 另存一张示例
    sample_png = ROOT / "detection_sample.png"
    if out_imgs:
        shutil.copyfile(out_imgs[0], sample_png)
    LOGGER.info(f"[OK] Images saved to: {out_dir}")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description="Detect objects and save MP4/images (YOLO .pt/.onnx)")
    ap.add_argument("--model", type=str, required=True, help="模型路径：.pt 或 .onnx")
    ap.add_argument("--source", type=str, required=True,
                    help="输入源：视频文件/摄像头编号/单图/图片文件夹")
    ap.add_argument("--imgsz", type=int, default=704, help="推理分辨率")
    ap.add_argument("--conf", type=float, default=0.35, help="置信度阈值")
    ap.add_argument("--device", type=str, default="cpu", help="设备：'cpu' 或 '0','1' 等GPU编号")
    ap.add_argument("--name", type=str, default="demo_inference", help="输出子目录名")
    ap.add_argument("--classes", type=str, default="", help="仅保留的类别id，逗号分隔，如 '0,1,2'")
    ap.add_argument("--stitch", action="store_true", help="当输入为图片或文件夹时，将多张图合成为 MP4")
    args = ap.parse_args()

    model_path = Path(args.model)
    source = args.source
    device = args.device
    imgsz = args.imgsz
    conf = args.conf
    classes = [int(i) for i in args.classes.split(",")] if args.classes else None

    out_dir = ensure_outdir(ROOT / "runs" / "detect", args.name)

    # .onnx + device自动修正：onnx 在多数环境仅支持 cpu
    if model_path.suffix.lower() == ".onnx" and device != "cpu":
        LOGGER.warning("检测到 ONNX 模型，已强制切换为 device='cpu' 以保证兼容性。")
        device = "cpu"

    # 加载模型
    model = load_model(model_path)

    # 分支：视频/摄像头 vs 图片/目录
    if is_video_source(source):
        annotate_and_write_video(
            model=model,
            source=source,
            out_dir=out_dir,
            imgsz=imgsz,
            conf=conf,
            device=device,
            classes=classes,
            save_sample=True,
            name=args.name,
        )
    else:
        annotate_images(
            model=model,
            src=Path(source),
            out_dir=out_dir,
            imgsz=imgsz,
            conf=conf,
            device=device,
            classes=classes,
            stitch=args.stitch,
            fps=10
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        LOGGER.error(f"[FATAL] {e}")
        sys.exit(1)
