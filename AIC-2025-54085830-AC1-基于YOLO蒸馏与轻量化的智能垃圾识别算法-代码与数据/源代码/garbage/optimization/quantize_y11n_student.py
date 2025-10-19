
from pathlib import Path
import os
import onnx
from ultralytics import YOLO
from onnxruntime.quantization import quantize_dynamic, QuantType

ROOT = Path("/root/autodl-tmp/SWS3009Assg")
PT_PATH = ROOT / "models/student_best.pt"
ONNX_FP32 = ROOT / "models/y11n_student_fp32.onnx"
ONNX_INT8 = ROOT / "models/y11n_student_int8.onnx"

os.makedirs(ROOT / "models", exist_ok=True)

# ====== 导出为 ONNX ======
print("[INFO] Exporting YOLO11n student model to ONNX (FP32)...")
yolo = YOLO(str(PT_PATH))
yolo.export(format="onnx", imgsz=704, simplify=True, dynamic=False, opset=13)

# 自动查找导出的 ONNX 文件（通常保存在 runs/export/ 或 models/）
export_candidates = list(ROOT.rglob("*.onnx"))
if not export_candidates:
    raise FileNotFoundError("❌ 没找到导出的 ONNX 文件，请检查 Ultralytics 导出路径。")
latest_onnx = max(export_candidates, key=os.path.getmtime)

# 复制/重命名到目标路径
os.rename(latest_onnx, ONNX_FP32)
print(f"[OK] Exported ONNX FP32 model → {ONNX_FP32}")

# ====== 量化 (INT8) ======
print("[INFO] Quantizing model to INT8 ...")
quantize_dynamic(
    model_input=str(ONNX_FP32),
    model_output=str(ONNX_INT8),
    weight_type=QuantType.QInt8
)
print(f"[OK] Quantized INT8 model saved → {ONNX_INT8}")

# ====== 验证 ONNX 完整性 ======
onnx_model = onnx.load(ONNX_INT8)
onnx.checker.check_model(onnx_model)
print("[TEST] ONNX INT8 model check passed ✅")
