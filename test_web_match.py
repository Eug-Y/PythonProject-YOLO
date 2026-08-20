"""和web viewer完全一致的测试脚本"""
import cv2, numpy as np
from web_viewer import yolo_predict

# 改成你的图片路径
IMG_PATH = r"datasets\Data_v2\train\images\DJI_0010.JPG"

img = cv2.imread(IMG_PATH)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img_rgb.shape[:2]

# 尝试从 _5ch.npy 加载 NDVI/NDRE
import re, os
from pathlib import Path
stem = os.path.splitext(os.path.basename(IMG_PATH))[0]
m = re.search(r"DJI_(\d{4})", stem)
img_num = int(m.group(1)) if m else None

ndvi = ndre = None
if img_num is not None:
    for search_dir in [
        Path(r"D:\PythonProject-YOLO\多光谱图片"),
        Path(r"D:\PythonProject-YOLO\datasets\Data_v2\train\images"),
    ]:
        npy_path = search_dir / f"DJI_{img_num:04d}_5ch.npy"
        if npy_path.exists():
            npy = np.load(str(npy_path))
            ndvi = npy[:,:,3].astype(np.float32)
            ndre = npy[:,:,4].astype(np.float32)
            if ndvi.shape[:2] != (h, w):
                ndvi = cv2.resize(ndvi, (w, h))
                ndre = cv2.resize(ndre, (w, h))
            print(f"已加载 NDVI/NDRE 从: {npy_path}")
            break

if ndvi is None:
    ndvi = np.zeros((h, w), dtype=np.float32)
    ndre = np.zeros((h, w), dtype=np.float32)
    print("警告: 未找到 _5ch.npy，使用全零 NDVI/NDRE")

# 和web viewer完全相同的推理
overlay, count, model = yolo_predict("exp_ca_5ch_v11", img_rgb, ndvi, ndre, conf=0.25)
out_path = f"test_match_{stem}.png"
cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
print(f"检测数: {count}, 模型: {model}")
print(f"已保存: {out_path}")
print(f"\n对比方法: 上传 {os.path.basename(IMG_PATH)} 到网页后，")
print(f"查看 debug_web_{stem}_yolo.png 和 test_match_{stem}.png 是否一致")
