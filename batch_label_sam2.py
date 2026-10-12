"""
YOLO v10 + SAM2 批量标注脚本 (多 bbox box prompt)
质量对标已有 100 张 SAM2 标注: stride=150 grid point segmentation
1. v10 5ch YOLO 检出田垄 bbox
2. 所有 bbox 作为 box prompt 单次发给 SAM2
3. SAM2 单次 generate 生成精细 mask
4. 转换 mask -> YOLO 分割多边形 TXT

与之前 100 张标注质量一致。
"""
import os
import re
import sys
import time
import cv2
import numpy as np
from pathlib import Path

TIF_DIR = Path(r"D:\PythonProject-YOLO\多光谱图片")
OUT_DIR = Path(r"D:\PythonProject-YOLO\datasets\Data_v2\sam2_labels")
os.makedirs(OUT_DIR, exist_ok=True)

GRID_STRIDE = 150  # 与之前标注一致
BATCH_SIZE = 10
CONF_THRESHOLD = 0.25


def load_v10_model():
    """加载 5ch v10 YOLO 模型"""
    sys.path.insert(0, str(Path(__file__).parent))
    import torch
    from custom_modules import CoordAtt, ECA
    from ultralytics.nn.tasks import SegmentationModel, load_checkpoint

    pt = Path(r"D:\PythonProject\runs\segment\land_seg\exp_ca_5ch_v11\weights\best.pt")
    ckpt_model, _ = load_checkpoint("yolo11n-seg.pt")
    base_model = SegmentationModel(ckpt_model.yaml, nc=1)
    base_model.load(ckpt_model)

    def _get_output_channels(layer):
        if hasattr(layer, "out_channels"):     return layer.out_channels
        if hasattr(layer, "cv2") and hasattr(layer.cv2, "conv"): return layer.cv2.conv.out_channels
        if hasattr(layer, "cv3") and hasattr(layer.cv3, "conv"): return layer.cv3.conv.out_channels
        if hasattr(layer, "conv"):             return layer.conv.out_channels
        return None

    old_layers = list(base_model.model)
    backbone_end = 7
    new_layers, old_to_new = [], {}
    for oi, layer in enumerate(old_layers):
        old_to_new[oi] = len(new_layers)
        new_layers.append(layer)
        if "C3k2" in layer.__class__.__name__ and oi < backbone_end:
            ch = _get_output_channels(layer)
            if ch:
                ca = CoordAtt(ch, ch); ca.f = -1; ca.i = -1
                new_layers.append(ca)
                eca = ECA(ch, k_size=3); eca.f = -1; eca.i = -1
                new_layers.append(eca)
    for i, layer in enumerate(new_layers):
        if hasattr(layer, "f"):
            f = layer.f
            if isinstance(f, (int, np.integer)):
                if f != -1: layer.f = old_to_new.get(int(f), int(f))
            else:
                new_f = []
                for idx in f:
                    if idx == -1: new_f.append(-1)
                    else: new_f.append(old_to_new.get(int(idx), int(idx)))
                layer.f = new_f
        layer.i = i
    base_model.model = torch.nn.Sequential(*new_layers)
    save = set()
    for layer in new_layers:
        f = getattr(layer, "f", -1)
        if f == -1: continue
        for idx in ([f] if isinstance(f, (int, np.integer)) else f):
            if idx != -1: save.add(int(idx))
    base_model.save = sorted(save)

    first = base_model.model[0]
    old_conv = first.conv
    old_weight = old_conv.weight.data
    new_conv = torch.nn.Conv2d(5, old_conv.out_channels, old_conv.kernel_size,
                               old_conv.stride, old_conv.padding, bias=False)
    new_conv.weight.data[:, :3].copy_(old_weight)
    torch.nn.init.kaiming_uniform_(new_conv.weight.data[:, 3:], a=1.0)
    first.conv = new_conv

    ckpt = torch.load(str(pt), map_location="cpu", weights_only=False)
    base_model.load_state_dict(ckpt["model"].float().state_dict())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model.to(device)
    base_model.eval()
    return base_model, device


def load_sam2_model():
    """加载 SAM2 small 模型"""
    from osam._models.sam2._models import Sam2Small
    return Sam2Small()


def get_yolo_bboxes(model, img_rgb, ndvi_norm, ndre_norm, device, conf=0.25, imgsz=1024):
    """v10 5ch YOLO 推理 -> bbox 列表 [x1,y1,x2,y2,score]"""
    import torch

    h0, w0 = img_rgb.shape[:2]
    ch5 = np.stack([
        img_rgb[:, :, 0].astype(np.float32) / 255.0,
        img_rgb[:, :, 1].astype(np.float32) / 255.0,
        img_rgb[:, :, 2].astype(np.float32) / 255.0,
        ndvi_norm.astype(np.float32),
        ndre_norm.astype(np.float32) if ndre_norm is not None else np.zeros((h0, w0)),
    ], axis=-1)
    r = imgsz / max(h0, w0)
    if r != 1.0:
        ch5 = cv2.resize(ch5, (int(w0 * r), int(h0 * r)))
    input_h, input_w = ch5.shape[:2]
    tensor = torch.from_numpy(ch5).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model._predict_once(tensor)
    preds = outputs[0][0]

    from ultralytics.utils.nms import non_max_suppression
    pred = non_max_suppression(preds, conf, 0.45, nc=1)

    scale_x, scale_y = w0 / input_w, h0 / input_h
    bboxes = []
    for det in pred:
        if det is None or len(det) == 0:
            continue
        for i in range(len(det)):
            x1, y1, x2, y2 = det[i, :4].cpu().numpy()
            x1, x2 = x1 * scale_x, x2 * scale_x
            y1, y2 = y1 * scale_y, y2 * scale_y
            score = det[i, 4].item()
            bboxes.append([x1, y1, x2, y2, score])
    bboxes.sort(key=lambda b: b[4], reverse=True)
    return bboxes


def sam2_per_box_masks(sam2, img_rgb, bboxes):
    """逐 bbox SAM2 box prompt 分割，每框 max_annotations=1
    返回: list of masks (每个 bbox 最多一个 mask)
    """
    if not bboxes:
        return []

    h, w = img_rgb.shape[:2]
    from osam.types._generate import GenerateRequest, Prompt

    img_emb = sam2.encode_image(img_rgb)
    all_masks = []

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox[:4]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            continue

        prompt = Prompt(
            points=np.array([[x1, y1], [x2, y2]], dtype=np.float64),
            point_labels=np.array([2, 3], dtype=np.int32),
            iou_threshold=0.5,
            score_threshold=0.0,
            max_annotations=1,
        )
        req = GenerateRequest(model="sam2:small", image_embedding=img_emb, prompt=prompt)
        resp = sam2.generate(req)

        if resp.annotations:
            mask = resp.annotations[0].mask
            # 只保留 bbox 区域内的 mask（去除外部噪声）
            mask_clean = np.zeros_like(mask)
            mask_clean[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
            all_masks.append(mask_clean)

    return all_masks


def mask_to_yolo_polygons(mask, img_w, img_h):
    """二值 mask -> YOLO 分割归一化多边形列表"""
    mask_u8 = mask.astype(np.uint8)
    # 形态学平滑
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    # 高斯模糊去锯齿
    mask_blur = cv2.GaussianBlur(mask_u8.astype(np.float32), (3, 3), 0.8)
    mask_u8 = (mask_blur > 0.5).astype(np.uint8)

    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 40:
            continue
        epsilon = 0.002 * cv2.arcLength(cnt, True)
        smooth = cv2.approxPolyDP(cnt, epsilon, True)
        if len(smooth) < 3:
            continue
        norm = smooth.reshape(-1, 2).astype(np.float32)
        norm[:, 0] /= img_w
        norm[:, 1] /= img_h
        polygons.append(norm.reshape(-1).tolist())
    return polygons


def compute_ndvi_from_jpg(img_num, h, w):
    """计算 NDVI 和 NDRE"""
    def load_tif(offset):
        p = TIF_DIR / f"DJI_{img_num + offset:04d}.TIF"
        if not p.exists():
            return None
        raw = np.fromfile(str(p), dtype=np.uint16)
        band = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
        return band.astype(np.float32).squeeze() if band is not None else None

    red = load_tif(3)
    nir = load_tif(5)
    if red is None or nir is None:
        return None, None
    if red.shape[:2] != (h, w):
        red = cv2.resize(red, (w, h))
        nir = cv2.resize(nir, (w, h))
    eps = 1e-6
    ndvi = (nir - red) / (nir + red + eps)
    ndvi_norm = (ndvi + 1.0) / 2.0
    reg = load_tif(4)
    if reg is not None:
        if reg.shape[:2] != (h, w):
            reg = cv2.resize(reg, (w, h))
        ndre = (nir - reg) / (nir + reg + eps)
        ndre_norm = (ndre + 1.0) / 2.0
    else:
        ndre_norm = None
    return ndvi_norm, ndre_norm


def get_unlabeled_images():
    """返回未标注图片路径列表"""
    all_jpgs = set()
    for f in TIF_DIR.glob("DJI_*.JPG"):
        all_jpgs.add(f.stem)

    labeled = set()
    for split in ["train", "valid", "test"]:
        label_dir = Path(r"D:\PythonProject-YOLO\datasets\Data_v2") / split / "labels"
        for txt in label_dir.glob("*.txt"):
            labeled.add(txt.stem)
    for txt in OUT_DIR.glob("*.txt"):
        labeled.add(txt.stem)

    unlabeled = sorted(all_jpgs - labeled)
    return [TIF_DIR / f"{u}.JPG" for u in unlabeled]


def main():
    print("=" * 60)
    print("  YOLO v10 + SAM2 grid-point 批量标注 (stride=150)")
    print("=" * 60)

    unlabeled = get_unlabeled_images()
    print(f"\n  未标注图片: {len(unlabeled)} 张")
    if not unlabeled:
        print("  所有图片已标注完毕！")
        return

    try:
        n = int(input(f"\n  要标注多少张？(1-{len(unlabeled)}): ").strip())
    except (ValueError, EOFError):
        n = min(100, len(unlabeled))
    n = max(1, min(n, len(unlabeled)))
    targets = unlabeled[:n]

    print("\n  加载 YOLO v10 5ch 模型...")
    t0 = time.time()
    yolo, device = load_v10_model()
    print(f"  YOLO 加载完成 ({time.time()-t0:.1f}s, device={device})")

    print("  加载 SAM2 small 模型...")
    t0 = time.time()
    sam2 = load_sam2_model()
    print(f"  SAM2 加载完成 ({time.time()-t0:.1f}s)")

    print(f"\n  开始标注 {n} 张图片 (grid stride={GRID_STRIDE})...")
    total_start = time.time()
    ok, fail, skipped = 0, 0, 0

    for idx, jpg_path in enumerate(targets):
        t_img = time.time()
        stem = jpg_path.stem
        out_txt = OUT_DIR / f"{stem}.txt"
        if out_txt.exists():
            skipped += 1
            continue

        m = re.search(r"DJI_(\d{4})", stem)
        if not m:
            fail += 1
            continue
        img_num = int(m.group(1))

        try:
            # 读取图片
            raw = np.fromfile(str(jpg_path), dtype=np.uint8)
            img_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if img_bgr is None:
                fail += 1
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]

            # 计算 NDVI/NDRE
            ndvi, ndre = compute_ndvi_from_jpg(img_num, h, w)
            if ndvi is None:
                fail += 1
                continue

            # YOLO 检出 bbox
            bboxes = get_yolo_bboxes(yolo, img_rgb, ndvi, ndre, device)

            # SAM2 逐 bbox 分割
            masks = sam2_per_box_masks(sam2, img_rgb, bboxes)

            # 转换 masks -> polygons
            all_polygons = []
            if masks:
                for mask in masks:
                    polygons = mask_to_yolo_polygons(mask, w, h)
                    all_polygons.extend(polygons)

            # 写入 YOLO TXT
            with open(out_txt, "w", encoding="utf-8") as f:
                for poly in all_polygons:
                    f.write(f"0 {' '.join(f'{v:.6f}' for v in poly)}\n")

            ok += 1
            elapsed = time.time() - t_img
            eta = (time.time() - total_start) / ok * (n - ok - skipped) if ok > 0 else 0
            if (idx + 1) % BATCH_SIZE == 0 or idx == n - 1:
                print(f"  [{idx+1}/{n}] {stem}  {len(bboxes)}框"
                      f"→{len(all_polygons)}段  {elapsed:.1f}s  预计剩余{eta/60:.0f}min")

        except Exception as e:
            fail += 1
            print(f"  [{idx+1}/{n}] {stem} 失败: {e}")

    total_time = time.time() - total_start
    print(f"\n  完成! ok={ok} fail={fail} skip={skipped}  总耗时 {total_time/60:.1f}min")
    print(f"  标注文件: {OUT_DIR}")
    print("\n  下一步: 将标注文件移动到 datasets/Data_v2/train/labels/ 并更新数据集")


if __name__ == "__main__":
    main()
