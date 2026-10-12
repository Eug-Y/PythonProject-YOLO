"""
单张图片田垄检测 — CoordAtt+ECA 5ch YOLO
用法: python test_land_seg.py [图片名或路径]
示例: python test_land_seg.py DJI_0080
"""
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn

from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import process_mask
from ultralytics.nn.tasks import SegmentationModel, load_checkpoint
from custom_modules import CoordAtt, ECA

# ====== 配置 ======
CKPT_PATH = r"D:\PythonProject\runs\segment\land_seg\exp_ca_5ch_v11\weights\best.pt"
DEVICE = "cuda:0"
IMGSZ = 1024
CONF = 0.25
IOU_THRESH = 0.45
OUT_DIR = "test_vis_v11"
os.makedirs(OUT_DIR, exist_ok=True)


# ====== 模型构建 ======
def _get_output_channels(layer):
    if hasattr(layer, "out_channels"):     return layer.out_channels
    if hasattr(layer, "cv2") and hasattr(layer.cv2, "conv"): return layer.cv2.conv.out_channels
    if hasattr(layer, "cv3") and hasattr(layer.cv3, "conv"): return layer.cv3.conv.out_channels
    if hasattr(layer, "conv"):             return layer.conv.out_channels
    return None


def build_ca_model_from_base(detection_model):
    old_layers = list(detection_model.model)
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
    detection_model.model = nn.Sequential(*new_layers)
    save = set()
    for layer in new_layers:
        f = getattr(layer, "f", -1)
        if f == -1: continue
        for idx in ([f] if isinstance(f, (int, np.integer)) else f):
            if idx != -1: save.add(int(idx))
    detection_model.save = sorted(save)
    return detection_model


def adapt_first_conv_5ch(model):
    first = model.model[0]
    old_conv = first.conv
    old_weight = old_conv.weight.data
    new_conv = nn.Conv2d(5, old_conv.out_channels, old_conv.kernel_size,
                         old_conv.stride, old_conv.padding, bias=False)
    new_conv.weight.data[:, :3].copy_(old_weight)
    nn.init.kaiming_uniform_(new_conv.weight.data[:, 3:], a=1.0)
    first.conv = new_conv
    return model


def load_model(ckpt_path, device):
    ckpt_model, _ = load_checkpoint("yolo11n-seg.pt")
    base_model = SegmentationModel(ckpt_model.yaml, nc=1)
    base_model.load(ckpt_model)
    model = build_ca_model_from_base(base_model)
    model = adapt_first_conv_5ch(model)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"].float().state_dict())
    model.to(device)
    model.eval()
    return model


# ====== 数据加载 ======
def load_gt(label_path, h, w):
    gt_masks = []
    if not os.path.exists(label_path):
        return gt_masks
    with open(label_path) as f:
        for line in f:
            if not line.strip(): continue
            parts = list(map(float, line.split()))
            pts = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
            pts_abs = (pts * np.array([w, h])).astype(np.int32)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts_abs], 1)
            gt_masks.append(mask)
    return gt_masks


# ====== 推理 ======
def infer_one(model, im_5ch, device):
    h0, w0 = im_5ch.shape[:2]
    r = IMGSZ / max(h0, w0)
    im_resized = cv2.resize(im_5ch, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR) if r != 1.0 else im_5ch
    tensor = torch.from_numpy(im_resized).permute(2, 0, 1).unsqueeze(0).to(device)
    _, _, input_h, input_w = tensor.shape

    with torch.no_grad():
        outputs = model._predict_once(tensor)
    pred = non_max_suppression(outputs[0][0], CONF, IOU_THRESH, nc=1)
    proto = outputs[0][1]

    scale_x, scale_y = w0 / input_w, h0 / input_h
    results = []
    for det in pred:
        if det is None or len(det) == 0: continue
        if det.shape[1] > 6:
            masks_in = det[:, 6:].cpu()
            bboxes = det[:, :4].cpu()
            conf = det[:, 4].cpu()
            cls = det[:, 5].cpu()
            bboxes_orig = bboxes.clone()
            bboxes_orig[:, [0, 2]] *= scale_x
            bboxes_orig[:, [1, 3]] *= scale_y
            pred_masks = (process_mask(proto[0].cpu(), masks_in, bboxes_orig, (h0, w0), upsample=True) > 0.5).byte()
            for i in range(len(conf)):
                results.append({
                    "bbox": bboxes_orig[i].numpy(),
                    "score": conf[i].item(),
                    "cls": int(cls[i].item()),
                    "mask": pred_masks[i].numpy(),
                })
    return results


# ====== 可视化 ======
def mask_iou(a, b):
    """两个二值mask的IoU"""
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / (union + 1e-7)


def draw_results(rgb_img, results, ndvi_map=None, gt_masks=None, out_path=None):
    h, w = rgb_img.shape[:2]
    overlay = rgb_img.copy()

    # 预定义颜色
    colors = [(0, 255, 0), (255, 200, 0), (0, 200, 255), (200, 0, 255),
              (0, 140, 255), (255, 120, 0), (120, 255, 0), (255, 0, 100)]

    # ---- 绘制 GT（蓝色轮廓） ----
    if gt_masks:
        gt_combined = np.zeros((h, w), dtype=np.uint8)
        for m in gt_masks:
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            gt_combined = np.maximum(gt_combined, m)
        contours, _ = cv2.findContours(gt_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 80, 80), 2)

    # ---- 绘制预测（彩色填充 + 框 + 分数） ----
    results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
    for i, r in enumerate(results_sorted):
        color = colors[i % len(colors)]
        mask = r["mask"]
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        overlay[mask > 0] = color

        bx = r["bbox"]
        x1, y1, x2, y2 = int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])
        # 粗边框 + 外发光
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 5)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
        # 分数：白底 + 彩色字，大字号
        text = f"{r['score']:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 3)
        cv2.rectangle(overlay, (x1, y1 - th - 8), (x1 + tw + 6, y1), (255, 255, 255), -1)
        cv2.rectangle(overlay, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, 2)
        cv2.putText(overlay, text, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)

    # ---- 合成 ----
    alpha = 0.5
    vis = cv2.addWeighted(rgb_img, 1 - alpha, overlay, alpha, 0)

    # ---- 图例 ----
    y0 = h - 10
    if gt_masks:
        cv2.putText(vis, "GT (blue)  |  Pred (color fill)", (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3)
        cv2.putText(vis, "GT (blue)  |  Pred (color fill)", (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    # ---- 右侧 NDVI 热力图（可选，注释掉即不显示） ----
    # ndvi_uint8 = (np.clip(ndvi_map, 0, 1) * 255).astype(np.uint8)
    # ndvi_color = cv2.applyColorMap(ndvi_uint8, cv2.COLORMAP_JET)
    # vis = np.hstack([vis, ndvi_color])

    cv2.imwrite(out_path, vis)
    print(f"  结果图: {out_path}")
    return vis


# ====== 查找图片路径 ======
def find_file(name, splits=("test", "valid", "train"), exts=(".JPG", ".jpg", "")):
    """在数据集各split中查找文件"""
    dataset_root = r"D:\PythonProject-YOLO\datasets\Data_v2"
    for split in splits:
        d = os.path.join(dataset_root, split, "images")
        if not os.path.isdir(d): continue
        for ext in exts:
            probe = name + ext if ext else name
            path = os.path.join(d, probe)
            if os.path.exists(path):
                return path, split
    return None, None


# ====== 主流程 ======
def main():
    # ---- 解析参数 ----
    if len(sys.argv) < 2:
        test_dir = r"D:\PythonProject-YOLO\datasets\Data_v2\test\images"
        candidates = sorted([f for f in os.listdir(test_dir) if f.endswith(".JPG")])
        if not candidates:
            print("未找到测试图片。用法: python test_land_seg.py <图片名或路径>")
            return
        jpg_path = os.path.join(test_dir, candidates[0])
        print(f"未指定图片，默认使用: {candidates[0]}")
    else:
        arg = sys.argv[1]
        if os.path.exists(arg):
            jpg_path = arg
        else:
            # 尝试在数据集中查找
            base = arg.rsplit(".", 1)[0] if "." in arg else arg
            jpg_path, split = find_file(base)
            if jpg_path is None:
                print(f"找不到图片: {arg}")
                return

    base = os.path.splitext(os.path.basename(jpg_path))[0]
    print(f"\n{'='*55}")
    print(f"  单张测试 — {base}.JPG")
    print(f"  Conf={CONF}  IoU={IOU_THRESH}  ImgSz={IMGSZ}")
    print(f"{'='*55}")

    # ---- 查找 5ch npy ----
    jpg_dir = os.path.dirname(jpg_path)
    npy_path = os.path.join(jpg_dir, base + "_5ch.npy")
    if not os.path.exists(npy_path):
        npy_path, _ = find_file(base + "_5ch.npy", exts=[""])
    if not npy_path or not os.path.exists(npy_path):
        print("错误: 找不到 _5ch.npy！先运行 prepare_5ch.py 生成。")
        return
    print(f"  图片: {jpg_path}")
    print(f"  5ch : {npy_path}")

    # ---- 查找 GT 标签 ----
    label_dir = jpg_dir.replace("images", "labels")
    label_path = os.path.join(label_dir, base + ".txt")
    if not os.path.exists(label_path):
        for split in ["test", "valid", "train"]:
            probe = os.path.join(r"D:\PythonProject-YOLO\datasets\Data_v2", split, "labels", base + ".txt")
            if os.path.exists(probe):
                label_path = probe
                break

    # ---- 加载 ----
    print("  加载模型中...")
    model = load_model(CKPT_PATH, DEVICE)

    rgb = cv2.cvtColor(cv2.imread(jpg_path), cv2.COLOR_BGR2RGB)
    im_5ch = np.load(npy_path).astype(np.float32)
    if im_5ch.max() > 1.0:
        im_5ch /= 255.0
    ndvi = im_5ch[..., 3]
    gt_masks = load_gt(label_path, rgb.shape[0], rgb.shape[1])

    # ---- 推理 ----
    print("  推理中...")
    results = infer_one(model, im_5ch, DEVICE)

    # ---- 统计 ----
    n_gt = len(gt_masks)
    n_pred = len(results)
    print(f"\n  GT 实例: {n_gt}    检测结果: {n_pred}")

    if n_pred > 0:
        scores = np.array([r["score"] for r in results])
        print(f"  置信度范围: [{scores.min():.3f}, {scores.max():.3f}]  均值: {scores.mean():.3f}")

        # 按置信度分层统计
        for lo, hi, label in [(0.5, 1.0, "高置信(>0.5)"), (0.25, 0.5, "中置信(0.25-0.5)")]:
            cnt = ((scores >= lo) & (scores < hi)).sum()
            if cnt > 0:
                print(f"    {label}: {cnt} 个")

    # ---- GT 匹配分析 ----
    if n_gt > 0 and n_pred > 0:
        # 简单的 IoU 匹配
        matched = 0
        gt_matched = [False] * n_gt
        for r in sorted(results, key=lambda x: x["score"], reverse=True):
            best_iou, best_gt = 0, -1
            for j, gm in enumerate(gt_masks):
                if gt_matched[j]: continue
                iou = mask_iou(r["mask"], gm)
                if iou > best_iou:
                    best_iou, best_gt = iou, j
            if best_iou > 0.3:
                gt_matched[best_gt] = True
                matched += 1
        print(f"\n  IoU>0.3 匹配: {matched}/{n_gt} GT = {matched/n_gt*100:.1f}%")
        print(f"  未匹配 GT: {sum(1 for m in gt_matched if not m)}")
        fp = n_pred - matched
        if fp > 0:
            print(f"  未匹配预测 (可能FP): {fp}")

    # ---- 详细列表 ----
    if n_pred > 0 and n_pred <= 80:
        show = min(n_pred, 30)
        print(f"\n  Top {show} 检测结果:")
        for i, r in enumerate(sorted(results, key=lambda x: x["score"], reverse=True)[:show]):
            bx = r["bbox"]
            w_box, h_box = bx[2] - bx[0], bx[3] - bx[1]
            aspect = w_box / (h_box + 0.1)
            print(f"  [{i:2d}] conf={r['score']:.3f}  bbox=({bx[0]:5.0f},{bx[1]:5.0f},{bx[2]:5.0f},{bx[3]:5.0f})  "
                  f"aspect={aspect:.1f}  area={r['mask'].sum():6d}px")

    # ---- 可视化 ----
    out_path = os.path.join(OUT_DIR, f"{base}_result.jpg")
    draw_results(rgb, results, ndvi, gt_masks, out_path)

    # ---- 总结 ----
    print(f"\n  {'='*40}")
    if n_gt > 0:
        r = n_pred / n_gt * 100
        print(f"    检测数/GT = {n_pred}/{n_gt} ({r:.1f}%)")
    print(f"    结果图 -> {OUT_DIR}/{base}_result.jpg")
    print(f"  {'='*40}")


if __name__ == "__main__":
    main()
