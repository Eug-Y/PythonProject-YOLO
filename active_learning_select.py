"""
主动学习筛选100张最有价值的未标注图片
策略：用v10对229张未标注图片推理，按平均置信度从低到高排序
置信度最低的 = 模型最不确定 = 人工标注收益最大
"""
import os, sys, re, shutil
import numpy as np
import cv2
from pathlib import Path
import torch
from ultralytics.utils.nms import non_max_suppression

PROJECT_DIR = Path(__file__).resolve().parent
MULTI_DIR = PROJECT_DIR / "多光谱图片"
DATA_V2 = PROJECT_DIR / "datasets" / "Data_v2"
OUTPUT_DIR = PROJECT_DIR / "待标注_100张"
CKPT_NAME = "exp_ca_5ch_v11"

def load_model():
    from custom_modules import CoordAtt, ECA
    from ultralytics.nn.tasks import SegmentationModel, load_checkpoint

    pt = Path(r"D:\PythonProject\runs\segment\land_seg") / CKPT_NAME / "weights" / "best.pt"
    ckpt_model, _ = load_checkpoint("yolo11n-seg.pt")
    base_model = SegmentationModel(ckpt_model.yaml, nc=1)
    base_model.load(ckpt_model)

    def _get_output_channels(layer):
        if hasattr(layer, "out_channels"): return layer.out_channels
        if hasattr(layer, "cv2") and hasattr(layer.cv2, "conv"): return layer.cv2.conv.out_channels
        if hasattr(layer, "cv3") and hasattr(layer.cv3, "conv"): return layer.cv3.conv.out_channels
        if hasattr(layer, "conv"): return layer.conv.out_channels
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
    return base_model


def find_unlabeled_images():
    """找出有_5ch.npy但没有标注的图片编号"""
    labeled = set()
    for split in ['train', 'valid', 'test']:
        lbl_dir = DATA_V2 / split / 'labels'
        if lbl_dir.exists():
            for f in lbl_dir.glob('*.txt'):
                labeled.add(f.stem)

    npy_files = list(MULTI_DIR.glob('*_5ch.npy'))
    unlabeled = []
    for npy_path in sorted(npy_files):
        stem = npy_path.stem.replace('_5ch', '')
        if stem not in labeled:
            # 找对应JPG
            jpg = MULTI_DIR / f"{stem}.JPG"
            if not jpg.exists():
                jpg = MULTI_DIR / f"{stem}.jpg"
            if jpg.exists():
                unlabeled.append((stem, jpg, npy_path))
    return unlabeled


def infer_one(model, img_rgb, ndvi, ndre, device, imgsz=1024):
    """单次推理，返回检测数和平均置信度"""
    h0, w0 = img_rgb.shape[:2]
    ch5 = np.stack([
        img_rgb[:, :, 0].astype(np.float32) / 255.0,
        img_rgb[:, :, 1].astype(np.float32) / 255.0,
        img_rgb[:, :, 2].astype(np.float32) / 255.0,
        ndvi.astype(np.float32),
        ndre.astype(np.float32),
    ], axis=-1)

    r = imgsz / max(h0, w0)
    if r != 1.0:
        ch5 = cv2.resize(ch5, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)

    tensor = torch.from_numpy(ch5).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model._predict_once(tensor)
    preds = outputs[0][0]
    pred = non_max_suppression(preds, 0.05, 0.45, nc=1)

    detections = []
    for det in pred:
        if det is None or len(det) == 0:
            continue
        confs = det[:, 4].cpu().numpy()
        detections.extend(confs.tolist())

    num_det = len(detections)
    avg_conf = np.mean(detections) if detections else 0.0
    min_conf = np.min(detections) if detections else 0.0
    # Higher score means more useful for review: low confidence and unstable predictions.
    conf_std = np.std(detections) if detections else 0.0
    uncertainty = (1.0 - avg_conf) + conf_std * 0.5 if detections else 2.0
    return num_det, avg_conf, min_conf, uncertainty


def main():
    print("=" * 60)
    print("主动学习：筛选100张最有标注价值的图片")
    print("=" * 60)

    # 1. 找未标注图片
    unlabeled = find_unlabeled_images()
    print(f"\n找到 {len(unlabeled)} 张未标注图片（有_5ch.npy）")

    # 2. 加载模型
    print("\n加载模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model()
    print(f"模型已加载，设备: {device}")

    # 3. 对所有未标注图片推理
    print(f"\n推理中 ({len(unlabeled)} 张)...")
    results = []
    for idx, (stem, jpg_path, npy_path) in enumerate(unlabeled):
        img_bgr = cv2.imread(str(jpg_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        npy = np.load(str(npy_path))
        npy_h, npy_w = npy.shape[:2]
        if (npy_h, npy_w) != (h, w):
            ndvi = cv2.resize(npy[:, :, 3], (w, h)).astype(np.float32)
            ndre = cv2.resize(npy[:, :, 4], (w, h)).astype(np.float32)
        else:
            ndvi = npy[:, :, 3].astype(np.float32)
            ndre = npy[:, :, 4].astype(np.float32)

        num_det, avg_conf, min_conf, uncertainty = infer_one(
            model, img_rgb, ndvi, ndre, device)

        results.append({
            'stem': stem,
            'jpg': jpg_path,
            'npy': npy_path,
            'num_det': num_det,
            'avg_conf': avg_conf,
            'min_conf': min_conf,
            'uncertainty': uncertainty,
        })

        if (idx + 1) % 20 == 0:
            print(f"  进度: {idx+1}/{len(unlabeled)}")

    # 4. Sort highest uncertainty first.
    results.sort(key=lambda x: x['uncertainty'], reverse=True)

    # 5. 选前100张
    selected = results[:100]
    print(f"\n筛选结果:")
    print(f"  总未标注: {len(unlabeled)}")
    print(f"  已选: {len(selected)}")
    print(f"  平均检测数: {np.mean([r['num_det'] for r in selected]):.1f}")
    print(f"  平均置信度: {np.mean([r['avg_conf'] for r in selected]):.3f}")

    # 置信度分布
    conf_bins = {'<0.2': 0, '0.2-0.3': 0, '0.3-0.4': 0, '0.4-0.5': 0, '>0.5': 0}
    for r in selected:
        c = r['avg_conf']
        if c < 0.2: conf_bins['<0.2'] += 1
        elif c < 0.3: conf_bins['0.2-0.3'] += 1
        elif c < 0.4: conf_bins['0.3-0.4'] += 1
        elif c < 0.5: conf_bins['0.4-0.5'] += 1
        else: conf_bins['>0.5'] += 1
    print(f"  置信度分布: {conf_bins}")

    # 6. 复制到标注目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    img_out = OUTPUT_DIR / "images"
    img_out.mkdir(exist_ok=True)

    print(f"\n复制图片到: {OUTPUT_DIR}")
    manifest = []
    for r in selected:
        shutil.copy2(str(r['jpg']), str(img_out / f"{r['stem']}.JPG"))
        manifest.append(f"{r['stem']}: det={r['num_det']}, avg_conf={r['avg_conf']:.3f}")

    # 写入清单
    manifest_path = OUTPUT_DIR / "标注清单.txt"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        f.write("主动学习筛选100张 — 按标注价值排序（最上面=最值得标注）\n")
        f.write("=" * 60 + "\n\n")
        for i, line in enumerate(manifest):
            f.write(f"{i+1:3d}. {line}\n")

    # 写入标注指南
    guide_path = OUTPUT_DIR / "标注指南.txt"
    with open(guide_path, 'w', encoding='utf-8') as f:
        f.write("标注指南 — 保持与现有标注一致\n")
        f.write("=" * 60 + "\n\n")
        f.write("工具: Labelme (推荐) 或 X-AnyLabeling (带SAM辅助)\n\n")
        f.write("标注规范:\n")
        f.write("1. 用多边形标注每条田垄的完整轮廓\n")
        f.write("2. 标签类别统一为 '0'（单类别: 田垄）\n")
        f.write("3. 多边形顶点数平均60-70个（高精度标注）\n")
        f.write("4. 田垄边界对齐像素边缘\n")
        f.write("5. 被遮挡或模糊的田垄也标注（宁可多标不要漏标）\n")
        f.write("6. 导出为YOLO分割TXT格式\n\n")
        f.write("参考现有标注:\n")
        f.write(f"  {DATA_V2}/train/labels/ 下有140个样例\n")
        f.write(f"  建议先看 DJI_0010.txt, DJI_0030.txt 了解格式\n\n")
        f.write("格式: class_id x1 y1 x2 y2 ... (坐标归一化到0-1)\n")
        f.write("示例: 0 0.6219 0.0000 0.6063 0.0177 ...\n")

    print(f"清单: {manifest_path}")
    print(f"指南: {guide_path}")
    print(f"\n完成! 图片在: {img_out}")
    print(f"共 {len(selected)} 张，建议按清单顺序标注（排前面的模型最不确定）")


if __name__ == "__main__":
    main()
