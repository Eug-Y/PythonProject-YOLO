"""
自愈训练管道：标注完成 → 集成 → 训练
1. 等待 SAM2 标注完成
2. 生成缺失的 _5ch.npy
3. 合并 + 重分割数据集 (70/20/10)
4. 启动训练，失败自动重试
"""
import sys
import time
import numpy as np
from pathlib import Path

from merge_and_train import build_dataset, create_yaml, merge_datasets, split_dataset
from multispectral import ensure_5ch


PROJECT = Path(__file__).resolve().parent
TIF_DIR = PROJECT / "多光谱图片"
SAM2_LABELS = PROJECT / "待标注_100张" / "labels"

MAX_TRAIN_RETRIES = 3
SEED = 42


def wait_for_labeling(expected_count=100):
    """等待 SAM2 标注完成"""
    print(f"[pipeline] 等待 SAM2 标注完成 (目标 {expected_count} 张)...")
    while True:
        done = list(SAM2_LABELS.glob("*.txt"))
        n = len(done)
        print(f"  sam2_labels: {n}/{expected_count}")
        if n >= expected_count:
            print(f"[pipeline] 标注完成! {n} 张")
            return n
        # Check if batch_label_sam2 is still running
        time.sleep(30)


def check_label_quality(txt_path):
    """验证标注文件有效"""
    if not txt_path.exists():
        return False
    content = txt_path.read_text(encoding="utf-8").strip()
    if not content:
        return False  # empty = no detections, acceptable
    for line in content.split("\n"):
        parts = line.strip().split()
        if len(parts) < 7:  # class + at least 3 points (6 coords)
            return False
        try:
            coords = [float(v) for v in parts[1:]]
            if len(coords) % 2 != 0:
                return False
        except ValueError:
            return False
    return True


def gen_5ch_npy(jpg_stem, img_num):
    """Generate or repair one normalized _5ch.npy file."""
    npy_path = TIF_DIR / f"{jpg_stem}_5ch.npy"
    jpg_path = TIF_DIR / f"{jpg_stem}.JPG"
    if not jpg_path.exists():
        return False
    try:
        ensure_5ch(jpg_path, TIF_DIR, npy_path, repair=True)
        return True
    except (FileNotFoundError, ValueError):
        return False


def integrate_labels():
    """将 sam2_labels 集成到训练数据集"""
    print("[pipeline] 集成标注...")
    txts = list(SAM2_LABELS.glob("*.txt"))
    integrated = []

    for txt in txts:
        stem = txt.stem
        npy_path = TIF_DIR / f"{stem}_5ch.npy"
        if not npy_path.exists():
            # 尝试生成 _5ch.npy
            import re
            m = re.search(r"DJI_(\d{4})", stem)
            if m:
                img_num = int(m.group(1))
                ok = gen_5ch_npy(stem, img_num)
                if not ok:
                    print(f"  [WARN] 无法生成 {stem}_5ch.npy，跳过")
                    continue

        if check_label_quality(txt):
            integrated.append(stem)
        else:
            print(f"  [WARN] {stem} 标注质量异常，跳过")

    print(f"  有效标注: {len(integrated)}/{len(txts)}")
    return integrated


def run_training(data_yaml):
    """运行训练，失败自动重试"""
    print("[pipeline] 启动训练...")
    sys.path.insert(0, str(PROJECT))

    for attempt in range(1, MAX_TRAIN_RETRIES + 1):
        print(f"\n{'='*60}")
        print(f"  训练第 {attempt}/{MAX_TRAIN_RETRIES} 次")
        print(f"{'='*60}")

        try:
            import torch.nn as nn
            from custom_modules import CoordAtt, ECA
            from ultralytics.nn.tasks import SegmentationModel, load_checkpoint
            from yolo_5ch_data import FiveChannelSegmentationTrainer

            # Model building functions
            def _get_output_channels(layer):
                if hasattr(layer, "out_channels"): return layer.out_channels
                if hasattr(layer, "cv2") and hasattr(layer.cv2, "conv"): return layer.cv2.conv.out_channels
                if hasattr(layer, "cv3") and hasattr(layer.cv3, "conv"): return layer.cv3.conv.out_channels
                if hasattr(layer, "conv"): return layer.conv.out_channels
                return None

            ckpt_model, _ = load_checkpoint("yolo11n-seg.pt")
            base_model = SegmentationModel(ckpt_model.yaml, nc=1)
            base_model.load(ckpt_model)

            # Insert CoordAtt + ECA
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
            base_model.model = nn.Sequential(*new_layers)
            save = set()
            for layer in new_layers:
                f = getattr(layer, "f", -1)
                if f == -1: continue
                for idx in ([f] if isinstance(f, (int, np.integer)) else f):
                    if idx != -1: save.add(int(idx))
            base_model.save = sorted(save)

            # 5ch first conv
            first = base_model.model[0]
            old_conv = first.conv
            old_weight = old_conv.weight.data
            new_conv = nn.Conv2d(5, old_conv.out_channels, old_conv.kernel_size,
                                 old_conv.stride, old_conv.padding, bias=False)
            new_conv.weight.data[:, :3].copy_(old_weight)
            nn.init.kaiming_uniform_(new_conv.weight.data[:, 3:], a=1.0)
            first.conv = new_conv

            args = {
                "model": "yolo11n-seg.pt",
                "data": str(data_yaml),
                "epochs": 300,
                "imgsz": 1024,
                "batch": 2,
                "device": "0",
                "workers": 2,
                "patience": 50,
                "optimizer": "SGD",
                "lr0": 0.002,
                "momentum": 0.937,
                "weight_decay": 0.0005,
                "warmup_epochs": 10,
                "cos_lr": True,
                "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
                "fliplr": 0.5, "flipud": 0,
                "mosaic": 1.0, "mixup": 0.1,
                "close_mosaic": 10,
                "scale": 0.5, "translate": 0.1,
                "amp": True, "cache": True,
                "save": True, "save_period": 10,
                "project": "land_seg",
                "name": "exp_ca_5ch_v11",
                "exist_ok": True,
                "seed": SEED,
                "deterministic": False,
                "verbose": True,
            }

            trainer = FiveChannelSegmentationTrainer(overrides=args)
            trainer.model = base_model
            trainer.train()

            print("[pipeline] 训练完成!")
            return True

        except Exception as e:
            print(f"[pipeline] 训练失败 (attempt {attempt}): {e}")
            if attempt < MAX_TRAIN_RETRIES:
                wait_s = 30 * attempt
                print(f"  等待 {wait_s}s 后重试...")
                time.sleep(wait_s)

    print("[pipeline] 训练多次失败，请手动排查")
    return False


def main():
    print("=" * 60)
    print("  自愈训练管道")
    print("=" * 60)

    SAM2_LABELS.mkdir(parents=True, exist_ok=True)

    # Step 1: 等待标注完成
    wait_for_labeling(100)

    # Step 2: 检查并准备五通道数据
    integrate_labels()

    # Step 3: 使用连续序列分组流程构建 Data_v3
    pairs = merge_datasets()
    splits = split_dataset(pairs)
    build_dataset(pairs, splits)
    data_yaml = create_yaml()

    # Step 4: 训练
    success = run_training(data_yaml)

    if success:
        print("\n[Pipeline] 全部完成!")
        print("  模型: land_seg/exp_ca_5ch_v11/weights/best.pt")
    else:
        print("\n[Pipeline] 训练失败，需手动排查")

    return success


if __name__ == "__main__":
    main()
