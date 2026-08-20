"""
合并 200张(Data_v2) + 100张(SAM2新标注) → 300张 → 重分割 → 训练v11
"""
import csv
import shutil
from pathlib import Path
from collections import defaultdict

from grouped_split import adjacent_cross_split_pairs, grouped_split
from multispectral import ensure_5ch, read_image


PROJECT = Path(__file__).resolve().parent
MULTI = PROJECT / "多光谱图片"
DATA_V2 = PROJECT / "datasets" / "Data_v2"
SAM2_LABELS = PROJECT / "待标注_100张" / "labels"
SAM2_IMGS = PROJECT / "待标注_100张" / "images"
NEW_DATASET = PROJECT / "datasets" / "Data_v3"
TRAIN_RATIO = 0.70
VALID_RATIO = 0.20
TEST_RATIO = 0.10

def merge_datasets():
    """收集所有图片-标注对"""
    pairs = {}  # stem -> {'jpg': path, 'txt': path}

    # 1. 现有Data_v2标注
    for split in ['train', 'valid', 'test']:
        lbl_dir = DATA_V2 / split / 'labels'
        img_dir = DATA_V2 / split / 'images'
        if not lbl_dir.exists():
            continue
        for txt in lbl_dir.glob('*.txt'):
            stem = txt.stem
            jpg = img_dir / f"{stem}.JPG"
            if not jpg.exists():
                jpg = img_dir / f"{stem}.jpg"
            if jpg.exists():
                pairs[stem] = {'jpg': jpg, 'txt': txt, 'source': f'Data_v2/{split}'}

    # 2. 新的SAM2标注
    if SAM2_LABELS.exists():
        for txt in SAM2_LABELS.glob('*.txt'):
            stem = txt.stem
            if stem in pairs:
                continue  # 不覆盖已有的
            jpg = SAM2_IMGS / f"{stem}.JPG"
            if not jpg.exists():
                jpg = SAM2_IMGS / f"{stem}.jpg"
            if jpg.exists():
                pairs[stem] = {'jpg': jpg, 'txt': txt, 'source': 'sam2_new'}

    print(f"总计图片-标注对: {len(pairs)}")
    return pairs


def split_dataset(pairs):
    """按连续航拍序列分组，避免相邻帧跨数据集泄漏。"""
    is_new = {stem: pair['source'] == 'sam2_new' for stem, pair in pairs.items()}
    splits = grouped_split(
        list(pairs),
        is_new,
        ratios=(TRAIN_RATIO, VALID_RATIO, TEST_RATIO),
        max_gap=10,
        seed=42,
    )
    crossed = adjacent_cross_split_pairs(splits, max_gap=10)
    if crossed:
        raise RuntimeError(f"分组划分后仍有 {len(crossed)} 对相邻帧跨 split")
    for k, v in splits.items():
        new_count = sum(is_new[stem] for stem in v)
        print(f"  {k}: {len(v)} (base={len(v) - new_count}, sam2_new={new_count})")
    return splits


def ensure_npy(stem, jpg_path, h, w):
    """Ensure a normalized _5ch.npy exists in the shared data directory."""
    npy_path = MULTI / f"{stem}_5ch.npy"
    try:
        path, status = ensure_5ch(jpg_path, MULTI, npy_path, repair=True)
        if status in {"generated", "repaired"}:
            print(f"    {status} npy: {path.name}")
        return path
    except (FileNotFoundError, ValueError) as exc:
        print(f"    failed npy: {stem}: {exc}")
        return None


def build_dataset(pairs, splits):
    """在临时目录构建并校验后替换旧 split。"""
    staging = NEW_DATASET / "_build_staging"
    backup = NEW_DATASET / "_previous_split"
    if staging.exists():
        shutil.rmtree(staging)
    if backup.exists():
        raise FileExistsError(f"请先处理上次保留的目录: {backup}")
    for split in ['train', 'valid', 'test']:
        for sub in ['images', 'labels']:
            (staging / split / sub).mkdir(parents=True, exist_ok=True)

    stats = defaultdict(int)
    for split, stems in splits.items():
        for stem in stems:
            pair = pairs[stem]
            src_jpg = pair['jpg']
            src_txt = pair['txt']

            # 读取图片获取尺寸（用 imdecode 兼容部分 JPG）
            try:
                img = read_image(src_jpg)
            except ValueError:
                continue
            h, w = img.shape[:2]

            # 复制图片
            dst_jpg = staging / split / 'images' / f"{stem}.JPG"
            shutil.copy2(str(src_jpg), str(dst_jpg))

            # 复制标注
            dst_txt = staging / split / 'labels' / f"{stem}.txt"
            shutil.copy2(str(src_txt), str(dst_txt))

            # 确保npy
            ensure_npy(stem, src_jpg, h, w)

            stats[split] += 1
            stats[f'{split}_from_{pair["source"]}'] += 1

    for split, stems in splits.items():
        image_count = len(list((staging / split / 'images').glob('*.JPG')))
        label_count = len(list((staging / split / 'labels').glob('*.txt')))
        if image_count != len(stems) or label_count != len(stems):
            raise RuntimeError(f"{split} 构建不完整: images={image_count}, labels={label_count}, expected={len(stems)}")

    report_dir = PROJECT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "Data_v3_previous_split.csv").open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['stem', 'split'])
        for split in ['train', 'valid', 'test']:
            old_label_dir = NEW_DATASET / split / 'labels'
            for label in sorted(old_label_dir.glob('*.txt')):
                writer.writerow([label.stem, split])

    existing = [NEW_DATASET / split for split in ['train', 'valid', 'test'] if (NEW_DATASET / split).exists()]
    try:
        if existing:
            backup.mkdir(parents=True)
            for directory in existing:
                shutil.move(str(directory), str(backup / directory.name))
        for split in ['train', 'valid', 'test']:
            shutil.move(str(staging / split), str(NEW_DATASET / split))
        staging.rmdir()
    except Exception:
        for split in ['train', 'valid', 'test']:
            current = NEW_DATASET / split
            previous = backup / split
            if previous.exists():
                if current.exists():
                    shutil.rmtree(current)
                shutil.move(str(previous), str(current))
        raise
    if backup.exists():
        shutil.rmtree(backup)

    return stats


def create_yaml():
    """生成 data_5ch_v3.yaml"""
    yaml_path = NEW_DATASET / "data_5ch_v3.yaml"
    content = f"""# Data_v3: 300 images (200 original + 100 SAM2 new)
path: {NEW_DATASET.as_posix()}
train: train/images
val: valid/images
test: test/images

nc: 1
names: ['land']
channels: 5
"""
    with open(yaml_path, 'w') as f:
        f.write(content)
    print(f"\nYAML: {yaml_path}")
    return yaml_path


def main():
    print("=" * 60)
    print("  合并数据集 → Data_v3 (300张)")
    print("=" * 60)

    # 1. 合并
    print("\n[1/4] 收集标注...")
    pairs = merge_datasets()

    # 2. 分割
    print("\n[2/4] 连续序列分组划分 (70/20/10)...")
    splits = split_dataset(pairs)

    # 3. 构建
    print("\n[3/4] 构建 Data_v3...")
    stats = build_dataset(pairs, splits)

    print(f"\n  数据集统计:")
    for k, v in sorted(stats.items()):
        if not k.startswith('train_from') and not k.startswith('valid_from') and not k.startswith('test_from'):
            print(f"    {k}: {v}")
    print(f"    来源分布:")
    for k, v in sorted(stats.items()):
        if '_from_' in k:
            print(f"      {k}: {v}")

    # 4. YAML
    print("\n[4/4] 生成配置...")
    create_yaml()

    print(f"\n{'='*60}")
    print(f"  完成! 数据集: {NEW_DATASET}")
    print("  训练命令: python train_ca_5ch.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
