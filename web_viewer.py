"""
农智眼：无人机农田智能体检 — 多光谱图片提取工具 v3
"""
import os, re, json, io, base64, csv, uuid
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# 中文字体
_CN_FONT = None


def _get_cn_font(size):
    global _CN_FONT
    if _CN_FONT is None or _CN_FONT.size != size:
        try:
            _CN_FONT = ImageFont.truetype("msyh.ttc", size)
        except Exception:
            _CN_FONT = ImageFont.load_default()
    return _CN_FONT


def put_cn_text(img, text, pos, color, font_size=20, thickness=0):
    """用 PIL 在 OpenCV 图像上绘制中文文本"""
    if not text:
        return
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = _get_cn_font(font_size)
    # PIL 的 thickness 处理
    if thickness > 0:
        for dx in (-thickness, 0, thickness):
            for dy in (-thickness, 0, thickness):
                if dx == 0 and dy == 0:
                    continue
                draw.text((pos[0] + dx, pos[1] + dy), text, font=font, fill=color)
    draw.text(pos, text, font=font, fill=color)
    result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    img[:] = result[:]
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory

from spectral_indices import INDEX_REGISTRY, compute_indices, normalize_dn, required_bands

app = Flask(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent
TIF_DIR = PROJECT_ROOT / "多光谱图片"
MODEL_DIR = PROJECT_ROOT / "runs" / "segment" / "land_seg"
LEGACY_MODEL_DIR = Path(r"D:\PythonProject\runs\segment\land_seg")
MODEL_DIRS = (MODEL_DIR, LEGACY_MODEL_DIR)
DEPLOYMENT_CONFIG_PATH = PROJECT_ROOT / "configs" / "deployment.json"
SHADOW_DEPLOYMENT_CONFIG_PATH = PROJECT_ROOT / "configs" / "shadow_deployment_v27.json"


def _load_deployment_settings():
    fallback = {
        "model_name": "exp_ca_5ch_grouped_reviewed_v1",
        "confidence": 0.25,
        "sha256": "589a1d5600e58b1b5932a472137bddc057821daac4c8a5f98bb1e3a94715649b",
        "release_version": "incumbent_pre_v20",
    }
    try:
        payload = json.loads(DEPLOYMENT_CONFIG_PATH.read_text(encoding="utf-8"))
        model_name = str(payload["model_name"]).strip()
        confidence = float(payload["confidence"])
        if not model_name or not 0.0 < confidence < 1.0:
            raise ValueError("invalid deployment model/confidence")
        return {**fallback, **payload, "model_name": model_name, "confidence": confidence}
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return fallback


DEPLOYMENT_SETTINGS = _load_deployment_settings()
PREFERRED_MODEL_NAME = DEPLOYMENT_SETTINGS["model_name"]
DEFAULT_YOLO_CONF = float(DEPLOYMENT_SETTINGS["confidence"])


def _load_shadow_deployment_settings():
    try:
        payload = json.loads(SHADOW_DEPLOYMENT_CONFIG_PATH.read_text(encoding="utf-8"))
        if payload.get("state") != "shadow" or payload.get("validation_gate_passed") is not True:
            return None
        if payload.get("every_output_requires_yolo11") is not True:
            return None
        return payload
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


SHADOW_DEPLOYMENT_SETTINGS = _load_shadow_deployment_settings()
SHADOW_FUSION_MODEL_NAME = (
    SHADOW_DEPLOYMENT_SETTINGS["model_name"] if SHADOW_DEPLOYMENT_SETTINGS else None
)
DEFAULT_GSD = 0.028
REPORT_DIR = PROJECT_ROOT / "reports"

WEB_INDEX_NAMES = ("NDVI", "NDRE", "GNDVI", "EVI2", "RE_BSI_PROXY")
WEB_INDEX_METADATA = {
    "NDVI": {"label": "NDVI 植被指数", "formula": "(NIR - Red) / (NIR + Red)", "kind": "vegetation"},
    "NDRE": {"label": "NDRE 红边指数", "formula": "(NIR - RedEdge) / (NIR + RedEdge)", "kind": "vegetation"},
    "GNDVI": {"label": "GNDVI 绿光植被指数", "formula": "(NIR - Green) / (NIR + Green)", "kind": "vegetation"},
    "EVI2": {"label": "EVI2 增强植被指数", "formula": "2.5(NIR - Red) / (NIR + 2.4Red + 1)", "kind": "vegetation"},
    "RE_BSI_PROXY": {
        "label": "RE-BSI 土壤代理",
        "formula": "(RedEdge + Red - NIR - Blue) / sum",
        "kind": "soil_proxy",
        "warning": "实验性 RedEdge 裸土代理，不是依赖 SWIR 的标准 BSI。",
    },
}
REPORT_CATALOG = (
    ("V5 特征初筛", "Data_v5_feature_factor_analysis/index.html"),
    ("V5 单因子消融", "Data_v5_yolo_stage1_ablation/index.html"),
    ("V5 三 Seed 复验", "Data_v5_yolo_multiseed_validation/index.html"),
    ("V5 组合复验", "Data_v5_yolo_combo_validation/index.html"),
    ("V5 高分辨率筛选", "Data_v5_yolo_highres_screen/index.html"),
    ("V5 正式训练", "Data_v5_final_training/index.html"),
    ("V5 最终独立测试", "Data_v5_final_model_comparison/index.html"),
    ("V4 特征影响因子", "Data_v4_feature_factor_analysis/index.html"),
    ("V4 通道分组消融", "Data_v4_yolo_group_ablation/index.html"),
    ("V4 单因子消融", "Data_v4_yolo_single_feature_ablation/index.html"),
    ("V4 三 Seed 复验", "Data_v4_yolo_multiseed_validation/index.html"),
    ("V4 最终独立测试", "Data_v4_final_model_comparison/index.html"),
)
RESEARCH_MODEL_PREFIXES = ("v4_", "smoke_v4_", "v5_", "smoke_v5_")

# 模型缓存 {model_name: YOLO_instance}
_model_cache = {}


def discover_models():
    """从当前项目和历史项目中自动发现所有 YOLO 模型。"""
    models = []
    candidates = [pt for root in MODEL_DIRS if root.is_dir() for pt in root.glob("*/weights/best.pt")]
    seen_names = set()
    candidates.sort(
        key=lambda p: (
            p.parent.parent.name == PREFERRED_MODEL_NAME,
            p.stat().st_mtime,
        ),
        reverse=True,
    )
    for pt in candidates:
        name = pt.parent.parent.name
        if name in seen_names:
            continue
        if name.startswith(RESEARCH_MODEL_PREFIXES):
            continue
        # Research checkpoints carry an explicit feature contract.
        # The current web inference path intentionally supports only the
        # deployed 5-channel model and ordinary RGB models. Do not expose
        # rejected 4/6-channel experiments as selectable models.
        contract_path = pt.parent.parent / "feature_contract.json"
        if contract_path.is_file():
            try:
                contract = json.loads(contract_path.read_text(encoding="utf-8"))
                if int(contract.get("channel_count", 3)) != 3 and name != PREFERRED_MODEL_NAME:
                    continue
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        seen_names.add(name)
        size_mb = pt.stat().st_size / 1e6
        # 判断类别
        if "5ch" in name:
            kind = "5ch 多光谱"
        elif "ca" in name:
            kind = "3ch CoordAtt+ECA"
        elif "high_res" in name or "1280" in name:
            kind = "3ch 高分辨率"
        elif "1024" in name:
            kind = "3ch 1024"
        else:
            kind = "3ch 标准"
        source = "当前项目" if MODEL_DIR in pt.parents else "历史项目"
        models.append({
            "name": name,
            "path": str(pt),
            "kind": kind,
            "source": source,
            "size_mb": round(size_mb, 1),
            "recommended": name == PREFERRED_MODEL_NAME,
        })
    if SHADOW_DEPLOYMENT_SETTINGS:
        virtual = {
            "name": SHADOW_FUSION_MODEL_NAME,
            "path": str(SHADOW_DEPLOYMENT_CONFIG_PATH),
            "kind": "YOLO11主模型 + YOLOv8辅助（1:1融合）",
            "source": "V27正式部署" if PREFERRED_MODEL_NAME == SHADOW_FUSION_MODEL_NAME else "V27影子部署",
            "size_mb": None,
            "recommended": PREFERRED_MODEL_NAME == SHADOW_FUSION_MODEL_NAME,
            "shadow": PREFERRED_MODEL_NAME != SHADOW_FUSION_MODEL_NAME,
            "primary": "YOLO11",
            "auxiliary": "YOLOv8",
        }
        # Keep the active production model as the first/default option.
        insert_at = 1 if models and models[0]["name"] == PREFERRED_MODEL_NAME else 0
        models.insert(insert_at, virtual)
    return models


def model_requires_multispectral(model_name):
    return bool(model_name and ("5ch" in model_name or model_name == SHADOW_FUSION_MODEL_NAME))


def get_model(model_name):
    """惰性加载指定模型（支持自定义CoordAtt+ECA的5ch模型）"""
    if not model_name or model_name.startswith(RESEARCH_MODEL_PREFIXES):
        return None
    if model_name in _model_cache:
        return _model_cache[model_name]
    pt = next(
        (root / model_name / "weights" / "best.pt" for root in MODEL_DIRS
         if (root / model_name / "weights" / "best.pt").is_file()),
        None,
    )
    if pt is None:
        return None

    contract_path = pt.parent.parent / "feature_contract.json"
    if contract_path.is_file():
        try:
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            if int(contract.get("channel_count", 3)) != 3 and model_name != PREFERRED_MODEL_NAME:
                return None
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    if "5ch" in model_name:
        # 使用自定义模块加载 5ch 模型
        import torch
        from custom_modules import CoordAtt, ECA
        from ultralytics.nn.tasks import SegmentationModel, load_checkpoint

        ckpt_model, _ = load_checkpoint("yolo11n-seg.pt")
        base_model = SegmentationModel(ckpt_model.yaml, nc=1)
        base_model.load(ckpt_model)

        # build_ca_model_from_base
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

        # adapt_first_conv_5ch
        first = base_model.model[0]
        old_conv = first.conv
        old_weight = old_conv.weight.data
        new_conv = torch.nn.Conv2d(5, old_conv.out_channels, old_conv.kernel_size,
                                   old_conv.stride, old_conv.padding, bias=False)
        new_conv.weight.data[:, :3].copy_(old_weight)
        torch.nn.init.kaiming_uniform_(new_conv.weight.data[:, 3:], a=1.0)
        first.conv = new_conv

        # 加载训练好的权重
        ckpt = torch.load(str(pt), map_location="cpu", weights_only=False)
        base_model.load_state_dict(ckpt["model"].float().state_dict())
        device = "cuda" if torch.cuda.is_available() else "cpu"
        base_model.to(device)
        base_model.eval()
        _model_cache[model_name] = base_model
        return base_model
    else:
        from ultralytics import YOLO
        m = YOLO(str(pt))
        _model_cache[model_name] = m
        return m


def _load_tif(num):
    p = TIF_DIR / f"DJI_{num:04d}.TIF"
    if not p.exists():
        return None
    try:
        raw = np.fromfile(str(p), dtype=np.uint16)
        band = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    except Exception:
        return None
    if band is None:
        return None
    return band.astype(np.float32).squeeze()


def make_thermal_view(img_num, h, w, nir=None):
    """NIR 近红外视图：NIR 波段 + INFERNO 热力色映射"""
    if nir is None:
        nir = _load_tif(img_num + 5)
    if nir is None:
        return None
    if nir.shape[:2] != (h, w):
        nir = cv2.resize(nir, (w, h))
    # 归一化到 [0, 255]
    nir_norm = cv2.normalize(nir, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # INFERNO 热力色
    colored = cv2.applyColorMap(nir_norm, cv2.COLORMAP_INFERNO)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    # 底部图例
    bar_h = 24
    bar = np.zeros((bar_h + 16, w, 3), dtype=np.uint8)
    bar[:] = (25, 25, 25)
    for x in range(w):
        v = int(255 * x / w)
        color = cv2.applyColorMap(np.array([[v]], dtype=np.uint8), cv2.COLORMAP_INFERNO)[0, 0]
        bar[6:bar_h + 2, x] = color[::-1]
    put_cn_text(bar, "低温", (4, bar_h + 2), (180, 180, 180), font_size=11)
    put_cn_text(bar, "NIR 近红外", (w // 2 - 30, bar_h + 2), (180, 180, 180), font_size=11)
    put_cn_text(bar, "高温", (w - 50, bar_h + 2), (180, 180, 180), font_size=11)
    return np.vstack([colored_rgb, bar])


def compute_ndvi(img_num, h, w):
    """Backward-compatible NDVI/NDRE wrapper over the shared registry."""
    values = compute_web_indices(img_num, h, w, ("NDVI", "NDRE"))
    return values.get("NDVI"), values.get("NDRE")


_WEB_BAND_OFFSETS = {"blue": 1, "green": 2, "red": 3, "rededge": 4, "nir": 5}


def compute_web_indices(img_num, h, w, names=WEB_INDEX_NAMES, band_cache=None):
    """Load P4M bands once and compute encoded web layers via spectral_indices.py."""
    names = tuple(names)
    bands = {}
    for band_name in required_bands(names):
        band = band_cache.get(band_name) if band_cache is not None else None
        if band is None:
            band = _load_tif(img_num + _WEB_BAND_OFFSETS[band_name])
        if band is None:
            return {}
        if band.shape[:2] != (h, w):
            band = cv2.resize(band, (w, h), interpolation=cv2.INTER_LINEAR)
        if band_cache is not None:
            band_cache[band_name] = band
        bands[band_name] = normalize_dn(band)
    return compute_indices(list(names), bands, encoded=True)


def _plant_mask(ndvi, manual_threshold=0):
    """共享：双边滤波 + Otsu/手动阈值 + 形态学 + 高斯平滑 -> clean mask
    manual_threshold: 0=自动Otsu, 1-255=手动NDVI阈值
    """
    ndvi_f32 = ndvi.astype(np.float32)
    ndvi_smooth = cv2.bilateralFilter(ndvi_f32, 9, 0.15, 0.15)
    ndvi_u8 = (np.clip(ndvi_smooth, 0, 1) * 255).astype(np.uint8)
    if manual_threshold > 0:
        thresh_val = float(manual_threshold)
    else:
        thresh_val, _ = cv2.threshold(ndvi_u8, 0, 255, cv2.THRESH_OTSU)
    thresh = thresh_val / 255.0
    plant_mask = ndvi_smooth > thresh
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    plant_clean = cv2.morphologyEx(plant_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1)
    plant_clean = cv2.morphologyEx(plant_clean, cv2.MORPH_OPEN, kernel, iterations=1)
    plant_clean = cv2.GaussianBlur(plant_clean.astype(np.float32), (3, 3), 1.0)
    plant_clean = (plant_clean > 0.5).astype(np.uint8)
    return plant_clean, thresh


def _field_colors(n):
    """生成 n 个区分度高的 HSV 颜色 -> BGR"""
    cols = []
    for i in range(n):
        hue = int(180 * i / max(n, 1))
        hsv = np.uint8([[[hue, 210, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        cols.append(bgr.astype(np.int32))
    return cols


def calc_planting_area(ndvi, gsd=0.028, min_field_mu=0.005, ndvi_threshold=0):
    """种植面积统计 + 每块田明细"""
    h, w = ndvi.shape[:2]
    plant_clean, thresh = _plant_mask(ndvi, ndvi_threshold)
    cnts, _ = cv2.findContours(plant_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area_px = max(h * w * 0.0002, min_field_mu * 666.67 / (gsd * gsd))
    fields = []
    plant_px = 0
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_area_px:
            continue
        epsilon = 0.002 * cv2.arcLength(cnt, True)
        smooth = cv2.approxPolyDP(cnt, epsilon, True)
        area_mu = round(area * gsd * gsd / 666.67, 3)
        M = cv2.moments(smooth)
        cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else 0
        cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else 0
        fields.append({"id": len(fields) + 1, "area_px": int(area), "area_mu": area_mu,
                        "cx": cx, "cy": cy})
        plant_px += int(area)
    fields.sort(key=lambda f: f["area_mu"], reverse=True)
    # renumber after sort
    for i, f in enumerate(fields):
        f["id"] = i + 1
    plant_mu = round(plant_px * gsd * gsd / 666.67, 3)
    plant_pct = round(plant_px / (h * w) * 100, 1)
    return {
        "plant_total_px": plant_px, "plant_total_mu": plant_mu,
        "plant_pct": plant_pct, "plant_thresh": round(thresh, 3),
        "field_count": len(fields), "fields": fields,
    }


def make_planting_overlay(img_rgb, ndvi, gsd=0.028, min_field_mu=0.005, ndvi_threshold=0):
    """种植面积着色：每块田不同颜色 + 编号 + 图例"""
    overlay = img_rgb.copy()
    h, w = img_rgb.shape[:2]
    plant_clean, thresh = _plant_mask(ndvi, ndvi_threshold)

    cnts, _ = cv2.findContours(plant_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area_px = max(h * w * 0.0002, min_field_mu * 666.67 / (gsd * gsd))
    fields = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_area_px:
            continue
        epsilon = 0.002 * cv2.arcLength(cnt, True)
        smooth = cv2.approxPolyDP(cnt, epsilon, True)
        M = cv2.moments(smooth)
        cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else w // 2
        cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else h // 2
        area_mu = round(area * gsd * gsd / 666.67, 3)
        fields.append({"area_px": int(area), "area_mu": area_mu, "cnt": smooth, "cx": cx, "cy": cy})
    fields.sort(key=lambda f: f["area_mu"], reverse=True)

    colors = _field_colors(len(fields))

    # 非种植区统一灰色
    non_plant = plant_clean == 0
    overlay[non_plant] = (overlay[non_plant] * 0.65 + np.array([100, 100, 100], dtype=np.float32) * 0.35).astype(np.uint8)

    # 每块田着色 + 轮廓
    for idx, fd in enumerate(fields):
        color = colors[idx].astype(np.float32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [fd["cnt"]], -1, 255, -1)
        field_bool = mask > 0
        overlay[field_bool] = (overlay[field_bool] * 0.30 + color * 0.70).astype(np.uint8)
        cv2.drawContours(overlay, [fd["cnt"]], -1, (255, 255, 255), 2, cv2.LINE_AA)

    # PIL 渲染：编号 + 图例
    img_pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font_num = _get_cn_font(18)
    font_sm = _get_cn_font(12)
    font_title = _get_cn_font(14)

    for idx, fd in enumerate(fields):
        color_rgb = tuple(colors[idx][::-1].tolist())
        label = str(fd["area_mu"])
        bbox = draw.textbbox((0, 0), label, font=font_num)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        cx, cy = fd["cx"], fd["cy"]
        pad = 5
        # 白色圆角标签
        draw.rounded_rectangle(
            [cx - tw // 2 - pad - 2, cy - th // 2 - pad - 14,
             cx + tw // 2 + pad + 2, cy + th // 2 + pad + 2],
            radius=8, fill=(255, 255, 255, 240), outline=color_rgb, width=2)
        draw.text((cx - tw // 2, cy - th // 2 - 10), str(fd["area_mu"]), font=font_num, fill=(0, 0, 0))
        draw.text((cx - tw // 2 + 2, cy + th // 2 - 8), "亩", font=font_sm, fill=(80, 80, 80))

    # 图例
    y0 = 28
    draw.text((12, y0 - 2), f"种植区 共{len(fields)}块  (NDVI>{thresh:.2f})", font=font_title, fill=(255, 255, 255))
    y0 += 26
    max_legend = min(len(fields), 14)
    for i in range(max_legend):
        fd = fields[i]
        color_rgb = tuple(colors[i][::-1].tolist())
        draw.rectangle([12, y0 - 8, 28, y0 + 8], fill=color_rgb, outline=(255, 255, 255), width=1)
        draw.text((34, y0 - 4), f"#{i + 1}  {fd['area_mu']:.2f}亩  ({fd['area_px']}px)", font=font_sm, fill=(220, 220, 220))
        y0 += 20
    if len(fields) > max_legend:
        draw.text((12, y0 - 4), f"... 还有 {len(fields) - max_legend} 块", font=font_sm, fill=(150, 150, 150))
        y0 += 20
    y0 += 4
    draw.rectangle([12, y0 - 8, 28, y0 + 8], fill=(100, 100, 100), outline=(180, 180, 180), width=1)
    draw.text((34, y0 - 4), f"非种植区  NDVI<{thresh:.2f}", font=font_sm, fill=(180, 180, 180))

    overlay[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return overlay




def img_to_b64(img_rgb, fmt="jpg", quality=95):
    ext = ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buf = cv2.imencode(ext, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), params)
    return base64.b64encode(buf).decode("utf-8")

def img_to_b64_png(img_rgb):
    _, buf = cv2.imencode('.png', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode("utf-8")


def make_ndvi_heatmap(ndvi_norm, manual_threshold=0):
    """NDVI 二值化栅格：手动/Otsu 阈值，植被=绿色(0,255,0)，非植被=黑色(0,0,0)"""
    h, w = ndvi_norm.shape[:2]
    ndvi_u8 = (np.clip(ndvi_norm, 0, 1) * 255).astype(np.uint8)
    if manual_threshold > 0:
        thresh_val = float(manual_threshold)
    else:
        thresh_val, _ = cv2.threshold(ndvi_u8, 0, 255, cv2.THRESH_OTSU)
    binary = (ndvi_u8 > thresh_val).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    # 植被=绿色，非植被=黑色
    result = np.zeros((h, w, 3), dtype=np.uint8)
    result[binary > 0] = (0, 255, 0)

    bar_h = 22
    bar = np.full((bar_h + 14, w, 3), 20, dtype=np.uint8)
    cv2.rectangle(bar, (6, 6), (24, 6 + bar_h - 4), (0, 255, 0), -1)
    cv2.rectangle(bar, (6, 6), (24, 6 + bar_h - 4), (100, 100, 100), 1)
    put_cn_text(bar, "植被=255", (30, bar_h - 2), (200, 200, 200), font_size=12)
    cv2.rectangle(bar, (120, 6), (138, 6 + bar_h - 4), (0, 0, 0), -1)
    cv2.rectangle(bar, (120, 6), (138, 6 + bar_h - 4), (100, 100, 100), 1)
    put_cn_text(bar, "非植被=0", (144, bar_h - 2), (200, 200, 200), font_size=12)
    put_cn_text(bar, f"Otsu阈值={thresh_val}", (250, bar_h - 2), (150, 150, 150), font_size=11)

    return np.vstack([result, bar])


def make_index_heatmap(encoded_values, index_name):
    """Render a continuous encoded index with formula and fixed-range legend."""
    if index_name not in WEB_INDEX_METADATA:
        raise KeyError(f"Unsupported web index: {index_name}")
    values = np.asarray(encoded_values, dtype=np.float32)
    values = np.clip(np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    image_u8 = np.rint(values * 255.0).astype(np.uint8)
    color_map = cv2.COLORMAP_TURBO if index_name != "RE_BSI_PROXY" else cv2.COLORMAP_BONE
    colored_bgr = cv2.applyColorMap(image_u8, color_map)
    h, w = values.shape[:2]
    legend_h = 48
    legend = np.full((legend_h, w, 3), 22, dtype=np.uint8)
    gradient = np.tile(np.arange(256, dtype=np.uint8), (10, 1))
    gradient = cv2.resize(gradient, (max(w - 20, 1), 10), interpolation=cv2.INTER_LINEAR)
    gradient_bgr = cv2.applyColorMap(gradient, color_map)
    if w > 20:
        legend[7:17, 10:w - 10] = gradient_bgr
    metadata = WEB_INDEX_METADATA[index_name]
    put_cn_text(legend, "低值", (8, 18), (180, 180, 180), font_size=11)
    put_cn_text(legend, "高值", (max(w - 42, 8), 18), (180, 180, 180), font_size=11)
    title = f"{metadata['label']}  {metadata['formula']}"
    put_cn_text(legend, title, (8, 32), (220, 220, 220), font_size=11)
    return cv2.cvtColor(np.vstack([colored_bgr, legend]), cv2.COLOR_BGR2RGB)


def _render_shadow_fusion(image_rgb, predictions, model_name):
    """Render frozen fusion predictions without changing the legacy renderer."""
    height, width = image_rgb.shape[:2]
    overlay = image_rgb.copy()
    colors = (
        (46, 204, 113), (52, 152, 219), (241, 196, 15), (155, 89, 182),
        (230, 126, 34), (26, 188, 156), (231, 76, 60), (149, 165, 166),
    )
    for index, prediction in enumerate(predictions):
        color = colors[index % len(colors)]
        mask = np.asarray(prediction.mask, dtype=bool)
        if mask.shape != (height, width):
            raise ValueError(f"Fused mask/image mismatch: {mask.shape} versus {(height, width)}")
        overlay[mask] = color
        x1, y1, x2, y2 = prediction.bbox
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(width - 1, int(x2)), min(height - 1, int(y2))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 5)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
        label = f"YOLO11 {prediction.score:.2f}"
        text_y = max(18, y1 - 7)
        cv2.putText(overlay, label, (x1 + 3, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3)
        cv2.putText(overlay, label, (x1 + 3, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    rendered = cv2.addWeighted(image_rgb, 0.5, overlay, 0.5, 0)
    if not predictions:
        cv2.putText(
            rendered, "No field detected (YOLO11 primary)", (20, 34),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 180, 80), 2,
        )
    return rendered, len(predictions), model_name


def yolo_predict(model_name, image_rgb, ndvi_norm, ndre_norm, conf=DEFAULT_YOLO_CONF, imgsz=1024):
    """根据模型名自动选择 3ch 或 5ch 推理 — 与 test_land_seg.py 完全一致"""
    if model_name == SHADOW_FUSION_MODEL_NAME:
        if ndvi_norm is None or ndre_norm is None:
            return None, None, "V27融合模型需要NDVI和NDRE多光谱输入"
        from deployment_fusion import build_web_five_channel, infer_shadow_fusion

        image_5ch = build_web_five_channel(image_rgb, ndvi_norm, ndre_norm)
        predictions, _metadata = infer_shadow_fusion(image_5ch)
        return _render_shadow_fusion(image_rgb, predictions, model_name)

    import torch
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import process_mask

    model = get_model(model_name)
    if model is None:
        return None, None, "模型加载失败"

    is_5ch = "5ch" in model_name
    h0, w0 = image_rgb.shape[:2]

    if is_5ch:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        ch5 = np.stack([
            image_rgb[:, :, 0].astype(np.float32) / 255.0,
            image_rgb[:, :, 1].astype(np.float32) / 255.0,
            image_rgb[:, :, 2].astype(np.float32) / 255.0,
            ndvi_norm.astype(np.float32) if ndvi_norm is not None else np.zeros((h0, w0)),
            ndre_norm.astype(np.float32) if ndre_norm is not None else np.zeros((h0, w0)),
        ], axis=-1)
        r = imgsz / max(h0, w0)
        if r != 1.0:
            ch5 = cv2.resize(ch5, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
        input_h, input_w = ch5.shape[:2]

        # 单次推理（与 test_land_seg 一致，无TTA）
        tensor = torch.from_numpy(ch5).permute(2, 0, 1).unsqueeze(0).to(device)
        _, _, input_h, input_w = tensor.shape
        with torch.no_grad():
            outputs = model._predict_once(tensor)
        preds = outputs[0][0]
        proto = outputs[0][1] if len(outputs[0]) > 1 else None
        scale_x, scale_y = w0 / input_w, h0 / input_h
    else:
        model_obj = model.model
        model_obj.to("cpu")
        model_obj.eval()
        img_r = cv2.resize(image_rgb, (imgsz, imgsz)) if max(h0, w0) != imgsz else image_rgb
        input_h, input_w = img_r.shape[:2]
        tensor = torch.from_numpy(img_r.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            outputs = model_obj._predict_once(tensor)
        preds = outputs[0][0]
        proto = outputs[0][1] if len(outputs[0]) > 1 else None
        scale_x, scale_y = w0 / input_w, h0 / input_h

    pred = non_max_suppression(preds, conf, 0.45, nc=1)

    overlay = image_rgb.copy()
    box_count = 0
    colors = [
        (0, 255, 0), (255, 200, 0), (0, 200, 255), (200, 0, 255),
        (0, 140, 255), (255, 120, 0), (120, 255, 0), (255, 0, 100),
    ]

    results_sorted = []
    for det in pred:
        if det is None or len(det) == 0:
            continue
        has_mask = det.shape[1] > 6 and proto is not None

        if has_mask:
            masks_in = det[:, 6:].cpu()
            bboxes = det[:, :4].cpu()
            bboxes_orig = bboxes.clone()
            bboxes_orig[:, [0, 2]] *= scale_x
            bboxes_orig[:, [1, 3]] *= scale_y
            pred_masks = (process_mask(proto[0].cpu(), masks_in, bboxes_orig, (h0, w0), upsample=True) > 0.5).byte()
            confs = det[:, 4].cpu()
            for i in range(len(confs)):
                results_sorted.append({
                    "bbox": bboxes_orig[i].numpy(),
                    "score": confs[i].item(),
                    "mask": pred_masks[i].numpy(),
                })
        else:
            bboxes = det[:, :4].cpu()
            bboxes_orig = bboxes.clone()
            bboxes_orig[:, [0, 2]] *= scale_x
            bboxes_orig[:, [1, 3]] *= scale_y
            confs = det[:, 4].cpu()
            for i in range(len(confs)):
                results_sorted.append({
                    "bbox": bboxes_orig[i].numpy(),
                    "score": confs[i].item(),
                    "mask": None,
                })

    results_sorted.sort(key=lambda x: x["score"], reverse=True)

    # 渲染 — 与 test_land_seg draw_results 一致，但用中文标签
    text_labels = []
    cn_font = _get_cn_font(18)

    for i, r in enumerate(results_sorted):
        color = colors[i % len(colors)]
        mask = r.get("mask")
        if mask is not None:
            if mask.shape != (h0, w0):
                mask = cv2.resize(mask, (w0, h0), interpolation=cv2.INTER_NEAREST)
            overlay[mask > 0] = color
        else:
            bx = r["bbox"]
            x1, y1, x2, y2 = int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w0, x2), min(h0, y2)
            if x2 > x1 and y2 > y1:
                color_f = np.array(color, dtype=np.float32)
                overlay[y1:y2, x1:x2] = (overlay[y1:y2, x1:x2] * 0.5 + color_f * 0.5).astype(np.uint8)

        bx = r["bbox"]
        x1, y1, x2, y2 = int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 5)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)

        label = f"田垄 {r['score']:.2f}"
        bbox = cn_font.getbbox(label)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        margin = 10
        if y1 - th - margin >= 0:
            # 标签在框上方
            ly1, ly2 = y1 - th - margin, y1
            tx, ty = x1 + 5, y1 - th - 5
        else:
            # 标签在框内顶部
            ly1, ly2 = y1 + 2, y1 + th + margin
            tx, ty = x1 + 5, y1 + 5
        cv2.rectangle(overlay, (x1, ly1), (x1 + tw + margin, ly2), (255, 255, 255), -1)
        cv2.rectangle(overlay, (x1, ly1), (x1 + tw + margin, ly2), color, 2)
        text_labels.append((label, tx, ty, (0, 0, 0)))
        box_count += 1

    # PIL 渲染中文标签
    if text_labels:
        img_pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        for text, tx, ty, fill in text_labels:
            draw.text((tx, ty), text, font=cn_font, fill=fill)
        overlay[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # 最终混合（与 test_land_seg 一致: alpha=0.5）
    overlay = cv2.addWeighted(image_rgb, 0.5, overlay, 0.5, 0)

    if box_count == 0:
        msg = "未检测到农田目标"
        font = _get_cn_font(22)
        img_pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        bbox = draw.textbbox((0, 0), msg, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        cx, cy = w0 // 2, 30
        cv2.rectangle(overlay, (cx - tw // 2 - 12, cy - th // 2 - 8),
                      (cx + tw // 2 + 12, cy + th // 2 + 8), (0, 0, 0), -1)
        put_cn_text(overlay, msg, (cx - tw // 2, cy - th // 2), (100, 200, 255), font_size=22)

    return overlay, box_count, model_name


def process_one_image_file(file, mode, min_field_mu, gsd, model_name=None, yolo_conf=DEFAULT_YOLO_CONF, ndvi_threshold=0):
    import time
    _t0 = time.time()
    tmp_name = f"tmp_{uuid.uuid4().hex[:8]}.jpg"
    tmp_path = str(PROJECT_ROOT / tmp_name)
    file.save(tmp_path)

    raw = np.fromfile(tmp_path, dtype=np.uint8)
    img_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img_bgr is None:
        os.remove(tmp_path)
        return {"error": "无法读取图片"}
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    m = re.search(r"DJI_(\d{4})", file.filename)
    img_num = int(m.group(1)) if m else None
    print(f"  [计时] 图片加载: {time.time()-_t0:.1f}s  img_num={img_num}  size={w}x{h}")

    results = {"filename": file.filename, "img_w": w, "img_h": h, "gsd_m": gsd}
    _t1 = time.time()
    images = {"original": img_to_b64_png(img_rgb)}
    print(f"  [计时] original base64: {time.time()-_t1:.1f}s")

    # 提示非 DJI 图片
    if img_num is None:
        results["notice"] = "此图片非 DJI 航拍图，无法获取多光谱指数，仅显示原图。请上传 DJI 无人机拍摄的图片以使用植被、土壤代理和 YOLO 功能。"

    ndvi = ndre = None
    spectral_layers = {}
    raw_band_cache = {}
    if img_num is not None:
        _t2 = time.time()
        requested_indices = WEB_INDEX_NAMES + (("ExG",) if mode in ("weed", "all") else ())
        spectral_layers = compute_web_indices(img_num, h, w, requested_indices, band_cache=raw_band_cache)
        ndvi = spectral_layers.get("NDVI")
        ndre = spectral_layers.get("NDRE")
        # 后备：TIF不存在时尝试加载 _5ch.npy
        if ndvi is None:
            npy_path = TIF_DIR / f"DJI_{img_num:04d}_5ch.npy"
            if not npy_path.exists():
                npy_path = PROJECT_ROOT / "datasets" / "Data_v2" / "train" / "images" / f"DJI_{img_num:04d}_5ch.npy"
            if npy_path.exists():
                try:
                    npy = np.load(str(npy_path))
                    npy_h, npy_w = npy.shape[:2]
                    if (npy_h, npy_w) != (h, w):
                        ndvi_raw = cv2.resize(npy[:, :, 3], (w, h))
                        ndre_raw = cv2.resize(npy[:, :, 4], (w, h))
                    else:
                        ndvi_raw = npy[:, :, 3]
                        ndre_raw = npy[:, :, 4]
                    ndvi = ndvi_raw.astype(np.float32)
                    ndre = ndre_raw.astype(np.float32)
                    print(f"  [计时] NDVI从npy加载: {time.time()-_t2:.1f}s")
                except Exception as e:
                    print(f"  [警告] 加载npy失败: {e}")
        print(f"  [计时] NDVI计算: {time.time()-_t2:.1f}s")

    # === NDVI ===
    ndvi_modes = ("vegetation", "ndvi", "all", "predict")
    if ndvi is not None and mode in ndvi_modes:
        _t3 = time.time()
        images["ndvi"] = img_to_b64_png(make_ndvi_heatmap(ndvi, ndvi_threshold))
        print(f"  [计时] NDVI热力图: {time.time()-_t3:.1f}s")

    # === 连续植被指数图层 ===
    if mode in ("indices", "all") and spectral_layers:
        names = ("NDVI", "NDRE", "GNDVI", "EVI2") if mode == "indices" else ("GNDVI", "EVI2")
        for name in names:
            if name in spectral_layers:
                images[f"index_{name.lower()}"] = img_to_b64_png(make_index_heatmap(spectral_layers[name], name))
        results["vegetation_indices"] = [name for name in names if name in spectral_layers]

    # === 土壤代理图层（明确非标准 BSI） ===
    if mode in ("soil", "all") and "RE_BSI_PROXY" in spectral_layers:
        images["soil_proxy"] = img_to_b64_png(make_index_heatmap(spectral_layers["RE_BSI_PROXY"], "RE_BSI_PROXY"))
        results["soil_proxy"] = "RE_BSI_PROXY"
        results["soil_proxy_warning"] = WEB_INDEX_METADATA["RE_BSI_PROXY"]["warning"]

    # === 种植面积 ===
    veg_modes = ("vegetation", "all")
    if ndvi is not None and mode in veg_modes:
        _t4 = time.time()
        plant_data = calc_planting_area(ndvi, gsd, min_field_mu, ndvi_threshold)
        images["vegetation"] = img_to_b64_png(make_planting_overlay(img_rgb, ndvi, gsd, min_field_mu, ndvi_threshold))
        print(f"  [计时] 种植面积分析: {time.time()-_t4:.1f}s")
        total_m2 = h * w * gsd * gsd
        results.update({
            "total_m2": round(total_m2, 2),
            "total_mu": round(total_m2 / 666.67, 3),
            "plant_total_mu": plant_data["plant_total_mu"],
            "plant_pct": plant_data["plant_pct"],
            "plant_thresh": plant_data["plant_thresh"],
            "field_count": plant_data.get("field_count", 0),
            "fields": plant_data.get("fields", []),
        })

    # === 作物行约束的杂草候选（复核辅助，不作为自动施药处方） ===
    if mode in ("weed", "all"):
        if ndvi is not None and ndre is not None:
            _t_weed = time.time()
            try:
                from weed_candidate_analysis import analyze_weed_candidates

                weed = analyze_weed_candidates(
                    img_rgb, ndvi, ndre, exg=spectral_layers.get("ExG"), gsd=gsd,
                )
                images["weed_candidates"] = img_to_b64_png(weed["overlay"])
                metrics = weed["metrics"]
                results.update({
                    "weed_candidate_count": metrics["weed_candidate_count"],
                    "weed_candidate_pct": metrics["weed_candidate_pct"],
                    "weed_candidate_area_m2": metrics["weed_candidate_area_m2"],
                    "off_row_candidate_count": metrics["off_row_candidate_count"],
                    "in_row_suspect_count": metrics["in_row_suspect_count"],
                    "crop_row_angle_deg": metrics["row_angle_deg"],
                    "crop_row_confidence": metrics["row_orientation_confidence"],
                    "weed_analysis_method": metrics["method"],
                    "weed_candidate_warning": weed["warning"],
                    "weed_candidates": weed["candidates"][:50],
                })
            except Exception as exc:
                results["weed_error"] = str(exc)[:160]
            print(f"  [计时] 作物行约束杂草候选: {time.time()-_t_weed:.1f}s")
        else:
            results["weed_error"] = "杂草候选分析需要 NDVI 与 NDRE 多光谱图层"

    # === 热红外（NIR 波段） ===
    if img_num is not None:
        _t_thermal = time.time()
        thermal_img = make_thermal_view(img_num, h, w, nir=raw_band_cache.get("nir"))
        if thermal_img is not None:
            images["thermal"] = img_to_b64_png(thermal_img)
        print(f"  [计时] NIR近红外: {time.time()-_t_thermal:.1f}s")

    # === YOLO ===
    if mode in ("predict", "all") and model_name:
        needs_multispectral = model_requires_multispectral(model_name)
        if not needs_multispectral or (ndvi is not None and ndre is not None):
            _t6 = time.time()
            try:
                overlay, box_count, used_model = yolo_predict(model_name, img_rgb, ndvi, ndre, conf=yolo_conf)
                if overlay is not None:
                    images["yolo"] = img_to_b64_png(overlay)
                    results["yolo_model"] = used_model
                    results["yolo_detections"] = box_count
                    if os.environ.get("YOLO_WEB_DEBUG_OUTPUT") == "1":
                        _debug_dir = PROJECT_ROOT
                        cv2.imwrite(str(_debug_dir / f"debug_web_{file.filename.rsplit('.',1)[0]}_yolo.png"),
                                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                        print(f"  [调试] 已保存 debug_web_{file.filename.rsplit('.',1)[0]}_yolo.png")
                else:
                    results["yolo_error"] = used_model or "加载失败"
            except Exception as e:
                results["yolo_error"] = str(e)[:100]
            print(f"  [计时] YOLO推理: {time.time()-_t6:.1f}s")

    os.remove(tmp_path)
    _t7 = time.time()
    resp = {"images": images, "results": results, "status": "ok"}
    # 计算响应大小
    total_kb = sum(len(v) if isinstance(v, str) else 0 for v in images.values())
    print(f"  [计时] 总处理={time.time()-_t0:.1f}s  响应图片={total_kb/1024:.0f}KB  ({file.filename})")
    return resp


# ==================== Routes ====================

@app.after_request
def add_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/")
def index():
    return render_template("index.html", models=discover_models(), project_status=build_project_status())


@app.route("/api/models")
def api_models():
    return jsonify(discover_models())


def _read_report_json(relative_path, default=None):
    path = REPORT_DIR / relative_path
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {} if default is None else default


def build_project_status():
    final = _read_report_json("Data_v5_final_model_comparison/summary.json")
    guard = _read_report_json("Data_v5_final_candidate_test/evaluation_guard.json")
    baseline = final.get("baseline", {})
    candidate = final.get("candidate", {})
    shadow = SHADOW_DEPLOYMENT_SETTINGS or {}
    release_version = str(DEPLOYMENT_SETTINGS.get("release_version", ""))
    is_v27 = release_version.startswith("data_v27")
    blind = {}
    blind_summary = DEPLOYMENT_SETTINGS.get("release_blind_summary")
    if blind_summary:
        try:
            blind = json.loads((PROJECT_ROOT / blind_summary).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            blind = {}
    blind_metrics = blind.get("metrics", {})
    fusion_metrics = blind_metrics.get("fusion", {})
    incumbent_metrics = blind_metrics.get("incumbent_yolov8", {})
    blind_values = blind.get("values", {})
    return {
        "state": "completed" if final else "in_progress",
        "status_title": "V27 YOLO11 主导融合生产状态" if is_v27 else "Data_v5 特征优化最终结果",
        "deployment_model": PREFERRED_MODEL_NAME,
        "deployment_confidence": DEFAULT_YOLO_CONF,
        "deployment_sha256": DEPLOYMENT_SETTINGS.get("sha256"),
        "deployment_release": release_version,
        "deployment_primary": DEPLOYMENT_SETTINGS.get("primary", {}).get("architecture"),
        "deployment_auxiliary": DEPLOYMENT_SETTINGS.get("auxiliary", {}).get("architecture"),
        "shadow_model": shadow.get("model_name"),
        "shadow_state": shadow.get("state"),
        "shadow_primary": shadow.get("primary", {}).get("architecture"),
        "shadow_auxiliary": shadow.get("auxiliary", {}).get("architecture"),
        "shadow_validation_gate_passed": shadow.get("validation_gate_passed", False),
        "shadow_production_switch_allowed": shadow.get("production_switch_allowed", False),
        "shadow_every_output_requires_yolo11": shadow.get("every_output_requires_yolo11", False),
        "deployment_mask_map": None if is_v27 else baseline.get("metrics/mAP50-95(M)"),
        "deployment_metric_label": "30 张封存人工盲测融合 F1" if is_v27 else "独立测试 Mask mAP50-95",
        "deployment_metric_value": fusion_metrics.get("f1") if is_v27 else baseline.get("metrics/mAP50-95(M)"),
        "incumbent_metric_value": incumbent_metrics.get("f1") if is_v27 else None,
        "deployment_f1_delta": blind_values.get("fusion_f1_delta_vs_incumbent") if is_v27 else None,
        "release_gate_passed": blind.get("gate_passed") if is_v27 else None,
        "candidate_model": None if is_v27 else final.get("candidate_model"),
        "candidate_mask_map": None if is_v27 else candidate.get("metrics/mAP50-95(M)"),
        "candidate_delta": final.get("candidate_minus_baseline", {}).get("metrics/mAP50-95(M)"),
        "decision": final.get("decision", "keep existing website model"),
        "decision_zh": (
            "V27 已通过独立人工盲测终审并切换为 YOLO11 主导融合生产模型"
            if is_v27
            else "V20 已通过终审并切换为网站默认模型"
            if str(DEPLOYMENT_SETTINGS.get("release_version", "")).startswith("v20")
            else "生产模型保持当前冻结版本"
        ),
        "test_evaluation_count": guard.get("evaluation_count"),
        "research_note": (
            "V27 终审仅执行 1 次；盲测结果已封存，禁止用于继续调参。"
            if is_v27 else "RGB+MCARI 仅作为研究结果归档，禁止重复使用独立 test 调参。"
        ),
        "soil_proxy_warning": "RE_BSI_PROXY 是实验性 RedEdge 土壤代理，不是标准 SWIR-BSI。",
        "weed_candidate_warning": "杂草图层仅为作物行约束候选，必须人工复核，不得直接生成施药处方。",
        "reports": [{"label": label, "url": f"/reports/{path}"} for label, path in REPORT_CATALOG],
    }


@app.route("/api/project-status")
def api_project_status():
    return jsonify(build_project_status())


@app.route("/api/shadow-status")
def api_shadow_status():
    if not SHADOW_DEPLOYMENT_SETTINGS:
        return jsonify({"state": "unavailable"}), 404
    try:
        from deployment_fusion import shadow_status

        return jsonify(shadow_status())
    except Exception as exc:
        return jsonify({"state": "invalid", "error": str(exc)}), 503


@app.route("/reports/<path:filename>")
def report_file(filename):
    return send_from_directory(REPORT_DIR, filename)


@app.route("/api/extract", methods=["POST"])
def extract():
    mode = request.form.get("mode", "vegetation")
    min_field_mu = float(request.form.get("threshold", 0.005))
    ndvi_threshold = int(request.form.get("ndvi_threshold", 0))
    gsd = float(request.form.get("gsd", DEFAULT_GSD))
    model_name = request.form.get("model") or None
    yolo_conf = float(request.form.get("yolo_conf", DEFAULT_YOLO_CONF))

    files = request.files.getlist("images")
    if not files:
        f = request.files.get("image")
        if not f:
            return jsonify({"error": "未上传图片"}), 400
        files = [f]

    all_results = []
    for f in files:
        if not f.filename.lower().endswith((".jpg", ".jpeg")):
            continue
        r = process_one_image_file(f, mode, min_field_mu, gsd, model_name, yolo_conf, ndvi_threshold)
        all_results.append(r)

    if not all_results:
        return jsonify({"error": "无有效图片"}), 400

    csv_data = [r["results"] for r in all_results if r.get("status") == "ok"]
    return jsonify({
        "results": all_results,
        "csv_ready": len(csv_data) > 0,
        "mode": mode,
        "min_field_mu": min_field_mu,
        "gsd_m": gsd,
        "model_used": model_name,
    })


@app.route("/api/export_csv", methods=["POST"])
def export_csv():
    data = request.get_json()
    if not data or not isinstance(data, list):
        return jsonify({"error": "无数据"}), 400
    keys = ["filename", "img_w", "img_h", "gsd_m",
            "plant_total_mu", "plant_pct", "field_count",
            "total_m2", "total_mu",
            "yolo_model", "yolo_detections",
            "weed_candidate_count", "weed_candidate_pct", "weed_candidate_area_m2",
            "off_row_candidate_count", "in_row_suspect_count",
            "crop_row_angle_deg", "crop_row_confidence"]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=keys, extrasaction="ignore")
    writer.writeheader()
    for row in data:
        writer.writerow(row)
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode("utf-8-sig")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="extraction_results.csv",
    )


# ==================== 公网隧道 ====================

TUNNEL_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tunnel_config.json")


def load_tunnel_config():
    """读取隧道配置文件"""
    if os.path.exists(TUNNEL_CONFIG_FILE):
        with open(TUNNEL_CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_tunnel_config(config):
    """保存隧道配置文件"""
    with open(TUNNEL_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def start_sunny_tunnel(port, config):
    """启动 Sunny-Ngrok 隧道"""
    import subprocess, threading, sys
    sunny_exe = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "sunny_ngrok", "windows_amd64", "sunny.exe")
    if not os.path.exists(sunny_exe):
        return None, None
    server = config.get("server", "")
    tunnel_id = config.get("tunnel_id", "")
    if not server or not tunnel_id:
        return None, None

    result = {"url": None, "proc": None}
    def run():
        try:
            proc = subprocess.Popen(
                [sunny_exe, f"-s={server}", f"-k={tunnel_id}"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            result["proc"] = proc
            for line in proc.stdout:
                line_s = line.strip()
                # Sunny-Ngrok 输出格式: "隧道启动成功: http://xxxx.ngrok.cc"
                if "http" in line_s.lower() and "成功" in line_s:
                    for part in line_s.split():
                        if part.startswith("http"):
                            result["url"] = part
                            break
                elif "http" in line_s.lower() and "ngrok" in line_s.lower():
                    for part in line_s.split():
                        if part.startswith("http") and "ngrok" in part:
                            result["url"] = part
                            break
        except Exception:
            pass

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=12)
    return result["url"], result["proc"]


def start_natapp_tunnel(port, config):
    """启动 natapp 隧道。返回 (url, proc)。"""
    import subprocess, threading, sys

    natapp_exe = os.path.join(os.path.dirname(os.path.abspath(__file__)), "natapp.exe")
    if not os.path.exists(natapp_exe):
        print("  [隧道] natapp.exe 不存在")
        return None, None
    authtoken = config.get("authtoken", "")
    if not authtoken:
        print("  [隧道] authtoken 未配置")
        return None, None

    # 先杀掉所有旧 natapp 进程
    if sys.platform == "win32":
        os.system("taskkill /F /IM natapp.exe >nul 2>&1")
        import time; time.sleep(0.5)

    result = {"url": None, "proc": None, "done": False}

    def run():
        try:
            proc = subprocess.Popen(
                [natapp_exe, f"-authtoken={authtoken}", "-log=stdout"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            result["proc"] = proc
            for line in proc.stdout:
                line_s = line.strip()
                if not result["url"] and "natappfree.cc" in line_s:
                    for part in line_s.split():
                        if part.startswith("http") and "natappfree.cc" in part:
                            result["url"] = part
                            result["done"] = True
                            break
                if result["done"]:
                    break
        except Exception as e:
            print(f"  [隧道] 启动异常: {e}")

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=15)

    if result["url"]:
        print(f"  [隧道] 已连接: {result['url']}")
        return result["url"], result["proc"]
    else:
        print(f"  [隧道] 启动失败，请检查 authtoken 是否正确")
        if result["proc"]:
            result["proc"].kill()
        return None, None


def start_public_tunnel(port=5000):
    """根据配置文件启动公网隧道"""
    config = load_tunnel_config()
    if not config:
        return None, None

    ttype = config.get("type", "")
    if ttype == "sunny":
        return start_sunny_tunnel(port, config)
    elif ttype == "natapp":
        return start_natapp_tunnel(port, config)
    return None, None


if __name__ == "__main__":
    import socket, atexit
    os.makedirs("templates", exist_ok=True)

    # 获取局域网 IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "127.0.0.1"

    print(f"\n{'='*60}")
    print(f"  农智眼：无人机农田智能体检")
    print(f"  本机访问: http://127.0.0.1:5000")
    print(f"  局域网访问: http://{local_ip}:5000 (同 WiFi)")

    # 尝试启动公网隧道（仅在 reloader 子进程中启动，避免 debug 双进程冲突）
    is_reloader = os.environ.get("WERKZEUG_RUN_MAIN") == "true"
    tunnel_ok = []

    config = load_tunnel_config()
    if config:
        if is_reloader:
            ttype = config.get("type", "")
            print(f"  启动公网隧道 ({ttype})...")
            public_url, tunnel_proc = start_public_tunnel(5000)
            if public_url:
                atexit.register(lambda: tunnel_proc.kill() if tunnel_proc else None)
                print(f"  公网访问: {public_url}")
                print(f"  (任何设备在任何网络均可访问)")
                tunnel_ok.append(True)
            else:
                print(f"  公网隧道: 启动失败，检查 tunnel_config.json")
    else:
        if is_reloader:
            print(f"  公网隧道: 未配置")
    print(f"{'='*60}")
    if is_reloader and not tunnel_ok:
        print(f"  如需公网访问:")
        print(f"  1. 注册 https://natapp.cn 获取 authtoken")
        print(f"  2. 或注册 https://www.ngrok.cc 获取隧道ID和服务器")
        print(f"  3. 告诉我 token，我自动配置好")
    print()

    # 预加载模型，避免首次请求等 6s
    if is_reloader:
        models = discover_models()
        if models:
            print(f"  预加载模型中...")
            for m in models:
                print(f"    - {m['name']} ...", end=" ", flush=True)
                try:
                    get_model(m["name"])
                    print("OK")
                except Exception as e:
                    print(f"失败: {e}")
            print()

    app.run(host="0.0.0.0", port=5000, debug=True)
