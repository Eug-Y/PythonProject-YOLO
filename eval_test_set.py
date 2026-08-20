"""
Test set evaluation for CoordAtt+ECA 5ch YOLO model.
Computes Box mAP and Mask mAP on 10 held-out test images.
"""
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import process_mask, scale_masks, segments2boxes
from ultralytics.nn.tasks import SegmentationModel, load_checkpoint
from custom_modules import CoordAtt, ECA


TEST_DIR = r"D:\PythonProject-YOLO\datasets\Data_v2\test"
CKPT_PATH = r"D:\PythonProject\runs\segment\land_seg\exp_ca_5ch_v11\weights\best.pt"
DEVICE = "cuda:0"
IMGSZ = 1024
CONF = 0.001  # low conf for mAP
IOU_THRESH = 0.45


# ============================================================================
# Model building (same as train_ca_5ch.py)
# ============================================================================
def _get_output_channels(layer):
    if hasattr(layer, "out_channels"):
        return layer.out_channels
    if hasattr(layer, "cv2") and hasattr(layer.cv2, "conv"):
        return layer.cv2.conv.out_channels
    if hasattr(layer, "cv3") and hasattr(layer.cv3, "conv"):
        return layer.cv3.conv.out_channels
    if hasattr(layer, "conv"):
        return layer.conv.out_channels
    return None


def build_ca_model_from_base(detection_model):
    old_layers = list(detection_model.model)
    backbone_end = 7
    new_layers = []
    old_to_new = {}

    for oi, layer in enumerate(old_layers):
        old_to_new[oi] = len(new_layers)
        new_layers.append(layer)

        name = layer.__class__.__name__
        if "C3k2" in name and oi < backbone_end:
            ch = _get_output_channels(layer)
            if ch:
                ca = CoordAtt(ch, ch)
                ca.f = -1
                ca.i = -1
                new_layers.append(ca)
                eca = ECA(ch, k_size=3)
                eca.f = -1
                eca.i = -1
                new_layers.append(eca)

    # Fix f/i indices
    for i, layer in enumerate(new_layers):
        if hasattr(layer, "f"):
            f = layer.f
            if isinstance(f, (int, np.integer)):
                if f != -1:
                    layer.f = old_to_new.get(int(f), int(f))
            else:
                new_f = []
                for idx in f:
                    if idx == -1:
                        new_f.append(-1)
                    else:
                        new_f.append(old_to_new.get(int(idx), int(idx)))
                layer.f = new_f
        layer.i = i

    detection_model.model = nn.Sequential(*new_layers)

    save = set()
    for layer in new_layers:
        f = getattr(layer, "f", -1)
        if f == -1:
            continue
        indices = [f] if isinstance(f, (int, np.integer)) else f
        for idx in indices:
            if idx != -1:
                save.add(int(idx))
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

    # Load trained weights
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"].float().state_dict())
    model.to(device)
    model.eval()
    return model


# ============================================================================
# Label loading
# ============================================================================
def load_label(label_path, h, w):
    """Load YOLO segmentation label, return bboxes (xyxy normalized), cls, mask (H,W)."""
    if not os.path.exists(label_path):
        return None, None, None

    with open(label_path) as f:
        lines = [x.strip() for x in f.readlines() if x.strip()]

    segments = []
    for line in lines:
        parts = list(map(float, line.split()))
        cls = int(parts[0])
        pts = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)  # normalized x,y
        segments.append((cls, pts))

    if not segments:
        return None, None, None

    # Build mask from polygon
    gt_mask = np.zeros((h, w), dtype=np.uint8)
    gt_cls = []
    gt_boxes = []
    for inst_id, (cls, pts) in enumerate(segments, 1):
        pts_abs = (pts * np.array([w, h])).astype(np.int32)
        cv2.fillPoly(gt_mask, [pts_abs], inst_id)
        gt_cls.append(cls)
        # bbox from polygon points
        x = pts[:, 0]
        y = pts[:, 1]
        gt_boxes.append([x.min(), y.min(), x.max(), y.max()])

    gt_boxes = np.array(gt_boxes, dtype=np.float32)  # normalized xyxy
    gt_cls = np.array(gt_cls, dtype=np.int32)
    gt_mask_t = torch.from_numpy(gt_mask).long()
    return gt_boxes, gt_cls, gt_mask_t


# ============================================================================
# Inference on a single image
# ============================================================================
def infer_one(model, npy_path, device):
    """Run inference on a 5ch npy file. Returns (pred_boxes, pred_scores, pred_cls, pred_masks)."""
    im = np.load(npy_path, allow_pickle=True).astype(np.float32)
    if im.max() > 1.0:
        im /= 255.0
    h0, w0 = im.shape[:2]

    r = IMGSZ / max(h0, w0)
    if r != 1.0:
        new_w, new_h = int(w0 * r), int(h0 * r)
        im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    tensor = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0).to(device)
    _, _, input_h, input_w = tensor.shape  # model input dimensions

    with torch.no_grad():
        outputs = model._predict_once(tensor)
    preds_tensor = outputs[0][0]  # [1, 37, N]
    proto = outputs[0][1]         # [1, 32, H/4, W/4]

    pred = non_max_suppression(preds_tensor, CONF, IOU_THRESH, nc=1)

    # Scaling factors: NMS boxes are in input coordinate space
    scale_x = w0 / input_w
    scale_y = h0 / input_h

    boxes = []
    scores = []
    classes = []
    masks = []

    for det in pred:
        if det is None or len(det) == 0:
            continue
        if det.shape[1] > 6:
            masks_in = det[:, 6:].cpu()
            bboxes = det[:, :4].cpu()  # in input coordinate space
            conf = det[:, 4].cpu()
            cls = det[:, 5].cpu()

            # Scale bboxes to original image coordinates before process_mask
            bboxes_orig = bboxes.clone()
            bboxes_orig[:, [0, 2]] *= scale_x
            bboxes_orig[:, [1, 3]] *= scale_y

            # process_mask needs bboxes in the shape coordinate space
            pred_masks = process_mask(proto[0].cpu(), masks_in, bboxes_orig,
                                      (h0, w0), upsample=True)
            pred_masks = (pred_masks > 0.5).byte()

            boxes.append(bboxes_orig)
            scores.append(conf)
            classes.append(cls)
            masks.append(pred_masks)

    if boxes:
        boxes = torch.cat(boxes, dim=0)
        scores = torch.cat(scores, dim=0)
        classes = torch.cat(classes, dim=0)
        masks = torch.cat(masks, dim=0)
    else:
        boxes = torch.zeros((0, 4))
        scores = torch.zeros(0)
        classes = torch.zeros(0)
        masks = torch.zeros((0, h0, w0), dtype=torch.uint8)

    return boxes, scores, classes, masks, (h0, w0)


# ============================================================================
# Metrics computation (using ultralytics built-in)
# ============================================================================
from ultralytics.utils.metrics import ap_per_class


def mask_iou(gt_masks, pred_masks):
    """Compute pairwise mask IoU. gt_masks: (N, H*W), pred_masks: (M, H*W).
    Returns (N, M) tensor."""
    intersection = torch.matmul(gt_masks.float(), pred_masks.float().T).clamp_(0)
    union = (gt_masks.sum(1)[:, None] + pred_masks.sum(1)[None]) - intersection
    return intersection / (union + 1e-7)


def match_predictions(pred_classes, true_classes, iou, iouv):
    """Replicate ultralytics match_predictions.
    iou: (N_gt, N_pred) tensor."""
    correct = np.zeros((pred_classes.shape[0], len(iouv)), dtype=bool)
    correct_class = true_classes[:, None] == pred_classes  # (N_gt, N_pred)
    iou = iou * correct_class  # zero out wrong classes
    iou_np = iou.cpu().numpy()
    for i, threshold in enumerate(iouv.tolist()):
        matches = np.nonzero(iou_np >= threshold)
        matches = np.array(matches).T  # (M, 2), col0=gt_idx, col1=pred_idx
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou_np[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct


# ============================================================================
# Main evaluation
# ============================================================================
def evaluate():
    print("=" * 60)
    print("Test Set Evaluation — exp_ca_5ch_v11")
    print("=" * 60)

    model = load_model(CKPT_PATH, DEVICE)
    print("Model loaded.")

    img_dir = os.path.join(TEST_DIR, "images")
    label_dir = os.path.join(TEST_DIR, "labels")

    all_tp_box = []
    all_conf_box = []
    all_pred_cls = []
    all_gt_cls_box = []

    all_tp_mask = []
    all_conf_mask = []
    all_pred_cls_m = []
    all_gt_cls_mask = []

    iouv = torch.linspace(0.5, 0.95, 10)

    test_files = sorted([f for f in os.listdir(img_dir) if f.endswith("_5ch.npy")])
    print(f"Found {len(test_files)} test images.\n")

    for fname in test_files:
        base = fname.replace("_5ch.npy", "")
        npy_path = os.path.join(img_dir, fname)
        label_path = os.path.join(label_dir, base + ".txt")

        print(f"  Processing: {base}.JPG ...", end=" ")

        # Load ground truth (get image size from npy)
        im_sample = np.load(npy_path, allow_pickle=True)
        h_img, w_img = im_sample.shape[:2]
        gt_boxes_norm, gt_cls, gt_mask_t = load_label(label_path, h_img, w_img)
        if gt_boxes_norm is None:
            print("no GT labels, skipping")
            continue

        # Inference
        pred_boxes, pred_scores, pred_cls, pred_masks, (h0, w0) = infer_one(model, npy_path, DEVICE)

        n_gt = len(gt_cls)
        n_pred = len(pred_scores)
        print(f"{n_gt} GT, {n_pred} pred")

        if n_pred == 0 and n_gt == 0:
            continue

        # === Box mAP ===
        if n_pred == 0:
            all_gt_cls_box.extend(gt_cls.tolist())
            continue
        if n_gt == 0:
            all_conf_box.extend(pred_scores.tolist())
            all_pred_cls.extend(pred_cls.tolist())
            # all false positives
            fp_box = np.zeros((n_pred, 10), dtype=bool)
            all_tp_box.append(fp_box)
            continue

        # Compute box IoU (both in absolute original coordinates)
        from torchvision.ops import box_iou as torch_box_iou
        gt_boxes_abs = torch.from_numpy(gt_boxes_norm).clone()
        gt_boxes_abs[:, [0, 2]] *= w0
        gt_boxes_abs[:, [1, 3]] *= h0

        iou_box = torch_box_iou(gt_boxes_abs, pred_boxes)  # (N_gt, N_pred)
        tp_box = match_predictions(pred_cls, torch.from_numpy(gt_cls), iou_box, iouv)

        all_tp_box.append(tp_box)
        all_conf_box.extend(pred_scores.tolist())
        all_pred_cls.extend(pred_cls.tolist())
        all_gt_cls_box.extend(gt_cls.tolist())

        # === Mask mAP ===
        if n_pred == 0:
            all_gt_cls_mask.extend(gt_cls.tolist())
            continue

        # Compute mask IoU at original resolution
        gt_masks_bin = torch.stack([
            (gt_mask_t == (i + 1)).flatten() for i in range(n_gt)
        ]).bool()  # (N_gt, H*W)
        pred_masks_flat = pred_masks.reshape(n_pred, -1).bool()  # (N_pred, H*W)

        iou_mask = mask_iou(gt_masks_bin, pred_masks_flat)  # (N_gt, N_pred)
        tp_mask = match_predictions(pred_cls, torch.from_numpy(gt_cls), iou_mask, iouv)

        all_tp_mask.append(tp_mask)
        all_conf_mask.extend(pred_scores.tolist())
        all_pred_cls_m.extend(pred_cls.tolist())
        all_gt_cls_mask.extend(gt_cls.tolist())

    # === Aggregate metrics ===
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    def compute_map(tp_list, conf_list, pred_cls_list, gt_cls_list):
        if not tp_list:
            return {"mAP50": 0.0, "mAP50-95": 0.0, "P": 0.0, "R": 0.0}

        tp = np.concatenate(tp_list, axis=0)
        conf = np.array(conf_list)
        pred_cls_arr = np.array(pred_cls_list, dtype=np.int32)
        target_cls = np.array(gt_cls_list, dtype=np.int32)

        if len(conf) == 0 or len(target_cls) == 0:
            return {"mAP50": 0.0, "mAP50-95": 0.0, "P": 0.0, "R": 0.0}

        tp_c, fp_c, p_arr, r_arr, f1_arr, ap_arr, _, _, _, _, _, _ = \
            ap_per_class(tp, conf, pred_cls_arr, target_cls)

        ap50 = ap_arr[:, 0].mean() if len(ap_arr) else 0.0
        ap50_95 = ap_arr.mean() if len(ap_arr) else 0.0

        return {"mAP50": ap50, "mAP50-95": ap50_95, "P": p_arr.mean(), "R": r_arr.mean()}

    box_metrics = compute_map(all_tp_box, all_conf_box, all_pred_cls, all_gt_cls_box)
    mask_metrics = compute_map(all_tp_mask, all_conf_mask, all_pred_cls_m, all_gt_cls_mask)

    print(f"  Box  — P={box_metrics['P']:.4f}  R={box_metrics['R']:.4f}  "
          f"mAP50={box_metrics['mAP50']:.4f}  mAP50-95={box_metrics['mAP50-95']:.4f}")
    print(f"  Mask — P={mask_metrics['P']:.4f}  R={mask_metrics['R']:.4f}  "
          f"mAP50={mask_metrics['mAP50']:.4f}  mAP50-95={mask_metrics['mAP50-95']:.4f}")

    print(f"\n  Total GT instances: {len(all_gt_cls_box)}")
    print(f"  Total predictions:  {len(all_conf_box)}")

    return box_metrics, mask_metrics


if __name__ == "__main__":
    evaluate()
