# YOLO 5-channel segmentation project

This project trains a YOLO segmentation model with five normalized input channels:
`[R, G, B, NDVI, NDRE]`.

## Current production release: V27

The deployed model is `v27_yolo11_primary_fusion_v1`: YOLO11n-seg is the
required primary model and YOLOv8n-seg is an equal-weight auxiliary model.
Every fused output must contain a YOLO11 source. The one-time sealed 30-image
blind evaluation reached F1 `0.5282`, versus `0.4992` for the previous YOLOv8
model. The blind result is frozen and must not be reused for tuning.

Install the deployment dependencies and start the production WSGI server:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-deployment.txt
.\.venv\Scripts\python.exe serve_web_5055.py
```

Then open `http://127.0.0.1:5055`. The weed layer is a crop-row-constrained
review aid, not a validated weed classifier or an automatic spraying
prescription. Raw datasets, multispectral imagery, experimental runs and local
credentials are intentionally excluded from the repository.

## Common commands

Build missing five-channel files and repair inconsistent files:

```powershell
python prepare_5ch.py --repair
```

Audit label quality and generate an HTML report:

```powershell
python label_quality_audit.py datasets/Data_v2 --output reports/Data_v2_label_quality
python label_quality_audit.py datasets/Data_v3 --output reports/Data_v3_label_quality
```

Generate focused crops for the P1 geometry review queue:

```powershell
python geometry_review.py
```

Start the local correction UI:

```powershell
python label_review_app.py
```

For image-level difficult samples selected from train/validation error analysis,
build and open the whole-image correction queue:

```powershell
python build_difficult_sample_correction.py
python difficult_sample_correction_app.py --port 5057
```

The whole-image editor supports adding, redrawing, and deleting polygons. Drafts
are persisted on the server (including an unfinished contour), geometric issues
are highlighted in purple, and final saves are synchronized to both Data_v3 and
the canonical source label with an undo record. This workflow must only use
train/validation queues; the frozen test split is not part of correction or
feature selection.

Create a dry-run plan for strictly mask-preserving geometry repairs:

```powershell
python repair_label_geometry.py
```

Rebuild Data_v3 with adjacent flight frames kept in the same split:

```powershell
python merge_and_train.py
```

Train the five-channel segmentation model:

```powershell
.\.venv\Scripts\python.exe train_ca_5ch.py
```

The default experiment uses YOLO11n-seg, CoordAtt + ECA, mean initialization
for the NDVI/NDRE kernels, and the `spectral_aug` profile. This profile applies
color jitter only to RGB and small gain/bias/noise perturbations only to the
two vegetation-index channels. Validation data is never photometrically
augmented.

Verify the model and real Data_v3 pipeline before a long run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
.\.venv\Scripts\python.exe train_ca_5ch.py --profile spectral_aug --model-size n --verify-only
.\.venv\Scripts\python.exe verify_5ch_pipeline.py --profile spectral_aug --imgsz 640 --batch 2
```

Run a one-epoch smoke test first:

```powershell
.\.venv\Scripts\python.exe train_ca_5ch.py `
  --profile spectral_aug `
  --model-size n `
  --spectral-init mean `
  --smoke-test
```

`--smoke-test` keeps the requested image size, batch size, and complete train
split, but limits the run to one epoch, disables RAM caching, and uses zero
data-loader workers. This makes it useful for detecting an OOM at the real
training resolution without changing the normal full-dataset label cache. Use
`--no-cache` for a full run when RAM is limited, and `--fraction 0.25` only for
a deliberate partial-data experiment.

Recommended controlled experiments use the same split, image size, batch size,
and seed. Start with these YOLO11n comparisons:

```powershell
# Previous augmentation behavior (control)
.\.venv\Scripts\python.exe train_ca_5ch.py --profile baseline --model-size n --spectral-init kaiming --seed 0

# New channel-safe augmentation
.\.venv\Scripts\python.exe train_ca_5ch.py --profile spectral_aug --model-size n --spectral-init mean --seed 0

# Single-variable test: baseline geometry + mild NDVI/NDRE-only augmentation
.\.venv\Scripts\python.exe train_ca_5ch.py --profile spectral_only_mild --model-size n --spectral-init kaiming --seed 0

# Attention ablation
.\.venv\Scripts\python.exe train_ca_5ch.py --profile spectral_aug --model-size n --attention none --spectral-init mean --seed 0
```

After selecting the better configuration, repeat it with `--seed 0`,
`--seed 1`, and `--seed 2`. YOLO11s can be tested with `--model-size s`
after `yolo11s-seg.pt` is available; its default batch size is 1 on the
current 6 GB GPU. Deterministic training is enabled by default to improve
repeatability; PyTorch may still warn about CUDA operations that do not have a
fully deterministic implementation. Use `--no-deterministic` when maximum
throughput is more important than repeatable comparisons.

Validate the current checkpoint with the strict five-channel loader:

```powershell
python validate_5ch.py
```

Select samples for active learning:

```powershell
python active_learning_select.py
```

## Reports

- `reports/Data_v2_label_quality/index.html`
- `reports/Data_v3_label_quality/index.html`
- `reports/Data_v3_label_quality/review_queue.csv`
- `reports/Data_v3_geometry_review/index.html`

The audit is read-only. Geometry warnings such as self-intersections, duplicate
vertices, and tiny polygons require visual review before changing labels.

## Data_v4 spectral-factor outcome

Data_v4 adds a configurable 17-channel candidate store and evaluates raw
spectral bands, vegetation indices, soil-adjusted indices, and an experimental
RedEdge bare-soil proxy. Feature selection uses train/valid only; the held-out
test split is evaluated once after the final candidate is frozen.

The final `RGB + RE_BSI_PROXY` candidate did not outperform the deployed model
on the independent test set (Mask mAP50-95 `0.3774` versus `0.3886`). The web
viewer therefore continues to recommend `exp_ca_5ch_grouped_reviewed_v1`.
`RE_BSI_PROXY` is not standard SWIR-based BSI and must always be described as
an experimental proxy.

Start the updated local site:

```powershell
.\.venv\Scripts\python.exe serve_web.py --host 127.0.0.1 --port 5055
```

The site exposes NDVI, NDRE, GNDVI, EVI2, and RE-BSI explanatory layers plus
links to all Data_v4 reports. Data_v4 research checkpoints are intentionally
hidden from the deployment model selector.

Final evidence and keep/discard decisions are summarized in:

- `reports/Data_v4_final_summary.md`
- `reports/Data_v4_feature_factor_analysis/index.html`
- `reports/Data_v4_yolo_group_ablation/index.html`
- `reports/Data_v4_yolo_single_feature_ablation/index.html`
- `reports/Data_v4_yolo_multiseed_validation/index.html`
- `reports/Data_v4_final_model_comparison/index.html`

## Data_v5 compact feature screening

Data_v5 keeps the deployed model unchanged and starts a new train/valid-only
candidate track. It adds six complementary spectral indices (`VARI`, `ARVI`,
`SIPI`, `PSRI`, `MCARI`, `MTCI`) plus two spatial features (`NDVI_GRAD`,
`NDVI_STD5`). Statistical screening promotes only `MCARI`, `SIPI`, `VARI`, and
`NDVI_STD5` to YOLO ablation; the other four are retained in the evidence
store but removed from the training queue.

- `configs/data_v5_channels.json`
- `configs/feature_ablation_groups_v5.json`
- `reports/Data_v5_feature_factor_analysis/index.html`
- `reports/Data_v5_optimization_plan.md`

Data_v5 is now complete. The formal `RGB + MCARI` candidate stopped early at
epoch 218 (best epoch 160, validation Mask mAP50-95 `0.3740`). It was then
evaluated exactly once on the frozen independent test set. Its test Mask
mAP50-95 was `0.3782`, below the deployed model's `0.3886`; precision, recall,
Mask mAP50, and Box mAP50-95 also declined. The candidate is therefore archived
as a research result and must not be evaluated on test again for tuning.

The website continues to recommend `exp_ca_5ch_grouped_reviewed_v1` with
`R,G,B,NDVI,NDRE`. All `v4_*` and `v5_*` research checkpoints are hidden from
the model selector to prevent incompatible-channel inference.

Final Data_v5 evidence:

- `reports/Data_v5_final_summary.md`
- `reports/Data_v5_yolo_stage1_ablation/index.html`
- `reports/Data_v5_yolo_multiseed_validation/index.html`
- `reports/Data_v5_yolo_combo_validation/index.html`
- `reports/Data_v5_yolo_highres_screen/index.html`
- `reports/Data_v5_final_training/index.html`
- `reports/Data_v5_final_model_comparison/index.html`

## Data_v6 terrain-factor track

Data_v6 is an optional train/valid-only research track for images that have a
real pixel-aligned DEM, DSM, or DTM. RGB texture and multispectral reflectance
must not be treated as elevation. The terrain pipeline adds encoded slope,
aspect sine/cosine, TRI, TPI, and local relief. Absolute elevation is excluded
to reduce flight/site leakage.

Prepare candidates after aligned elevation rasters are available (one
`train/DJI_xxxx.npy|tif` or `valid/DJI_xxxx.npy|tif` per image):

```powershell
# Read-only coverage/alignment check; this must reach 100% before preparation.
.\.venv\Scripts\python.exe audit_terrain_sources.py `
  --terrain-root D:\path\to\aligned_elevation `
  --pixel-size-m 0.05

.\.venv\Scripts\python.exe prepare_terrain_multichannel.py `
  --terrain-root D:\path\to\aligned_elevation `
  --pixel-size-m 0.05
```

Run a terrain-channel verification with either YOLO11 or YOLOv8 segmentation:

```powershell
# YOLO11n + slope
.\.venv\Scripts\python.exe train_feature_ablation.py `
  --group slope_6ch `
  --groups-config configs\feature_ablation_groups_v6_terrain.json `
  --manifest datasets\Data_v6_terrain_candidates\manifest.json `
  --feature-root datasets\Data_v6_terrain_candidates `
  --architecture yolo11 --model-size n --verify-only

# YOLOv8n + slope (downloads yolov8n-seg.pt on first use if absent)
.\.venv\Scripts\python.exe train_feature_ablation.py `
  --group slope_6ch `
  --groups-config configs\feature_ablation_groups_v6_terrain.json `
  --manifest datasets\Data_v6_terrain_candidates\manifest.json `
  --feature-root datasets\Data_v6_terrain_candidates `
  --architecture yolov8 --model-size n --verify-only
```

Supported pretrained segmentation families are YOLO11n/s, YOLOv8n/s,
YOLO12n/s, YOLO26n/s, and YOLOv9c. Use `--base-model`, for example
`--base-model yolo12n-seg.pt`. All reuse the RGB kernels and initialize added
spectral/terrain kernels explicitly. YOLO11, YOLOv8, YOLO12, and YOLO26 can use
the project CoordAtt+ECA insertion at compatible C2f/C3k2 backbone blocks.
YOLOv9c must use `--attention none`; it is a much larger fallback for batch-1
experiments on the current 6 GB GPU. Compare each model independently on the
same grouped train/valid split before prediction-level fusion. The frozen test
split remains sealed until one final candidate is selected.

Prediction-level fusion is implemented in `yolo_mask_ensemble.py`. It matches
same-class instances by mask IoU, confidence-weights the aligned masks, keeps
same-model instances separate, and can require agreement from two models.
This allows, for example, the deployed YOLO11 5-channel model and a validated
YOLOv8 terrain model to be combined without averaging incompatible weights.

After the Data_v6 manifest exists, run the paired default screen (YOLO11n,
YOLOv8n, YOLO12n, and YOLO26n; current 5ch versus slope 6ch) with:

```powershell
.\.venv\Scripts\python.exe run_terrain_model_screen.py
```

The runner writes resumable progress to
`logs/v6_terrain_model_screen.status.json`. It never exposes the test split.
After completion it automatically generates
`reports/Data_v6_terrain_model_screen/index.html`, pairs every terrain result
with the same architecture's 5-channel control, and promotes only gains of at
least 0.3 Mask mAP50-95 percentage points.

Run paired three-seed stability validation for the automatically promoted
candidate with:

```powershell
.\.venv\Scripts\python.exe run_terrain_multiseed.py
```

This reuses completed seed-0 runs, trains missing 5-channel/candidate pairs for
seeds 1 and 2, and writes `reports/Data_v6_terrain_multiseed/index.html`.
Fusion is allowed only when the mean gain is at least 0.3 pp, at least two of
three seeds are positive, and no seed regresses by more than 0.3 pp.

For a candidate that passes multi-seed validation, export original-resolution
valid predictions from the two frozen component checkpoints with
`export_multichannel_predictions.py`. Each export stores compressed instance
masks and a valid-only provenance manifest. Then calibrate late fusion with:

```powershell
.\.venv\Scripts\python.exe calibrate_yolo_ensemble.py `
  --prediction current5=reports\predictions_current5 `
  --prediction terrain=reports\predictions_terrain
```

Calibration searches model weights, mask-matching IoU, vote threshold,
confidence threshold, and one-model/two-model agreement using valid labels.
The selected parameters are saved as
`reports/Data_v6_yolo_ensemble_calibration/locked_fusion_config.json`; test is
not exposed during export or calibration.

### Pre-terrain YOLOv8 control

A 30-epoch YOLOv8n `R,G,B,NDVI,NDRE` control completed on train/valid at 640
resolution (batch 4, seed 0). Independent validation Mask mAP50-95 is 0.29257.
This is about 0.32 pp above the paired short YOLO11 screen but 7.65 pp below
the long-run deployed model, so it is retained only as a research/future-fusion
candidate. See `reports/Data_v6_preterrain_yolov8n_current5_valid/README.md`.
No test images were read. Terrain training remains gated on real aligned
DEM/DSM/DTM data, and final training remains gated on completing the
high-priority label review queue.

## Data_v7–v29 research track

Everything after the terrain gate stayed inside a train/valid-only protocol: the
grouped split keeps adjacent flight frames in the same split, the training
overlap mask stays disabled, and the frozen test split and the one-shot release
blind holdout are never read during selection. The full narrative, the per-image
metrics and the annotated production-effect figures are in
`reports/Project_progress_report/index.html`.

| Stage | Goal | Outcome | Evidence |
| --- | --- | --- | --- |
| V1–V3 | five-channel data build, label audit, geometry review | read-only audit plus a correction workflow established; test split sealed | `reports/Data_v3_label_quality/` |
| V4–V5 | spectral factor screening (17-channel candidate store, compact screening) | `RGB + RE_BSI_PROXY` and `RGB + MCARI` both stayed below the deployed model and were archived as research results | `reports/Data_v5_final_summary.md` |
| V6–V8 | terrain factors, teacher-refined labels, dual-branch multispectral fusion | terrain training remains gated on real aligned DEM/DSM/DTM; teacher refinement and fusion screening became the later data base | `reports/Data_v8_yolo11_paired_multiseed/` |
| V9–V18 | grouped cross-validation, boundary masks, standardization, threshold robustness | fixed the protocol: group by flight segment, keep the training overlap mask off | `reports/Data_v13_overlap_precision_analysis/` |
| V19 | YOLO11n boundary training without overlap masks, multi-seed | passed the pre-registered gate and was frozen: seed-0 Mask mAP50-95 `0.39596`, sha256 `eea619ab…` | `reports/Data_v19_final_report/README.md` |
| V22–V27 | fusion matrix, full-resolution fusion, recall recovery, production switch | locked 1:1 strict YOLO11-primary fusion; blind F1 `0.5282` versus `0.4992` for the previous model, then promoted | `logs/v27_production_promotion.summary.json` |
| V28 (rejected) | train-only hard-example mining: 4 hard-negative crops plus 8 hard-positive replays | 60 epochs, best epoch 32; fusion F1 `0.6225` (−0.0199) and recall `−0.0690`, so the gate failed | `logs/v28_yolo11_hard_examples.summary.json` |
| V29 (rejected) | recall priority: low-learning-rate fine-tune from the frozen V19 weights plus train-only false-negative weighted sampling | 12 epochs, best epoch 1; fusion F1 `0.6301` (−0.0123) and recall `−0.0238`, so the gate failed | `logs/v29_yolo11_recall_finetune.summary.json` |

### Deployment decision log

The deployed model is still `v27_yolo11_primary_fusion_v1`. Both rejected runs
recorded `production_deployment_unchanged = true`, and the deployment config
hash is identical before and after each one. V28 bought precision (`+0.0395`) by
giving up recall (`−0.0690`); V29 never improved on its first epoch. Two
consecutive rejections mean the 60-image validation split is exhausted for
hyper-parameter, sampling and weighting variants: further gains have to come
from label quality, hard-example definition or genuinely new data.

### Label density audit (Data_v30)

V28 and V29 were both motivated by the same guess: the SAM2-assisted batch has
about twice the polygons per image of the hand-labelled batch (79.3 versus 38.3
pooled over train+valid), so its densely labelled scenes were suspected of
inflating the false negatives. `build_data_v30_label_density_audit.py` tests
that guess with three indicators that were fixed before the real data was read,
using the hand-labelled batch as the control.

The result rejects the hypothesis:

- **No over-segmentation.** The SAM2 batch has *fewer* mutually overlapping
  polygons (overlap share `0.090` versus `0.108`) and *fewer* tiling contacts
  (`0.108` versus `0.157`) than the hand-labelled batch, and its union-to-sum
  area ratio is closer to 1 (`0.92` versus `0.85`). Its polygons are smaller
  (median equivalent radius `23.8` versus `35.4` grid units) but they are
  spatially disjoint, so instances were not split into touching or duplicated
  pieces. Decision: `real_density`.
- **Recall does not degrade with density.** On the 60 validation images the
  Spearman correlation between polygon count and frozen-production FN rate is
  `+0.026`, and FN *rate* is flat across density terciles (`0.369`, `0.320`,
  `0.364`). Absolute FN grows with instance count because there are simply more
  instances to miss — the per-instance recall ceiling is uniform, so the
  recall-priority route V29 took had nothing to recover.
- **The label geometry warnings sit in the hand-labelled batch, not in the SAM2
  batch.** Re-running `label_quality_audit.py` on train+valid leaves 328 warnings
  across 136 of 270 images, dominated by 302 `self_intersection` rows, and 326 of
  those warnings come from the 180 hand-labelled images (1.81 per image) against
  19 from the 90 SAM2 images (0.21 per image).

Two consequences for the next round: label merging is not justified, and neither
is another sampling or weighting variant on the same split. The open items are
the 302 unresolved self-intersections concentrated in the hand-labelled labels
(the review queue still lists 91 train `P3-review` and 21 train `P2-geometry`
images), cross-region generalization, and the delivery form of the result.

To reproduce the audit (train and valid only; `test` is not an accepted value):

```powershell
.\.venv\Scripts\python.exe build_data_v30_label_density_audit.py
.\.venv\Scripts\python.exe label_quality_audit.py datasets/Data_v3 `
  --output reports/Data_v3_label_quality_trainvalid_v30 --splits train valid
```

Only one acceptance CSV (`reports/Data_v3_geometry_review/geometry_review.csv`,
36 items) survives for the seven review state files, so merging the available
acceptances removes just 17 of the 345 raw warning rows. A merged acceptance
index is worth rebuilding before the next audit is treated as a queue.

### Evidence index

- `reports/Project_progress_report/index.html` – production replay, per-image
  metrics, annotated effect figures and the V1→V29 timeline.
- `reports/Data_v19_final_report/README.md` – frozen primary checkpoint, fixed
  threshold slice audit, and the `sparse_low_coverage` regression list.
- `reports/Data_v30_label_density_audit/summary.json` – the density decision, its
  pre-registered thresholds and the valid recall linkage.
- `reports/Data_v3_label_quality_trainvalid_v30_reviewed/summary.json` –
  refreshed train/valid label audit with the surviving review acceptances merged.
- `logs/v27_release_blind_final.summary.json` – the one-shot sealed blind
  evaluation. It is frozen evidence and must not be reused for tuning.
- `logs/v28_*.summary.json`, `logs/v29_*.summary.json` – rejected-experiment
  protocols, gates and decisions (tracked in Git on purpose).
- `reports/Data_v27_yolo11_fusion_recall_recovery/locked_fusion_config.json` –
  the locked fusion parameters referenced by `configs/deployment.json`.
