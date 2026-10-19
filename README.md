# fMRI Emotion Recognition (CLIP-NeuroPredict)

Code for **fMRI Emotion Recognition**, a two-stage framework that (1) decodes facial emotions from ROI-level fMRI activity and (2) predicts voxel responses to emotional-face images using CLIP features.

## 1. Environment

Python **3.10** is recommended. A CUDA-enabled GPU is recommended for CLIP encoding; ROI decoding can run on CPU.

```bash
conda create -n fmri-emotion python=3.10 -y
conda activate fmri-emotion

# Install PyTorch appropriate for your system: https://pytorch.org/get-started/locally/
pip install torch torchvision
pip install numpy pandas scipy scikit-learn nibabel nilearn openpyxl h5py pillow tqdm matplotlib statsmodels joblib openai
pip install git+https://github.com/openai/CLIP.git
```

Run the commands below **from the folder containing `p_PCA.py` and `train_clip_single_roi_fdr_v2.py`** (in the supplied archive: `Supplementary_coding/code/`). No `requirements.txt` or command-line argument parser is included; paths and experiment settings are configured near the top of each script.

## 2. Dataset and preparation

**Natural Scenes Dataset (NSD)** — [Official website and access agreement](https://naturalscenesdataset.org/) · [Data manual](https://naturalscenesdataset.org/NSD_Data_Manual_v1.5.pdf) · [AWS Open Data](https://registry.opendata.aws/nsd/).

1. Complete the **NSD Data Access Agreement** on the official website before accessing the data.
2. Download the **MNI-space `betas_fithrf`** volumes, NSD stimulus images, and stimulus/experiment metadata. The MNI-space betas are appropriate for the MNI-space HCPex atlas used by these scripts.
3. Obtain the [HCPex atlas](https://github.com/wayalan/HCPex) (`HCPex_2mm.nii` and `HCPex_LookUpTable.txt`), then run `HCP_360_TO_22.py` after updating `HCPex_DIR` to generate `HCPex_22Systems.nii` and `HCPex_22Systems_Labels.csv`.

Example AWS CLI downloads (subject 01; repeat for other subjects):

```bash
# Install AWS CLI first: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
aws s3 cp s3://natural-scenes-dataset/nsddata_betas/ppdata/subj01/MNI/betas_fithrf/ \
  data/nsd_betas/sub01/betas_fithrf/ --recursive --exclude '*' --include 'betas_session*.nii.gz' --no-sign-request

aws s3 cp s3://natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5 \
  data/nsd_stimuli.hdf5 --no-sign-request
aws s3 cp s3://natural-scenes-dataset/nsddata/experiments/nsd/nsd_stim_info_merged.csv \
  data/nsd_stim_info_merged.csv --no-sign-request
```

The repository **does not include the complete NSD data or the study-specific facial-emotion annotations**. Prepare the following inputs before running the experiments:

```text
data/
├── nsd_betas/sub01/betas_fithrf/betas_session01.nii.gz  # ...sub01–sub08
├── fmri/emotion/sub01_output.xlsx                       # ...sub01–sub08
├── fmri/emotion/coco(nsd)/000000000123_face_0.jpg       # example cropped face
├── HCPex/HCPex_v1.0/HCPex_22Systems.nii
└── HCPex/HCPex_v1.0/HCPex_22Systems_Labels.csv
```

Each `subXX_output.xlsx` must contain **`orderid`** (zero-based chronological trial index; 750 trials/session), **`cocoid`** (corresponding image ID), and **`predicted_emotion`** (`happy`, `sad`, `angry`, `fear`, `neutral`, or `surprise`). These labels/face crops must be prepared separately: they are **not provided as ready-to-use labels or crops by NSD**. The CLIP script expects images named `<12-digit-cocoid>_face_0.jpg` (also accepts `.jpeg`, `.png`, and `.webp`). Maintain correct trial-to-image alignment.

## 3. Quick reproduction

**Before running:** update the hard-coded paths in `batch_extract.py`, `p_PCA.py`, `emotion_extract.py`, and `train_clip_single_roi_fdr_v2.py` (`cfg` **and** `cfg.update`). The example layout above places `data/` under the working directory; use paths consistent with your machine. In particular, change:

- `batch_extract.py`: `BASE_DATA_DIR`, `BASE_NII_DIR`, `OUTPUT_DIR` (set the last two to `data/nsd_betas` and `data/extract_fmri`).
- `p_PCA.py` / `emotion_extract.py`: `FMRI_DIR`, label directory, HCPex atlas/labels paths, and output directories.
- `train_clip_single_roi_fdr_v2.py`: `image_dir`, `fmri_path_pattern`, `output_dir`, and the atlas/label/significant-ROI paths in `cfg.update`.

### Step 1 — Extract emotion-related fMRI trials

```bash
python batch_extract.py
```

Output: `data/extract_fmri/sub01_emotion_trials.nii.gz` (and corresponding files for the remaining subjects, if their inputs are available).

### Step 2 — Decode emotions and select ROIs

```bash
python p_PCA.py
```

Runs **SVM / DT / RF / MLP** on **3-class and 6-class** tasks with ROI-level cross-validation and permutation testing. Main outputs include `GROUP_ALL_3class.csv`, `GROUP_ALL_6class.csv`, and `GROUP_LEVEL_FISHER_FDR.csv` in the configured `OUTPUT_DIR` (e.g., `data/decoding_selection/all_model/`). This full run uses **10-fold CV and 3,000 permutations** and can take considerable time.

**ROI-list adapter:** The subsequent scripts require `data/emotion_decoding_3c/GROUP_LEVEL_PERMUTATION_UAR.csv` with a column named `ROI`, whereas `p_PCA.py` writes `GROUP_LEVEL_FISHER_FDR.csv`. To use the *shared significant SVM/MLP ROIs from the script's 3-class Fisher-FDR summary*, run:

```bash
python - <<'PY'
import os
import pandas as pd
f = pd.read_csv('data/decoding_selection/all_model/GROUP_LEVEL_FISHER_FDR.csv')
f = f[(f['Task'] == '3Class') & (f['Model'].isin(['SVM', 'MLP'])) &
      (f['Significant'].astype(str).str.lower() == 'true')]
common = f.groupby('ROI_Name')['Model'].nunique()
os.makedirs('data/emotion_decoding_3c', exist_ok=True)
pd.DataFrame({'ROI': common[common == 2].index}).to_csv(
    'data/emotion_decoding_3c/GROUP_LEVEL_PERMUTATION_UAR.csv', index=False)
PY
```

*Note:* This adapter matches the **file format** required downstream. Fisher-combined per-subject p-values are **not identical** to the paper's group-level sign-reversal testing; exact statistical reproduction requires matching the paper's selection procedure.

### Step 3 — Export emotion-conditioned ROI responses

```bash
python emotion_extract.py
```

Output: `data/12_ROI_emotion/sub01_positive_fmri.npz`, `sub01_negative_fmri.npz`, `sub01_neutral_fmri.npz`, etc. Each file contains `fmri`, `image_ids`, `cocoids`, and ROI metadata.

### Step 4 — Train/evaluate CLIP-to-fMRI encoding

```bash
python train_clip_single_roi_fdr_v2.py
```

In the script's `cfg`, set `backbone` to **`RN50`** or **`ViT-B/32`**. `finetune_stage1=False` uses the pretrained CLIP model (the current default); set it to **`True`** to run emotion-aware image–text fine-tuning first. Fine-tuning requires a caption CSV with `image_id` and `caption` columns, plus the corresponding face images; `LLM_Push.py` is an optional caption-generation utility that requires a separately configured API endpoint/model.

The encoding stage fits voxel-wise ridge readouts and evaluates held-out predictions, including voxel-level significance and FDR correction. Main outputs under `cfg['output_dir']`:

- `summary_roi_subject_level.csv`
- `summary_roi_cross_subject_by_emotion.csv`
- `summary_roi_overall.csv`
- Per-ROI `y_pred.npy`, `y_true.npy`, and `voxelwise_stats_with_fdr.csv`