# iMAGE ? Image Denoiser

FastAPI website with preview filters and real inference using Restormer, DnCNN, and NAFNet. Three Jupyter notebooks prepare paired data, fine-tune the models, and evaluate/deploy them.

## Setup (Windows PowerShell)

Use Python 3.11 with the VS Code Python and Jupyter extensions. Open this project folder in VS Code, then run from its root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

For GPU training, install a CUDA-enabled PyTorch build using the [official PyTorch selector](https://pytorch.org/get-started/locally/). Select `.venv` as the notebook kernel. A CPU can run previews and small smoke tests; full Restormer/NAFNet training is computationally expensive.

## Website

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000. The default `preview` mode uses three labeled demonstration filters. To run actual models, restart with `IMAGE_MODE=real`:

```powershell
$env:IMAGE_MODE="real"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Real mode retrieves model architecture files and pretrained checkpoints on first use, then caches them under `cache/`. Internet access, disk space, and trust in upstream code are required. Restormer and DnCNN baseline checkpoints target Gaussian noise; the NAFNet baseline targets SIDD camera noise. Their baseline scores therefore do not constitute a matched training comparison. Local `weights/{model}.pth` checkpoints take precedence for inference.

## Paired real-noise experiment

The current experiment uses **5,000 aligned noisy/clean pairs**: 3,500 SIDD and 1,500 PolyU. The fixed quotas are 4,000 training (2,800 SIDD, 1,200 PolyU), 500 validation (350, 150), and 500 test (350, 150). No COCO images or synthetic Gaussian noise are used. Splits are assigned by original scene group to avoid patch leakage.

1. Obtain SIDD and PolyU paired data under their respective terms. Extract the paired subsets under `data/raw/`, then open [notebook 01](notebooks/01_prepare_5000_images.ipynb). Set the SIDD/PolyU directories to the actual `GT` (clean) and `LQ` (noisy) folders. Verify filename matching, alignment, and scene IDs. The notebook writes `data/inventory/SIDD.csv` and `PolyU.csv`, then uses `scripts/prepare_mixed_dataset.py` to write `data/splits.csv` once. Inventory paths are relative to `data/`. The generator uses a greedy scene allocator and fails without writing if it cannot fill the exact quotas. That failure does not prove no valid allocation exists; review group definitions and inventory ordering before changing the fixed split seed. Keep the manifest unchanged across all model runs.
2. Open [notebook 02](notebooks/02_finetune_models.ipynb). Start in `RUN_MODE="smoke"` with DnCNN, then use `"full"`; repeat for NAFNet and Restormer. Each full run starts from a pretrained checkpoint and saves `weights/{model}_finetuned.pth`, training CSV logs, and plots. Smoke checkpoints overwrite the same filenames, so complete full training before final evaluation. Use validation results for model choices; reserve test pairs for the final comparison.
3. Open [notebook 03](notebooks/03_evaluate_and_deploy.ipynb). Start in smoke mode, then switch to `RUN_MODE="full"` to evaluate all three fine-tuned checkpoints on the same 500 test pairs. `app.experiments.evaluate_model` evaluates **256 px center crops by default**, with tiled model inference. PSNR/SSIM are patch-based, not full-resolution image metrics. Outputs are `results/{model}_test.csv`, `results/{model}_summary.json`, and a comparison plot. Review the full results, then set `DEPLOY_FINETUNED=True` if you want to copy checkpoints to `weights/{model}.pth`. Restart the real-mode server afterward.

The API recommends a model from test results only when all three deployed checkpoints match full evaluation summaries under the same manifest and settings. For a single upload with a clean reference, it can compute metrics for that upload. Without a clean reference or matching benchmark, it makes no quality claim.

## Project layout

- `app/`: API, paired-image loader, training/evaluation methods, inference, model/cache adapters.
- `notebooks/`: one notebook per preparation, fine-tuning, and evaluation step.
- `scripts/prepare_mixed_dataset.py`: paired inventory validation and scene-group split generation.
- `data/raw/`, `data/inventory/`, `data/splits.csv`: local data, excluded from Git.
- `cache/`, `weights/`, `results/`: downloaded sources/weights and local outputs, excluded from Git.
- `tests/`: API, notebook, and dataset workflow checks.

## Limits and provenance

The repository does not include the datasets, manifest, or trained checkpoints. This checkout cannot be fully trained or evaluated until paired data and a compatible Python environment are available. Inventory validation checks image readability and dimensions; inspect representative pairs to confirm spatial alignment and the meaning of each scene ID. Exact group quotas can also be impossible for a given inventory.

Online source files come from the [Restormer](https://github.com/swz30/Restormer), [KAIR/DnCNN](https://github.com/cszn/KAIR), and [NAFNet](https://github.com/megvii-research/NAFNet) projects; the NAFNet checkpoint is fetched from a [third-party mirror](https://huggingface.co/mikestealth/nafnet-models). The mirror checkpoint is SHA-256 checked by the loader. Real model downloads and checkpoint compatibility still need verification in the target environment. Report dataset provenance, grouping policy, checkpoint provenance, center-crop size, device, and measured results when publishing comparisons.
