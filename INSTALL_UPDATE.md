# Applying the iMAGE experiment-folder update

Copy the files in this patch into the matching locations in your existing `CCS250_CompVis_Final` project, replacing the three `app/*.py` files and adding the notebook, launch scripts, and README. Keep your current `app/main.py`, `app/data.py`, frontend, dataset, and checkpoints.

Before replacing files, commit or back up your current working tree. Then run:

```bash
python -m compileall app
python -c "from app.experiments import setup_device, dataset_counts; print(setup_device()); print(dataset_counts())"
```

The second command requires the 5,000-pair manifest and image dataset to be present. If your current checkout uses different manifest source labels or split quotas, stop and reconcile before training.

Windows: `./scripts/run_app.ps1`
macOS: `chmod +x scripts/run_app.sh && ./scripts/run_app.sh`

The launch scripts default to `weights/`. To use an experiment, pass its `checkpoints` directory. They require all three model files before starting.
