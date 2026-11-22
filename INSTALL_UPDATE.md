# Versioned fine-tuning workflow

The experiment-folder update is installed in this repository. No file copying is needed.

1. Keep the fixed data/splits.csv manifest and verified SIDD/PolyU data in place.
2. Open notebooks/04_finetune_new_version.ipynb. Run smoke mode first, then use a GPU runtime for full training. Each run writes to a new experiments/<run-id>/ folder with checkpoints, logs, and optional test summaries.
3. The notebook starts every method from a published checkpoint unless INITIALIZATION is changed. Changing the starting checkpoint restarts the optimizer; it does not resume optimizer state.
4. For a fully evaluated experiment, start the app with its checkpoint folder:
   - Windows: scripts/run_app.ps1 -CheckpointDir experiments/<run-id>/checkpoints
   - macOS/Linux: ./scripts/run_app.sh experiments/<run-id>/checkpoints
5. The launcher requires all three model files. The app checks checkpoint hashes against full summaries in that experiment folder before making a PolyU-based benchmark recommendation.

The current submitted baseline run and report are in HeraldFolder/. The new experiment workflow does not replace them. A one-step DnCNN smoke run verified that the notebook's output_dir path writes a checkpoint and training log.

Run project checks after changes:

    .\.venv\Scripts\python.exe -m compileall -q app methods scripts
    .\.venv\Scripts\python.exe -m pytest -q
