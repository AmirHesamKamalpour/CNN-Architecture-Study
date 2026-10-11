# Experimental record and provenance

The CSVs in this directory preserve the measurements recorded in the supplied project archive.

- `mnist_model_comparison.csv`: four MNIST model results, with absolute local Windows paths replaced by portable checkpoint filenames. Values were not recomputed or revised.
- `table_*.history.csv`, `mnist_res_inception_best.history.csv`: epoch histories **recovered from the best saved PyTorch checkpoints**. Each file stops at its selected best-validation-loss epoch and therefore may not contain the full original training run. Do not interpret these as full-run logs.
- `fashion_head_only_best.history.csv`: restored **all 30 epochs** from the Fashion-MNIST checkpoint. The CSV in the original archive stopped at epoch 29, while the checkpoint contains the complete history and identifies epoch 30 as the best epoch.
- `fashion_transfer_summary.csv`: Fashion-MNIST test measurement with portable checkpoint filenames.

**Verification performed:** The four MNIST state dictionaries loaded into their corresponding model classes without missing or unexpected keys; their parameter counts and forward tensor shapes agreed with the code. The supplied checkpoint metadata agrees with the recorded best validation losses and epochs. Since the original `mnist.pkl.gz` was not present, the archived MNIST test metrics were **not independently re-evaluated**; they are reported as saved experimental results. The same restriction applies to the Fashion-MNIST test accuracy.

The original source used one reported run per architecture; no multi-seed estimates, statistical significance claims, or scratch-trained Fashion-MNIST control are provided.
