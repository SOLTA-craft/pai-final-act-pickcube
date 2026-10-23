# Reproduction verification — 2026-10-23 (local machine)

This is a **record of commands/results supplied from the user's Ubuntu PC**, not an independent clean-room audit or newly trained model. The experimental workspace was not modified; the release candidate was expanded to a separate directory.

## Tested

- Candidate archive `PAI_ACT_PickCube_GitHub_PREPUBLIC_20261023.zip` was extracted under `~/pai_final/github_repro_check_20261023/`.
- `python3 tools/summarize_results.py`: success; CSV totals matched the reference figures: ACT-06 0/30; ACT-07 3/30 & 3/50; ACT-08 4k 13/30 & 17/50; ACT-08 5k 13/30 & 18/50, once 38/50.
- The original ACT-08 5k checkpoint was **symlinked from the existing private experiment directory** to `runs/act_pickcube_train_08_phase/checkpoints/5000.pt`. No weights were copied into the public candidate ZIP.
- In the candidate directory with existing `.venv-maniskill` Python 3.11.16 and `PYTHONPATH` including bundled ACT baseline: `python inspect_act08_seed2024.py`.
- Actual output: 150 steps, max lift **188.14 mm**, nearest target distance **5.81 mm**, first success **Step 68**, last success **Step 150**, success_at_end **True**. Recorded success was continuous for **83 rows** Step 68…150 inclusive.
- `uv pip freeze --python .venv-maniskill/bin/python` completed for 111 installed packages; the complete list was saved locally as `~/pai_final/github_repro_check_20261023/environment/requirements-actual.txt`, **not included here** (contents not independently audited beyond a narrow grep search).
- Git source commit `a4a4f9272ad64b1564035874b605ceb687b63ed8`, `git status --short` empty in the local ManiSkill sparse-checkout.

## ACT-08 historical 4k/best checkpoint comparison (user log, 2026-10-23)

- Local files: `4000.pt` and `best_eval_success_at_end.pt` from the ACT-08 training run.
- `ema_agent` key structure matched, and **every EMA tensor was exactly equal** (`torch.equal`: True).
- `norm_stats` was `None` for **both** files.
- The user-reported comparison ended with **PASS**.
- In conjunction with the saved 4,000-iteration best-update log, this supports
  identifying the evaluated `best_eval_success_at_end.pt` EMA model as the 4k model.
- It does not show that the full checkpoint files are byte-for-byte identical,
  and does not establish model hashes; original model files were not shared here.
- The comparison was run by the user using an inline Python snippet, not by
  executing the separately bundled `tools/verify_act08_4k_checkpoint.py`.

## Verified Python package versions

| Package | Version |
|---|---|
| Python | 3.11.16 |
| uv | 0.12.9 |
| torch | 2.7.1+cu128 |
| torchvision | 0.22.1+cu128 |
| ManiSkill | 3.0.1 |
| numpy | 1.26.4 |
| opencv-python | 4.11.0.86 |
| diffusers | 0.33.1 |
| tensorboard | 2.21.0 |

GPU: NVIDIA GeForce RTX 5060 Laptop GPU; torch.cuda.is_available() True, torch.version.cuda 12.8.

## Important limits

- This **does not** prove that a third party can run from the code ZIP alone: the checkpoint is not distributed and the existing Python environment was reused.
- A fresh venv installation, 100-demo generation/replay, complete training cycle, and 50-seed re-evaluation have **not** been reproduced from the public package.
- The 111-line dependency freeze file has not been supplied to this artifact-generation session. It must be inspected in full before any optional publication.
- Model and dataset SHA-256 hashes have not been supplied or independently verified.
- Final video / dataset / checkpoint provenance should be cross-checked during publication.
