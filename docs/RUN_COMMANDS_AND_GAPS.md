# Execution commands and remaining provenance checks

**IMPORTANT:** The review ZIP did NOT include the actual historical shell
commands, trained weights, demonstration HDF5 or JSON metadata. The examples
below document the script interface, **not proven historical commands**.
Do not describe them in a paper as an exact log of the completed training.

## Dataset

The historical training log references this relative path:

`data/PickCube-v1/motionplanning/act_pickcube_100.state.pd_ee_delta_pos.physx_cpu.h5`

The corresponding JSON is required by the training script (same basename with
`.json` extension). Do not place actual HDF5 data in GitHub by default.

An example from ManiSkill's own documentation for motion planning is:

```bash
python -m mani_skill.examples.motionplanning.panda.run -e PickCube-v1 --save-video
```

For state / EE delta action conversion, see the official trajectory-replay CLI
for the installed ManiSkill version, and use `--use-first-env-state`,
`--obs-mode state`, `--target-control-mode pd_ee_delta_pos`,
`--sim-backend physx_cpu`, `--save-traj` as appropriate. **The original
100-demo generation command was not available for verification.**

## Training script mapping

- ACT-06: `train_chat.py` and `train_chat_v2.py` are both present in the
  collected research files; which actually produced the final ACT-06 weight
  file must be cross-checked with saved TensorBoard hyperparameters / history.
- ACT-07: `train_chat_xyz.py` contains L1 channel weights 5, 5, 2, 1.
- ACT-08: `train_chat_act08.py` additionally adds time-weighted sampling.

The expected original checkpoint locations in the evaluation scripts are:

```
runs/act_pickcube_train_06/checkpoints/5000.pt
runs/act_pickcube_train_07_xyz/checkpoints/5000.pt
runs/act_pickcube_train_08_phase/checkpoints/5000.pt
runs/act_pickcube_train_08_phase/checkpoints/best_eval_success_at_end.pt
```

Those files are not in the public code ZIP. The evaluator loads the
`ema_agent` state from each checkpoint. No checkpoint hashes have been verified.

### Training command TEMPLATE (not an original command)

From the repository root, after installing the dependencies and placing the
compatible demo file (and JSON) at the path listed above:

```bash
export PYTHONPATH="$PWD/ManiSkill-src/examples/baselines/act:$PWD"

# Use a fresh run name; this is a TEMPLATE to inspect/adapt, not a record of
# the real training. Check --help for the correct tyro boolean flag format.
python train_chat_act08.py \
  --env-id PickCube-v1 \
  --demo-path data/PickCube-v1/motionplanning/act_pickcube_100.state.pd_ee_delta_pos.physx_cpu.h5 \
  --control-mode pd_ee_delta_pos --sim-backend physx_cpu \
  --num-demos 100 --max-episode-steps 150 \
  --total-iters 5001 --num-queries 30 \
  --batch-size 128 --num-eval-envs 1 --num-eval-episodes 5 \
  --eval-freq 1000 --save-freq 1000 \
  --no-temporal-agg --no-capture-video \
  --exp-name example_ACT08_NEWRUN
```

**Caution:** batch size and flags above are illustrative, NOT verified from
training runs. The code's defaults are NOT evidence of actual runtime values.

### Re-evaluating historical checkpoints

1. Place the four checkpoint files into the exact relative paths shown above.
2. Ensure the runtime uses ManiSkill 3.0.1 and expected CPU physics.
3. CSV files are installed under `reference_results/`; evaluation scripts
   create new files under `results/` and REFUSE to overwrite existing ones.

```bash
export PYTHONPATH="$PWD/ManiSkill-src/examples/baselines/act:$PWD"
python compare_act_06_07_5000.py
python compare_act_07_08_30seeds.py
python compare_act_07_08_new50seeds.py
```

Note the seed ranges: 1000–1029 and 2000–2049, 150 environment steps,
action chunk refresh every 30 steps, `temporal_agg=False` in the comparison
scripts. Re-running these requires real weights / simulation dependencies.

## Outstanding for independent reproduction

- Exact executed training commands / parameters for 06–08, especially ACT-06
  `train_chat.py` versus `train_chat_v2.py`.
- ManiSkill source commit was identified as `a4a4f9272ad64b1564035874b605ceb687b63ed8` on 2026-10-23 (verify bundled source matches).
- Original 100-demo metadata (`.json`), integrity and hashes of `.h5`.
- Checkpoint SHA256 remains unverified. Runtime environment versions and a local 111-package `uv pip freeze` were recorded on 2026-10-23, but the complete freeze file was not provided for this release.
- Local seed=2024 runtime re-execution succeeded using existing Python environment and model; behavior on a fresh machine remains unverified.
- Final YouTube link and GitHub repository URL after publishing.

## ACT-08 4k / best checkpoint provenance (user-verified EMA match)

Training log `reference_results/training_logs/act_pickcube_train_08_phase.txt`
shows that `success_at_end` reached a new best value of `0.8` at the evaluation
that immediately precedes `Iteration 4000, loss: ...`; the next evaluation,
immediately preceding `Iteration 5000`, yielded `0.4` and did not save best.
`train_chat_act08.py` evaluates before logging the iteration number, and
`save_ckpt(run_name, f"best_eval_{k}")` saves at a new best. This supports
identifying the recorded best checkpoint as being saved at iteration 4000.
On 2026-10-23, the user compared the current two local checkpoint files on the original
PC. Their `ema_agent` tensor keys and all tensors were exactly equal, and both
`norm_stats` entries were `None` (PASS). This supports the historical 4k-to-best
model mapping alongside the training log. It does **not** establish byte-level
equality or equality of every checkpoint field. The actual checkpoint files and
hashes are not present in this public code archive. The separate
`tools/verify_act08_4k_checkpoint.py` is provided for future rechecks but was not
itself run during the user-reported comparison.
