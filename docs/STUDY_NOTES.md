# Evaluation details and interpretation

- `PickCube-v1`, `state`, `pd_ee_delta_pos`, `physx_cpu`.
- Evaluation lasts at most 150 steps in the local comparison scripts.
- A new action chunk is computed every 30 steps, no temporal ensembling.
- The checkpoint `ema_agent` is used.
- `success_once`: succeeded at least once at any intermediate evaluation step.
- `success_at_end`: success state at episode end (primary reported metric).
- The scripts retrieve success flags from `info["episode"]`; the ManiSkill
  3.0.1 internal success function was not separately reviewed in this package.
- ACT-07 `compute_loss`: `(abs(action-pred)*[5,5,2,1]).mean()` + KL term.
  It does NOT divide by sum(weights). This changes the L1/KL scale too.
- ACT-08: `WeightedRandomSampler(replacement=True)` with 2.5x on starting
  timestep `t/(L-1)` within `[0.35,0.70)` of demonstration length; this is
  NOT a claim that all such steps are grasping actions.
- 30-seed set was used in development, 50-seed set in comparison and choice
  between 4k and 5k; results do NOT imply a universal success probability.
- ACT-08 5k: success once 38/50 but at end 18/50. 20 episodes no longer
  satisfied success at end; cause not established by aggregate CSV alone.
- Four preserved successful traces: seeds 2003, 2013, 2016, 2024.
- Checkpoints and demonstration data are not part of this code package.
