# Third-party code and attribution

The files under `ManiSkill-src/examples/baselines/act/` are copied from the
ManiSkill ACT baseline included in the author's local sparse checkout at the
start of this experiment. The exact Git revision of the local sparse checkout was verified on
2026-10-23 as `a4a4f9272ad64b1564035874b605ceb687b63ed8`
(`git rev-parse HEAD`, with no modifications from `git status --short`).
The user subsequently reported that all 11 bundled ManiSkill baseline files
match that revision and that the bundled LICENSE matches after normalization.
Work records this as a user-performed comparison, not an independently repeated
upstream byte comparison.

Original project: https://github.com/haosulab/ManiSkill
Baseline: https://github.com/haosulab/ManiSkill/tree/main/examples/baselines/act
Original license: Apache License 2.0. See `LICENSE`. Copyright and
third-party attributions in individual source files are preserved.
Some DETR modules carry Facebook/Meta copyrights, preserved verbatim.

The bundled baseline README identifies the original ACT implementation as an
upstream source: https://github.com/tonyzhaozh/act . That project is distributed
under the MIT License, Copyright (c) 2023 Tony Z. Zhao. Its copyright and complete
permission notice are retained in `LICENSES/ACT-MIT.txt` for the ACT-derived
portions. The root Apache-2.0 license does not replace third-party notices.
License source checked on 2026-10-23:
https://github.com/tonyzhaozh/act/blob/main/LICENSE

The Facebook DETR implementation is licensed under Apache-2.0:
https://github.com/facebookresearch/detr/blob/main/LICENSE . Its copyright
headers remain in the bundled DETR-derived modules; the complete Apache-2.0
text is in the root LICENSE. The sampler reference in act/utils.py names
facebookresearch/maskrcnn-benchmark as an additional source. Its MIT License,
Copyright (c) 2018 Facebook, is retained in `LICENSES/MASKRCNN-MIT.txt` for
that sampler-derived portion. Source checked on 2026-10-23:
https://github.com/facebookresearch/maskrcnn-benchmark/blob/main/LICENSE

The scripts `train_chat.py`, `train_chat_v2.py`, `train_chat_xyz.py`,
`train_chat_act08.py` are research adaptations of the ManiSkill ACT baseline,
not independent clean-room implementations. Changes include evaluation timing,
weighted L1 channel error, an evaluation helper, and time-dependent sampling.
Please retain this attribution when distributing derivatives.

ACT research: Zhao, Kumar, Levine, Finn, "Learning Fine-Grained Bimanual
Manipulation with Low-Cost Hardware", Robotics: Science and Systems, 2023,
https://doi.org/10.15607/RSS.2023.XIX.016

ManiSkill3 research: Tao et al., "ManiSkill3: GPU Parallelized Robotics
Simulation and Rendering for Generalizable Embodied AI", RSS 2025.

Robotics assets / datasets are **not** included in this archive. Review
ManiSkill's separate asset licensing before sharing visual or dataset assets.

No claim is made that this archive has been re-run on a fresh machine.
