# Checklist BEFORE making a GitHub repository public

This is a **pre-publication candidate**, not an authorization to upload.

- [ ] Review every tracked file for personal names, corporate secrets,
      credentials, absolute paths, source-code provenance and licenses.
- [x] Record the local ManiSkill source Git commit in `THIRD_PARTY_NOTICES.md`.
- [ ] Confirm the copied ACT baseline files match the pinned upstream commit and attribution obligations.
- [ ] Ensure source authors' notices are retained; confirm upstream licenses
      for all bundled `act` / DETR code.
- [ ] Confirm ACT-06 historical training entrypoint and all real CLI options.
- [ ] Verify 100 demonstrations and HDF5/JSON SHA-256 privately.
- [x] User compared ACT-08 `4000.pt` and `best_eval_success_at_end.pt` EMA tensors
      and `norm_stats` on the original PC (2026-10-23); both matched. Bundled
      `tools/verify_act08_4k_checkpoint.py` itself remains untested.
- [ ] Confirm checkpoint SHA-256 and consider appropriate redistribution
      rights before uploading model files separately.
- [x] On the reported Ubuntu PC, reaggregate CSV and rerun seed=2024 inference using the existing environment and model (see `REPRO_VERIFICATION_20261023.md`).
- [ ] Fresh environment installation, model distribution, training and full 50-condition re-evaluation remain unverified.
- [ ] Do not claim 50-seed results are an untouched independent final test
      after selecting between 4,000- and 5,000-iteration checkpoints.
- [ ] Resolve the video's “100 successful demonstrations” claim against the
      actual demonstration metadata, or replace with “100 demonstrations”.
- [x] Local video: H.264/1080p/30fps, 127.67 s, 0.5x playback label for ten-panel insert (see `VIDEO_QC_20261023.md`).
- [ ] Confirm images and videos do not display account details/hostnames (manual full-frame review).
- [ ] Add public YouTube URL when ready, and check it can be viewed.
- [ ] Make the repo public only after the review; submit URL + PDF + video.

**Current evaluation:** 50 seeds (2000–2049), ACT-07 3/50 final successes;
ACT-08 4k 17/50; ACT-08 5k 18/50. ACT-08 5k success once 38/50.
No repeated training-seed study: causality / generalization not established.
