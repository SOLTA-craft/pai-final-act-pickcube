# Files manifest and verification

`SOURCE_SHA256SUMS.txt` records SHA-256 hashes of each file in this candidate archive (except the manifest itself). Historical research Python sources and evaluation CSVs were retained byte-for-byte from the earlier review package; **documentation was edited for publication-readiness**.

To validate the extracted archive:

```bash
sha256sum --check SOURCE_SHA256SUMS.txt
```

This file-integrity check does NOT authenticate checkpoints or demonstrations; neither is included in the archive.
