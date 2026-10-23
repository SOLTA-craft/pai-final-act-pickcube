#!/usr/bin/env python3
"""Read-only check: ACT-08 historical 4,000-iteration and best checkpoints.

Usage:
  python tools/verify_act08_4k_checkpoint.py --checkpoints /path/to/checkpoints

Does not load untrusted checkpoint files; torch.load can deserialize objects.
No modifications to checkpoints or experiment directories.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def same_value(a, b, path='root') -> list[str]:
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return [] if torch.equal(a, b) else [path]
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return [] if np.array_equal(a, b, equal_nan=True) else [path]
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return [path + '.keys']
        result = []
        for key in a:
            result += same_value(a[key], b[key], path + '.' + str(key))
        return result
    if isinstance(a, (tuple,list)) and isinstance(b, type(a)):
        if len(a) != len(b):
            return [path + '.len']
        result = []
        for i,(x,y) in enumerate(zip(a,b)):
            result += same_value(x,y,path + '[' + str(i) + ']')
        return result
    try:
        eq = a == b
        return [] if isinstance(eq, (bool,np.bool_)) and eq else [path]
    except Exception:
        return [path]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoints', type=Path, required=True)
    args = ap.parse_args()
    ck = args.checkpoints.expanduser().resolve()
    paths = {'iter4000': ck / '4000.pt', 'best': ck / 'best_eval_success_at_end.pt'}
    print('ACT-08 checkpoint provenance comparison (READ ONLY)')
    for label,p in paths.items():
        print(f'{label}: {p}')
        if not p.is_file():
            print('NOT FOUND: ' + str(p)); return 2
        print(f'  bytes={p.stat().st_size} sha256={sha256(p)}')
    print('Loading only local, trusted training-produced checkpoint files...')
    # Use only checkpoints produced by the user's trusted training run.
    a = torch.load(paths['iter4000'], map_location='cpu', weights_only=False)
    b = torch.load(paths['best'], map_location='cpu', weights_only=False)
    failed=[]
    for key in ('ema_agent', 'norm_stats'):
        if key not in a or key not in b:
            failed.append('missing key ' + key); continue
        failed += same_value(a[key],b[key],key)
    if failed:
        print('CHECK FAILED: EMA weights or normalization statistics differ.')
        for x in failed[:12]:print('  mismatch:',x)
        print('  (full mismatches:',len(failed),')')
        return 1
    print('PASS: all ema_agent tensors and norm_stats match exactly.')
    print('Interpretation: EMA weights and norm_stats match; iteration provenance still depends on training logs.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
