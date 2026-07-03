import sys
import os
import torch
import sys, os, time
from pathlib import Path
from pymatgen.io.cif import CifWriter

sys.path.insert(0, '/srv/scratch/ml4matdis/Quanli_Project/z5561341/projectA/cdvae')

import importlib, pathlib
for pyc in pathlib.Path('/srv/scratch/ml4matdis/Quanli_Project/z5561341/projectA/cdvae').rglob('*.pyc'):
    pyc.unlink()
os.chdir('/srv/scratch/ml4matdis/Quanli_Project/z5561341/projectA/cdvae')

from cdvae.pl_modules.model import WyckoffCDVAE
from cdvae.pl_data.wyckoff_utils import wyckoff_to_structure, W2S_PATH_COUNTER

CKPT = '/srv/scratch/ml4matdis/Quanli_Project/z5561341/cdvae_outputs/hydra/singlerun/2026-06-27/test_wyckoff/epoch=144-step=15369.ckpt'

model = WyckoffCDVAE.load_from_checkpoint(CKPT, map_location='cpu')
model.eval()

# 只生成100个
t0 = time.time()
structs = model.generate(num_samples=100)
t1 = time.time()

print(f'\n=== 结果 ===')
print(f'100个结构耗时: {t1-t0:.1f}s ({(t1-t0)/100:.2f}s/struct)')
print(f'有效: {sum(s is not None for s in structs)}/100')
w2s_report()
print(f'\n推算 2500 个需要: {(t1-t0)/100*2500/3600:.1f}h')