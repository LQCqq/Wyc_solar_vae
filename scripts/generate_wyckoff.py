import sys
import os
import torch
from pathlib import Path
from pymatgen.io.cif import CifWriter

sys.path.insert(0, '/srv/scratch/ml4matdis/Quanli_Project/z5561341/projectA/cdvae')

# 强制清除 pyc 缓存
import importlib, pathlib
for pyc in pathlib.Path('/srv/scratch/ml4matdis/Quanli_Project/z5561341/projectA/cdvae').rglob('*.pyc'):
    pyc.unlink()
os.chdir('/srv/scratch/ml4matdis/Quanli_Project/z5561341/projectA/cdvae')

CKPT = '/srv/scratch/ml4matdis/Quanli_Project/z5561341/cdvae_outputs/hydra/singlerun/2026-06-07/test_wyckoff/epoch=89-step=9539.ckpt'
NUM_SAMPLES = 1000
OUT_DIR = '/srv/scratch/ml4matdis/Quanli_Project/z5561341/generated_structures'

os.makedirs(OUT_DIR, exist_ok=True)

from cdvae.pl_modules.model import WyckoffCDVAE

print('加载模型...')
model = WyckoffCDVAE.load_from_checkpoint(CKPT)
model.eval()
# 强制 CPU（避免 H200 sm_90 兼容性问题）
print('使用 CPU')

print(f'生成 {NUM_SAMPLES} 个结构...')
structures = model.generate(num_samples=NUM_SAMPLES)

valid = [(i, s) for i, s in enumerate(structures) if s is not None]
print(f'有效结构: {len(valid)}/{NUM_SAMPLES}')

print('保存 CIF 文件...')
for i, struct in valid:
    try:
        cif_path = os.path.join(OUT_DIR, f'struct_{i:04d}.cif')
        CifWriter(struct).write_file(cif_path)
    except Exception as e:
        print(f'保存失败 {i}: {e}')

print(f'完成！保存到 {OUT_DIR}')
print(f'有效率: {len(valid)/NUM_SAMPLES*100:.1f}%')
