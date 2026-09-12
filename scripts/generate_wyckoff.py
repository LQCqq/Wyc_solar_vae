import sys
import os
import argparse
import torch
from pathlib import Path
from pymatgen.io.cif import CifWriter
from pymatgen.core import Element

sys.path.insert(0, '/srv/scratch/ml4matdis/Quanli_Project/z5561341/projectA/cdvae')

import pathlib
for pyc in pathlib.Path('/srv/scratch/ml4matdis/Quanli_Project/z5561341/projectA/cdvae').rglob('*.pyc'):
    pyc.unlink()
os.chdir('/srv/scratch/ml4matdis/Quanli_Project/z5561341/projectA/cdvae')

from cdvae.pl_modules.model import WyckoffCDVAE
from cdvae.pl_data.wyckoff_utils import w2s_report


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str,
                   default='/srv/scratch/ml4matdis/Quanli_Project/z5561341/cdvae_outputs/hydra/singlerun/2026-09-07/wyckoff_new/epoch=178-step=18974.ckpt')
    p.add_argument('--out_dir', type=str,
                   default='/srv/scratch/ml4matdis/Quanli_Project/z5561341/generated_structures/charge_refactor_structure')
    p.add_argument('--num_samples', type=int, default=3500)
    p.add_argument('--target_elements', type=str, default=None,
                   help='目标元素,逗号分隔,如 "S,Se,Te"。不传=无条件生成')
    p.add_argument('--cfg_w', type=float, default=0.0,
                   help='联合CFG强度：0=普通条件生成，>0增强元素/Ehull条件')
    p.add_argument('--stability_class', type=int, default=None,
                   choices=[0, 1, 2, 3],
                   help=('Ehull条件类别；不传=unconditional。默认分箱：'
                         '0:Ehull<=0, 1:(0,0.01], '
                         '2:(0.01,0.03], 3:>0.03 eV/atom'))
    p.add_argument('--must_contain', type=str, default='off',
                   choices=['off', 'any', 'all'],
                   help='硬过滤: off=不过滤; any=含目标元素至少一个; all=全含')
    return p.parse_args()


def build_elem_cond(target_elements):
    if not target_elements:
        return None, set()
    zs = set()
    mh = torch.zeros(1, 100)
    for sym in target_elements.split(','):
        sym = sym.strip()
        if not sym:
            continue
        z = Element(sym).Z
        mh[0, z - 1] = 1.0
        zs.add(z)
    return mh, zs


def structure_element_zs(struct):
    return {e.Z for e in struct.composition.elements}


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print('加载模型...')
    model = WyckoffCDVAE.load_from_checkpoint(args.ckpt, map_location='cpu')
    model.eval()

    elem_cond, target_zs = build_elem_cond(args.target_elements)
    stability_cond = None
    if args.stability_class is not None:
        stability_cond = torch.tensor(
            [args.stability_class], dtype=torch.long
        )
    if elem_cond is not None:
        print(f'目标元素: {args.target_elements} (Z={sorted(target_zs)})  '
              f'cfg_w={args.cfg_w}  must_contain={args.must_contain}')
    else:
        print('无条件生成（未指定 target_elements）')
    if stability_cond is None:
        print('Ehull条件: unconditional')
    else:
        print(f'Ehull条件类别: {args.stability_class}  cfg_w={args.cfg_w}')

    print(f'生成 {args.num_samples} 个结构...')
    with torch.no_grad():
        structures = model.generate(
            num_samples=args.num_samples,
            elem_cond=elem_cond,
            stability_cond=stability_cond,
            cfg_w=args.cfg_w,
        )

    w2s_report()

    def keep(struct):
        if struct is None:
            return False
        if elem_cond is None or args.must_contain == 'off':
            return True
        els = structure_element_zs(struct)
        if args.must_contain == 'any':
            return len(els & target_zs) > 0
        else:
            return target_zs.issubset(els)

    valid = [(i, s) for i, s in enumerate(structures) if keep(s)]
    total_nonnull = sum(1 for s in structures if s is not None)
    print(f'生成成功(非空): {total_nonnull}/{args.num_samples}')
    if elem_cond is not None and args.must_contain != 'off':
        print(f'满足"必须含"({args.must_contain})的: {len(valid)}/{total_nonnull} '
              f'= {len(valid)/max(total_nonnull,1)*100:.1f}%')

    print('保存 CIF...')
    n_saved = 0
    for i, struct in valid:
        try:
            cif_path = os.path.join(args.out_dir, f'struct_{i:04d}.cif')
            CifWriter(struct).write_file(cif_path)
            n_saved += 1
        except Exception as e:
            print(f'保存失败 {i}: {e}')

    print(f'完成！保存 {n_saved} 个到 {args.out_dir}')


if __name__ == '__main__':
    main()
