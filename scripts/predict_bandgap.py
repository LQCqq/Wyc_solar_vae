#!/usr/bin/env python3
"""
批量预测一个目录下所有 CIF 的带隙（MEGNet, matgl 4.x, PyG）。
用法:
  python predict_bandgap.py --cif_dir <目录> [--out <csv路径>] [--method 0] [--pattern "*.cif"]

method: 0=PBE(默认,和mp_20标签一致), 1=GLLB-SC, 2=HSE, 3=SCAN
"""
import os
import sys
import glob
import argparse
import warnings
import json

warnings.filterwarnings('ignore')


def parse_args():
    p = argparse.ArgumentParser(description="Batch band gap prediction with MEGNet")
    p.add_argument('--cif_dir', type=str, required=True,
                   help='包含 CIF 文件的目录')
    p.add_argument('--out', type=str, default=None,
                   help='结果 CSV 输出路径（默认 <cif_dir>/bandgap_predictions.csv）')
    p.add_argument('--method', type=int, default=0, choices=[0, 1, 2, 3],
                   help='泛函: 0=PBE(默认), 1=GLLB-SC, 2=HSE, 3=SCAN')
    p.add_argument('--pattern', type=str, default='*.cif',
                   help='文件匹配模式（默认 *.cif）')
    p.add_argument('--recursive', action='store_true',
                   help='递归搜索子目录')
    p.add_argument('--model', type=str,
                   default='materialyze/MEGNet-BandGap-mfi-MP-2019.4.1',
                   help='matgl 带隙模型的 HF repo id')
    return p.parse_args()


def main():
    args = parse_args()

    import matgl
    import torch
    import numpy as np
    from pymatgen.core import Structure

    # 收集 CIF
    if args.recursive:
        cifs = sorted(glob.glob(os.path.join(args.cif_dir, '**', args.pattern), recursive=True))
    else:
        cifs = sorted(glob.glob(os.path.join(args.cif_dir, args.pattern)))

    if not cifs:
        print(f"[错误] 在 {args.cif_dir} 下没找到匹配 {args.pattern} 的文件")
        sys.exit(1)

    print(f"目录: {args.cif_dir}")
    print(f"找到 {len(cifs)} 个 CIF，方法={['PBE','GLLB-SC','HSE','SCAN'][args.method]}")

    # 加载模型
    print(f"加载带隙模型: {args.model} ...")
    model = matgl.load_model(args.model)
    state = torch.tensor([args.method])
    print("模型加载完成，开始预测...\n")

    results = []
    n_fail = 0
    for i, f in enumerate(cifs):
        try:
            s = Structure.from_file(f)
            bg = float(model.predict_structure(s, state_attr=state))
            results.append({
                'file': os.path.basename(f),
                'formula': s.composition.reduced_formula,
                'n_atoms': len(s),
                'band_gap_eV': round(bg, 4),
            })
        except Exception as e:
            n_fail += 1
        if (i + 1) % 200 == 0:
            print(f"  have been done {i+1}/{len(cifs)}")

    if not results:
        print("[erro] 没有成功预测任何结构")
        sys.exit(1)

    bgs = np.array([r['band_gap_eV'] for r in results])

    # 汇总
    print(f"\n{'='*50}")
    print(f"success: {len(results)}/{len(cifs)}  (failed {n_fail})")
    print(f"band gap analysis (eV):")
    print(f"  average   {bgs.mean():.3f}")
    print(f"  medium   {np.median(bgs):.3f}")
    print(f"  minimum   {bgs.min():.3f}    Maximum {bgs.max():.3f}")
    print(f"\ndistribution:")
    print(f"  metal (<0.1eV):      {(bgs<0.1).sum():5d}  ({(bgs<0.1).mean()*100:.1f}%)")
    print(f"  0.1-1.0 eV:        {((bgs>=0.1)&(bgs<1.0)).sum():5d}")
    print(f"  pv 1.0-1.8 eV:  {((bgs>=1.0)&(bgs<=1.8)).sum():5d}  ({((bgs>=1.0)&(bgs<=1.8)).mean()*100:.1f}%)")
    print(f"  tandem bottom ~1.1eV(0.9-1.3): {((bgs>=0.9)&(bgs<=1.3)).sum():5d}")
    print(f"  tandem top ~1.7eV(1.5-1.9): {((bgs>=1.5)&(bgs<=1.9)).sum():5d}")
    print(f"  wide (>1.8eV):    {(bgs>1.8).sum():5d}")

    # 写 CSV
    out_path = args.out or os.path.join(args.cif_dir, 'bandgap_predictions.csv')
    import csv
    with open(out_path, 'w', newline='') as fp:
        w = csv.DictWriter(fp, fieldnames=['file', 'formula', 'n_atoms', 'band_gap_eV'])
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\n逐结构结果 → {out_path}")

    # 汇总 JSON
    summary = {
        'cif_dir': args.cif_dir,
        'method': ['PBE', 'GLLB-SC', 'HSE', 'SCAN'][args.method],
        'n_total': len(cifs),
        'n_success': len(results),
        'mean_eV': float(bgs.mean()),
        'median_eV': float(np.median(bgs)),
        'frac_metal': float((bgs < 0.1).mean()),
        'frac_solar_1_1p8': float(((bgs >= 1.0) & (bgs <= 1.8)).mean()),
    }
    sum_path = os.path.join(os.path.dirname(out_path), 'bandgap_summary.json')
    with open(sum_path, 'w') as fp:
        json.dump(summary, fp, indent=2)
    print(f"conclusion → {sum_path}")


if __name__ == '__main__':
    main()