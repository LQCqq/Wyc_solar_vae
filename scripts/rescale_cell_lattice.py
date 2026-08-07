import argparse
import json
from pathlib import Path

import numpy as np
from pymatgen.core import Structure, Lattice
from tqdm import tqdm


BASE = Path("/srv/scratch/ml4matdis/Quanli_Project/z5561341")
DEFAULT_INPUT  = BASE / "SMACT_structure" / "cifs"
DEFAULT_OUTPUT = BASE / "Rescale_lattice_structure"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT,
                    help="待处理的CIF目录（默认 SMACT_structure/cifs/）")
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT,
                    help="放大晶胞后的CIF输出目录（默认 Rescale_lattice_structure/）")
    p.add_argument("--target", type=float, default=20.0,
                    help="目标每原子体积 Å³（MP 典型约 20）")
    p.add_argument("--thresh", type=float, default=12.0,
                    help="每原子体积低于此值才放大（Å³），高于则原样保留")
    return p.parse_args()


def main():
    args = parse_args()
    # 和 SMACT_structure/cifs、MACE_structure/cifs 保持同样的子目录习惯
    cif_dir = args.output_dir / "cifs"
    cif_dir.mkdir(parents=True, exist_ok=True)

    cif_files = sorted(args.input_dir.rglob("*.cif"))
    if not cif_files:
        print(f"[错误] {args.input_dir} 下没有找到 CIF 文件")
        return
    print(f"[输入] 共 {len(cif_files)} 个 CIF，来自 {args.input_dir}")
    print(f"[rescale] target={args.target} Å³/atom, thresh={args.thresh} Å³/atom")

    results = []
    n_scaled = n_kept = n_error = 0

    for cif_path in tqdm(cif_files, desc="放大过小晶胞"):
        try:
            struct = Structure.from_file(str(cif_path))
        except Exception as e:
            print(f"  [读取失败] {cif_path.name}: {e}")
            n_error += 1
            continue

        vpa_before = struct.volume / len(struct)

        if vpa_before < args.thresh:
            # 分数坐标不变，只等比放大晶格矩阵 → 撑开晶胞、原子相对位置不动
            scale = (args.target / vpa_before) ** (1.0 / 3.0)
            new_lat = Lattice(struct.lattice.matrix * scale)
            struct = Structure(
                new_lat, struct.species, struct.frac_coords,
                coords_are_cartesian=False,
            )
            scaled = True
            n_scaled += 1
        else:
            scaled = False
            n_kept += 1

        vpa_after = struct.volume / len(struct)
        results.append({
            "source": cif_path.name,
            "vpa_before": round(vpa_before, 3),
            "vpa_after": round(vpa_after, 3),
            "scaled": scaled,
        })

        out_path = cif_dir / cif_path.name
        struct.to(str(out_path))

    summary_path = args.output_dir / "rescale_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total = len(cif_files)
    print(f"\n[完成] 放大: {n_scaled}/{total} ({n_scaled/total*100:.1f}%)"
          f"  原样保留: {n_kept}  读取错误: {n_error}")
    print(f"  输出CIF → {cif_dir}")
    print(f"  汇总JSON → {summary_path}")
    print(f"\n[提示] 放大是粗暴等比到 {args.target} Å³/atom，"
          f"建议对输出再跑一次 MACE relax，收缩到各组分的平衡密度。")


if __name__ == "__main__":
    main()