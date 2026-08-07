"""
smact_charge_filter.py
───────────────────────
用 SMACT 对已生成的 CIF 结构做电荷中性硬过滤：
 
"""

import argparse
import json
from pathlib import Path

from pymatgen.core import Structure
from smact.screening import smact_validity
from tqdm import tqdm


BASE = Path("/srv/scratch/ml4matdis/Quanli_Project/z5561341")
DEFAULT_INPUT  = BASE / "generated_structures"
DEFAULT_OUTPUT = BASE / "SMACT_structure"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT,
                    help="待过滤的CIF目录（默认 generated_structures/）")
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT,
                    help="通过SMACT检验的CIF输出目录（默认 SMACT_structure/）")
    p.add_argument("--use_pauling_test", action="store_true", default=True,
                    help="是否同时检查Pauling电负性排序（默认开启，更严格）")
    p.add_argument("--no_pauling_test", dest="use_pauling_test", action="store_false")
    return p.parse_args()


def main():
    args = parse_args()
    # 和 MACE_structure/cifs/ 保持同样的子目录习惯
    cif_dir = args.output_dir / "cifs"
    cif_dir.mkdir(parents=True, exist_ok=True)

    cif_files = sorted(args.input_dir.rglob("*.cif"))
    if not cif_files:
        print(f"[错误] {args.input_dir} 下没有找到 CIF 文件")
        return
    print(f"[输入] 共 {len(cif_files)} 个 CIF，来自 {args.input_dir}")
    print(f"[SMACT] use_pauling_test={args.use_pauling_test}")

    results = []
    n_pass = n_fail = n_error = 0

    for cif_path in tqdm(cif_files, desc="SMACT电荷中性过滤"):
        try:
            struct = Structure.from_file(str(cif_path))
        except Exception as e:
            print(f"  [读取失败] {cif_path.name}: {e}")
            n_error += 1
            continue

        formula = struct.composition.reduced_formula
        try:
            valid = smact_validity(
                struct.composition,
                use_pauling_test=args.use_pauling_test,
                include_alloys=True,
            )
        except Exception as e:
            # SMACT 对某些罕见元素组合(比如稀有气体、超重元素)可能直接报错，
            # 视为不通过处理，不让脚本中断
            valid = False
            print(f"  [SMACT检查异常，判为不通过] {cif_path.name} ({formula}): {e}")

        results.append({
            "source": cif_path.name,
            "formula": formula,
            "smact_valid": bool(valid),
        })

        if valid:
            out_path = cif_dir / cif_path.name
            struct.to(str(out_path))
            n_pass += 1
        else:
            n_fail += 1

    summary_path = args.output_dir / "smact_filter_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total = len(cif_files)
    print(f"\n[完成] 通过: {n_pass}/{total} ({n_pass/total*100:.1f}%)"
          f"  未通过: {n_fail}  读取错误: {n_error}")
    print(f"  通过的CIF → {cif_dir}")
    print(f"  汇总JSON  → {summary_path}")
    print(f"\n[产出率提示] 若要保证最终拿到 N 个电荷合规结构，"
          f"生成阶段建议按当前通过率({n_pass/total*100:.0f}%)反推超采样数量，"
          f"比如目标2500个 → 建议生成约 {int(2500/(n_pass/total)) if n_pass else '?'} 个。")


if __name__ == "__main__":
    main()