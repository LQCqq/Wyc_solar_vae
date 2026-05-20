#import numpy as np
#from pymatgen.core import Structure
#from pyxtal import pyxtal

#def parse_structure_to_wyckoff(pmg_struct: Structure, tol=0.01):
#    """
#    pymatgen to Wyckoff
#    """
#    xtal = pyxtal()
#    try:
#        # get symmetry
#        xtal.from_seed(pmg_struct, tol=tol)
#    except Exception as e:
#        print(f"fail to get symmetry: {e}")
#        return None

#    # get space groups info
#    sg_number = xtal.group.number
#    sg_symbol = xtal.group.symbol
    
#    # get asymmetry unit
#    wyc_sites = []
    
#    for site in xtal.atom_sites:
#        specie = site.specie
#        wp = site.wp  # get Wyckoff_position
        
#        # Wyckoff position info
#        letter = wp.letter
#        multiplicity = wp.multiplicity
        
#        # site.position 是它在 ASU 中的精确坐标
#        generator_coord = site.position
        
#        # get DOF and Mask
#        dof = site.dof
        
#        wyc_sites.append({
#            "specie": specie,
#            "multiplicity": multiplicity,
#            "letter": letter,
#            "generator_coord": generator_coord,
#            "dof": dof,
#        })
        
#    return {
#        "sg_number": sg_number,
#        "sg_symbol": sg_symbol,
#        "lattice": xtal.lattice.matrix,
#        "wyc_sites": wyc_sites
#    }

## test demo
#if __name__ == "__main__":
#    # build cif test
#    cif_test = """# generated using pymatgen
#    data_PrS2
#    _symmetry_space_group_name_H-M   'P 1'
#    _cell_length_a   5.62273210
#    _cell_length_b   5.62273210
#    _cell_length_c   5.62273210
#    _cell_angle_alpha   60.00000000
#    _cell_angle_beta   60.00000000
#    _cell_angle_gamma   60.00000000
#    _symmetry_Int_Tables_number   1
#    _chemical_formula_structural   PrS2
#    _chemical_formula_sum   'Pr2 S4'
#    _cell_volume   125.69765576
#    _cell_formula_units_Z   2
#    loop_
#    _symmetry_equiv_pos_site_id
#    _symmetry_equiv_pos_as_xyz
#     1  'x, y, z'
#    loop_
#    _atom_site_type_symbol
#    _atom_site_label
#    _atom_site_symmetry_multiplicity
#    _atom_site_fract_x
#    _atom_site_fract_y
#    _atom_site_fract_z
#    _atom_site_occupancy
#      Pr  Pr0  1  0.50000000  0.50000000  0.50000000  1
#      Pr  Pr1  1  0.75000000  0.75000000  0.75000000  1
#      S  S2  1  0.12500000  0.12500000  0.12500000  1
#      S  S3  1  0.62500000  0.12500000  0.12500000  1
#      S  S4  1  0.12500000  0.62500000  0.12500000  1
#      S  S5  1  0.12500000  0.12500000  0.62500000  1"""


#    cif_struct = Structure.from_str(cif_test, fmt="cif")
    
#    print("the number of atoms of pymatgen before parser:", len(cif_struct))
#    result = parse_structure_to_wyckoff(cif_struct)
    
#    print(f"\nspace group: {result['sg_number']} ({result['sg_symbol']})")
#    print("atoms need prediction:")
#    for idx, site in enumerate(result['wyc_sites']):
#        print(f"atoms {idx+1}: elements={site['specie']}, Wyckoff={site['multiplicity']}{site['letter']}, "
#              f"coord={site['generator_coord']}, dof={site['dof']}")

import numpy as np
import os
import glob
import pandas as pd
from pathlib import Path
from pymatgen.core import Structure
from pyxtal import pyxtal


def parse_structure_to_wyckoff(pmg_struct: Structure, tol=0.01):
    """
    pymatgen to Wyckoff
    """
    xtal = pyxtal()
    try:
        xtal.from_seed(pmg_struct, tol=tol)
    except Exception as e:
        print(f"  [WARN] fail to get symmetry: {e}")
        return None

    sg_number = xtal.group.number
    sg_symbol = xtal.group.symbol

    wyc_sites = []
    for site in xtal.atom_sites:
        specie = site.specie
        wp = site.wp
        letter = wp.letter
        multiplicity = wp.multiplicity
        generator_coord = site.position
        dof = site.dof

        wyc_sites.append({
            "specie": specie,
            "multiplicity": multiplicity,
            "letter": letter,
            "generator_coord": generator_coord,
            "dof": dof,
        })

    return {
        "sg_number": sg_number,
        "sg_symbol": sg_symbol,
        "lattice": xtal.lattice.matrix,
        "wyc_sites": wyc_sites
    }


def process_structure(struct: Structure, label: str = "", tol: float = 0.01):
    """
    single structure
    """
    atoms_before = len(struct)
    result = parse_structure_to_wyckoff(struct, tol=tol)

    if result is None:
        return {
            "label": label,
            "formula": struct.formula,
            "atoms_before": atoms_before,
            "atoms_after": None,   # 转换失败
            "sg_number": None,
            "sg_symbol": None,
            "reduction_ratio": None,
            "success": False,
        }

    atoms_after = len(result["wyc_sites"])   # 不对称单元中的独立原子数
    reduction_ratio = atoms_after / atoms_before

    return {
        "label": label,
        "formula": struct.formula,
        "atoms_before": atoms_before,
        "atoms_after": atoms_after,
        "sg_number": result["sg_number"],
        "sg_symbol": result["sg_symbol"],
        "reduction_ratio": reduction_ratio,
        "success": True,
        "_wyckoff_detail": result["wyc_sites"],  
    }



# 方式一：读取多个 CIF 文件

def process_cif_files(cif_dir: str = ".", pattern: str = "*.cif",
                      tol: float = 0.01, verbose: bool = True):
    """
    读取目录下所有 CIF 文件并批量转换
    
    Args:
        cif_dir:  CIF 文件所在目录
        pattern:  文件匹配模式，默认 *.cif
        tol:      对称性识别容差
        verbose:  是否打印每个结构的详细信息
    
    Returns:
        pd.DataFrame: 包含所有结构的统计信息
    """
    cif_paths = sorted(glob.glob(os.path.join(cif_dir, pattern)))
    if not cif_paths:
        print(f"[ERROR] 在 {cif_dir} 下未找到匹配 {pattern} 的文件")
        return pd.DataFrame()

    print(f"找到 {len(cif_paths)} 个 CIF 文件，开始处理...\n")
    records = []

    for i, cif_path in enumerate(cif_paths):
        label = Path(cif_path).stem
        print(f"[{i+1}/{len(cif_paths)}] 处理: {label}")

        try:
            struct = Structure.from_file(cif_path)
        except Exception as e:
            print(f"  [ERROR] 读取 CIF 失败: {e}")
            records.append({
                "label": label, "formula": "N/A",
                "atoms_before": None, "atoms_after": None,
                "sg_number": None, "sg_symbol": None,
                "reduction_ratio": None, "success": False,
            })
            continue

        info = process_structure(struct, label=label, tol=tol)

        if verbose and info["success"]:
            print(f"  化学式: {info['formula']}")
            print(f"  转换前原子数: {info['atoms_before']}")
            print(f"  转换后独立原子数（ASU）: {info['atoms_after']}")
            print(f"  空间群: {info['sg_number']} ({info['sg_symbol']})")
            print(f"  压缩比: {info['reduction_ratio']:.2f}")
            if "_wyckoff_detail" in info:
                for s in info["_wyckoff_detail"]:
                    print(f"    {s['specie']}  {s['multiplicity']}{s['letter']}  "
                          f"coord={np.round(s['generator_coord'], 4)}  dof={s['dof']}")

        # 去掉内部详细字段再存表
        row = {k: v for k, v in info.items() if k != "_wyckoff_detail"}
        records.append(row)
        print()

    df = pd.DataFrame(records)
    return df



def print_summary(df: pd.DataFrame):
    """打印整体统计信息"""
    total = len(df)
    success = df["success"].sum()
    failed = total - success

    print("=" * 55)
    print("               汇总统计")
    print("=" * 55)
    print(f"总结构数:          {total}")
    print(f"转换成功:          {success}")
    print(f"转换失败:          {failed}")

    if success > 0:
        df_ok = df[df["success"]]
        print(f"\n转换前原子数:  均值={df_ok['atoms_before'].mean():.1f}  "
              f"最小={df_ok['atoms_before'].min()}  最大={df_ok['atoms_before'].max()}")
        print(f"转换后原子数:  均值={df_ok['atoms_after'].mean():.1f}  "
              f"最小={df_ok['atoms_after'].min()}  最大={df_ok['atoms_after'].max()}")
        print(f"平均压缩比:    {df_ok['reduction_ratio'].mean():.3f}  "
              f"（转换后/转换前，越小说明对称性越高）")

        # 空间群分布 Top 10
        print(f"\n出现最多的空间群 Top 10:")
        sg_counts = df_ok["sg_symbol"].value_counts().head(10)
        for sg, cnt in sg_counts.items():
            print(f"  {sg:20s}  {cnt} 个")

    print("=" * 55)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="批量解析晶体结构的 Wyckoff 信息")
    parser.add_argument("--mode", choices=["cif", "mp"], default="cif",
                        help="cif: 读取 CIF 文件目录；mp: 读取 CDVAE MP csv 数据集")
    parser.add_argument("--input", type=str, required=True,
                        help="CIF 目录路径 或 MP csv 文件路径")
    parser.add_argument("--pattern", type=str, default="*.cif",
                        help="CIF 文件匹配模式（mode=cif 时有效），默认 *.cif")
    parser.add_argument("--max", type=int, default=None,
                        help="最多处理多少个结构（调试用）")
    parser.add_argument("--tol", type=float, default=0.01,
                        help="对称性识别容差，默认 0.01")
    parser.add_argument("--output", type=str, default="wyckoff_stats.csv",
                        help="输出 CSV 文件路径，默认 wyckoff_stats.csv")
    parser.add_argument("--quiet", action="store_true",
                        help="不打印每个结构的详细信息")
    args = parser.parse_args()

   
    if args.mode == "cif":
        df_result = process_cif_files(
            cif_dir=args.input,
            pattern=args.pattern,
            tol=args.tol,
            verbose=not args.quiet,
        )
        if args.max:
            df_result = df_result.head(args.max)
    else:
        df_result = process_mp_csv(
            csv_path=args.input,
            tol=args.tol,
            max_structures=args.max,
            verbose=not args.quiet,
        )

    if not df_result.empty:
        print_summary(df_result)
        out_cols = ["label", "formula", "atoms_before", "atoms_after",
                    "reduction_ratio", "sg_number", "sg_symbol", "success"]
        df_result[out_cols].to_csv(args.output, index=False)
        print(f"\n结果已保存到: {args.output}")