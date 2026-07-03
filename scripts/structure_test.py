import numpy as np
from pyxtal import pyxtal
from pyxtal.symmetry import Group

print("pyxtal version:", end=" ")
try:
    import pyxtal as _p
    print(_p.__version__)
except Exception:
    print("unknown")

# 测试案例：空间群 225 (Fm-3m), 元素 Na Cl
spg = 225
g = Group(spg)
print(f"\nSPG {spg} 的 Wyckoff 位:")
for i, wp in enumerate(g.Wyckoff_positions[:6]):
    print(f"  index {i}: letter={wp.letter}, mult={wp.multiplicity}, dof={wp.get_dof()}")

# 选两个有自由度的位测试
# 找一个 general position（dof=3）
test_letter = None
for wp in g.Wyckoff_positions:
    if wp.get_dof() >= 3:
        test_letter = wp.letter
        test_mult = wp.multiplicity
        break
print(f"\n用 letter='{test_letter}' (mult={test_mult}, 通用位) 测试坐标格式")

target_coord = [0.123, 0.456, 0.789]

def check_coords(crystal, label):
    """打印生成结构的原子坐标，看是否接近 target"""
    try:
        struct = crystal.to_pymatgen()
        while isinstance(struct, list):
            struct = struct[0] if struct else None
        if struct is None:
            print(f"  [{label}] to_pymatgen 返回 None")
            return
        coords = struct.frac_coords
        print(f"  [{label}] 成功，{len(coords)} 原子，前2个坐标:")
        for c in coords[:2]:
            print(f"        [{c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}]")
    except Exception as e:
        print(f"  [{label}] to_pymatgen 失败: {e}")

# 格式A：单字母 {letter: coord}
print(f"\n--- 格式A: sites=[[{{'{test_letter}': {target_coord}}}]] (单字母) ---")
try:
    c = pyxtal()
    c.from_random(3, spg, ['Si'], [test_mult],
                  sites=[[{test_letter: target_coord}]],
                  lattice=[5,5,5,90,90,90])
    check_coords(c, "单字母dict")
except Exception as e:
    print(f"  失败: {type(e).__name__}: {e}")

# 格式B：multiplicity+letter {f'{mult}{letter}': coord}
label_full = f"{test_mult}{test_letter}"
print(f"\n--- 格式B: sites=[[{{'{label_full}': {target_coord}}}]] (带mult) ---")
try:
    c = pyxtal()
    c.from_random(3, spg, ['Si'], [test_mult],
                  sites=[[{label_full: target_coord}]],
                  lattice=[5,5,5,90,90,90])
    check_coords(c, "带mult_dict")
except Exception as e:
    print(f"  失败: {type(e).__name__}: {e}")

# 格式C：单字母 letter（无坐标，原代码格式）
print(f"\n--- 格式C: sites=[['{test_letter}']] (单字母无坐标) ---")
try:
    c = pyxtal()
    c.from_random(3, spg, ['Si'], [test_mult],
                  sites=[[test_letter]],
                  lattice=[5,5,5,90,90,90])
    check_coords(c, "单字母无坐标")
except Exception as e:
    print(f"  失败: {type(e).__name__}: {e}")

# 格式D：带mult无坐标
print(f"\n--- 格式D: sites=[['{label_full}']] (带mult无坐标) ---")
try:
    c = pyxtal()
    c.from_random(3, spg, ['Si'], [test_mult],
                  sites=[[label_full]],
                  lattice=[5,5,5,90,90,90])
    check_coords(c, "带mult无坐标")
except Exception as e:
    print(f"  失败: {type(e).__name__}: {e}")

print("\n=== 结论 ===")
print("看哪个格式：(1)不报错 (2)生成坐标接近 [0.123,0.456,0.789]")
print("那个格式就是 wyckoff_to_structure 该用的。")