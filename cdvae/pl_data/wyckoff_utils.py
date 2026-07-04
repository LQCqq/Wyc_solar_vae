# cdvae/pl_data/wyckoff_utils.py
import numpy as np
import torch
from pyxtal import pyxtal
from pymatgen.core import Structure, Element
from pyxtal.symmetry import Group

WYCKOFF_LETTERS = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
LETTER_TO_IDX = {l: i for i, l in enumerate(WYCKOFF_LETTERS)}
MAX_WYCKOFF_SITES = 27


def structure_to_wyckoff(structure: Structure, tol: float = 0.1):
    crystal = pyxtal()
    last_err = None
    for t in [tol, 0.3, 0.5]:
        try:
            crystal.from_seed(structure, tol=t)
            break
        except Exception as e:
            last_err = e
    else:
        raise RuntimeError(f"PyXtal failed at all tolerances: {last_err}")

    spg_num = crystal.group.number
    lattice_params = np.array([
        crystal.lattice.a, crystal.lattice.b, crystal.lattice.c,
        crystal.lattice.alpha, crystal.lattice.beta, crystal.lattice.gamma,
    ], dtype=np.float32)

    site_elements, site_letters, site_multiplicities, site_free_params = [], [], [], []
    for site in crystal.atom_sites:
        site_elements.append(site.specie)
        site_letters.append(site.wp.letter)
        site_multiplicities.append(site.wp.multiplicity)
        site_free_params.append(np.array(site.position, dtype=np.float32))

    return {
        'spacegroup_num': spg_num,
        'site_elements': site_elements,
        'site_letters': site_letters,
        'site_multiplicities': site_multiplicities,
        'site_free_params': site_free_params,
        'lattice_params': lattice_params,
        'num_sites': len(site_elements),
    }


def encode_wyckoff_tensors(wyckoff_dict):
    spg_idx = torch.tensor(
        [wyckoff_dict['spacegroup_num'] - 1], dtype=torch.long
    )
    n = wyckoff_dict['num_sites']
    atom_types = torch.tensor(
        [Element(e).Z for e in wyckoff_dict['site_elements']], dtype=torch.long
    )
    letter_idx = torch.tensor(
        [LETTER_TO_IDX.get(l, 0) for l in wyckoff_dict['site_letters']],
        dtype=torch.long
    )
    multiplicities = torch.tensor(
        wyckoff_dict['site_multiplicities'], dtype=torch.float32
    )
    free_params = torch.zeros(n, 3, dtype=torch.float32)
    for i, fp in enumerate(wyckoff_dict['site_free_params']):
        fp_t = torch.tensor(fp, dtype=torch.float32)
        free_params[i, :len(fp_t)] = fp_t

    return {
        'spg_idx':        spg_idx,
        'atom_types':     atom_types,
        'letter_idx':     letter_idx,
        'multiplicities': multiplicities,
        'free_params':    free_params,
        'lattice_params': torch.tensor(wyckoff_dict['lattice_params']),
        'num_sites':      n,
    }


def _unwrap_pymatgen(result):

    while isinstance(result, list):
        if len(result) == 0:
            return None
        result = result[0]
    if isinstance(result, Structure):
        return result
    return None


import collections as _collections
W2S_PATH_COUNTER = _collections.Counter()
W2S_ERROR_SAMPLES = _collections.defaultdict(list)
W2S_BAD_LETTERS = []  # 记录非法 (spg, letter, 错误类型)

def _w2s_log(path, err=None):
    W2S_PATH_COUNTER[path] += 1
    if err is not None and len(W2S_ERROR_SAMPLES[path]) < 6:
        W2S_ERROR_SAMPLES[path].append(f"{type(err).__name__}: {err}")

def w2s_report():
    print("\n===== wyckoff_to_structure 路径统计 =====")
    total = sum(v for k,v in W2S_PATH_COUNTER.items() if k in
                ['尝试1_ops投影','尝试2_pyxtal_build','尝试3_random+晶格','尝试4_全随机','返回None'])
    for path in ['尝试1_ops投影','尝试2_pyxtal_build','尝试3_random+晶格','尝试4_全随机','返回None']:
        n = W2S_PATH_COUNTER.get(path, 0)
        print(f"  {path:20s}: {n:4d} ({100*n/total if total else 0:5.1f}%)")
    print(f"  {'总计':20s}: {total}")
    for path in ['尝试1_ops投影手动']:
        if W2S_ERROR_SAMPLES[path]:
            print(f"\n  [{path}] 整体失败错误样本:")
            for e in W2S_ERROR_SAMPLES[path]:
                print(f"    - {e}")
    if W2S_BAD_LETTERS:
        print(f"\n  跳过的非法site样本 (spg, letter, 错误): {W2S_BAD_LETTERS[:10]}")


def wyckoff_to_structure(spacegroup_num, site_elements, site_letters,
                          site_free_params, lattice_params):
  
    from pymatgen.core import Lattice, Structure

    if hasattr(lattice_params, 'cpu'):
        lp = lattice_params.cpu().numpy()
    else:
        lp = np.array(lattice_params)
    a, b, c, alpha, beta, gamma = [float(x) for x in lp]

    g = Group(spacegroup_num)

    # 预处理：每个site的 (elem, letter, 预测坐标)
    sites_info = []
    for elem, letter, fp in zip(site_elements, site_letters, site_free_params):
        fp_arr = fp.cpu().numpy() if hasattr(fp, 'cpu') else np.array(fp)
        coord = [float(fp_arr[k]) % 1.0 for k in range(3)]
        sites_info.append((elem, letter, coord))

    # 尝试1：ops[0]投影 
    try:
        lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
        valid_letters = set(wp.letter for wp in g.Wyckoff_positions)
        all_sp, all_co = [], []
        n_skip = 0
        for elem, letter, coord in sites_info:
            try:
                if letter not in valid_letters:
                    raise KeyError(f"letter {letter} 不在SG{spacegroup_num}")
                wp = g[letter]
                rep = wp.ops[0].operate(coord)
                rep = [float(x) % 1.0 for x in rep]
                for op in wp.ops:
                    pos = np.array(op.operate(rep)) % 1.0
                    all_sp.append(elem)
                    all_co.append(pos)
            except Exception as se:
                n_skip += 1
                if len(W2S_BAD_LETTERS) < 15:
                    W2S_BAD_LETTERS.append((spacegroup_num, letter, type(se).__name__))
                continue
        
        if all_sp and n_skip <= len(sites_info) // 2:
            s = Structure(lattice, all_sp, all_co)
            s.merge_sites(tol=0.01, mode='delete')
            #ok = True
            #if len(s) > 1:
            #    dm = s.distance_matrix.copy()
            #    np.fill_diagonal(dm, 999.0)
            #    if dm.min() < 0.5:
            #        ok = False
            #if ok and len(s) > 0:
            #    _w2s_log('尝试1_ops投影手动')
            #    return s
            if len(s) > 0:
                # 密度校正：给 MACE 更合理的初始构型
                # LeMat physical_plausibility 判定范围：0.01-25 g/cm³
                try:
                    density = s.density
                    if density < 0.01:
                        # 太稀（晶胞过大）：压缩晶格
                        new_vol = s.volume * (density / 0.02)
                        s.scale_lattice(new_vol)
                    elif density > 25.0:
                        # 太密（晶胞过小）：膨胀晶格
                        new_vol = s.volume * (density / 20.0)
                        s.scale_lattice(new_vol)
                except Exception:
                    pass  # 密度计算失败不影响主流程
                _w2s_log('尝试1_ops投影手动')
                return s
    except Exception as e:
        _w2s_log('尝试1_ops投影手动', e)

    # pyxtal
    from pyxtal.lattice import Lattice as PyxtalLattice
    ltype = getattr(g, 'lattice_type', 'triclinic')

    element_sites = {}
    for elem, letter, coord in sites_info:
        if elem not in element_sites:
            element_sites[elem] = []
        try:
            mult = g[letter].multiplicity
        except Exception:
            mult = 1
        element_sites[elem].append((f"{mult}{letter}", coord, mult))

    species = list(element_sites.keys())
    numIons = [sum(m for _, _, m in element_sites[e]) for e in species]
    sites_coords = [[{lab: crd} for lab, crd, _ in element_sites[e]] for e in species]

    latt = None
    for try_ltype in [ltype, 'triclinic']:
        try:
            latt = PyxtalLattice.from_para(a, b, c, alpha, beta, gamma, ltype=try_ltype)
            break
        except Exception:
            latt = None
    if latt is None:
        try:
            latt = PyxtalLattice.from_para(8.0, 8.0, 8.0, 90, 90, 90, ltype='cubic')
        except Exception:
            latt = None

    # 尝试2：pyxtal build
    if latt is not None:
        try:
            crystal = pyxtal()
            crystal.build(g, species, numIons, lattice=latt, sites=sites_coords)
            result = _unwrap_pymatgen(crystal.to_pymatgen())
            if result is not None:
                _w2s_log('尝试2_pyxtal_build')
                return result
        except Exception:
            pass

    # 尝试3：from_random + 预测晶格
    if latt is not None:
        try:
            crystal = pyxtal()
            crystal.from_random(3, spacegroup_num, species, numIons, lattice=latt)
            result = _unwrap_pymatgen(crystal.to_pymatgen())
            if result is not None:
                _w2s_log('尝试3_random+晶格')
                return result
        except Exception:
            pass

    # 尝试4：全随机
    try:
        crystal = pyxtal()
        crystal.from_random(3, spacegroup_num, species, numIons)
        result = _unwrap_pymatgen(crystal.to_pymatgen())
        _w2s_log('尝试4_全随机')
        return result
    except Exception:
        pass

    _w2s_log('None')
    return None