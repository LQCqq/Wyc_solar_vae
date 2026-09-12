# cdvae/pl_data/wyckoff_utils.py
import numpy as np
import torch
import json
from pathlib import Path
from pyxtal import pyxtal
from pymatgen.core import Structure, Element
from pyxtal.symmetry import Group

WYCKOFF_LETTERS = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
LETTER_TO_IDX = {l: i for i, l in enumerate(WYCKOFF_LETTERS)}
MAX_WYCKOFF_SITES = 27
_ATOMIC_RADII = None


def _get_atomic_radii():
    global _ATOMIC_RADII
    if _ATOMIC_RADII is None:
        with open(Path(__file__).parent / 'atomic_radii.json') as f:
            raw = json.load(f)
        _ATOMIC_RADII = np.zeros(101, dtype=np.float64)
        for z_str, info in raw.items():
            _ATOMIC_RADII[int(z_str)] = float(info['radius_A'])
    return _ATOMIC_RADII


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
    lattice_params = np.array(
        crystal.lattice.get_para(degree=True),
        dtype=np.float32,
    )

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
W2S_FAILURE_PATHS = (
    'invalid_space_group',
    'invalid_letter',
    'duplicate_fixed_orbit',
    'atom_budget',
    'atom_count_mismatch',
    'overlap_before_scaling',
    'overlap_after_scaling',
    'site_projection_failure',
    'lattice_build_error',
)
W2S_OVERLAP_DIAGNOSTICS = (
    'overlap_intra_orbit',
    'overlap_inter_orbit',
    'exact_duplicate',
    'close_contact',
)

def _w2s_log(path, err=None):
    W2S_PATH_COUNTER[path] += 1
    if err is not None and len(W2S_ERROR_SAMPLES[path]) < 6:
        W2S_ERROR_SAMPLES[path].append(f"{type(err).__name__}: {err}")

def w2s_report():
    print("\n===== wyckoff_to_structure 路径统计 =====")
    success = W2S_PATH_COUNTER.get('尝试1_ops投影手动', 0)
    failed = sum(W2S_PATH_COUNTER.get(path, 0) for path in W2S_FAILURE_PATHS)
    total = success + failed
    print(f"  {'success':24s}: {success:4d} ({100*success/total if total else 0:5.1f}%)")
    print(f"  {'failed':24s}: {failed:4d} ({100*failed/total if total else 0:5.1f}%)")
    print(f"  {'总计':20s}: {total}")
    print("\n  失败原因:")
    for path in W2S_FAILURE_PATHS:
        n = W2S_PATH_COUNTER.get(path, 0)
        print(f"  {path:24s}: {n:4d} ({100*n/total if total else 0:5.1f}%)")
    overlap_failed = (
        W2S_PATH_COUNTER.get('overlap_before_scaling', 0)
        + W2S_PATH_COUNTER.get('overlap_after_scaling', 0)
    )
    print("\n  overlap 诊断 (同一结构可同时计入多项):")
    for path in W2S_OVERLAP_DIAGNOSTICS:
        n = W2S_PATH_COUNTER.get(path, 0)
        print(f"  {path:24s}: {n:4d} ({100*n/overlap_failed if overlap_failed else 0:5.1f}%)")
    for path in W2S_FAILURE_PATHS:
        if W2S_ERROR_SAMPLES[path]:
            print(f"\n  [{path}] 整体失败错误样本:")
            for e in W2S_ERROR_SAMPLES[path]:
                print(f"    - {e}")
    if W2S_BAD_LETTERS:
        print(f"\n  跳过的非法site样本 (spg, letter, 错误): {W2S_BAD_LETTERS[:10]}")


def wyckoff_to_structure(spacegroup_num, site_elements, site_letters,
                          site_free_params, lattice_params, max_atoms=20,
                          distance_margin=0.05):
  
    from pymatgen.core import Lattice, Structure

    if hasattr(lattice_params, 'cpu'):
        lp = lattice_params.cpu().numpy()
    else:
        lp = np.array(lattice_params)
    a, b, c, alpha, beta, gamma = [float(x) for x in lp]

    try:
        g = Group(spacegroup_num)
    except Exception as e:
        _w2s_log('invalid_space_group', e)
        return None
    valid_wp = {wp.letter: wp for wp in g.Wyckoff_positions}
    expected_atoms = 0
    used_fixed_letters = set()
    for letter in site_letters:
        if letter not in valid_wp:
            _w2s_log('invalid_letter')
            return None
        wp = valid_wp[letter]
        if int(wp.get_dof()) == 0:
            if letter in used_fixed_letters:
                _w2s_log('duplicate_fixed_orbit')
                return None
            used_fixed_letters.add(letter)
        expected_atoms += int(wp.multiplicity)
    if expected_atoms < 1 or expected_atoms > int(max_atoms):
        _w2s_log('atom_budget')
        return None

    def _validated(result, orbit_ids=None, record_overlap=False):
        if result is None or len(result) != expected_atoms:
            return None
        if len(result) > 1:
            dm = result.distance_matrix.copy()
            zs = np.array([int(site.specie.Z) for site in result], dtype=np.int64)
            radii = _get_atomic_radii()[zs]
            threshold = (0.7 + radii[:, None] + radii[None, :]) * 0.5
            threshold = np.maximum(threshold + float(distance_margin), 0.5)
            np.fill_diagonal(dm, np.inf)
            overlap_mask = np.triu(dm < threshold, k=1)
            if np.any(overlap_mask):
                if record_overlap:
                    if orbit_ids is not None and len(orbit_ids) == len(result):
                        orbit_ids = np.asarray(orbit_ids, dtype=np.int64)
                        same_orbit = orbit_ids[:, None] == orbit_ids[None, :]
                        if np.any(overlap_mask & same_orbit):
                            W2S_PATH_COUNTER['overlap_intra_orbit'] += 1
                        if np.any(overlap_mask & ~same_orbit):
                            W2S_PATH_COUNTER['overlap_inter_orbit'] += 1
                    if np.any(overlap_mask & (dm < 1e-3)):
                        W2S_PATH_COUNTER['exact_duplicate'] += 1
                    if np.any(overlap_mask & (dm >= 1e-3)):
                        W2S_PATH_COUNTER['close_contact'] += 1
                return None
        return result

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
        all_sp, all_co, all_orbit_ids = [], [], []
        n_skip = 0
        for orbit_id, (elem, letter, coord) in enumerate(sites_info):
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
                    all_orbit_ids.append(orbit_id)
            except Exception as se:
                n_skip += 1
                if len(W2S_BAD_LETTERS) < 15:
                    W2S_BAD_LETTERS.append((spacegroup_num, letter, type(se).__name__))
                continue
        
        if all_sp and n_skip <= len(sites_info) // 2:
            s = Structure(lattice, all_sp, all_co)
            if len(s) != expected_atoms:
                _w2s_log('atom_count_mismatch')
                return None
            if _validated(s, all_orbit_ids, record_overlap=True) is None:
                _w2s_log('overlap_before_scaling')
                return None
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
                s = _validated(s, all_orbit_ids, record_overlap=True)
                if s is None:
                    _w2s_log('overlap_after_scaling')
                    return None
                _w2s_log('尝试1_ops投影手动')
                return s
    except Exception as e:
        _w2s_log('lattice_build_error', e)
        return None

    _w2s_log('site_projection_failure')
    return None
