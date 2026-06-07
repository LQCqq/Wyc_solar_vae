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
    """递归展开嵌套 list，直到得到 Structure 或 None"""
    while isinstance(result, list):
        if len(result) == 0:
            return None
        result = result[0]
    if isinstance(result, Structure):
        return result
    return None


def wyckoff_to_structure(spacegroup_num, site_elements, site_letters,
                          site_free_params, lattice_params):
    """
    Wyckoff to pymatgen Structure.
    """
    crystal = pyxtal()

    if hasattr(lattice_params, 'cpu'):
        lp = lattice_params.cpu().numpy()
    else:
        lp = np.array(lattice_params)
    a, b, c, alpha, beta, gamma = [float(x) for x in lp]

    # 按元素分组，保留 Wyckoff letter
    element_sites = {}
    for elem, letter in zip(site_elements, site_letters):
        if elem not in element_sites:
            element_sites[elem] = []
        element_sites[elem].append(letter)

    species = list(element_sites.keys())
    sites   = [element_sites[e] for e in species]

    # 计算 numIons
    g = Group(spacegroup_num)
    numIons = []
    for elem_letters in sites:
        count = 0
        for letter in elem_letters:
            try:
                count += g[letter].multiplicity
            except Exception:
                count += 1
        numIons.append(count)

    # 先尝试指定 sites
    try:
        crystal.from_random(
            3, spacegroup_num, species, numIons,
            sites=sites,
            lattice=[a, b, c, alpha, beta, gamma],
        )
        result = _unwrap_pymatgen(crystal.to_pymatgen())
        if result is not None:
            return result
    except Exception:
        pass

    # fallback: 不指定 sites
    try:
        crystal.from_random(
            3, spacegroup_num, species, numIons,
            lattice=[a, b, c, alpha, beta, gamma],
        )
        return _unwrap_pymatgen(crystal.to_pymatgen())
    except Exception:
        pass

    # 最终 fallback: 从 atom_sites 直接构建
    try:
        from pymatgen.core import Lattice
        lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
        sp, coords = [], []
        for site in crystal.atom_sites:
            for coord in site.coords:
                sp.append(site.specie)
                coords.append(coord)
        if sp:
            return Structure(lattice, sp, coords)
    except Exception as e:
        print(f'[DEBUG] fallback failed: {e}')

    # 最终 fallback：完全不指定 lattice，让 PyXtal 自由生成
    try:
        crystal2 = pyxtal()
        crystal2.from_random(3, spacegroup_num, species, numIons)
        result = _unwrap_pymatgen(crystal2.to_pymatgen())
        if result is not None:
            return result
    except Exception:
        pass

    return None
