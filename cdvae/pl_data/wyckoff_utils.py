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
    """
    pymatgen Structure to wyckoff 
    tol 用 0.1 /0.5, MP_20
    """
    crystal = pyxtal()

    # 三个tol级别
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
    """
    Wyckoff to tensor
    """
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

    # free要3维
    free_params = torch.zeros(n, 3, dtype=torch.float32)
    for i, fp in enumerate(wyckoff_dict['site_free_params']):
        fp_t = torch.tensor(fp, dtype=torch.float32)
        free_params[i, :len(fp_t)] = fp_t

    return {
        'spg_idx':       spg_idx,
        'atom_types':    atom_types,
        'letter_idx':    letter_idx,
        'multiplicities': multiplicities,
        'free_params':   free_params,
        'lattice_params': torch.tensor(wyckoff_dict['lattice_params']),
        'num_sites':     n,
    }


def wyckoff_to_structure(spacegroup_num, site_elements, site_letters,
                          site_free_params, lattice_params):
    """
    Wyckoff to pymatgen Structure。
    """
    crystal = pyxtal()
    a, b, c, alpha, beta, gamma = lattice_params.tolist()

    element_sites = {}
    for elem, letter, free in zip(site_elements, site_letters, site_free_params):
        if elem not in element_sites:
            element_sites[elem] = []
        element_sites[elem].append((letter, free.tolist()))

    species, numIons = [], []
    for elem, sites in element_sites.items():
        g = Group(spacegroup_num)
        count = sum(g[letter].multiplicity for letter, _ in sites)
        species.append(elem)
        numIons.append(count)

    crystal.from_random(
        3, spacegroup_num, species, numIons,
        lattice=[a, b, c, alpha, beta, gamma],
    )
    return crystal.to_pymatgen()