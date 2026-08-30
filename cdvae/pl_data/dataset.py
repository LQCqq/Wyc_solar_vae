import os
import io
import pickle
import hydra
import omegaconf
import torch
import pandas as pd

from omegaconf import ValueNode
from torch.utils.data import Dataset
from torch_geometric.data import Data
from pymatgen.core import Structure

from cdvae.common.utils import PROJECT_ROOT
from cdvae.common.data_utils import (
    preprocess, preprocess_tensors, add_scaled_lattice_prop)
from cdvae.pl_data.wyckoff_utils import structure_to_wyckoff, encode_wyckoff_tensors


class CrystDataset(Dataset):
    def __init__(self, name: ValueNode, path: ValueNode,
                 prop: ValueNode, niggli: ValueNode, primitive: ValueNode,
                 graph_method: ValueNode, preprocess_workers: ValueNode,
                 lattice_scale_method: ValueNode, **kwargs):
        super().__init__()
        self.path = path
        self.name = name
        self.df = pd.read_csv(path)
        self.prop = prop
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method

        self.cached_data = preprocess(
            self.path,
            preprocess_workers,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method,
            prop_list=[prop])

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

        self.use_wyckoff = kwargs.get('use_wyckoff', True)
        if self.use_wyckoff:
            self._precompute_wyckoff()

    def _precompute_wyckoff(self):
        from pymatgen.io.cif import CifParser
        cache_path = self.path.replace('.csv', '_wyckoff.pkl')

        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                self.wyckoff_cache = pickle.load(f)
            print(f'[Wyckoff] Loaded cache: {len(self.wyckoff_cache)} entries')
            return

        self.wyckoff_cache = {}
        failed = 0

        for idx, row in self.df.iterrows():
            try:
                cif_str = row['cif']
                if isinstance(cif_str, str) and (
                    cif_str.strip().startswith('#') or
                    cif_str.strip().startswith('data_')
                ):
                    parser = CifParser(io.StringIO(cif_str))
                    struct = parser.get_structures(primitive=False)[0]
                else:
                    struct = Structure.from_dict(eval(cif_str))
                w = structure_to_wyckoff(struct, tol=0.1)
                self.wyckoff_cache[idx] = encode_wyckoff_tensors(w)
            except Exception:
                try:
                    parser = CifParser(io.StringIO(row['cif']))
                    struct = parser.get_structures(primitive=False)[0]
                    w = structure_to_wyckoff(struct, tol=0.5)
                    self.wyckoff_cache[idx] = encode_wyckoff_tensors(w)
                except Exception:
                    self.wyckoff_cache[idx] = None
                    failed += 1

        print(f'[Wyckoff] Done. Failed={failed}/{len(self.df)}')
        with open(cache_path, 'wb') as f:
            pickle.dump(self.wyckoff_cache, f)

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        prop = self.scaler.transform(data_dict[self.prop])
        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']

        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,
            y=prop.view(1, -1),
        )

        if self.use_wyckoff:
            wyk = self.wyckoff_cache.get(index)
            if wyk is not None:
                data.spg_idx = wyk['spg_idx']
                data.wyk_atom_types = wyk['atom_types']
                data.wyk_letters = wyk['letter_idx']
                data.wyk_multi = wyk['multiplicities']
                data.wyk_free = wyk['free_params']
                data.wyk_lattice = wyk['lattice_params']
                data.num_wyk_sites = torch.tensor(wyk['num_sites'])
            else:
                #
                data.spg_idx = torch.tensor([0], dtype=torch.long)
                data.wyk_atom_types = torch.tensor([1], dtype=torch.long)
                data.wyk_letters = torch.tensor([0], dtype=torch.long)
                data.wyk_multi = torch.tensor([1.0])
                data.wyk_free = torch.zeros(1, 3)
                data.wyk_lattice = torch.tensor([
                    float(lengths[0]), float(lengths[1]), float(lengths[2]),
                    float(angles[0]), float(angles[1]), float(angles[2])
                ])
                data.num_wyk_sites = torch.tensor(1)

            # ── CFG 元素条件（对齐论文类别标签语义）──
            # 条件表示"想要含哪些元素"（生成意图），不是"结构含全部元素"（成分描述）。
            # 若用全部元素，masked-diffusion 会把 multihot=1 学成"这些已被别处占用"
            # 从而反向压低目标元素。改为：每个结构随机选 1-2 个真实元素作为目标条件。
            zs = data.wyk_atom_types.long().view(-1)
            valid_zs = zs[(zs >= 1) & (zs <= 100)].unique()
            elem_mh = torch.zeros(100)
            if valid_zs.numel() > 0:
                k = 1 if valid_zs.numel() == 1 else int(torch.randint(1, 3, (1,)).item())  # 1或2
                k = min(k, valid_zs.numel())
                perm = torch.randperm(valid_zs.numel())[:k]
                chosen = valid_zs[perm]
                elem_mh[chosen - 1] = 1.0
            data.elem_multihot = elem_mh.view(1, 100)  # (1,100) 便于按batch拼接

        return data

    def __repr__(self) -> str:
        return f"CrystDataset({self.name=}, {self.path=})"


class TensorCrystDataset(Dataset):
    def __init__(self, crystal_array_list, niggli, primitive,
                 graph_method, preprocess_workers,
                 lattice_scale_method, **kwargs):
        super().__init__()
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method

        self.cached_data = preprocess_tensors(
            crystal_array_list,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method)

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']

        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,
        )
        return data

    def __repr__(self) -> str:
        return f"TensorCrystDataset(len: {len(self.cached_data)})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    from torch_geometric.data import Batch
    from cdvae.common.data_utils import get_scaler_from_data_list
    dataset: CrystDataset = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.train, _recursive_=False
    )
    lattice_scaler = get_scaler_from_data_list(
        dataset.cached_data, key='scaled_lattice')
    scaler = get_scaler_from_data_list(
        dataset.cached_data, key=dataset.prop)

    dataset.lattice_scaler = lattice_scaler
    dataset.scaler = scaler
    data_list = [dataset[i] for i in range(len(dataset))]
    batch = Batch.from_data_list(data_list)
    return batch


if __name__ == "__main__":
    main()