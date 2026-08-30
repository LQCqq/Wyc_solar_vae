#!/usr/bin/env python3
"""
生成扩散去噪轨迹（支持 CFG 定向 + 目标元素筛选）。

对每个候选结构跑完整去噪，在指定时间步(默认 T=100,75,50,25,0)保存中间态。
用 CFG 时最终结构不一定含目标元素，因此可生成多个候选，
只保留【最终结构(t=0)含目标元素】的，每个保留结构导出其完整时间步轨迹。

去噪帧：位点级(不展开)，每个Wyckoff位点一个代表原子，数量不变，
        MASK位点用占位元素(默认He，渲染时显示灰色)，逐步上色。
最终帧：位点全确定后按Wyckoff对称展开成完整晶体(traj_final_expanded.cif)。
这样避免"原子数暴增"的视觉跳变，忠实体现"位点并行去噪 → 最后展开"。

用法:
  # 生成10个候选，保留含硫族的，每个导出完整轨迹
  python generate_trajectory.py --ckpt <ckpt> --out_dir <dir> \
      --target_elements "S,Se,Te" --cfg_w 2.0 \
      --n_candidates 10 --need_target any \
      --save_steps 100,75,50,25,0

产出: <out_dir>/struct_XXX/traj_t{T}.cif   每个保留结构一个子目录
"""
import sys, os, argparse
import numpy as np
import torch

sys.path.insert(0, '/srv/scratch/ml4matdis/Quanli_Project/z5561341/projectA/cdvae')
os.chdir('/srv/scratch/ml4matdis/Quanli_Project/z5561341/projectA/cdvae')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--target_elements', type=str, default=None,
                   help='目标元素,逗号分隔,如 "S,Se,Te"。用于CFG条件+筛选')
    p.add_argument('--cfg_w', type=float, default=0.0)
    p.add_argument('--n_candidates', type=int, default=10,
                   help='生成多少个候选结构（每个不同seed）')
    p.add_argument('--need_target', type=str, default='any',
                   choices=['off', 'any', 'all'],
                   help='筛选: off=全保留; any=最终结构含目标元素至少一个; all=全含')
    p.add_argument('--max_keep', type=int, default=0,
                   help='最多保留几个含目标元素的结构(0=不限)')
    p.add_argument('--seed_start', type=int, default=0,
                   help='候选的起始seed(候选i用seed_start+i)')
    p.add_argument('--save_steps', type=str, default='100,75,50,25,0')
    p.add_argument('--placeholder', type=str, default='He',
                   help='MASK位点占位元素(渲染时显示成灰色代表未确定)')
    p.add_argument('--temperature', type=float, default=0.5)
    return p.parse_args()


def build_elem_cond(target_elements, device):
    if not target_elements:
        return None, set()
    from pymatgen.core import Element
    zs = set()
    mh = torch.zeros(1, 100, device=device)
    for sym in target_elements.split(','):
        sym = sym.strip()
        if sym:
            z = Element(sym).Z
            mh[0, z - 1] = 1.0
            zs.add(z)
    return mh, zs


def assemble_structure(spg_num, noisy_elem_ids_i, noisy_letter_ids_i,
                       free_params_pred_i, lattice_pred_i, n_sites, placeholder_Z,
                       expand=True):
    """
    用当前(可能部分MASK)的 elem/letter 组装结构。MASK(=0)用占位元素。
    expand=True : 按 Wyckoff 对称展开成完整晶体(最终帧用)
    expand=False: 只画每个位点的代表原子(位点级，不展开)——数量=位点数，
                  用于去噪过程帧，避免"原子数暴增"的视觉跳变
    """
    from pymatgen.core import Element, Lattice, Structure
    from cdvae.pl_data.wyckoff_utils import WYCKOFF_LETTERS, wyckoff_to_structure
    from cdvae.pl_modules.wyckoff_decoder import _fix_lattice

    elements, letters, free_params = [], [], []
    for s in range(n_sites):
        elem_z = int(noisy_elem_ids_i[s].item())
        if elem_z == 0:
            elem_z = placeholder_Z
        letter_id = int(noisy_letter_ids_i[s].item())
        letter_idx = 0 if letter_id == 0 else (letter_id - 1)
        try:
            elem = Element.from_Z(elem_z).symbol
        except Exception:
            elem = 'He'
        letters.append(WYCKOFF_LETTERS[letter_idx])
        elements.append(elem)
        free_params.append(free_params_pred_i[s])

    # 晶格
    try:
        from pyxtal.symmetry import Group as _Group
        _g = _Group(spg_num)
        _valid = {wp.letter for wp in _g.Wyckoff_positions}
        _n_atoms = sum(_g[lt].multiplicity if lt in _valid else 1 for lt in letters)
    except Exception:
        _n_atoms = n_sites * 2
    lat_arr = _fix_lattice(lattice_pred_i * np.array([10., 10., 10., 90., 90., 90.]),
                           n_atoms=_n_atoms)

    if expand:
        # 完整展开成晶体（最终帧）
        struct = wyckoff_to_structure(spg_num, elements, letters, free_params, lat_arr)
        return struct, elements
    else:
        # 位点级：每个位点一个代表原子，不展开（去噪过程帧）
        a, b, c, al, be, ga = [float(x) for x in lat_arr]
        latt = Lattice.from_parameters(a, b, c, al, be, ga)
        coords = []
        species = []
        for s in range(n_sites):
            fp = np.array(free_params[s], dtype=float).flatten()
            # 取前3个作为分数坐标，越界取模
            xyz = [float(fp[k]) % 1.0 if k < len(fp) else 0.0 for k in range(3)]
            coords.append(xyz)
            species.append(elements[s])
        try:
            struct = Structure(latt, species, coords)
            return struct, elements
        except Exception:
            return None, elements


def run_one_trajectory(model, dec, device, seed, elem_cond, cfg_w, temperature,
                       save_steps, placeholder_Z):
    """跑一条完整去噪轨迹，返回 {step: (struct, final_element_Zs)}。"""
    torch.manual_seed(seed)
    B = 1
    z = torch.randn(B, model.hparams.latent_dim, device=device)

    ec = elem_cond
    if ec is not None and ec.size(0) == 1 and B > 1:
        ec = ec.expand(B, -1)

    def _cfg(pc, pu):
        if pu is None or cfg_w <= 0:
            return pc['elem_logits']
        return pu['elem_logits'] + (1.0 + cfg_w) * (pc['elem_logits'] - pu['elem_logits'])

    h_init = dec.fc_shared(z)
    spg_nums = dec.spg_head(h_init).argmax(-1) + 1

    def build_mask(spg_num, num_letters):
        try:
            from pyxtal.symmetry import Group
            from cdvae.pl_data.wyckoff_utils import WYCKOFF_LETTERS
            g = Group(int(spg_num))
            mask = torch.zeros(num_letters, device=device)
            avail = {wp.letter for wp in g.Wyckoff_positions}
            for i, ch in enumerate(WYCKOFF_LETTERS[:num_letters]):
                if ch in avail:
                    mask[i] = 1.0
            return mask if mask.sum() > 0 else torch.ones(num_letters, device=device)
        except Exception:
            return torch.ones(num_letters, device=device)

    letter_masks = torch.stack([build_mask(spg_nums[b].item(), dec.num_letters) for b in range(B)])
    z_proj = dec.site_z_projector(z).unsqueeze(1).expand(B, dec.max_sites, -1)
    per_site_z = z_proj + torch.randn_like(z_proj) * 0.2

    noisy_elem_ids = torch.zeros(B, dec.max_sites, dtype=torch.long, device=device)
    noisy_letter_ids = torch.zeros(B, dec.max_sites, dtype=torch.long, device=device)

    i = 0
    spg_num = int(spg_nums[i].item())
    snapshots = {}  # step -> (struct, elements)

    def snapshot(step, preds, expand=False):
        n_sites = preds['num_sites_logits'][i].argmax().item() + 1
        fp = preds['free_params'][i].cpu().numpy()
        lat = preds['lattice_pred'][i].cpu().numpy()
        try:
            struct, elems = assemble_structure(spg_num, noisy_elem_ids[i], noisy_letter_ids[i],
                                               fp, lat, n_sites, placeholder_Z, expand=expand)
            snapshots[step] = (struct, elems, n_sites,
                               sum(1 for s in range(n_sites) if int(noisy_elem_ids[i, s].item()) == 0))
        except Exception as e:
            snapshots[step] = (None, None, 0, 0)

    last_preds = None
    with torch.no_grad():
        t_full = torch.full((B,), dec.T, dtype=torch.long, device=device)
        preds0 = dec.forward(z, t=t_full, noisy_elem_ids=noisy_elem_ids,
                             noisy_letter_ids=noisy_letter_ids, per_site_z=per_site_z, elem_cond=ec)
        last_preds = preds0
        if dec.T in save_steps:
            snapshot(dec.T, preds0, expand=False)  # 位点级(不展开)

        for step in range(dec.T, 0, -1):
            t = torch.full((B,), step, dtype=torch.long, device=device)
            preds = dec.forward(z, t=t, noisy_elem_ids=noisy_elem_ids,
                                noisy_letter_ids=noisy_letter_ids, per_site_z=per_site_z, elem_cond=ec)
            if ec is not None and cfg_w > 0:
                preds_u = dec.forward(z, t=t, noisy_elem_ids=noisy_elem_ids,
                                      noisy_letter_ids=noisy_letter_ids, per_site_z=per_site_z, elem_cond=None)
                preds['elem_logits'] = _cfg(preds, preds_u)
            last_preds = preds

            elem_probs = torch.softmax(preds['elem_logits'] / temperature, dim=-1)
            B_, S_, V_ = elem_probs.shape
            sampled_elems = torch.multinomial(elem_probs.view(-1, V_), 1).view(B_, S_) + 1

            letter_probs = torch.softmax(preds['letter_logits'] / temperature, dim=-1)
            letter_probs = letter_probs * letter_masks.unsqueeze(1)
            letter_probs = letter_probs / letter_probs.sum(-1, keepdim=True).clamp(min=1e-8)
            L_ = letter_probs.shape[-1]
            sampled_letters = torch.multinomial(letter_probs.view(-1, L_), 1).view(B_, S_) + 1

            unmask_prob = 1.0 / step
            should_unmask = torch.rand(B, dec.max_sites, device=device) < unmask_prob
            still_masked = (noisy_elem_ids == 0)
            update = still_masked & should_unmask
            noisy_elem_ids = torch.where(update, sampled_elems, noisy_elem_ids)
            noisy_letter_ids = torch.where(update, sampled_letters, noisy_letter_ids)

            if (step - 1) in save_steps:
                snapshot(step - 1, preds, expand=False)  # 位点级(不展开)

        # 额外一帧：位点全确定后，按 Wyckoff 对称展开成完整晶体
        snapshot('expanded', last_preds, expand=True)

    # 最终结构的元素Z集合(用展开后的完整结构，排除占位)
    final_zs = set()
    exp = snapshots.get('expanded')
    if exp is not None and exp[0] is not None:
        final_zs = {e.Z for e in exp[0].composition.elements}
    return snapshots, spg_num, final_zs


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    save_steps = set(int(x) for x in args.save_steps.split(','))

    from cdvae.pl_modules.model import WyckoffCDVAE
    from pymatgen.core import Element
    from pymatgen.io.cif import CifWriter

    placeholder_Z = Element(args.placeholder).Z

    print('加载模型...')
    model = WyckoffCDVAE.load_from_checkpoint(args.ckpt, map_location='cpu')
    model.eval()
    dec = model.decoder
    device = 'cpu'

    elem_cond, target_zs = build_elem_cond(args.target_elements, device)
    if elem_cond is not None:
        print(f'目标元素: {args.target_elements} (Z={sorted(target_zs)})  cfg_w={args.cfg_w}  '
              f'筛选={args.need_target}')
    else:
        print('无条件（未指定 target_elements）')
    print(f'生成 {args.n_candidates} 个候选，seed {args.seed_start}..{args.seed_start+args.n_candidates-1}\n')

    def passes(final_zs):
        if elem_cond is None or args.need_target == 'off':
            return True
        if args.need_target == 'any':
            return len(final_zs & target_zs) > 0
        return target_zs.issubset(final_zs)

    n_kept = 0
    for ci in range(args.n_candidates):
        seed = args.seed_start + ci
        snapshots, spg_num, final_zs = run_one_trajectory(
            model, dec, device, seed, elem_cond, args.cfg_w,
            args.temperature, save_steps, placeholder_Z)

        # 最终结构组成(用展开后的完整晶体)
        exp_struct = snapshots.get('expanded', (None,))[0]
        formula = exp_struct.composition.reduced_formula if exp_struct is not None else 'FAILED'
        ok = passes(final_zs)
        tag = '✓保留' if ok else '✗跳过(不含目标)'
        el_syms = sorted(Element.from_Z(z).symbol for z in final_zs) if final_zs else []
        print(f"候选{ci} (seed{seed}): 展开后 {formula} 元素{el_syms}  spg{spg_num}  {tag}")

        if not ok:
            continue

        # 保存该结构的完整轨迹
        sub = os.path.join(args.out_dir, f'struct_{ci:03d}_seed{seed}')
        os.makedirs(sub, exist_ok=True)
        # 去噪帧（位点级，不展开）
        for step in sorted(save_steps, reverse=True):
            entry = snapshots.get(step)
            if entry is None or entry[0] is None:
                print(f"    t={step:3d}: 组装失败，跳过")
                continue
            struct, elems, n_sites, n_masked = entry
            path = os.path.join(sub, f'traj_t{step:03d}_sites.cif')
            try:
                CifWriter(struct).write_file(path)
                print(f"    t={step:3d}: {path}  (位点{n_sites}, 仍MASK {n_masked})")
            except Exception as e:
                print(f"    t={step:3d}: 保存失败 {e}")
        # 最终展开帧（完整晶体）
        exp_entry = snapshots.get('expanded')
        if exp_entry is not None and exp_entry[0] is not None:
            path = os.path.join(sub, 'traj_final_expanded.cif')
            try:
                CifWriter(exp_entry[0]).write_file(path)
                print(f"    展开: {path}  (完整晶体 {exp_entry[0].composition.reduced_formula}, "
                      f"{len(exp_entry[0])} 原子)")
            except Exception as e:
                print(f"    展开: 保存失败 {e}")

        n_kept += 1
        if args.max_keep and n_kept >= args.max_keep:
            print(f"\n已达 max_keep={args.max_keep}，停止。")
            break

    print(f"\n完成。保留 {n_kept} 个含目标元素的结构，各含完整时间步轨迹，在 {args.out_dir}")


if __name__ == '__main__':
    main()