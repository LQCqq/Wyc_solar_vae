import argparse
import json
import os
import gc
from pathlib import Path

import torch
from tqdm import tqdm
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
from mace.calculators import mace_mp


BASE = Path("/srv/scratch/ml4matdis/Quanli_Project/z5561341")
DEFAULT_INPUT  = BASE / "generated_structures"
DEFAULT_OUTPUT = BASE / "MACE_structure"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir",  type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--device",     default="cuda")
    p.add_argument("--mace_model", default="medium-mpa-0")
    p.add_argument("--fmax",       type=float, default=0.05)
    p.add_argument("--max_steps",  type=int,   default=300)
    p.add_argument("--relax_cell", action="store_true")
    p.add_argument("--reset_every",type=int,   default=100,
                   help="每N个结构重建 calculator，防止显存泄漏")
    return p.parse_args()


def make_calc(mace_model, device):
    return mace_mp(model=mace_model, dispersion=False,
                   default_dtype="float64", device=device)


def relax_one(struct_pmg, calc, fmax, max_steps, relax_cell):
    try:
        adaptor = AseAtomsAdaptor()
        atoms   = adaptor.get_atoms(struct_pmg)
        atoms.calc = calc
        target = FrechetCellFilter(atoms) if relax_cell else atoms
        opt = FIRE(target, logfile=None)
        converged = opt.run(fmax=fmax, steps=max_steps)
        energy_per_atom = atoms.get_potential_energy() / len(atoms)
        return adaptor.get_structure(atoms), float(energy_per_atom), bool(converged)
    except Exception as e:
        print(f"    [relaxing error] {e}")
        return None, None, False


def main():
    args = parse_args()

    cif_dir = args.output_dir / "cifs"
    cif_dir.mkdir(exist_ok=True)

    cif_files = sorted(args.input_dir.rglob("*.cif"))
    if not cif_files:
        print(f"[error] {args.input_dir} no CIF")
        return
    print(f"[input] total {len(cif_files)} CIF, from {args.input_dir}")

    # 断点续跑：跳过已存在的输出文件
    todo = [f for f in cif_files if not (cif_dir / f.name).exists()]
    skipped = len(cif_files) - len(todo)
    print(f"[resume] already done: {skipped}  remaining: {len(todo)}")
    if not todo:
        print("[done] all files processed")
        return

    print(f"[MACE] loading {args.mace_model} on {args.device}...")
    calc = make_calc(args.mace_model, args.device)

    results   = []
    n_success = n_fail = 0

    for i, cif_path in enumerate(tqdm(todo, desc="MACE relax")):

        # 每 reset_every 个结构重建 calculator，防止显存泄漏
        if i > 0 and i % args.reset_every == 0:
            del calc
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()
            print(f"\n[reset calculator @ {i}/{len(todo)}]")
            calc = make_calc(args.mace_model, args.device)

        try:
            struct = Structure.from_file(str(cif_path))
        except Exception as e:
            print(f"  [read fail] {cif_path.name}: {e}")
            n_fail += 1
            continue

        relaxed, e_per_atom, converged = relax_one(
            struct, calc, args.fmax, args.max_steps, args.relax_cell
        )

        if relaxed is not None:
            out_cif = cif_dir / cif_path.name
            relaxed.to(str(out_cif))
            results.append({
                "source":          cif_path.name,
                "relaxed_cif":     str(out_cif),
                "energy_per_atom": e_per_atom,
                "converged":       converged,
                "formula":         relaxed.composition.reduced_formula,
                "n_sites":         len(relaxed),
            })
            n_success += 1
        else:
            n_fail += 1

        # 每个结构后清理显存
        if args.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # 合并已有结果（断点续跑时追加，不覆盖）
    summary_path = args.output_dir / "relaxation_summary.json"
    existing = []
    if summary_path.exists():
        with open(summary_path) as f:
            existing = json.load(f)
    all_results = existing + results
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[success] success: {n_success}  failed: {n_fail}")
    print(f"  total accumulated: {len(all_results)}")
    print(f"  relaxing CIF → {cif_dir}")
    print(f"  JSON → {summary_path}")


if __name__ == "__main__":
    main()