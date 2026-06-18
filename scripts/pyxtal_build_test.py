"""
诊断 overlap penalty 是否导致 repeats/索引错误。强制 CPU 运行，
CPU 上的越界索引会同步报错且信息准确（不像 CUDA 异步会错位到 encoder）。

用法（PBS）：python diag_penalty.py
"""
import os
import torch
import hydra
from hydra import compose, initialize_config_dir

PROJECT_ROOT = os.environ.get(
    'PROJECT_ROOT',
    '/srv/scratch/ml4matdis/Quanli_Project/z5561341/projectA/cdvae'
)
CONF = os.path.join(PROJECT_ROOT, 'conf')


def main():
    with initialize_config_dir(config_dir=CONF):
        cfg = compose(
            config_name='default',
            overrides=['data=mp_20', 'model=wyckoff_cdvae',
                       'expname=t', 'logging.wandb.mode=disabled'],
        )
        dm = hydra.utils.instantiate(cfg.data.datamodule, _recursive_=False)
        dm.setup()
        model = hydra.utils.instantiate(cfg.model, _recursive_=False)
        model.eval()

        # 先看 lookup 能否加载
        from cdvae.pl_modules.wyckoff_loss import _load_wyckoff_ops, _WYCKOFF_OPS_PATH
        print('lookup 路径:', _WYCKOFF_OPS_PATH, flush=True)
        print('lookup 存在?', os.path.exists(_WYCKOFF_OPS_PATH), flush=True)
        lut = _load_wyckoff_ops(torch.device('cpu'))
        print('lookup 加载?', lut is not None, flush=True)
        if lut is not None:
            print('  R shape:', tuple(lut['R'].shape), 'NUM_LETTERS:', lut['NUM_LETTERS'],
                  'N_KEY:', lut['N_KEY'], 'radii len:', lut['element_radii'].shape[0], flush=True)

        vl = dm.val_dataloader()
        vl = vl[0] if isinstance(vl, list) else vl

        for bi, batch in enumerate(vl):
            print(f'\n===== batch {bi} =====', flush=True)
            # 1) 先单独验证 encoder（不经过 loss）
            try:
                with torch.no_grad():
                    enc_out = model.encoder(batch)
                print('  encoder 单独: OK', flush=True)
            except Exception as e:
                print(f'  encoder 单独: FAIL -> {type(e).__name__}: {e}', flush=True)
                n = batch.num_wyk_sites
                print(f'    num_wyk_sites: shape={tuple(n.shape)} min={n.min().item()} '
                      f'max={n.max().item()} neg={(n<0).any().item()}', flush=True)
                break

            # 2) 完整 forward（含 loss + penalty）
            try:
                with torch.no_grad():
                    out = model(batch)
                if isinstance(out, tuple):
                    loss_dict = out[1] if len(out) > 1 else {}
                    if isinstance(loss_dict, dict):
                        print('  full forward: OK  loss_overlap =',
                              loss_dict.get('loss_overlap', 'N/A'), flush=True)
                    else:
                        print('  full forward: OK', flush=True)
                else:
                    print('  full forward: OK', flush=True)
            except Exception as e:
                print(f'  full forward: FAIL -> {type(e).__name__}: {e}', flush=True)
                # 既然 encoder 单独 OK，full forward 崩 → 问题在 loss/penalty
                # 打印 penalty 所需字段的真实范围，定位越界索引
                _inspect_targets(model, batch, lut)
                import traceback
                traceback.print_exc()
                break

            if bi >= 3:
                print('\n前若干 batch 全部 OK，未复现崩溃', flush=True)
                break

    print('\nDONE', flush=True)


def _inspect_targets(model, batch, lut):
    """打印 penalty 用到的 spg/letter/elem 的真实索引范围，找越界。"""
    print('  ---- 检查 penalty 输入字段范围 ----', flush=True)
    # 尝试从 model 构造 targets（不同实现字段名可能不同）
    # 直接从 batch 找常见字段
    cand = {}
    for name in ['spg_idx', 'spacegroup', 'letter_idx', 'wyk_letters',
                 'atom_types', 'wyk_atom_types', 'num_wyk_sites']:
        if hasattr(batch, name):
            v = getattr(batch, name)
            if torch.is_tensor(v):
                cand[name] = v
                print(f'    {name}: shape={tuple(v.shape)} dtype={v.dtype} '
                      f'min={v.min().item()} max={v.max().item()}', flush=True)

    if lut is not None:
        NL = lut['NUM_LETTERS']
        NK = lut['N_KEY']
        RL = lut['element_radii'].shape[0]
        print(f'    [lookup 限制] NUM_LETTERS={NL} N_KEY={NK} radii_len={RL}', flush=True)
        # letter 越界检查
        for lname in ['letter_idx', 'wyk_letters']:
            if lname in cand:
                lt = cand[lname]
                over = ((lt < 0) | (lt >= NL)).sum().item()
                print(f'    letter字段 {lname}: 越界(<0或>={NL})的元素数 = {over}', flush=True)
        # elem 越界检查
        for ename in ['atom_types', 'wyk_atom_types']:
            if ename in cand:
                et = cand[ename]
                over = ((et < 0) | (et >= RL)).sum().item()
                print(f'    elem字段 {ename}: 越界(<0或>={RL})的元素数 = {over}', flush=True)


if __name__ == '__main__':
    main()