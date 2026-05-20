
import torch
from torch_geometric.data import Data, Batch

#  测试数据，模拟2个晶体，有3,2个Wyckoff位点
def make_fake_batch():
    # 晶体0: 3个sites
    # 晶体1: 2个sites
    total_sites = 5

    data0 = Data()
    data0.spg_idx         = torch.tensor([74])          # spg74
    data0.wyk_atom_types  = torch.tensor([14, 8, 8])    # Si, O, O
    data0.wyk_letters     = torch.tensor([0, 1, 2])     # a, b, c
    data0.wyk_multi       = torch.tensor([4., 4., 8.])
    data0.wyk_free        = torch.zeros(3, 3)
    data0.wyk_lattice     = torch.tensor([5.0, 5.0, 7.0, 90., 90., 90.])
    data0.num_wyk_sites   = torch.tensor(3)
    data0.y               = torch.tensor([-1.5])        # data 0

    data1 = Data()
    data1.spg_idx         = torch.tensor([225])
    data1.wyk_atom_types  = torch.tensor([26, 8])       # Fe, O
    data1.wyk_letters     = torch.tensor([0, 1])
    data1.wyk_multi       = torch.tensor([4., 4.])
    data1.wyk_free        = torch.zeros(2, 3)
    data1.wyk_lattice     = torch.tensor([4.0, 4.0, 4.0, 90., 90., 90.])
    data1.num_wyk_sites   = torch.tensor(2)
    data1.y               = torch.tensor([-2.0])

    # 
    batch = Batch.from_data_list(
        [data0, data1],
        follow_batch=['wyk_atom_types']
    )
    return batch


def test_encoder(batch):
    from cdvae.pl_modules.wyckoff_encoder import WyckoffEmbedding
    encoder = WyckoffEmbedding(hidden_dim=64, latent_dim=64)
    encoder.eval()
    with torch.no_grad():
        mu, log_var = encoder(batch)
    assert mu.shape == (2, 64), f"mu shape error: {mu.shape}"
    assert log_var.shape == (2, 64), f"log_var shape error: {log_var.shape}"
    print(f"Encoder: mu {mu.shape}, log_var {log_var.shape}")
    return mu, log_var


def test_decoder(mu, log_var):
    from cdvae.pl_modules.wyckoff_decoder import WyckoffDecoder
    decoder = WyckoffDecoder(latent_dim=64, hidden_dim=64, max_sites=4)
    decoder.eval()
    std = torch.exp(0.5 * log_var)
    z = mu + std * torch.randn_like(std)
    with torch.no_grad():
        preds = decoder(z)
    assert preds['spg_logits'].shape    == (2, 230),     f"spg shape error"
    assert preds['lattice_pred'].shape  == (2, 6),       f"lattice shape error"
    assert preds['elem_logits'].shape   == (2, 4, 100),  f"elem shape error"
    assert preds['letter_logits'].shape == (2, 4, 27),   f"letter shape error"
    assert preds['free_params'].shape   == (2, 4, 3),    f"free shape error"
    print(f"Decoder: all correct")
    return z, preds


def test_loss(preds, batch):
    from cdvae.pl_modules.wyckoff_loss import WyckoffReconLoss
    import torch

    S = 4  # max_sites
    B = 2
    device = preds['spg_logits'].device

    # 
    site_batch_idx = batch.wyk_atom_types_batch
    elem_target   = torch.zeros(B, S, dtype=torch.long)
    letter_target = torch.zeros(B, S, dtype=torch.long)
    free_target   = torch.zeros(B, S, 3)
    site_mask     = torch.zeros(B, S, dtype=torch.bool)

    for i in range(B):
        sel = (site_batch_idx == i)
        n   = min(int(sel.sum()), S)
        elem_target[i,   :n]    = batch.wyk_atom_types[sel][:n]
        letter_target[i, :n]    = batch.wyk_letters[sel][:n]
        free_target[i,   :n, :] = batch.wyk_free[sel][:n]
        site_mask[i,     :n]    = True

    targets = {
        'spg_target':       batch.spg_idx.squeeze(-1),
        'lattice_target':   batch.wyk_lattice.view(-1, 6),
        'num_sites_target': (batch.num_wyk_sites - 1).clamp(0, S - 1),
        'elem_target':      elem_target,
        'letter_target':    letter_target,
        'free_target':      free_target,
    }

    loss_fn = WyckoffReconLoss()
    total, loss_dict = loss_fn(preds, targets, site_mask)
    assert not torch.isnan(total), " Loss has NaN！"
    assert total.item() > 0,       " Loss = 0，error"
    print(f" Loss: total={total.item():.4f}")
    for k, v in loss_dict.items():
        val = v.item() if hasattr(v, 'item') else float(v)
        print(f"   {k}: {val:.4f}")


def test_generate():
    from cdvae.pl_modules.wyckoff_decoder import WyckoffDecoder
    decoder = WyckoffDecoder(latent_dim=64, hidden_dim=64, max_sites=4)
    decoder.eval()
    z = torch.randn(3, 64)
    with torch.no_grad():
        results = decoder.decode_to_wyckoff(z)
    assert len(results) == 3
    print(f" Generate:  {len(results)} structures")
    for i, r in enumerate(results):
        print(f"   structure{i}: spg={r['spacegroup_num']}, "
              f"elements={r['site_elements']}, letters={r['site_letters']}")


if __name__ == '__main__':
    print("=" * 50)
    print("Wyckoff-CDVAE test")
    print("=" * 50)

    batch = make_fake_batch()
    print(f"\n[data] batch parameters: {batch.keys}")
    print(f"       wyk_atom_types_batch: {batch.wyk_atom_types_batch}")

    print("\n[1] Encoder test")
    mu, log_var = test_encoder(batch)

    print("\n[2] Decoder test")
    z, preds = test_decoder(mu, log_var)

    print("\n[3] Loss test")
    test_loss(preds, batch)

    print("\n[4] Generate test")
    test_generate()

    print("\n" + "=" * 50)
    print("pass，next training")
    print("=" * 50)