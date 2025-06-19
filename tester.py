import torch
import torch.optim as optim
import numpy as np


def test_model(model, test_loader, num_iterations=30, lr=1e-6, device=None):
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    results = []
    optimizer = optim.Adam(model.convlstm.parameters(), lr=lr)

    for batch_idx, (slices, transforms) in enumerate(test_loader):
        slices = slices.to(device)
        transforms = transforms.to(device)

        for i in range(num_iterations):
            pred_transforms, generated_volume = model(slices)
            ssl_loss = model.calculate_ssl_loss(slices, pred_transforms)

            real_volume = _get_random_real_volume(test_loader, device)
            adv_loss, _ = model.calculate_adversarial_loss(generated_volume, real_volume)

            total_loss = ssl_loss + adv_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        result = evaluate_model(pred_transforms, transforms, device)
        results.append(result)

    return results


def evaluate_model(pred_transforms, target_transforms, device):
    batch_size, seq_len, _ = pred_transforms.size()
    results = {}

    cumul_drift = torch.zeros(batch_size, seq_len, 3, device=device)
    for i in range(1, seq_len):
        pred_trans = pred_transforms[:, i, :3]
        target_trans = target_transforms[:, i, :3]
        cumul_drift[:, i] = cumul_drift[:, i - 1] + (pred_trans - target_trans)

    final_drift = torch.norm(cumul_drift[:, -1], dim=1)
    fdr = final_drift / seq_len
    results['fdr'] = torch.mean(fdr).item()

    path_lengths = torch.arange(1, seq_len + 1, device=device).float()
    adr = torch.mean(torch.norm(cumul_drift, dim=2) / path_lengths)
    results['adr'] = adr.item()

    md = torch.mean(torch.max(torch.norm(cumul_drift, dim=2), dim=1)[0]).item()
    sd = torch.mean(torch.sum(torch.norm(cumul_drift, dim=2), dim=1)).item()

    all_drifts = torch.norm(cumul_drift, dim=2)
    hd = torch.mean(torch.max(all_drifts, dim=1)[0] + torch.max(all_drifts, dim=1)[0]) / 2
    results['hd'] = hd.item()

    return results


def _get_random_real_volume(loader, device):
    dataiter = iter(loader)
    try:
        slices, _ = next(dataiter)
    except StopIteration:
        dataiter = iter(loader)
        slices, _ = next(dataiter)
    return torch.mean(slices, dim=1, keepdim=True).to(device)