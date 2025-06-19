import torch
import torch.nn as nn


class SelfSupervisedLearning(nn.Module):
    def __init__(self, reco_ratio=0.5):
        super(SelfSupervisedLearning, self).__init__()
        self.reco_ratio = reco_ratio

    def forward(self, slices, transforms, differentiable_reco):
        batch_size, seq_len, _, height, width = slices.size()
        volume_size = [height, width, seq_len]

        num_reco = int(seq_len * self.reco_ratio)
        reco_indices = torch.randint(0, seq_len, (batch_size, num_reco), device=slices.device)

        reconstructed_volume = differentiable_reco(slices, transforms, reco_indices)

        all_indices = torch.arange(seq_len, device=slices.device).unsqueeze(0).expand(batch_size, -1)
        remaining_indices = torch.zeros(batch_size, seq_len - num_reco, dtype=torch.long, device=slices.device)

        for b in range(batch_size):
            remaining = torch.tensor([i for i in all_indices[b] if i not in reco_indices[b]], device=slices.device)
            remaining_indices[b] = remaining[:seq_len - num_reco]

        reconstructed_slices = differentiable_reco.reconstruct_slices(reconstructed_volume, transforms,
                                                                      remaining_indices)

        original_remaining = torch.gather(slices, 1,
                                          remaining_indices.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(-1, -1,
                                                                                                          slices.size(
                                                                                                              2),
                                                                                                          slices.size(
                                                                                                              3),
                                                                                                          slices.size(
                                                                                                              4)))
        ssl_loss = torch.mean(torch.abs(reconstructed_slices - original_remaining))

        return ssl_loss