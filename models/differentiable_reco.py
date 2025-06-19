import torch
import torch.nn as nn


# class DifferentiableReconstruction(nn.Module):
    # def __init__(self, volume_size, epsilon=1e-6):
    #     super(DifferentiableReconstruction, self).__init__()
    #     self.volume_size = volume_size
    #     self.epsilon = epsilon
    #
    # def forward(self, slices, transforms,slice_indices=None): #,slice_indices=None
    #     # # 先打印张量形状用于调试
    #     # print(f"transforms shape: {transforms.shape}")
    #     # print(f"slice_indices shape: {slice_indices.shape}")
    #     # batch_size, seq_len, _, height, width = slices.size()
    #     #
    #     # if slice_indices is None:
    #     #     slice_indices = torch.arange(seq_len, device=slices.device)
    #     #     slice_indices = slice_indices.unsqueeze(0).expand(batch_size, -1)
    #     print(f"transforms shape: {transforms.shape}")
    #     batch_size, seq_len, _, height, width = slices.size()
    #
    #     if slice_indices is None:
    #         slice_indices = torch.arange(seq_len, device=slices.device)
    #         slice_indices = slice_indices.unsqueeze(0).expand(batch_size, -1)
    #
    #     # 仅当slice_indices不为None时打印
    #     if slice_indices is not None:
    #         print(f"slice_indices shape: {slice_indices.shape}")
    #
    #
    #
    #     selected_slices = torch.gather(slices, 1, slice_indices.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(-1, -1,
    #                                                                                                           slices.size(
    #                                                                                                               2),
    #                                                                                                           slices.size(
    #                                                                                                               3),
    #                                                                                                           slices.size(
    #                                                                                                               4)))
    #     selected_transforms = torch.gather(transforms, 1, slice_indices.unsqueeze(2).expand(-1, -1, transforms.size(2)))
    #
    #     volume = torch.zeros(batch_size, 1, self.volume_size[0], self.volume_size[1], self.volume_size[2],
    #                          device=slices.device)
    #
    #     for x in range(self.volume_size[0]):
    #         for y in range(self.volume_size[1]):
    #             for z in range(self.volume_size[2]):
    #                 distances = []
    #                 intensities = []
    #
    #                 for t in range(selected_slices.size(1)):
    #                     tx, ty, tz, rx, ry, rz = torch.split(selected_transforms[:, t], 1, dim=2)
    #                     tx, ty, tz = tx.squeeze(2), ty.squeeze(2), tz.squeeze(2)
    #
    #                     voxel_pos = torch.tensor([x, y, z], device=slices.device).unsqueeze(0).unsqueeze(2)
    #                     slice_pos = torch.stack([tx, ty, tz], dim=2)
    #
    #                     distance = torch.sqrt(torch.sum((voxel_pos - slice_pos) ** 2, dim=2))
    #                     distances.append(distance)
    #
    #                     intensity = selected_slices[:, t, :, y, x].squeeze(1)
    #                     intensities.append(intensity)
    #
    #                 distances = torch.stack(distances, dim=1)
    #                 intensities = torch.stack(intensities, dim=1)
    #
    #                 weights = torch.exp(1.0 / (distances + self.epsilon))
    #                 weights = weights / torch.sum(weights, dim=1, keepdim=True)
    #
    #                 reconstructed_intensity = torch.sum(weights * intensities, dim=1)
    #                 volume[:, :, x, y, z] = reconstructed_intensity.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    #
    #     return volume
from torch.cuda import amp


class DifferentiableReconstruction(nn.Module):
    def __init__(self, volume_size, epsilon=1e-6):
        super(DifferentiableReconstruction, self).__init__()
        self.volume_size = volume_size
        self.epsilon = epsilon

    def forward(self, slices, transforms, slice_indices=None):
        assert transforms.dim() == 3 and transforms.size(2) >= 3, \
            f"Expected transforms of shape [B, T, D≥3], got {transforms.shape}"

        batch_size, seq_len, channels, height, width = slices.size()
        vol_x, vol_y, vol_z = self.volume_size
        V = vol_x * vol_y * vol_z  # 体素总数

        # 若未提供索引，则默认使用顺序切片索引
        if slice_indices is None:
            slice_indices = torch.arange(seq_len, device=slices.device).unsqueeze(0).expand(batch_size, -1)
        print(f"transforms shape: {transforms.shape}")
        print(f"slice_indices shape: {slice_indices.shape}")

        # 选取切片和对应变换
        if transforms.dim() == 2:
            selected_transforms = torch.gather(transforms, 1, slice_indices)
        elif transforms.dim() == 3:
            expand_dim = transforms.size(2)
            expanded_indices = slice_indices.unsqueeze(2).expand(-1, -1, expand_dim)
            selected_transforms = torch.gather(transforms, 1, expanded_indices)
        else:
            raise ValueError("Unsupported transforms shape")

        selected_slices = torch.gather(
            slices, 1,
            slice_indices.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(-1, -1, channels, height, width)
        )  # [B, T, 1, H, W]

        # -------- 体素坐标生成 --------
        grid = torch.stack(torch.meshgrid(
            torch.arange(vol_x, device=slices.device),
            torch.arange(vol_y, device=slices.device),
            torch.arange(vol_z, device=slices.device),
            indexing='ij'
        ), dim=-1)  # [X, Y, Z, 3]
        voxel_coords = grid.reshape(1, 1, V, 3)  # [1, 1, V, 3]
        slice_coords = selected_transforms[..., :3]  # [B, T, 3]

        print("voxel_coords.shape:", voxel_coords.shape)
        print("slice_coords.shape:", slice_coords.shape)
        print("selected_slices.shape:", selected_slices.shape)

        # 提取切片像素值并展开为 intensity_matrix: [B, T, H*W]
        intensity_matrix = selected_slices.squeeze(2).reshape(batch_size, -1, height * width)  # [B, T, X*Y]
        print("intensity_matrix.shape:", intensity_matrix.shape)

        chunk_size = 100  # 体素块大小，可调
        reconstructed_chunks = []

        with amp.autocast():
            for start in range(0, V, chunk_size):
                end = min(start + chunk_size, V)
                voxel_chunk = voxel_coords[:, :, start:end, :]  # [1, 1, chunk, 3]

                # 距离计算
                distance_chunk = torch.sqrt(
                    torch.sum((voxel_chunk - slice_coords.unsqueeze(2)) ** 2, dim=-1)
                )  # [B, T, chunk]

                weight_chunk = torch.exp(1.0 / (distance_chunk + self.epsilon))
                weight_chunk = weight_chunk / torch.sum(weight_chunk, dim=1, keepdim=True)  # [B, T, chunk]

                # 对应体素值为切片加权像素值平均: [B, chunk, 1]
                weighted_intensity = torch.matmul(weight_chunk.transpose(1, 2), intensity_matrix)  # [B, chunk, H*W]
                recon_chunk = weighted_intensity.mean(dim=2, keepdim=True)  # [B, chunk, 1]
                reconstructed_chunks.append(recon_chunk)

                del voxel_chunk, distance_chunk, weight_chunk, weighted_intensity, recon_chunk
                torch.cuda.empty_cache()

        # 拼接后变为 [B, V, 1] → reshape 为体积 [B, 1, X, Y, Z]
        reconstructed = torch.cat(reconstructed_chunks, dim=1)  # [B, V, 1]
        volume = reconstructed.reshape(batch_size, 1, vol_x, vol_y, vol_z)
        print(f"volume shape: {volume.shape}")
        return volume

    def reconstruct_slices(self, volume, transforms, slice_indices):
        batch_size, _, vol_x, vol_y, vol_z = volume.size()
        slices = torch.zeros(batch_size, len(slice_indices), 1, vol_x, vol_y, device=volume.device)

        for b in range(batch_size):
            for i, idx in enumerate(slice_indices[b]):
                tx, ty, tz, rx, ry, rz = transforms[b, idx].split(1)
                tx, ty, tz = tx.item(), ty.item(), tz.item()

                # 简化提取切片
                z_idx = int(tz)
                z_idx = max(0, min(z_idx, vol_z - 1))
                slices[b, i, :, :, :] = volume[b, :, :, :, z_idx]

        return slices
