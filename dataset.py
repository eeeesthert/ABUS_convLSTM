from torch.utils.data import Dataset
import numpy as np
import torch
from torchvision import transforms
import random
import nrrd
import os


class EnsureNativeByteOrder:
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            if pic.dtype.byteorder != '=':
                # 新的字节序转换方式
                new_dtype = pic.dtype.newbyteorder('=')  # '=' 表示本地字节序
                pic = pic.view(new_dtype)
        return pic


class ConvertToFloat:
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            if pic.dtype == np.uint16:
                return pic.astype(np.float32) / 65535.0
            elif pic.dtype == np.uint8:
                return pic.astype(np.float32) / 255.0
        return pic


class UltrasoundDataset(Dataset):
    def __init__(self, data_dir, transform=None, sequence_length=30, target_size=(256,128)):
        self.data_dir = data_dir
        self.transform = transform
        self.sequence_length = sequence_length
        self.target_size = target_size
        self.nrrd_files = [f for f in os.listdir(data_dir) if f.endswith('.nrrd')]
        random.shuffle(self.nrrd_files)

    def __len__(self):
        return len(self.nrrd_files) * 10

    def __getitem__(self, idx):
        file_idx = idx // 10
        seq_idx = idx % 10
        data_path = os.path.join(self.data_dir, self.nrrd_files[file_idx])
        data, header = self._load_nrrd(data_path)

        sequence, transforms = self.generate_complex_sequence(data, seq_idx)

        # 确保所有切片形状一致并转换为NumPy数组
        sequence = self._ensure_uniform_shape(sequence)

        # 转换为PyTorch张量 [B, H, W] -> [B, 1, H, W]
        sequence = torch.FloatTensor(sequence).unsqueeze(1)
        # transforms = torch.FloatTensor(transforms)
        transforms = torch.FloatTensor(np.array(transforms))
        return sequence, transforms

    def _load_nrrd(self, path):
        data, header = nrrd.read(path)
        if data.ndim == 4:
            data = data[..., 0]  # 移除通道维度
        return data, header

    def _ensure_uniform_shape(self, sequence):
        """确保所有切片为2D且形状一致"""
        if not sequence:
            return np.array([], dtype=np.float32)

        # 转换所有元素为NumPy数组并检查维度
        uniform_sequence = []
        for slice in sequence:
            if isinstance(slice, torch.Tensor):
                slice = slice.numpy()

            # 确保是2D数组
            if slice.ndim == 3:
                slice = slice.squeeze(0)  # 移除通道维度

            # 调整尺寸
            if slice.shape[:2] != self.target_size:
                slice = self._resize_slice(slice)

            uniform_sequence.append(slice)

        # 转换为统一的NumPy数组
        return np.array(uniform_sequence, dtype=np.float32)

    def _resize_slice(self, slice):
        """将切片调整为目标尺寸并保持2D"""
        # 转换为张量并添加通道维度 (H, W) -> (1, H, W)
        if isinstance(slice, np.ndarray):
            slice = torch.from_numpy(slice).unsqueeze(0)

        # 调整尺寸
        slice = transforms.functional.resize(
            slice,
            size=self.target_size,
            interpolation=transforms.InterpolationMode.BILINEAR
        )

        # 移除通道维度并转回NumPy数组 (1, H, W) -> (H, W)
        return slice.squeeze(0).numpy()

    def generate_complex_sequence(self, volume, seq_idx):
        depth = volume.shape[2]
        sequence = []
        transforms = []
        scan_types = ['linear', 'loop', 'fast_slow', 'sector']
        current_scan = scan_types[seq_idx % len(scan_types)]

        pos = np.array([volume.shape[0] // 2, volume.shape[1] // 2, 0], dtype=np.float32)
        rot = np.array([0, 0, 0], dtype=np.float32)

        for i in range(self.sequence_length):
            if current_scan == 'linear':
                pos[2] += depth / self.sequence_length
                delta_rot = np.zeros(3, dtype=np.float32)
            elif current_scan == 'loop':
                angle = 2 * np.pi * i / self.sequence_length
                pos[0] = volume.shape[0] // 2 + 50 * np.cos(angle)
                pos[1] = volume.shape[1] // 2 + 50 * np.sin(angle)
                pos[2] = depth // 2
                delta_rot = np.array([0, 0, angle * 2], dtype=np.float32)
            elif current_scan == 'fast_slow':
                speed = 0.5 + 0.5 * np.sin(2 * np.pi * i / self.sequence_length)
                pos[2] += (depth / self.sequence_length) * speed
                delta_rot = np.zeros(3, dtype=np.float32)
            elif current_scan == 'sector':
                angle = np.pi * i / self.sequence_length
                pos[0] = volume.shape[0] // 2
                pos[1] = volume.shape[1] // 2
                pos[2] = depth // 2
                delta_rot = np.array([0, angle, 0], dtype=np.float32)

            rot += delta_rot

            slice_idx = int(pos[2])
            slice_idx = max(0, min(slice_idx, depth - 1))
            current_slice = volume[:, :, slice_idx]

            # 处理字节序和数据类型
            current_slice = EnsureNativeByteOrder()(current_slice)
            current_slice = ConvertToFloat()(current_slice)

            if self.transform:
                current_slice = self.transform(current_slice)

            sequence.append(current_slice)
            transforms.append(np.concatenate([pos, rot]))

        return sequence, transforms