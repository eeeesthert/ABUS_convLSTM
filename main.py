import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:1024,garbage_collection_threshold:0.8"
import torch
from torchvision import transforms
from dataset import *
from torch.utils.data import DataLoader
from models.model import Freehand3DUSReconstruction
from trainer import train_model, validate_model
from tester import test_model
from metrics import plot_loss_curve, print_test_results


def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    data_dir = "./data"  # 替换为实际数据路径
    os.makedirs("checkpoints", exist_ok=True)

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5], std=[0.5])
    # ])
    transform = transforms.Compose([
        EnsureNativeByteOrder(),
        ConvertToFloat(),
        transforms.ToTensor(),
        transforms.Resize((128, 64)),  # 添加尺寸统一
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = UltrasoundDataset(os.path.join(data_dir, "train"), transform=transform, sequence_length=30)
    val_dataset = UltrasoundDataset(os.path.join(data_dir, "val"), transform=transform, sequence_length=30)
    test_dataset = UltrasoundDataset(os.path.join(data_dir, "test"), transform=transform, sequence_length=30)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    input_dim = 1
    hidden_dim = [64]
    kernel_size = 3
    num_layers = 1
    volume_size = [256, 128, 30]

    model = Freehand3DUSReconstruction(input_dim, hidden_dim, kernel_size, num_layers, volume_size).to(device)

    print("开始训练模型...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=200, lr=1e-3)

    print("绘制损失曲线...")
    plot_loss_curve(train_losses, val_losses)

    print("加载最佳模型...")
    model.load_state_dict(torch.load("checkpoints/freehand_3d_us_model_epoch_200.pth"))

    print("开始测试模型...")
    test_results = test_model(model, test_loader, num_iterations=30, lr=1e-6, device=device)

    print_test_results(test_results)


if __name__ == "__main__":
    main()