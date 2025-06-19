import gc

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torch.cuda.amp as amp


def train_model(model, train_loader, val_loader, num_epochs=200, lr=1e-3):
    # 检查模型参数类型
    print("模型参数类型:", next(model.parameters()).dtype)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    disc_optimizer = optim.Adam(model.discriminator.parameters(), lr=1e-4)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    disc_scheduler = lr_scheduler.StepLR(disc_optimizer, step_size=30, gamma=0.5)

    train_losses = []
    val_losses = []

    device = next(model.parameters()).device
    scaler = amp.GradScaler()  # 创建一个GradScaler实例

    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        #半精度训练
        # model = model.half()
        # model = model.float()
        # for x in train_loader:
        #     x = x.half().to(device)
        model.train()
        epoch_loss = 0.0

        for batch_idx, (slices, transforms) in enumerate(train_loader):
            torch.cuda.empty_cache()
            # slices = slices.to(device)
            slices = slices.to(device, dtype=torch.float32)  # 明确指定为float32
            transforms = transforms.to(device)
            optimizer.zero_grad()
            disc_optimizer.zero_grad()

            with amp.autocast():
                pred_transforms, generated_volume = model(slices)

                mae_loss = torch.mean(torch.abs(pred_transforms - transforms))
                covariance = torch.mean(
                    (pred_transforms - torch.mean(pred_transforms)) * (transforms - torch.mean(transforms)))
                pred_std = torch.std(pred_transforms)
                target_std = torch.std(transforms)
                correlation_loss = 1 - (covariance / (pred_std * target_std + 1e-6))

                train_loss = mae_loss + correlation_loss

                if epoch > 50:
                    ssl_loss = model.calculate_ssl_loss(slices, pred_transforms)
                    train_loss += ssl_loss

                if epoch > 100:
                    real_volume = _get_random_real_volume(train_loader, device)
                    adv_loss, disc_loss = model.calculate_adversarial_loss(generated_volume, real_volume)

            # 判别器反向和优化放在 autocast 外
            if epoch > 100:
                scaler.scale(disc_loss).backward(retain_graph=True)
                scaler.step(disc_optimizer)
                scaler.update()

            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += train_loss.item()

            if batch_idx % 10 == 0:
                print(
                    f'Epoch: {epoch + 1}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {train_loss.item():.4f}')

            del slices, transforms, pred_transforms, generated_volume, train_loss
            gc.collect()
            torch.cuda.empty_cache()

        val_loss = validate_model(model, val_loader, device)
        train_losses.append(epoch_loss / len(train_loader))
        val_losses.append(val_loss)

        scheduler.step()
        disc_scheduler.step()

        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), f'checkpoints/freehand_3d_us_model_epoch_{epoch + 1}.pth')

    return train_losses, val_losses

    #         pred_transforms, generated_volume = model(slices)
    #
    #         mae_loss = torch.mean(torch.abs(pred_transforms - transforms))
    #
    #         covariance = torch.mean(
    #             (pred_transforms - torch.mean(pred_transforms)) * (transforms - torch.mean(transforms)))
    #         pred_std = torch.std(pred_transforms)
    #         target_std = torch.std(transforms)
    #         correlation_loss = 1 - (covariance / (pred_std * target_std + 1e-6))
    #
    #         train_loss = mae_loss + correlation_loss
    #
    #         if epoch > 50:
    #             ssl_loss = model.calculate_ssl_loss(slices, pred_transforms)
    #             train_loss += ssl_loss
    #
    #         if epoch > 100:
    #             real_volume = _get_random_real_volume(train_loader, device)
    #             adv_loss, disc_loss = model.calculate_adversarial_loss(generated_volume, real_volume)
    #
    #             disc_optimizer.zero_grad()
    #             disc_loss.backward(retain_graph=True)
    #             disc_optimizer.step()
    #
    #             train_loss += adv_loss
    #
    #
    #         optimizer.zero_grad()
    #         train_loss.backward()
    #         optimizer.step()
    #
    #         epoch_loss += train_loss.item()
    #
    #         if batch_idx % 10 == 0:
    #             print(
    #                 f'Epoch: {epoch + 1}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {train_loss.item():.4f}')
    #             # 释放不需要的中间变量
    #         del slices, transforms, pred_transforms, generated_volume, train_loss
    #         gc.collect()
    #         torch.cuda.empty_cache()  # 清理缓存
    #     torch.cuda.empty_cache()  # 清理缓存
    #
    #     val_loss = validate_model(model, val_loader, device)
    #     train_losses.append(epoch_loss / len(train_loader))
    #     val_losses.append(val_loss)
    #
    #     scheduler.step()
    #     disc_scheduler.step()
    #
    #     if (epoch + 1) % 50 == 0:
    #         torch.save(model.state_dict(), f'checkpoints/freehand_3d_us_model_epoch_{epoch + 1}.pth')
    #
    # return train_losses, val_losses



def validate_model(model, val_loader, device):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for slices, transforms in val_loader:
            slices = slices.to(device)
            transforms = transforms.to(device)

            pred_transforms, _ = model(slices)
            mae_loss = torch.mean(torch.abs(pred_transforms - transforms))

            covariance = torch.mean(
                (pred_transforms - torch.mean(pred_transforms)) * (transforms - torch.mean(transforms)))
            pred_std = torch.std(pred_transforms)
            target_std = torch.std(transforms)
            correlation_loss = 1 - (covariance / (pred_std * target_std + 1e-6))

            val_loss += (mae_loss + correlation_loss).item()

    return val_loss / len(val_loader)


def _get_random_real_volume(loader, device):
    dataiter = iter(loader)
    try:
        slices, _ = next(dataiter)
    except StopIteration:
        dataiter = iter(loader)
        slices, _ = next(dataiter)
    return torch.mean(slices, dim=1, keepdim=True).to(device)