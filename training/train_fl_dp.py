!pip install opacus

import torch
from opacus import PrivacyEngine

DP_CLIP_NORM = 1.0
DP_NOISE_MULTIPLIER = 1.0
DELTA = 1e-5

ROUNDS = 12
LOCAL_EPOCHS = 2
LR = 5e-5
BATCH_SIZE = 1

def dp_step(model, max_grad_norm, noise_multiplier):
    total_norm = torch.norm(
        torch.stack([
            p.grad.norm(2)
            for p in model.parameters()
            if p.grad is not None
        ])
    )

    clip_coef = max_grad_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(clip_coef)

    for p in model.parameters():
        if p.grad is not None:
            noise = torch.normal(
                mean=0.0,
                std=noise_multiplier * max_grad_norm,
                size=p.grad.shape,
                device=p.grad.device
            )
            p.grad.add_(noise)

def train_local_dp_manual(model, loader, optimizer, criterion):
    model.train()
    for _ in range(LOCAL_EPOCHS):
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()

            dp_step(model, DP_CLIP_NORM, DP_NOISE_MULTIPLIER)

            optimizer.step()

global_model = UNet().to(device)
global_model.load_state_dict(
    torch.load("/content/unet_centralized_best.pth", map_location=device)
)

criterion = BCEDiceLoss()
best_val_dice = 0.0

for rnd in range(1, ROUNDS + 1):
    print(f"\n🔐 DP-FL Round {rnd}/{ROUNDS}")

    client_weights = []
    client_sizes = []

    for client_id, loader in enumerate(client_loaders):
        local_model = UNet().to(device)
        set_model_weights(local_model, get_model_weights(global_model))

        optimizer = torch.optim.Adam(local_model.parameters(), lr=LR)

        train_local_dp_manual(
            local_model, loader, optimizer, criterion,
        )

        client_weights.append(get_model_weights(local_model))
        client_sizes.append(len(loader.dataset))

        del local_model, optimizer

    new_weights = fedavg(client_weights, client_sizes)
    set_model_weights(global_model, new_weights)

    val_loss, val_dice, val_iou = validate(
        global_model, val_loader, criterion
    )

    print(f"Val Dice: {val_dice:.4f}")

    if val_dice > best_val_dice:
        best_val_dice = val_dice
        torch.save(global_model.state_dict(), "unet_fl_dp_best.pth")

global_model.load_state_dict(
    torch.load("unet_fl_dp_best.pth", map_location=device)
)

test_loss, test_dice, test_iou = validate(
    global_model, test_loader, criterion
)

print("=== DP-FL Test Performance ===")
print(f"Test Dice: {test_dice:.4f}")
print(f"Test IoU:  {test_iou:.4f}")
