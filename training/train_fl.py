def get_model_weights(model):
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}

def set_model_weights(model, weights):
    model.load_state_dict(weights)

def fedavg(client_weights, client_sizes):
    total_size = sum(client_sizes)
    new_weights = {}

    for key in client_weights[0].keys():
        new_weights[key] = sum(
            client_weights[i][key] * (client_sizes[i] / total_size)
            for i in range(len(client_weights))
        )

    return new_weights

def train_local(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

ROUNDS = 50
LOCAL_EPOCHS = 12
LR = 5e-5

global_model = UNet().to(device)

global_model.load_state_dict(
    torch.load("/content/unet_centralized_best.pth", map_location=device)
)

criterion = BCEDiceLoss()

best_val_dice = 0.0

for rnd in range(1, ROUNDS + 1):
    print(f"\n🌍 Federated Round {rnd}/{ROUNDS}")

    client_weights = []
    client_sizes = []

    for client_id, loader in enumerate(client_loaders):
        local_model = UNet().to(device)
        set_model_weights(local_model, get_model_weights(global_model))

        optimizer = torch.optim.Adam(local_model.parameters(), lr=LR)

        train_local(local_model, loader, optimizer, criterion)

        client_weights.append(get_model_weights(local_model))
        client_sizes.append(len(loader.dataset))

        print(f"  Client {client_id} done")

    new_global_weights = fedavg(client_weights, client_sizes)
    set_model_weights(global_model, new_global_weights)

    val_loss, val_dice, val_iou = validate(
        global_model, val_loader, criterion
    )

    print(
        f"  🔎 Global Val Dice: {val_dice:.4f} | "
        f"IoU: {val_iou:.4f}"
    )

    if val_dice > best_val_dice:
        best_val_dice = val_dice
        torch.save(global_model.state_dict(), "unet_fl_best.pth")

global_model.load_state_dict(torch.load("/content/unet_fl_best (3).pth", map_location=device))

test_loss, test_dice, test_iou = validate(
    global_model, test_loader, criterion
)

print("=== Federated Learning Test Performance ===")
print(f"Test Dice: {test_dice:.4f}")
print(f"Test IoU:  {test_iou:.4f}")