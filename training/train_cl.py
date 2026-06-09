model = UNet().to(device)
criterion = BCEDiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=5
)

EPOCHS = 50

best_val_dice = 0.0

for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch(
        model, train_loader, optimizer, criterion
    )

    val_loss, val_dice, val_iou = validate(
        model, val_loader, criterion
    )

    scheduler.step(val_dice)

    print(
        f"Epoch {epoch:02d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Dice: {val_dice:.4f} | "
        f"IoU: {val_iou:.4f}"
    )

    if val_dice > best_val_dice:
        best_val_dice = val_dice
        torch.save(model.state_dict(), "unet_centralized_best.pth")

model.load_state_dict(torch.load("unet_centralized_best.pth"))

test_loss, test_dice, test_iou = validate(
    model, test_loader, criterion
)

print("=== Centralized Test Performance ===")
print(f"Test Dice: {test_dice:.4f}")
print(f"Test IoU:  {test_iou:.4f}")

