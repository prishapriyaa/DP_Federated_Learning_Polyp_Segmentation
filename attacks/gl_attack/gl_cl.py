model = UNet().to(device)

model.load_state_dict(
    torch.load(
        "unet_centralized_best.pth",
        map_location=device
    )
)

image, mask = next(iter(train_loader))

image = image[:1].to(device)
mask = mask[:1].to(device)

run_gradient_leakage_attack(
    model,
    image,
    mask,
    "Reconstructed (Centralized)"
)