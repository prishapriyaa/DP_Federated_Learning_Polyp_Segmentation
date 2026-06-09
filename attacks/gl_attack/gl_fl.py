model = UNet().to(device)

model.load_state_dict(
    torch.load(
        "unet_fl_best.pth",
        map_location=device
    )
)

client_loader = client_loaders[0]

image, mask = next(iter(client_loader))

image = image[:1].to(device)
mask = mask[:1].to(device)

run_gradient_leakage_attack(
    model,
    image,
    mask,
    "Reconstructed (FL – No DP)"
)