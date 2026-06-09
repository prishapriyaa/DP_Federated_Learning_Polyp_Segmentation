central_model = UNet().to(device)

central_model.load_state_dict(
    torch.load(
        "unet_centralized_best.pth",
        map_location=device
    )
)

run_mia_attack(
    central_model,
    "Centralized Learning",
    "mia_cl_results.json"
)