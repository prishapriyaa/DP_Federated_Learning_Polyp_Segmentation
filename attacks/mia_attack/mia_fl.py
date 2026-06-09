fl_model = UNet().to(device)

fl_model.load_state_dict(
    torch.load(
        "unet_fl_best.pth",
        map_location=device
    )
)

run_mia_attack(
    fl_model,
    "Federated Learning",
    "mia_fl_results.json"
)