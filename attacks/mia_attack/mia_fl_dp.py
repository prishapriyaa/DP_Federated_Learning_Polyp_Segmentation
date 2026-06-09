dp_fl_model = UNet().to(device)

dp_fl_model.load_state_dict(
    torch.load(
        "unet_fl_dp_best.pth",
        map_location=device
    )
)

run_mia_attack(
    dp_fl_model,
    "Federated Learning + DP",
    "mia_fl_dp_results.json"
)