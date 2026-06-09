import json
from sklearn.metrics import accuracy_score, roc_auc_score

def collect_losses(model, loader, criterion):
    model.eval()
    losses = []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = criterion(logits, masks)

            losses.append(loss.item())

    return np.array(losses)


def run_mia_attack(model, model_name, output_file):

    member_losses = collect_losses(
        model,
        train_loader,
        criterion
    )

    nonmember_losses = collect_losses(
        model,
        test_loader,
        criterion
    )

    y_true = np.concatenate([
        np.ones(len(member_losses)),
        np.zeros(len(nonmember_losses))
    ])

    scores = np.concatenate([
        -member_losses,
        -nonmember_losses
    ])

    threshold = np.median(scores)

    y_pred = (
        scores >= threshold
    ).astype(int)

    attack_acc = accuracy_score(
        y_true,
        y_pred
    )

    roc_auc = roc_auc_score(
        y_true,
        scores
    )

    loss_gap = (
        nonmember_losses.mean()
        - member_losses.mean()
    )

    print(f"\n=== MIA ({model_name}) ===")
    print(f"Attack Accuracy : {attack_acc:.4f}")
    print(f"ROC-AUC         : {roc_auc:.4f}")
    print(f"Loss Gap        : {loss_gap:.4f}")

    results = {
        "attack_accuracy": float(attack_acc),
        "roc_auc": float(roc_auc),
        "loss_gap": float(loss_gap)
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)