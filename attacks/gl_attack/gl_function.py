import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

attack_criterion = nn.BCEWithLogitsLoss()

def show_reconstruction(gt, recon, title):
    plt.figure(figsize=(8,4))

    plt.subplot(1,2,1)
    plt.imshow(gt)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(recon)
    plt.title(title)
    plt.axis("off")

    plt.show()


def gradient_leakage_attack(
    model,
    target_gradients,
    true_mask,
    image_shape,
    iters=1200,
    lr=0.05,
    tv_weight=1e-4
):

    dummy_image = torch.rand(
        image_shape,
        device=device,
        requires_grad=True
    )

    optimizer = torch.optim.Adam(
        [dummy_image],
        lr=lr
    )

    for _ in range(iters):

        optimizer.zero_grad()

        pred = model(dummy_image)

        loss = attack_criterion(
            pred,
            true_mask
        )

        grads = torch.autograd.grad(
            loss,
            model.parameters(),
            create_graph=True
        )

        grad_diff = sum(
            ((g - tg) ** 2).sum()
            for g, tg in zip(grads, target_gradients)
        )

        total_loss = grad_diff

        total_loss.backward()
        optimizer.step()

        dummy_image.data.clamp_(0, 1)

    return dummy_image.detach()


def run_gradient_leakage_attack(
    model,
    image,
    mask,
    title,
    use_dp=False,
    noise_std=1.5
):

    model.eval()

    model.zero_grad()

    pred = model(image)

    loss = attack_criterion(
        pred,
        mask
    )

    true_grads = torch.autograd.grad(
        loss,
        model.parameters()
    )

    attack_grads = true_grads

    if use_dp:

        attack_grads = []

        for g in true_grads:

            noise = torch.normal(
                mean=0.0,
                std=noise_std,
                size=g.shape,
                device=device
            )

            attack_grads.append(
                g + noise
            )

    recon_img = gradient_leakage_attack(
        model,
        attack_grads,
        mask,
        image.shape
    )

    show_reconstruction(
        image[0].permute(1,2,0).cpu(),
        recon_img[0].permute(1,2,0).cpu(),
        title
    )