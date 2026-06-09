def dice_score(logits, targets, smooth=1.0):
    probs = torch.sigmoid(logits)
    probs = (probs > 0.5).float()

    intersection = (probs * targets).sum()
    dice = (2. * intersection + smooth) / (
        probs.sum() + targets.sum() + smooth
    )
    return dice.item()

def iou_score(logits, targets, smooth=1.0):
    probs = torch.sigmoid(logits)
    probs = (probs > 0.5).float()

    intersection = (probs * targets).sum()
    union = probs.sum() + targets.sum() - intersection
    return ((intersection + smooth) / (union + smooth)).item()
