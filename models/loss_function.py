class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )
        return 1 - dice

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        return 0.3 * self.bce(logits, targets) + 0.7 * self.dice(logits, targets)

model = UNet()
x = torch.randn(2, 3, 256, 256)
y = model(x)

print("Output shape:", y.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
