BATCH_SIZE = 4

train_loader = DataLoader(
    PolypGenDataset(train_samples),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    PolypGenDataset(val_samples),
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    PolypGenDataset(test_samples),
    batch_size=BATCH_SIZE,
    shuffle=False
)

client_loaders = []

for client_data in client_samples:
    loader = DataLoader(
        PolypGenDataset(client_data),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    client_loaders.append(loader)
