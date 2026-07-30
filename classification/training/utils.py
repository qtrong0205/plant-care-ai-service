import torch
import random
import matplotlib.pyplot as plt


def plot_transformed_images(dataloader: torch.utils.data.DataLoader,
                             class_names,
                             n: int = 3,
                             seed: int = None):
    if seed:
        random.seed(seed)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images, labels = next(iter(dataloader))

    batch_size = images.shape[0]
    n = min(n, batch_size) 
    indices = random.sample(range(batch_size), n)

    for idx in indices:
        img = images[idx]
        label = labels[idx]

        img_unnormalized = img * std + mean
        img_unnormalized = img_unnormalized.permute(1, 2, 0).clamp(0, 1)

        plt.figure(figsize=(12, 7))
        plt.imshow(img_unnormalized)
        plt.title(class_names[label])
        plt.axis("off")