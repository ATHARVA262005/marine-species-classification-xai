import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import config

def get_transforms(image_size, mean, std):
    """
    Defines image transformations for training, validation, and testing.
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size), # Randomly crop and resize
        transforms.RandomHorizontalFlip(),        # Randomly flip horizontally
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Randomly change brightness, contrast, saturation, hue
        transforms.ToTensor(),                    # Convert image to PyTorch Tensor
        transforms.Normalize(mean=mean, std=std)  # Normalize pixel values
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.1)), # Resize to slightly larger
        transforms.CenterCrop(image_size),        # Crop the center
        transforms.ToTensor(),                    # Convert image to PyTorch Tensor
        transforms.Normalize(mean=mean, std=std)  # Normalize pixel values
    ])
    return train_transform, val_test_transform

def get_dataloaders(train_dir, test_dir, image_size, batch_size, mean, std, val_split_ratio=0.15):
    """
    Loads datasets and creates data loaders for training, validation, and testing.
    """
    train_transform, val_test_transform = get_transforms(image_size, mean, std)

    # Load the full training dataset
    full_train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)

    # Split the training dataset into training and validation sets
    num_train = len(full_train_dataset)
    num_val = int(val_split_ratio * num_train)
    num_train = num_train - num_val
    train_dataset, val_dataset = random_split(full_train_dataset, [num_train, num_val])

    # Apply validation/test transform to the validation dataset
    # Note: random_split gives a Subset, so we need to set its transform explicitly
    val_dataset.dataset.transform = val_test_transform

    # Load the test dataset
    test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count() // 2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count() // 2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count() // 2)

    # Get class names
    class_names = full_train_dataset.classes
    class_to_idx = full_train_dataset.class_to_idx

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")
    print(f"Classes: {class_names}")

    return train_loader, val_loader, test_loader, class_names, class_to_idx

if __name__ == '__main__':
    # Example usage:
    train_loader, val_loader, test_loader, class_names, class_to_idx = get_dataloaders(
        config.TRAIN_DIR, config.TEST_DIR, config.IMAGE_SIZE, config.BATCH_SIZE,
        config.IMAGENET_MEAN, config.IMAGENET_STD
    )
    print(f"First batch from train_loader: {next(iter(train_loader))[0].shape}")
    print(f"First batch from val_loader: {next(iter(val_loader))[0].shape}")
    print(f"First batch from test_loader: {next(iter(test_loader))[0].shape}")
