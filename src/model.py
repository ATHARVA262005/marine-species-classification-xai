import torch
import torch.nn as nn
from torchvision import models
import config

def get_model(model_name, num_classes, pretrained=True):
    """
    Loads a pre-trained CNN model and modifies its final classification layer.
    """
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        # Freeze all parameters except the final layer
        for param in model.parameters():
            param.requires_grad = False
        # Replace the final fully connected layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=pretrained)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f"Model '{model_name}' not supported. Choose from 'resnet50', 'efficientnet_b0', 'mobilenet_v2'.")

    return model

if __name__ == '__main__':
    # Example usage:
    model = get_model(config.MODEL_NAME, config.NUM_CLASSES)
    print(f"Model: {config.MODEL_NAME}")
    print(model)

    # Test with a dummy input
    dummy_input = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
