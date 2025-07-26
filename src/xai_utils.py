import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import os
from PIL import Image

import config

def get_target_layer(model, model_name):
    """
    Returns the target convolutional layer for Grad-CAM based on the model name.
    You might need to adjust this based on the specific model architecture.
    """
    if model_name == 'resnet50':
        # For ResNet50, use the last bottleneck block
        return model.layer4[2]  # Last bottleneck block in layer4
    elif model_name == 'efficientnet_b0':
        return model.features[-1] # Last block of EfficientNet features
    elif model_name == 'mobilenet_v2':
        return model.features[-1] # Last block of MobileNetV2 features
    else:
        raise ValueError(f"Target layer not defined for model: {model_name}")

def visualize_gradcam(model, input_tensor, original_image_path, target_category, class_names, model_name, output_path):
    """
    Generates and saves a Grad-CAM visualization for a given input.

    Args:
        model (torch.nn.Module): The trained model.
        input_tensor (torch.Tensor): The preprocessed input image tensor.
        original_image_path (str): Path to the original (unprocessed) image.
        target_category (int): The index of the predicted class.
        class_names (list): List of class names.
        model_name (str): Name of the model (e.g., 'resnet50').
        output_path (str): Directory to save the visualization.
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Enable gradients for all parameters temporarily for Grad-CAM
    for param in model.parameters():
        param.requires_grad = True

    try:
        # Get the target layer for Grad-CAM
        target_layer = get_target_layer(model, model_name)
        print(f"Using target layer: {target_layer}")

        # Create a GradCAM object (newer pytorch-grad-cam API doesn't use use_cuda parameter)
        cam = GradCAM(model=model, target_layers=[target_layer])

        # Define the target for CAM (e.g., the predicted class)
        targets = [ClassifierOutputTarget(target_category)]

        # Generate the CAM heatmap
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        
        if grayscale_cam is None:
            print(f"Warning: GradCAM returned None for {original_image_path}")
            return
            
    except Exception as e:
        print(f"Error generating GradCAM for {original_image_path}: {str(e)}")
        # Try alternative target layers for ResNet50
        if model_name == 'resnet50':
            try:
                print("Trying alternative target layer: model.layer4[1]")
                target_layer = model.layer4[1]
                cam = GradCAM(model=model, target_layers=[target_layer])
                targets = [ClassifierOutputTarget(target_category)]
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                if grayscale_cam is None:
                    print("Alternative layer also returned None")
                    return
            except Exception as e2:
                print(f"Alternative target layer also failed: {str(e2)}")
                return
        else:
            return

    # In this example, the image is passed to the model as a tensor, but we need
    # the original image to overlay the heatmap correctly.
    # Load the original image
    original_image = Image.open(original_image_path).convert('RGB')
    # Resize to match model input size for proper overlay
    original_image = original_image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
    # Convert to numpy array and normalize to [0, 1]
    rgb_img = np.float32(original_image) / 255

    # Resize grayscale_cam to match original image dimensions for overlay
    # This is handled by show_cam_on_image internally if dimensions differ,
    # but it's good to be aware.

    # Overlay the heatmap on the original image
    cam_image = show_cam_on_image(rgb_img, grayscale_cam[0, :], use_rgb=True)

    # Display and save the image
    plt.figure(figsize=(8, 8))
    plt.imshow(cam_image)
    plt.title(f"Predicted: {class_names[target_category]}")
    plt.axis('off')
    
    # Save the visualization
    image_filename = os.path.basename(original_image_path)
    save_filepath = os.path.join(output_path, f"gradcam_{class_names[target_category]}_{os.path.splitext(image_filename)[0]}.png")
    plt.savefig(save_filepath)
    plt.close() # Close the plot to free memory
    print(f"Saved Grad-CAM visualization to: {save_filepath}")

def preprocess_image_for_xai(image_path, image_size, mean, std, device):
    """
    Loads and preprocesses an image for model input and XAI.
    Returns the tensor and the original image (for overlay).
    """
    transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.1)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device) # Add batch dimension
    return input_tensor, img
