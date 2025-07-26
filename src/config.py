import os
import torch

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Updated DATA_DIR based on user's input
DATA_DIR = os.path.join(BASE_DIR, 'data', 'images') # Changed from 'marine-animal-images', 'images'
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
MODELS_DIR = os.path.join(OUTPUTS_DIR, 'models')
VISUALIZATIONS_DIR = os.path.join(OUTPUTS_DIR, 'visualizations')

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# --- Model Parameters ---
MODEL_NAME = 'resnet50' # or 'efficientnet_b0', 'mobilenet_v2'
NUM_CLASSES = 9 # Fish, Goldfish, Harbor seal, Jellyfish, Lobster, Oyster, Sea turtle, Squid, Starfish
IMAGE_SIZE = 224 # Standard input size for many pre-trained models
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 25 # You can increase this for better performance
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Data Augmentation Parameters ---
# Mean and Std Dev for ImageNet pre-trained models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- Training Configuration ---
SAVE_MODEL_PATH = os.path.join(MODELS_DIR, f'{MODEL_NAME}_best_model.pth')
