import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

import config
from dataset import get_dataloaders
from model import get_model

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_path):
    """
    Trains the deep learning model.
    """
    best_val_accuracy = 0.0
    model.to(device)

    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # Training loop
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad() # Zero the parameter gradients
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward() # Backpropagation
            optimizer.step() # Update weights

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples
        print(f"Epoch {epoch+1} - Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}")

        # Validation loop
        val_loss, val_accuracy, _, _, _ = evaluate_model(model, val_loader, criterion, device, phase="Validation")
        print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Save the best model based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with Val Accuracy: {best_val_accuracy:.4f} to {save_path}")

    print("Training complete!")

def evaluate_model(model, data_loader, criterion, device, phase="Test", class_names=None):
    """
    Evaluates the model on a given dataset.
    """
    model.eval() # Set model to evaluation mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad(): # Disable gradient calculation during evaluation
        for inputs, labels in tqdm(data_loader, desc=f"{phase} Evaluation"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    avg_loss = running_loss / total_samples
    accuracy = correct_predictions / total_samples

    print(f"\n{phase} Loss: {avg_loss:.4f}, {phase} Accuracy: {accuracy:.4f}")

    if phase == "Test" and class_names:
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions, target_names=class_names))

        # Plot Confusion Matrix
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {phase} Set')
        plt.tight_layout()
        plt.savefig(os.path.join(config.VISUALIZATIONS_DIR, f'{phase.lower()}_confusion_matrix.png'))
        plt.show()

    return avg_loss, accuracy, all_labels, all_predictions, model


if __name__ == '__main__':
    # 1. Get DataLoaders
    train_loader, val_loader, test_loader, class_names, class_to_idx = get_dataloaders(
        config.TRAIN_DIR, config.TEST_DIR, config.IMAGE_SIZE, config.BATCH_SIZE,
        config.IMAGENET_MEAN, config.IMAGENET_STD
    )

    # 2. Get Model
    model = get_model(config.MODEL_NAME, config.NUM_CLASSES)
    print(f"Using device: {config.DEVICE}")
    model.to(config.DEVICE)

    # 3. Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=config.LEARNING_RATE) # Only optimize the new FC layer

    # 4. Train the Model
    train_model(model, train_loader, val_loader, criterion, optimizer, config.NUM_EPOCHS, config.DEVICE, config.SAVE_MODEL_PATH)

    # 5. Load the best model and evaluate on the test set
    print("\nEvaluating the best model on the test set...")
    model.load_state_dict(torch.load(config.SAVE_MODEL_PATH))
    test_loss, test_accuracy, _, _, _ = evaluate_model(model, test_loader, criterion, config.DEVICE, phase="Test", class_names=class_names)
    print(f"Final Test Accuracy: {test_accuracy:.4f}")