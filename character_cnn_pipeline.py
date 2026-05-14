import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from character_dataset import HandwrittenCharacterSynthesizer, render_sample_character_grid

# Apply clean custom styling for presentation outputs
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

# Ensure highly reproducible arrays and models
torch.manual_seed(42)
np.random.seed(42)

class CharacterRecognitionCNN(nn.Module):
    """
    Standard highly expressive Convolutional Neural Network (CNN) architecture
    tailored to extract local edge orientations, crossbar topologies, and loop
    bounds from 28x28 grayscale character frames.
    
    CRNN Extension Readiness:
    The final feature extraction blocks map seamlessly to temporal Recurrent layers
    (GRU/LSTM) to decode connected continuous cursive word strings.
    """
    def __init__(self, num_classes=8):
        super(CharacterRecognitionCNN, self).__init__()
        
        # Convolutions block 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output: 16 x 14 x 14
        )
        
        # Convolutions block 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output: 32 x 7 x 7
        )
        
        # Classification projection heads
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, channels=1, height=28, width=28)
        out = self.layer1(x)
        out = self.layer2(out)
        
        # Flatten spatial tensors into contiguous embedding vectors
        out = out.reshape(out.size(0), -1)
        logits = self.fc(out)
        return logits


def generate_and_prepare_data(vis_dir='visualizations', samples_per_class=250):
    """
    Synthesizes normalized image data, generates exploratory grids,
    and returns stratified streaming PyTorch structures.
    """
    os.makedirs(vis_dir, exist_ok=True)
    print("=" * 70)
    print("STEP 1: Synthesizing Grayscale Character Arrays & Ingestion")
    print("=" * 70)
    
    synth = HandwrittenCharacterSynthesizer()
    X, y, classes = synth.generate_dataset(samples_per_class=samples_per_class)
    
    print(f"Ingested Custom Character Database: Shape={X.shape} | Data Type={X.dtype}")
    
    # Render visual layout reference
    render_sample_character_grid(X, y, classes, vis_dir=vis_dir)
    
    # Perform Stratified Partitions
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Tensor Wrapping
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Training Partitions: {len(train_dataset)} Image Arrays")
    print(f"Testing Partitions:  {len(test_dataset)} Image Arrays\n")
    
    return train_loader, test_loader, X_test, y_test, classes

def train_cnn_model(train_loader, test_loader, num_classes=8, epochs=20, vis_dir='visualizations'):
    """
    Executes core training optimization minimizing spatial Cross-Entropy loss
    while retaining classification convergence history.
    """
    print("=" * 70)
    print("STEP 2: Optimizing Convolutional Features Architecture")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Hardware Execution Core: [{device}]")
    
    model = CharacterRecognitionCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0015, weight_decay=1e-5)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        corrects = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == targets).item()
                
        epoch_val_loss = val_loss / len(test_loader.dataset)
        epoch_val_acc = corrects / len(test_loader.dataset)
        
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        
        if (epoch + 1) % 4 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:02d}/{epochs}] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Accuracy: {epoch_val_acc*100:.2f}%")
            
    # Render Optimization Curves
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    axes[0].plot(range(1, epochs+1), train_losses, label='Training Loss', color='#9b59b6', lw=2)
    axes[0].plot(range(1, epochs+1), val_losses, label='Validation Loss', color='#34495e', lw=2, linestyle='--')
    axes[0].set_title("CNN Loss Minimization Curves", fontweight='bold')
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    
    axes[1].plot(range(1, epochs+1), [acc * 100 for acc in val_accuracies], label='Val Accuracy', color='#2ecc71', lw=2)
    axes[1].set_title("Validation Subset Accuracy Trajectory", fontweight='bold')
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    
    plt.tight_layout()
    curve_path = os.path.join(vis_dir, 'cnn_training_curves.png')
    plt.savefig(curve_path, dpi=300)
    plt.close()
    print(f"\n Saved structural optimization convergence charts to '{curve_path}'\n")
    
    return model, device

def evaluate_and_visualize_predictions(model, device, X_test, y_test, classes, vis_dir='visualizations'):
    """
    Renders granular multiclass boundary metrics alongside sample graphical grids
    displaying final model predictions versus true array labels.
    """
    print("=" * 70)
    print("STEP 3: Model Prediction Metrics & Confusion Bounds")
    print("=" * 70)
    
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_test, dtype=torch.float32).to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        
    y_pred = preds.cpu().numpy()
    
    print("Detailed CNN Image Recognition Metric Report:\n")
    print(classification_report(y_test, y_pred, target_names=classes))
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", cbar=True,
                xticklabels=classes, yticklabels=classes, annot_kws={"size": 14})
    plt.title("Handwritten Character Confusion Matrix", pad=20, fontweight='bold')
    plt.xlabel("Predicted Class Label", fontweight='bold')
    plt.ylabel("True Input Label", fontweight='bold')
    plt.tight_layout()
    
    cm_path = os.path.join(vis_dir, 'character_confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f" Saved multi-class visual confusion bounds to '{cm_path}'")
    
    # Render Sample Prediction Verifications Grid
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    # Pick a sequence of 10 random inputs from the validation subset
    sample_indices = np.random.choice(len(y_test), size=10, replace=False)
    
    for i, idx in enumerate(sample_indices):
        img = X_test[idx, 0, :, :]
        true_label = classes[y_test[idx]]
        pred_label = classes[y_pred[idx]]
        
        # Color title based on correctness
        title_color = '#27ae60' if true_label == pred_label else '#c0392b'
        
        axes[i].imshow(img, cmap='gray_r', interpolation='nearest')
        axes[i].set_title(f"P: {pred_label} | T: {true_label}", fontweight='bold', color=title_color)
        axes[i].axis('off')
        
    plt.tight_layout()
    pred_path = os.path.join(vis_dir, 'prediction_samples.png')
    plt.savefig(pred_path, dpi=300)
    plt.close()
    print(f" Rendered evaluation prediction review frames to '{pred_path}'")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    train_ldr, test_ldr, X_tst, y_tst, cls_list = generate_and_prepare_data(samples_per_class=250)
    cnn_model, target_dev = train_cnn_model(train_ldr, test_ldr, num_classes=len(cls_list), epochs=20)
    evaluate_and_visualize_predictions(cnn_model, target_dev, X_tst, y_tst, cls_list)
    print(">> Character Recognition Deep Learning Pipeline execution successfully finalized!")
