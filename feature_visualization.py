import os
import cv2
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from tqdm import tqdm
import pandas as pd

# ==========================================
# 0. DATASET SETUP & CONFIGURATION
# ==========================================
# Automatically clone dataset if it doesn't exist locally
if not os.path.exists('GradCAM-Dataset'):
    print("Downloading dataset...")
    os.system('git clone https://github.com/parth1620/GradCAM-Dataset.git')

CSV_FILE = 'GradCAM-Dataset/train.csv'
DATA_DIR = 'GradCAM-Dataset/'

# Dynamic device selection: uses GPU if available, otherwise safely drops to CPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

BATCH_SIZE = 16
LR = 0.001
EPOCHS = 10  

# ==========================================
# 1. DATA PREPARATION (Dataset Class)
# ==========================================
df = pd.read_csv(CSV_FILE)

train_df = df.sample(frac=0.8, random_state=42)
valid_df = df.drop(train_df.index)

train_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

valid_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(DATA_DIR, row.img_path)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = row.label
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

trainset = ImageDataset(train_df, train_transform)
validset = ImageDataset(valid_df, valid_transform)

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
validloader = DataLoader(validset, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 2. MODEL ARCHITECTURE WITH GRAD-CAM HOOKS
# ==========================================
class ImageModel(torch.nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        # Clean execution across different torchvision versions without warnings
        try:
            vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
        except AttributeError:
            vgg = torchvision.models.vgg16(pretrained=True)
            
        self.feature_extractor = vgg.features[:29] # Up to the last conv layer
        self.maxpool = vgg.features[29:]
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(512 * 7 * 7, 3) # Output map for 3 classes
        )
        self.gradient = None
        
    def activations_hook(self, grad):
        self.gradient = grad
        
    def forward(self, images):
        x = self.feature_extractor(images)
        # FIXED: Only register the hook if gradient tracking is active (safely skips during validation)
        if x.requires_grad:
            x.register_hook(self.activations_hook)
        x = self.maxpool(x)
        x = self.classifier(x)
        return x
        
    def get_activation(self, x):
        return self.feature_extractor(x)

model = ImageModel()
model.to(DEVICE)

# ==========================================
# 3. TRAINING & EVALUATION FUNCTIONS
# ==========================================
def train_fn(dataloader, model, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(dataloader, desc="Training Batch"):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(dataloader)

def eval_fn(dataloader, model, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()

best_valid_loss = float('inf')

# Training Loop Execution
print("\n--- Starting Training Loop ---")
for i in range(EPOCHS):
    train_loss = train_fn(trainloader, model, optimizer, criterion)
    valid_loss = eval_fn(validloader, model, criterion)
    
    if valid_loss < best_valid_loss:
        torch.save(model.state_dict(), 'best_weights.pt')
        best_valid_loss = valid_loss
        print("--> Saved new optimal weights successfully.")
        
    print(f"EPOCH: {i + 1} | Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}\n")

# ==========================================
# 4. GENERATING THE GRAD-CAM HEATMAP (ERROR-PROOF VERSION)
# ==========================================
print("\n--- Generating Grad-CAM Heatmap ---")

# Step 1: Load the smartest version of our model
model.load_state_dict(torch.load('best_weights.pt', map_location=DEVICE))
model.eval() # Put the model in "test mode" so it doesn't try to learn

# Step 2: Grab the first image from our validation dataset
image, label = validset[0] 
# PyTorch expects a "batch" of images, so we wrap our single image in an extra bracket
input_tensor = image.unsqueeze(0).to(DEVICE) 

# Step 3: Ask the model what it thinks this image is
logits = model(input_tensor)
pred_class = logits.argmax(dim=1).item() # Get the winning category number

# Step 4: Tell the model to work backward from its guess to find out WHY it guessed that
logits[0, pred_class].backward()

# Step 5: Extract the visual "thoughts" the model had during the backward pass
# (.detach() safely unplugs the data from PyTorch's gradient tracker so numpy can use it)
gradients = model.gradient.detach().cpu().numpy()[0]
activations = model.get_activation(input_tensor).detach().cpu().numpy()[0]

# Step 6: Figure out which visual channels were the most important (giving them a weight)
weights = np.mean(gradients, axis=(1, 2))

# Step 7: Multiply the visual maps by their importance weight and combine them into one map
cam = np.zeros(activations.shape[1:], dtype=np.float32)
for i, w in enumerate(weights):
    cam += w * activations[i]

# Step 8: Erase anything that confused the model (keep only numbers greater than 0)
cam = np.max(cam, 0)

# Step 9: Convert our original input tensor back into a normal picture so we can see it
img_display = image.permute(1, 2, 0).numpy()
# Un-do the color normalization we applied during training
img_display = img_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
img_display = np.clip(img_display, 0, 1) # Keep colors within normal bounds
img_display = np.uint8(255 * img_display) # Convert to standard 8-bit image pixels

# Step 10: THE FIX! Stretch the small heatmap to perfectly match the size of our original picture
cam = cv2.resize(cam, (img_display.shape[1], img_display.shape[0]))

# Normalize the heatmap so its intensity goes smoothly from 0 to 1
cam = cam - np.min(cam)
cam = cam / np.max(cam)

# Step 11: Paint the heatmap with colors (Blue = Cold/Unimportant, Red = Hot/Important)
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

# Step 12: Blend the original picture and the heatmap together (60% picture, 40% heatmap)
output_image = cv2.addWeighted(img_display, 0.6, heatmap, 0.4, 0)

# Step 13: Save it!
plt.imsave('gradcam_output.jpg', output_image)
print("Success! 'gradcam_output.jpg' has been generated and saved to your directory.")
