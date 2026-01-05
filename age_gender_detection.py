
#%% import libraries
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import timm
from pathlib import Path
import numpy as np



# %% data preprocessing
"""
DATA_PATH = Path('C:\\Users\\anti_\\OneDrive\\Desktop\\btk_pytorch\\5_GANs')


image_files = []
for root, dirs, files in os.walk(DATA_PATH):
    print(root, dirs, files)
"""

DATA_PATH = Path('D:\\datasets\\archive\\UTKFace')


age_list = []
gender_list = []
file_name = []
for file in os.listdir(DATA_PATH):
    age = file.split('_')[0]
    age_list.append(int(age))
    gender = file.split('_')[1]
    gender_list.append(int(gender))
    file_name.append(os.path.join(DATA_PATH, file))


data = {'file_name': file_name, 'age': age_list, 'gender': gender_list}

data['age']

df = pd.DataFrame(data)


df.head()


list_of_models = [m for m in timm.list_models(pretrained=True) if 'resnet18' in m]

model = timm.create_model('resnet18.tv_in1k')

print(model)

cfg = model.default_cfg

mean = cfg['mean']
std = cfg['std']

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])



val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])




class AgeGenderDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform


    def __len__(self):
        return len(self.df)


    def __getitem__(self, index):
        row = self.df.iloc[index]

        img_path = row['file_name']

        img = Image.open(img_path).convert('RGB')

        if self.transform:
           img = self.transform(img)


        age = torch.tensor(row['age'], dtype=torch.float32)

        gender = torch.tensor(row['gender'], dtype=torch.float32)

        return img, {'age': age, 'gender': gender}
    


train_df, val_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)

train_dataset = AgeGenderDataset(train_df, train_transform)
val_dataset = AgeGenderDataset(val_df, val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


x = torch.randint(1, 10, (32, 512, 7, 7)).float()

avg_pool = nn.AdaptiveAvgPool2d((1, 1))

x = avg_pool(x)

torch.flatten(x,1).shape

# %% generate model

class AgeGenderDetection(nn.Module):
    def __init__(self):
        super(AgeGenderDetection, self).__init__()

        self.backbone = timm.create_model('resnet18.tv_in1k', pretrained=True, num_classes=0)

        input_features = self.backbone.num_features

        self.age_head = nn.Sequential(
            nn.Linear(input_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

        self.gender_head = nn.Sequential(
            nn.Linear(input_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )


    def forward(self, x):
        features = self.backbone(x)

        age_out = self.age_head(features)

        gender_out = self.gender_head(features)

        return {'age': age_out, 'gender': gender_out}
        

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AgeGenderDetection().to(device)

print(f'model {device} uzerinde calisacak')


# %% train model

num_epochs = 5
learning_rate = 1e-4
weight_decay = 1e-5

criterion_age = nn.MSELoss()
criterion_gender = nn.BCEWithLogitsLoss()

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

best_val_loss = float('inf') 

for epoch in range(num_epochs):
    
    model.train()
    total_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")

    for images, labels in loop:
        images = images.to(device)
        age_labels = labels['age'].to(device)
        gender_labels = labels['gender'].to(device)
        
        optimizer.zero_grad()

        output = model(images)

        loss_age = criterion_age(output['age'], age_labels.unsqueeze(-1))
        loss_gender = criterion_gender(output['gender'], gender_labels.unsqueeze(-1))

        train_loss = loss_age + loss_gender

        train_loss.backward()

        optimizer.step()

        total_loss += train_loss.item()
        

    model.eval()
    val_loss_age = 0.0
    val_loss_gender = 0.0
    val_age_mae = 0.0
    correct_gender = 0
    total_samples_val = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            
            age_labels = labels['age'].to(device)
            gender_labels = labels['gender'].to(device)

            output = model(images)

            l_age = criterion_age(output['age'], age_labels.unsqueeze(-1))
            l_gender = criterion_gender(output['gender'], gender_labels.unsqueeze(-1))

            val_loss_age += l_age.item() * images.size(0)
            val_loss_gender += l_gender.item() * images.size(0)

            

            val_age_mae += torch.abs(output['age'] - age_labels.unsqueeze(-1)).sum().item()



            probs = torch.sigmoid(output['gender'])
            preds = (probs > 0.5).long()

            correct_gender += (gender_labels.long() == preds.view(-1)).sum().item()

            total_samples_val += images.size(0)

        avg_train_loss = total_loss / len(train_loader)
        avg_loss_age = val_loss_age / total_samples_val
        avg_loss_gender = val_loss_gender / total_samples_val
        avg_val_loss = avg_loss_age + avg_loss_gender
        avg_mae_age = val_age_mae / total_samples_val
        gender_acc = 100 * correct_gender / total_samples_val

        print(f'<<<<<<epoch: {epoch + 1}/{num_epochs}>>>>>>')
        print(f'train loss: {avg_train_loss:.4f}')
        print(f'validation age loss: {avg_loss_age:.4f}')
        print(f'validation gender loss: {avg_loss_gender:.4f}')
        print(f'validation total loss: {avg_val_loss:.4f}')
        print(f'avg age loss (MAE): {avg_mae_age:.2f}')
        print(f'gender accuracy: %{gender_acc:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'age_gender_detect_best.pth')


[2] + [1]*1
"""
model.eval()
images, labels = next(iter(val_loader))

with torch.no_grad():
    images = images.to(device)
    output = model(images.detach())
    gender_labels =  labels['gender'].unsqueeze(-1).to(device)
    l_gender = criterion_gender(output['gender'], gender_labels)
    l_gender
    probs = torch.sigmoid(output['gender'])
    
logits = torch.tensor([[2.0], [-1.0], [0.2]])
labels = torch.tensor([[1], [1], [1]])


probs = torch.sigmoid(logits)

pred = (probs > 0.5).long()
pred = pred.unsqueeze(-1)

pred.shape
labels.view_as(pred)
"""

# %%
