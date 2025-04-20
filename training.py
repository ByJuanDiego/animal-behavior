import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import AnimalVideoDataset, get_labels, num_classes
from model import CNN_Model
from sklearn.metrics import f1_score
import sys
import numpy as np
from torch.cuda.amp import autocast, GradScaler

processed_videos = "subsample_videos/"
labels_csv = "train.csv"
labels = get_labels(labels_csv)

# Cargar datos
train_loader = DataLoader(AnimalVideoDataset(video_dir=processed_videos, label_data=labels),
                           batch_size=8,  
                           shuffle=True,
                           num_workers=4,  
                           pin_memory=True)

# Inicializar el modelo
model = CNN_Model(num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

model = model.to(device)

# Loss func y optimizador
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Scheduler para learning rate
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

num_epochs = 10  

# Para precisión mixta
scaler = GradScaler()

# Liberar memoria GPU
torch.cuda.empty_cache()

print("Iniciando entrenamiento...", file=sys.stdout, flush=True)

# Ciclo de entrenamiento
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    for step, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward
        with autocast():  
            outputs = model(inputs)
            labels = labels.squeeze(dim=1) 
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        # Recortar gradientes para evitar exploding
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Actualizar pesos
        scaler.step(optimizer)
        scaler.update()

        # -- Progreso --
        if (step + 1) % 50 == 0:
            print(f"Step [{step + 1}/{len(train_loader)}], Loss: {loss.item():.4f}", file=sys.stdout, flush=True)

        running_loss += loss.item()

        # Calcular precisiónpredicciones para F1
        preds = (outputs > 0.3).float()
        correct_predictions += torch.sum(preds == labels).item()
        total_samples += labels.size(0) * labels.size(1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / len(train_loader)
    accuracy = correct_predictions / total_samples
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    f1 = f1_score(all_labels, all_preds, average='micro')

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

    # Guardar modelo después de cad aepoch
    torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pth")
    scheduler.step()

    # Limpiar memoria de GPU
    torch.cuda.empty_cache()

print("Entrenamiento completado.")

