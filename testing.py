import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from model import CNN_Model  # Assuming this model is already defined as in your previous code
import numpy as np
from torch.utils.data import Dataset
import sys

class AnimalVideoDataset(Dataset):
    def __init__(self, video_dir, label_data):
        self.video_dir = video_dir
        self.label_data = label_data

        if self.label_data is not None:
            self.label_data = label_data[['Id']]
        
        self.x = 0

    def __len__(self):
        return len(self.label_data)

    def __getitem__(self, idx):
        video_id = self.label_data.iloc[idx]['Id']

        video_path = os.path.join(self.video_dir, f"{video_id}.npy")
        frames = np.load(video_path)

        frames = torch.tensor(frames, dtype=torch.float32)

        self.x += 1

        if self.x % 100 == 0:
            print(f"Processed videos {video_id}: {self.x}", file=sys.stdout, flush=True)

        return frames


# Assuming your test dataset is in a CSV file
test_csv = "test.csv"  
processed_videos = "subsample_videos/"

test_data = pd.read_csv(test_csv)
dataset = AnimalVideoDataset(video_dir=processed_videos, label_data=test_data)
test_loader = DataLoader(dataset, batch_size=24, shuffle=False)

# Initialize model and load checkpoint

model = CNN_Model(num_classes=140)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')  # Change to 'cuda' if using a GPU
model = model.to(device)

checkpoint_path = "model_epoch_10.pth" 
model.load_state_dict(torch.load(checkpoint_path, map_location=device)) 
model.eval() 

predictions = []

print("Starting inference...", file=sys.stdout, flush=True)

with torch.no_grad():
    for step, inputs in enumerate(test_loader):
        inputs = inputs.to(device)

        outputs = model(inputs)

        preds = torch.sigmoid(outputs)
        threshold = 0.24

        # Apply the threshold
        preds = (preds > threshold).float()

        for batch_idx, pred in enumerate(preds):
            # Get the video ID for the current sample in the batch
            video_id = test_data.iloc[step * test_loader.batch_size + batch_idx]['Id']

            # Find the classes that exceed the threshold
            predicted_labels = [str(idx) for idx, value in enumerate(pred.int()) if value == 1]

            # If no class exceeds the threshold, pick the class with the highest score
            if len(predicted_labels) == 0:
                # Find the index of the class with the highest score and include it in the output
                max_idx = torch.argmax(pred).item()
                predicted_labels = [str(max_idx)]

            predictions.append([video_id, ' '.join(predicted_labels)])

        # Free memory after each batch
        torch.cuda.empty_cache()  # Free GPU memory

# Save predictions
predictions_df = pd.DataFrame(predictions, columns=["Id", "Predicted"])
output_csv = "predictions_10_0.24.csv"
predictions_df.to_csv(output_csv, index=False)

print(f"Predictions saved to {output_csv}", file=sys.stdout, flush=True)

