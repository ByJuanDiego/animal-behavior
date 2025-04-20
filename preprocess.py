import os
import cv2
import numpy as np
from tqdm import tqdm
import sys
from multiprocessing import Pool

original_videos = "./videos/video"
subsample_videos = "./subsample_videos"
os.makedirs(subsample_videos, exist_ok=True)

video_files = os.listdir(original_videos)
fixed_frame_count = 16

print(len(video_files), "videos found in the original directory")

def process_video(video_file):
    cap = cv2.VideoCapture(os.path.join(original_videos, video_file))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    indices = np.linspace(0, total_frames - 1, num=fixed_frame_count, dtype=int)
    frames_list = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            print(f"Error reading frame {idx}. Filling with black frame.")
            frame = np.zeros((360, 640, 3), dtype=np.uint8)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame.astype(np.float32) / 255.0
        frame = np.expand_dims(frame, axis=-1)
        frames_list.append(frame)

    cap.release()

    video_array = np.stack(frames_list, axis=0)
    out_file = os.path.join(subsample_videos, os.path.splitext(video_file)[0] + ".npy")
    np.save(out_file, video_array)
    return video_file

with Pool(processes=16) as pool:
    results = list(tqdm(pool.imap(process_video, video_files), total=len(video_files), desc="Processing Videos"))

for result in results:
    print(f"Processed {result}", file=sys.stdout, flush=True)

print("Processing complete.")

