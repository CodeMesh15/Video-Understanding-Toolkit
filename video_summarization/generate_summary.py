import torch
import numpy as np
import cv2
from sklearn.cluster import KMeans
import argparse
import os
from tqdm import tqdm

# We will reuse the action recognition model as a feature extractor
from action_recognition.action_model import ActionClassifier
from utils.video_loader import VideoDataset # For its frame sampling logic

@torch.no_grad()
def extract_features_for_segments(video_path, model, device, segment_duration=2, sr=16000):
    """
    Divides a video into segments and extracts a feature vector for each.
    """
    model.eval()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames_per_segment = int(segment_duration * fps)
    
    all_features = []
    segment_start_frames = []
    
    # Use the same transform as the model was trained on
    transform = VideoDataset(None, None).transform

    for start_frame in tqdm(range(0, total_frames, frames_per_segment), desc="Extracting segment features"):
        end_frame = min(start_frame + frames_per_segment, total_frames)
        if end_frame - start_frame < 16: # Need enough frames for the model
            continue
            
        # Sample frames from this segment
        frame_indices = np.linspace(start_frame, end_frame - 1, 16, dtype=int)
        frames = []
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if not frames:
            continue
            
        # Create a tensor for the model
        video_tensor = torch.stack([transform(Image.fromarray(f)) for f in frames])
        video_tensor = video_tensor.unsqueeze(0).to(device) # Add batch dimension
        
        # --- Feature Extraction ---
        # We pass the segment through the model and grab the video embedding
        # (the output of the GRU before the final classifier)
        video_embedding = model.video_encoder(video_tensor)
        
        all_features.append(video_embedding.cpu().numpy().squeeze())
        segment_start_frames.append(start_frame)
        
    cap.release()
    return np.array(all_features), segment_start_frames


def summarize_video(video_path, model, device, num_keyframes):
    """
    Generates an extractive summary of a video.
    
    Returns:
        list: A list of keyframes (as numpy arrays).
    """
    # 1. Extract features from video segments
    segment_features, segment_start_frames = extract_features_for_segments(video_path, model, device)
    
    if len(segment_features) < num_keyframes:
        print("Video is too short or could not be segmented properly for the requested number of keyframes.")
        return []
        
    # 2. Cluster the feature vectors
    kmeans = KMeans(n_clusters=num_keyframes, random_state=42, n_init=10)
    kmeans.fit(segment_features)
    
    # 3. Find the segment closest to each cluster centroid
    keyframes = []
    cap = cv2.VideoCapture(video_path)
    
    for i in range(num_keyframes):
        cluster_indices = np.where(kmeans.labels_ == i)[0]
        if len(cluster_indices) == 0:
            continue
            
        # Find the feature vector in this cluster that is closest to the centroid
        centroid = kmeans.cluster_centers_[i]
        distances = np.linalg.norm(segment_features[cluster_indices] - centroid, axis=1)
        closest_index_in_cluster = np.argmin(distances)
        
        # Get the original index of that segment
        original_segment_index = cluster_indices[closest_index_in_cluster]
        
        # Get the start frame of that segment
        keyframe_start_pos = segment_start_frames[original_segment_index]
        
        # Go to the middle of that segment to grab the representative frame
        keyframe_pos = keyframe_start_pos + (int(cap.get(cv2.CAP_PROP_FPS)) // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, keyframe_pos)
        ret, frame = cap.read()
        if ret:
            keyframes.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
    cap.release()
    return keyframes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a keyframe summary of a video.")
    parser.add_argument('--video_path', type=str, required=True, help="Path to the input video file.")
    parser.add_argument('--num_keyframes', type=int, default=5, help="Number of keyframes in the summary.")
    parser.add_gument('--action_model_path', type=str, default='models/action_classifier.pth')
    
    args = parser.parse_args()

    # --- Load the pre-trained action recognition model ---
    # We will use it as a powerful feature extractor.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # We need to wrap the ActionClassifier in another module to easily access its components
    class FeatureExtractor(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.video_encoder = nn.Sequential(*list(model.children())[:-1])

    # Initialize the base model (assuming 20 classes for MSR-VTT)
    action_model = ActionClassifier(num_classes=20)
    # action_model.load_state_dict(torch.load(args.action_model_path, map_location=device))
    
    feature_extractor = FeatureExtractor(action_model).to(device)
    feature_extractor.eval()
    
    print("--- Generating Video Summary ---")
    keyframes = summarize_video(args.video_path, feature_extractor, device, args.num_keyframes)
    
    # --- Display the summary ---
    if keyframes:
        import matplotlib.pyplot as plt
        print(f"Generated {len(keyframes)} keyframes.")
        
        fig, axes = plt.subplots(1, len(keyframes), figsize=(15, 5))
        for i, frame in enumerate(keyframes):
            axes[i].imshow(frame)
            axes[i].set_title(f"Keyframe {i+1}")
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print("Could not generate a summary.")
