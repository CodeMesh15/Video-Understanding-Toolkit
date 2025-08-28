import torch
from torch.utils.data import Dataset
import cv2
import os
import pandas as pd
from PIL import Image
import torchvision.transforms as T

class VideoDataset(Dataset):
    """
    A PyTorch Dataset for loading video frames and their corresponding labels.
    """
    def __init__(self, video_dir, annotations_df, num_frames=16, transform=None):
        """
        Args:
            video_dir (str): Directory with all the video files.
            annotations_df (pd.DataFrame): DataFrame with 'video_id' and 'label' columns.
            num_frames (int): The number of frames to sample from each video.
            transform (callable, optional): Optional transform to be applied on a frame.
        """
        self.video_dir = video_dir
        self.annotations = annotations_df
        self.num_frames = num_frames
        
        # Default transform if none is provided
        if transform is None:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Resize((224, 224)),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        video_info = self.annotations.iloc[idx]
        video_path = os.path.join(self.video_dir, f"{video_info['video_id']}.mp4")
        label = video_info['label']
        
        frames = self._sample_frames(video_path)
        
        # Apply transformations to each frame
        processed_frames = torch.stack([self.transform(Image.fromarray(frame)) for frame in frames])
        
        return processed_frames, label

    def _sample_frames(self, video_path):
        """
        Samples a fixed number of frames uniformly from a video.
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < self.num_frames:
            # If video is shorter than num_frames, loop the frames
            indices = np.arange(total_frames)
            indices = np.tile(indices, int(np.ceil(self.num_frames / total_frames)))
            indices = indices[:self.num_frames]
        else:
            # Uniformly sample frame indices
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            
        frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                # Convert from BGR (OpenCV default) to RGB
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                # If a frame can't be read, duplicate the last successful one
                if frames:
                    frames.append(frames[-1])

        cap.release()
        return frames
