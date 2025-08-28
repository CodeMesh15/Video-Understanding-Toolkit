import torch
import torch.nn as nn
from transformers import ViTModel

class ActionClassifier(nn.Module):
    """
    A video action classifier using a pre-trained Vision Transformer (ViT)
    as a frame encoder and a GRU for temporal modeling.
    """
    def __init__(self, num_classes, hidden_dim=512, dropout=0.3):
        super(ActionClassifier, self).__init__()
        
        # --- Vision Transformer (Frame Encoder) ---
        # Load a pre-trained ViT model
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        # The output dimension of the [CLS] token from ViT-base is 768
        vit_output_dim = 768
        
        # --- GRU (Temporal Encoder) ---
        self.gru = nn.GRU(
            input_size=vit_output_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # --- Classifier Head ---
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input video tensor. Shape: (B, T, C, H, W)
        """
        batch_size, num_frames, C, H, W = x.shape
        
        # Reshape to process all frames at once with ViT
        x = x.view(batch_size * num_frames, C, H, W)
        
        # Get frame embeddings from ViT
        # We use the embedding of the [CLS] token
        with torch.no_grad(): # Freeze the ViT for faster training
             frame_embeddings = self.vit(x).last_hidden_state[:, 0, :]
        
        # Reshape back to a sequence for the GRU
        seq_embeddings = frame_embeddings.view(batch_size, num_frames, -1)
        
        # Pass sequence through GRU
        _, hidden_state = self.gru(seq_embeddings)
        
        # We use the hidden state of the last layer as the video representation
        video_embedding = hidden_state[-1]
        
        # Classify
        logits = self.classifier(video_embedding)
        
        return logits
