import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from action_recognition.action_model import ActionClassifier

class VQAModel(nn.Module):
    """
    A multi-modal model for Video Question Answering.
    """
    def __init__(self, num_answers, action_model_path, hidden_dim=512):
        super(VQAModel, self).__init__()
        
        # --- Video Encoder ---
        # Load the pre-trained action classifier
        # We need its num_classes to initialize it before loading weights
        # For simplicity, let's assume num_classes is known (e.g., 20 for MSR-VTT)
        video_encoder = ActionClassifier(num_classes=20) 
        # video_encoder.load_state_dict(torch.load(action_model_path))
        # We use the model up to the GRU output (the video embedding)
        self.video_encoder = nn.Sequential(*list(video_encoder.children())[:-1])
        video_embedding_dim = hidden_dim # This is the GRU hidden_dim
        
        # Freeze the video encoder
        for param in self.video_encoder.parameters():
            param.requires_grad = False
            
        # --- Question Encoder ---
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        text_embedding_dim = self.text_encoder.config.hidden_size # 768 for BERT-base
        
        # --- Fusion and Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(video_embedding_dim + text_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_answers)
        )
        
    def forward(self, video, question_text):
        """
        Forward pass.
        Args:
            video (torch.Tensor): Video tensor (B, T, C, H, W)
            question_text (list of str): List of questions
        """
        # Get video embedding
        video_emb = self.video_encoder(video)
        
        # Get question embedding
        inputs = self.tokenizer(question_text, return_tensors="pt", padding=True, truncation=True).to(video.device)
        text_emb = self.text_encoder(**inputs).last_hidden_state[:, 0, :] # [CLS] token
        
        # Fuse features by concatenating
        fused_features = torch.cat((video_emb, text_emb), dim=1)
        
        # Classify the answer
        logits = self.classifier(fused_features)
        
        return logits
