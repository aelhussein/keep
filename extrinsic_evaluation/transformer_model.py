import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import TransformerEncoderLayer, TransformerEncoder
import torch.optim as optim 
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report, average_precision_score
import numpy as np
import math
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, id_dict, n_embd, n_head, n_layer, num_classes, 
                 dropout=0.2, pooling_type='attention', embedding_weights=None, 
                 freeze_emb=True):
        super().__init__()
        self.pooling_type = pooling_type
        
        # Embedding layer
        if embedding_weights is not None:
            self.embedding = nn.Embedding.from_pretrained(
                embedding_weights,
                freeze=freeze_emb,
                padding_idx=id_dict["<PAD>"]
            )
        else:
            self.embedding = nn.Embedding(
                vocab_size,
                n_embd,
                padding_idx=id_dict["<PAD>"]
            )
        self.emb_dropout = nn.Dropout(p=dropout)
        
        # Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward= n_embd,
            dropout=dropout,
            activation='relu',
            batch_first=True 
        )
        
        self.transformer = TransformerEncoder(
            encoder_layer,
            num_layers=n_layer,
            norm=nn.LayerNorm(n_embd)
        )
        
        # Pooling mechanism
        if pooling_type == 'attention':
            self.attention_weights = nn.Linear(n_embd, 1)
        elif pooling_type == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, n_embd))
            
        # Final classification head
        self.classifier = nn.Linear(n_embd, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Initialize embedding if not using pretrained weights
        if not hasattr(self.embedding, 'weight'):
            nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
            
        # Initialize classifier
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        
        # Initialize attention weights
        if self.pooling_type == 'attention':
            nn.init.normal_(self.attention_weights.weight, std=0.02)
            nn.init.zeros_(self.attention_weights.bias)
    
    def pool_output(self, x, mask=None):
        if self.pooling_type == 'mean':
            if mask is not None:
                # Apply mask
                x = x * mask.unsqueeze(-1)
                return x.sum(dim=1) / mask.sum(dim=1, keepdim=True)
            return torch.mean(x, dim=1)
            
        elif self.pooling_type == 'attention':
            # Compute attention weights
            weights = self.attention_weights(x) 
            if mask is not None:
                weights = weights.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            weights = torch.softmax(weights, dim=1)
            return torch.sum(weights * x, dim=1)
            
        elif self.pooling_type == 'cls':
            return x[:, 0]  # Return CLS token representation
    
    def forward(self, x, mask=None):
        # Create padding mask for transformer if needed
        if mask is None:
            mask = (x != self.embedding.padding_idx)
        
        # Embedding
        x = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        x = self.emb_dropout(x)
        
        # Add CLS token if using cls pooling
        if self.pooling_type == 'cls':
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            mask = torch.cat((torch.ones(x.shape[0], 1, device=x.device), mask), dim=1)
        
        # Transform mask into attention mask format
        attention_mask = mask.logical_not()
        
        # Pass through transformer
        x = self.transformer(x, src_key_padding_mask=attention_mask)
        
        # Pool the output
        x = self.pool_output(x, mask)
        
        # Classify
        return self.classifier(x)
    
    @torch.no_grad()
    def predict(self, sentence):
        self.eval()
        logits = self(sentence)
        return torch.argmax(logits, dim=1)
    
    @torch.no_grad()
    def predict_proba(self, sentence):
        self.eval()
        logits, _ = self(sentence)
        return F.softmax(logits, dim=1)



@torch.no_grad()
def evaluate(model, dataloader_test, criterion):
    model.eval()
    predicted_labels = []
    predicted_probas = []
    true_labels = []
    total_loss = 0.0
    
    for batch_sentences, batch_labels in dataloader_test:
        batch_sentences = batch_sentences.to(device)
        batch_labels = batch_labels.to(device)
        
        logits = model(batch_sentences)
        loss = criterion(logits, batch_labels)
        total_loss += loss.item()
        
        predicted_labels.extend(torch.argmax(logits, dim=1).cpu().numpy())
        predicted_probas.extend(F.softmax(logits, dim=1).cpu().numpy())
        true_labels.extend(batch_labels.cpu().numpy())
    
    # Calculate metrics
    f1_macro = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)
    f1_weighted = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    accuracy = accuracy_score(true_labels, predicted_labels)
    
    # Only calculate AUC for binary classification
    auc_score = None
    auprc_score = None
    if len(np.unique(true_labels)) == 2:
        probas = np.array(predicted_probas)[:, 1]
        if np.isnan(probas).any() or np.isinf(probas).any():
            print("Probabilities contain NaN or Inf. Skipping AUC/AUPRC calculation.")
        else:
            auc_score = roc_auc_score(true_labels, probas)
            auprc_score = average_precision_score(true_labels, probas)
    
    avg_loss = total_loss / len(dataloader_test)
    
    model.train()
    return {
        'f1_macro': f1_macro,
        'f1_weighted' : f1_weighted,
        'accuracy': accuracy,
        'auc': auc_score,
        'auprc': auprc_score,
        'loss': avg_loss,
        'predicted_probs': np.array(predicted_probas)[:, 1],
        'true_labels': true_labels,

    }

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
            self.counter = 0
        return self.early_stop

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def count_trainable_parameters(model):
    """
    Counts and prints the number of trainable and total parameters in the model
    """
    trainable_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
            print(f"{name}: {num_params:,} parameters")
    
    print(f"\nTotal Trainable Parameters: {trainable_params:,}", flush = True)
    print(f"Total Parameters: {total_params:,}", flush = True)
    print(f"Percentage of parameters being trained: {(trainable_params/total_params)*100:.2f}%", flush = True)