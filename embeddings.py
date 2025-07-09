# ============================
# Import Setup
# ============================

import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from tqdm import tqdm

def get_bert_embeddings(texts):
    """
    Extracts BERT [CLS] token embeddings for a list of input texts.

    This function tokenizes each input string, passes it through a pre-trained
    BERT model (bert-base-uncased), and extracts the embedding of the [CLS] token
    from the final hidden state. The result is a 2D NumPy array where each row
    corresponds to a 768-dimensional embedding of a text.

    Args:
        texts (list of str): List of input text strings to embed.

    Returns:
        np.ndarray: A NumPy array of shape (len(texts), 768) containing
                    the [CLS] embeddings of the input texts.
    """
    
    # Use GPU if available, otherwise fallback to CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load pre-trained BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)
    model.eval()
    
    embeddings = []
    
    # Iterate through texts and compute embeddings    
    for text in tqdm(texts, desc="Extracting BERT embeddings"):
        # Tokenize and convert to tensors (max 512 tokens)
        inputs = tokenizer(text, 
                           return_tensors='pt', 
                           truncation=True, 
                           padding=True, 
                           max_length=512
                           ).to(device)
        
        # Forward pass without gradient tracking
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extract the [CLS] token embedding (position 0)
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_embedding[0])
    
    # Stack all embeddings into a single NumPy array (shape: num_texts × 768)
    return np.array(embeddings)