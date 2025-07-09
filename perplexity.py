# ============================
# Import Setup
# ============================

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
from tqdm import tqdm

def get_perplexity_scores(texts):
    """
    Computes the GPT-2 perplexity score for a list of input texts.

    Perplexity is a measure of how "surprised" a language model is by a given sequence.
    Lower values indicate more fluent, human-like text according to the model.
    This function uses GPT-2 to calculate perplexity, truncating input to a max of 512 tokens.

    Args:
        texts (List[str]): A list of text strings to evaluate.

    Returns:
        np.ndarray: An array of shape (N,) where each value is the perplexity
                    of the corresponding input text.
    """
    
    # Choose CUDA (GPU) if available, otherwise fallback to CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load GPT-2 tokenizer and language model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.eval()
    
    perplexities = []
    # Iterate over input texts and compute perplexity per item
    for text in tqdm(texts, desc="Computing GPT-2 perplexity scores"):
        # Tokenize input with truncation to avoid exceeding 512 token limit
        inputs = tokenizer(text, 
                           return_tensors='pt', 
                           truncation=True, 
                           max_length=512
                           ).to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])   # Forward pass
            loss = outputs.loss                                     # Compute loss              
        
        # Convert loss to perplexity: exp(loss)
        perplexity = torch.exp(loss).item()
        perplexities.append(perplexity)
        
    return np.array(perplexities)