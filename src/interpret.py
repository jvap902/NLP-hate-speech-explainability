from captum.attr import LayerIntegratedGradients
import torch
import matplotlib.pyplot as plt
from typing import Tuple, List
from src import config

def explainPrediction(modelc, inputs: List[Tuple[str, int]]):
    modelc.model.eval()
    
    # 1. Define a wrapper function that Captum can use
    def forward_func(input_ids):
        # We only care about the .logits tensor
        return modelc.model(input_ids).logits

    # 2. Initialize LayerIntegratedGradients using the wrapper
    # Using modelc.model.bert.embeddings for BERTimbau
    lig = LayerIntegratedGradients(forward_func, modelc.model.bert.embeddings)
    
    all_attr = []
    all_tokens = []
    
    for text, label_idx in inputs:
        # 1. WRAP the string in a dict so tokenizeInstance works
        tokenized = modelc.tokenizeInstance({"text": text})
        
        # 2. Extract and prepare input_ids as a Tensor
        # (Assuming tokenizeInstance returns a dictionary of lists/tensors)
        input_ids = torch.tensor(tokenized['input_ids']).unsqueeze(0).to(config.device)
        
        # 3. Calculate attribution on the embeddings
        # We target the specific label_idx
        attributions = lig.attribute(inputs=input_ids, target=label_idx, n_steps=50)
        
        # 4. Process for visualization
        # Sum along the embedding dimensions to get one score per token
        attributions = attributions.sum(dim=-1).squeeze(0)
        
        # Convert IDs back to words for the result
        tokens = modelc.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        all_attr.append(attributions.cpu().detach().numpy())
        all_tokens.append(tokens)
    
    return all_attr, all_tokens


def plotAttributions(attributions, tokens, label_name, save_path=None, show=True):
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(tokens)), attributions, align='center')
    plt.xticks(range(len(tokens)), tokens, rotation=45)
    plt.ylabel('Attribution Score')
    plt.title(f'Feature Importance for Label: {label_name}')
    if save_path != None: plt.savefig(save_path, format='png', dpi=100)
    if show: plt.show()