import torch
from . import config

def classifyInputs(modelc): #preliminar
    modelc.model.eval()
    all_preds = []

    for batch in modelc.test_loader: #para teste
        with torch.no_grad():
            # Move inputs to GPU
            input_ids = batch['input_ids'].to(modelc.model.device)
            attention_mask = batch['attention_mask'].to(modelc.model.device)
            
            logits = modelc.model(input_ids, attention_mask).logits
            probs = torch.sigmoid(logits)
            
            # Convert to 0 or 1 based on threshold
            binary_preds = (probs > 0.5).int()
            all_preds.append(binary_preds.cpu())

    # Combine all batches
    final_predictions = torch.cat(all_preds)