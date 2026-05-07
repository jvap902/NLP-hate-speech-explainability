import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, accuracy_score
import pandas as pd
import shutil
from . import config

try:
    from google.colab import files
    IN_COLAB = True
except (ImportError, ModuleNotFoundError):
    IN_COLAB = False
    files = None

def classifyInputs(modelc, loader): #preliminar
    all_preds = []
    all_labels = []

    modelc.model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="classifying"): #para teste
            # Move inputs to GPU
            input_ids = batch['input_ids'].to(modelc.model.device)
            attention_mask = batch['attention_mask'].to(modelc.model.device)
            
            #predictions
            logits = modelc.model(input_ids, attention_mask).logits
            probs = torch.sigmoid(logits)
            binary_preds = (probs > 0.5).int()
            all_preds.append(binary_preds.cpu())
            
            #true labels
            labels = torch.stack([batch[col] for col in config.class_cols], dim=1)
            all_labels.append(labels.cpu())

    # Combine all batches
    final_predictions = torch.cat(all_preds)
    final_labels = torch.cat(all_labels).numpy()
    
    return final_predictions, final_labels

def evaluateModel(modelc, loader):
    preds, labels = classifyInputs(modelc, loader)

    stats = classification_report(
        labels, 
        preds, 
        target_names=config.class_cols, 
        zero_division=0,
        output_dict=True
    )

    print(stats)

    micro_f1 = f1_score(labels, preds, average='micro')
    macro_f1 = f1_score(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)

    print(f"Micro F1: {micro_f1:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Accuracy: {acc:.4f}")
    
    return stats
    
    
def fineTuneModel(modelc, epochs, epoch_save = False):
    modelc.model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(modelc.model.parameters(), lr=0.01)
    losses = []
    
    for epoch in tqdm(range(epochs), desc=f"Fine Tuning"):
        for batch in tqdm(modelc.train_loader, desc=f"Epoch progress"):
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            
            # Stack your 13 labels into a single matrix
            labels = torch.stack([batch[col] for col in config.class_cols], dim=1).float()
            labels = labels.to(config.device)

            optimizer.zero_grad()

            outputs = modelc.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss)
        
        
        validate(modelc)
        
        if epoch_save: modelc.saveModel()
        
        if IN_COLAB: #caso esteja no colab já baixa uma cópia para não perder por limite de tempo
            # Zips the folder then triggers download
            shutil.make_archive(modelc.name, 'zip', {modelc.save_dir})
            files.download(f'{modelc.name}.zip')
            
    return modelc, losses


def validate(modelc):
    raise NotImplementedError