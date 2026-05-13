import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, accuracy_score
import pandas as pd
import shutil
from . import config
from .fileHandler import getJsonInfo
from .model.modelClass import Model

try:
    from google.colab import files
    IN_COLAB = True
except (ImportError, ModuleNotFoundError):
    IN_COLAB = False
    files = None

def classifyInputs(modelc, loader): #preliminar
    all_texts = []
    all_preds = []
    all_labels = []

    modelc.model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="classifying"): #para teste
            # Move inputs to GPU
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            
            #predictions
            logits = modelc.model(input_ids, attention_mask).logits
            probs = torch.sigmoid(logits)
            binary_preds = (probs > 0.5).int()
            all_preds.extend(binary_preds.cpu().tolist())
            
            #true labels
            labels = torch.stack([batch[col] for col in config.classes], dim=1)
            all_labels.extend(labels.cpu().tolist())
            
            # Se o loader não tiver a chave 'text', decodificamos os input_ids
            if 'text' in batch:
                texts = batch['text']
            else:
                texts = modelc.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            all_texts.extend(texts)
    
    df = pd.DataFrame({
        'text': all_texts,
        'preds': all_preds,
        'labels': all_labels
    })
    
    return df

def evaluateModel(modelc, loader):
    df_results = classifyInputs(modelc, loader)

    preds, labels = df_results["preds"].tolist(), df_results["labels"].tolist()

    stats = classification_report(
        labels, 
        preds, 
        target_names=config.classes, 
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
    
    
def fineTuneModel(modelc: Model, total_epochs, epoch_save=True, losses_csv=f'losses/BERTimbau-base.csv', lr=2e-5):
    modelc.model.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(modelc.model.parameters(), lr=lr)
    losses = pd.DataFrame(columns=["Iteration", "Loss"])
    
    trained_epochs = getJsonInfo(config.fine_tune_info_path, [modelc.name])[0]
    
    epochs = total_epochs - trained_epochs
    
    for epoch in tqdm(range(epochs), desc=f"Fine Tuning"):
        running_loss = 0.0
        for batch in tqdm(modelc.train_loader, desc=f"Epoch progress"):
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            
            # Stack your 13 labels into a single matrix
            labels = torch.stack([batch[col] for col in config.classes], dim=1).float()
            labels = labels.to(config.device)

            optimizer.zero_grad()

            outputs = modelc.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss = loss.item()
        
        #validate(modelc)
        
        losses.loc[len(losses)] = {'Iteration': epoch, 'Loss': running_loss}
        losses.to_csv(losses_csv, mode='a', index=True)
        
        if epoch_save: modelc.saveModel(trained_epochs=1)
        
        if IN_COLAB: #caso esteja no colab já baixa uma cópia para não perder por limite de tempo
            shutil.make_archive(modelc.name, 'zip', {modelc.save_dir})
            files.download(f'{modelc.name}.zip')
            
    return modelc


def validate(modelc):
    raise NotImplementedError
