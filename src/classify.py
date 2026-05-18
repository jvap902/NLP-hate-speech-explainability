import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import KFold
import pandas as pd
from . import config
from .model.modelClass import Model
from .fileHandler import getJsonInfo, updateJson, writeCsvLine

try:
    from google.colab import files
    IN_COLAB = True
except (ImportError, ModuleNotFoundError):
    IN_COLAB = False
    files = None

def compute_metrics(pred):
    logits = torch.Tensor(pred.predictions)
    labels = pred.label_ids.astype(int)
    
    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= 0.5).astype(int)

    f1_mi = f1_score(labels, preds, average='micro')
    f1_ma = f1_score(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1-macro': f1_ma,
        'f1-micro': f1_mi
    }

def fineTune(modelc: Model, repeat: int):
    
    for r in range(repeat):
    
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        
        for train_index, test_index in kf.split(modelc.train_tokenized):
            train_split = modelc.train_tokenized.select(train_index)
            val_split = modelc.train_tokenized.select(test_index)
            
            modelc.trainer.train_dataset = train_split
            train_output = modelc.trainer.train()
            loss = train_output.training_loss
            
            val_data = modelc.trainer.evaluate(eval_dataset=val_split)
            
            writeCsvLine(f"{config.losses_dir}/{modelc.name}", [r, val_data['accuracy'], val_data['f1-macro'], val_data['f1-micro'], loss])
            
        modelc.saveModel(1)
    
    return modelc

def testModel(modelc: Model):
    modelc.trainer.eval_dataset = modelc.test_tokenized
    eval_data = modelc.trainer.evaluate()
    
    model_data = getJsonInfo(config.fine_tune_info_path, [modelc.name])[0]
        
    model_data["eval"] = eval_data
    
    updateJson(json_path=config.fine_tune_info_path, fields=[modelc.name], values=[model_data])
    
    return eval_data

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
            labels = batch['labels'].cpu().numpy().astype(int)
            all_labels.extend(labels.tolist())
            
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