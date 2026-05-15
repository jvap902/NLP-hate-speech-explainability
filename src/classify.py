import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, accuracy_score
from transformers import Trainer, TrainingArguments
import pandas as pd
import shutil
from . import config
from .fileHandler import getJsonInfo
from .model.modelClass import Model
from pathlib import Path

try:
    from google.colab import files
    IN_COLAB = True
except (ImportError, ModuleNotFoundError):
    IN_COLAB = False
    files = None

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # print(preds)
    f1_mi = f1_score(labels, preds, average='micro')
    f1_ma = f1_score(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1-macro': f1_ma,
        'f1-micro': f1_mi
    }
    
def getTrainingArgs():
    batch_size = 8
    epochs = 5
    learning_rate = 3e-05
    steps_per_epoch = round(len(train_dataset_tokenized) / batch_size)
    print(len(train_dataset_tokenized), len(train_dataset_tokenized), steps_per_epoch)
    
    training_args = TrainingArguments(
    output_dir='test_trainer',
    #overwrite_output_dir=True,
    eval_strategy='epoch',
    save_strategy ='epoch',
    per_device_train_batch_size = batch_size,
    per_device_eval_batch_size = batch_size,
    logging_steps=20,
    report_to="none",
    learning_rate=learning_rate,
    num_train_epochs = epochs,
    load_best_model_at_end = True,
    #   report_to='tensorboard'
        )


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

def computePosWeight(train_loader, num_classes: int, device) -> torch.Tensor:
    """
    Calcula o pos_weight para BCEWithLogitsLoss baseado na frequência de cada classe.
    pos_weight[i] = (nº exemplos negativos da classe i) / (nº exemplos positivos da classe i)
    Isso compensa o desbalanceamento do TuPyE-E.
    """
    pos_counts = torch.zeros(num_classes)
    total = 0

    for batch in train_loader:
        labels = torch.stack([batch[col] for col in config.classes], dim=1).float()
        pos_counts += labels.sum(dim=0).cpu()
        total += labels.shape[0]

    neg_counts = total - pos_counts
    # Clamp para evitar divisão por zero em classes sem exemplos positivos
    pos_weight = (neg_counts / pos_counts.clamp(min=1)).to(device)
    return pos_weight
    
def fineTuneModel( modelc: Model, total_epochs: int, epoch_save: bool = True, losses_csv: str = 'losses/BERTimbau-base.csv', lr: float = 2e-5, warmup_ratio: float = 0.1, weight_decay: float = 0.01, max_grad_norm: float = 1.0, use_pos_weight: bool = True):
    modelc.model.train()

    # para compensar desbalanceamento de classes
    if use_pos_weight:
        pos_weight = computePosWeight(modelc.train_loader, len(config.classes), config.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in modelc.model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in modelc.model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=lr)

    trained_epochs = getJsonInfo(config.fine_tune_info_path, [modelc.name])[0]
    epochs = total_epochs - trained_epochs
    total_steps = epochs * len(modelc.train_loader)
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    losses = pd.DataFrame(columns=["Epoch", "Avg_Loss", "LR"])

    for epoch in tqdm(range(epochs), desc="Fine Tuning"):
        modelc.model.train()
        running_loss = 0.0

        for batch in tqdm(modelc.train_loader, desc=f"Epoch {trained_epochs + epoch + 1}"):
            input_ids      = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels         = torch.stack(
                [batch[col] for col in config.classes], dim=1
            ).float().to(config.device)

            optimizer.zero_grad()

            outputs = modelc.model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = outputs.logits

            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(modelc.model.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(modelc.train_loader)
        current_lr = scheduler.get_last_lr()[0]

        losses.loc[len(losses)] = {
            'Epoch': trained_epochs + epoch + 1,
            'Avg_Loss': avg_loss,
            'LR': current_lr,
        }
        losses.to_csv(losses_csv, mode='a', header=not Path(losses_csv).exists(), index=False)

        evaluateModel(modelc, modelc.test_loader)
        modelc.model.train()

        if epoch_save:
            modelc.saveModel(trained_epochs=1)

        if IN_COLAB:
            shutil.make_archive(modelc.name, 'zip', modelc.save_dir)
            files.download(f'{modelc.name}.zip')

    return modelc
