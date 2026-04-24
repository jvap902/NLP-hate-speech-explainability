import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm
from . import config
from sklearn.metrics import classification_report, f1_score, accuracy_score

def classifyInputs(modelc, loader): #preliminar
    all_preds = []
    all_labels = []

    modelc.model.eval()
    with torch.no_grad():
        for batch in loader: #para teste
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

    # Print a detailed report
    print(classification_report(
        labels, 
        preds, 
        target_names=config.class_cols, 
        zero_division=0
    ))

    # Specifically for your article summary:
    micro_f1 = f1_score(labels, preds, average='micro')
    macro_f1 = f1_score(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)


    print(f"Micro F1: {micro_f1:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Accuracy: {acc:.4f}")
    
    
def fineTuneModel(modelc, epochs):
    modelc.model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(modelc.model.parameters(), lr=0.01)
    
    for epoch in tqdm(range(epochs), desc=f"Training Head"):
        for inputs, labels in modelc.train_loader:
            inputs, labels = inputs.to(config.device), labels.to(config.device)

            optimizer.zero_grad()
            outputs = modelc.model(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

    # 4. Evaluation (on pre-extracted validation features)
    #accuracy = evaluateModel(modelc.validation_loader, modelc.model)