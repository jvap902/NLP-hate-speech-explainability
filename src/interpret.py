from captum.attr import LayerIntegratedGradients
from src import config
from .dataVisualization import plotAttributions

def explainPrediction(modelc, tokenized_dataset, show_graph=True):
    modelc.model.eval()
    
    # wrapper function that Captum can use
    def forward_func(input_ids):
        return modelc.model(input_ids).logits

    # Using modelc.model.bert.embeddings for BERTimbau
    lig = LayerIntegratedGradients(forward_func, modelc.model.bert.embeddings)
    
    all_attr = []
    all_tokens = []
    
    for i in range(len(tokenized_dataset)):
        
        input_ids = tokenized_dataset[i]['input_ids'].unsqueeze(0).to(config.device)
        
        label_values = [tokenized_dataset[i][cls].item() for cls in config.classes]
        active_classes = [idx for idx, val in enumerate(label_values) if val == 1]
        
        for target_idx in active_classes:
            # 3. Calcular atribuição
            attributions = lig.attribute(inputs=input_ids, target=target_idx, n_steps=50)
            
            # 4. Processar
            attributions = attributions.sum(dim=-1).squeeze(0)
            
            tokens = modelc.tokenizer.convert_ids_to_tokens(input_ids[0])
            
            all_attr.append(attributions.cpu().detach().numpy())
            all_tokens.append(tokens)
            
            plotAttributions(all_attr[0], all_tokens[0], "aggressive", save_path=f"output-images/{modelc.name}-aggressive-ig.png", show=show_graph)