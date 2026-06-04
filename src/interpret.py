from captum.attr import LayerIntegratedGradients
import pandas as pd
from tqdm import tqdm
from src import config
from src.fileHandler import findInCsv, writeCsvLine
from .dataVisualization import plotAttributions

def getEmbeddingsLayer(model):
    # cada arquitetura expõe embeddings com um nome diferente
    for attr in ['bert', 'deberta', 'roberta', 'albert', 'electra', 'xlnet', 'distilbert']:
        if hasattr(model, attr):
            return getattr(model, attr).embeddings
    raise AttributeError(f"Não foi possível encontrar a camada de embeddings em {type(model).__name__}. "
                         f"Atributos disponíveis: {[n for n, _ in model.named_children()]}")

def explainPrediction(modelc, tokenized_dataset, show_graph=True):
    modelc.model.eval()
    
    df_attributions = pd.DataFrame()
    
    #wrapper
    def forward_func(input_ids):
        return modelc.model(input_ids).logits

    # Using modelc.model.bert.embeddings for BERTimbau
    lig = LayerIntegratedGradients(forward_func, getEmbeddingsLayer(modelc.model))
    
    # pega tokens speciais de acordo com o modelo
    special_tokens = set(modelc.tokenizer.all_special_tokens)    
    
    for i in range(len(tokenized_dataset)):
        
        instance = tokenized_dataset[i]
        
        # Usamos a mask para saber onde a frase termina de verdade
        mask = instance['attention_mask']
        actual_length = int(mask.sum()) 
        
        # Cortamos o tensor para ignorar os [PAD] antes de qualquer cálculo
        input_ids = instance['input_ids'][:actual_length].unsqueeze(0).to(config.device)
        
        # Pegamos os tokens correspondentes a esse corte
        all_tokens = modelc.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        token_mask = [t not in special_tokens for t in all_tokens]
        tokens     = [t for t, keep in zip(all_tokens, token_mask) if keep]
        
        df_sentence = pd.DataFrame({'text_id': id, 'token': tokens})
        
        for target_idx, class_name in tqdm(enumerate(config.model_classes), desc="Attributing values"):
            attributions = lig.attribute(inputs=input_ids, target=target_idx, n_steps=50)
            attr_array = attributions.sum(dim=-1).squeeze(0).cpu().detach().numpy()
            
            df_sentence[class_name] = attr_array[token_mask] #remove tokens especiais como [cls, sep]
        
        plotAttributions(df_sentence, i, save_path=f"{config.ig_results_dir}/images/{modelc.name}-{i}-ig.png", show=show_graph)
        
        df_attributions = pd.concat([df_attributions, df_sentence], axis=0, ignore_index=True)
        
        del df_sentence
        
    df_attributions.to_csv(f"{config.ig_results_dir}/{modelc.name}.csv", header=True)