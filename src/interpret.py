import ast
import torch
import pandas as pd
from tqdm import tqdm
from collections import deque
from captum.attr import LayerIntegratedGradients
from src import config
from .dataVisualization import plotAttributions

def getEmbeddingsLayer(model):
    # cada arquitetura expõe embeddings com um nome diferente
    for attr in ['bert', 'deberta', 'roberta', 'albert', 'electra', 'xlnet', 'distilbert']:
        if hasattr(model, attr):
            return getattr(model, attr).embeddings
    raise AttributeError(f"Não foi possível encontrar a camada de embeddings em {type(model).__name__}. "
                         f"Atributos disponíveis: {[n for n, _ in model.named_children()]}")

def explainPrediction(modelc, tokenized_dataset, indices, plot_indices: deque[int], show_graph=True):
    modelc.model.eval()
    
    df_attributions = pd.DataFrame()
    
    #wrapper
    def forward_func(input_ids):
        return modelc.model(input_ids).logits

    # Using modelc.model.bert.embeddings for BERTimbau
    lig = LayerIntegratedGradients(forward_func, getEmbeddingsLayer(modelc.model))
    
    # pega tokens speciais de acordo com o modelo
    ignore_tokens = {modelc.tokenizer.cls_token, modelc.tokenizer.sep_token, modelc.tokenizer.pad_token}
    unk_token = modelc.tokenizer.unk_token
    
    for i in tqdm(range(len(tokenized_dataset)), desc="Generating attributions"):
        
        instance = tokenized_dataset[i]
        text_id = indices[i]
        
        # Usamos a mask para saber onde a frase termina de verdade
        mask = instance['attention_mask']
        actual_length = int(mask.sum()) 
        
        # Cortamos o tensor para ignorar os [PAD] antes de qualquer cálculo
        input_ids = instance['input_ids'][:actual_length].unsqueeze(0).to(config.device)
        
        # Pegamos os tokens correspondentes a esse corte
        all_tokens = modelc.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        token_mask = [t not in ignore_tokens for t in all_tokens]
        tokens = [f"⚠️ {t}" if t == unk_token else t for t, keep in zip(all_tokens, token_mask) if keep]
        
        df_sentence = pd.DataFrame({'text_id': text_id, 'token': tokens})
        
        internal_batch_size = 2  if 'albertina' in modelc.name.lower() else 8
        
        for target_idx, class_name in tqdm(enumerate(config.model_classes), desc="Attributing values"):
            attributions = lig.attribute(inputs=input_ids, target=target_idx, n_steps=50, internal_batch_size=internal_batch_size)
            attr_array = attributions.sum(dim=-1).squeeze(0).cpu().detach().numpy()
            
            del attributions
            
            df_sentence[class_name] = attr_array[token_mask] #remove tokens especiais como [cls, sep]

        if plot_indices:
            if text_id == plot_indices[0]:
                plot_indices.popleft()
                plotAttributions(df_sentence, save_path=f"{config.ig_dir}/images/{modelc.name}/{text_id}-ig.png", show=show_graph)
        
        df_attributions = pd.concat([df_attributions, df_sentence], axis=0, ignore_index=True)

        # add csv line to save the progress
        df_sentence.to_csv(f"{config.ig_dir}/{modelc.name}.csv", header=False, mode='a', index=False)
        
        del df_sentence
        torch.cuda.empty_cache()
        
    #df_attributions.to_csv(f"{config.ig_dir}/{modelc.name}.csv", header=True)

def compareHumanModel(modelc, model_attr, instances):
    
    indices = instances["id"].to_list()
    
    human_tok = tokenizeRelevantWords(instances, modelc.tokenizer)
    
    for i in indices:
        model_inst = model_attr[model_attr["text_id"] == i]
        human_inst = human_tok[human_tok["text_id"] == i]
        
        active_class = human_inst["class"].iloc[0]
        
        h_tokens = human_inst["tokens"].to_list()
        
        result = True
        
        for token_list in h_tokens:
            
            m_tokens = model_inst[model_inst["token"].isin(token_list)]
            
            imp_values = m_tokens[active_class].to_numpy()
            
            result = result and not (imp_values <= 0).any() # False se token que deveria contribuir para uma classe for < 0
            
            if not result:
                print(result)
                print(imp_values)
                raise
        
        print(result)
        raise
        

def tokenizeRelevantWords(human_attr: pd.DataFrame, tokenizer) -> pd.DataFrame:
    rows = []

    for _, row in human_attr.iterrows():
        instance_id = row['id']
        cls         = row['class']
        words       = ast.literal_eval(row['relevant_words'])  # "[""toma"", ...]" → lista Python

        for word in words:
            tokens = tokenizer.tokenize(word)  # tokeniza sem adicionar [CLS]/[SEP]
            rows.append({
                'text_id': instance_id,
                'class': cls,
                'word': word,
                'tokens': tokens,
            })

    return pd.DataFrame(rows)