from captum.attr import LayerIntegratedGradients
import pandas as pd
from tqdm import tqdm
from src import config
from src.fileHandler import findInCsv, writeCsvLine
from .dataVisualization import plotAttributions


def explainPrediction(modelc, tokenized_dataset, show_graph=True):
    modelc.model.eval()
    #wrapper
    def forward_func(input_ids):
        return modelc.model(input_ids).logits

    # Using modelc.model.bert.embeddings for BERTimbau
    lig = LayerIntegratedGradients(forward_func, modelc.model.bert.embeddings)
    
    for i in range(len(tokenized_dataset)):
        
        instance = tokenized_dataset[i]
        
        # Usamos a mask para saber onde a frase termina de verdade
        mask = instance['attention_mask']
        actual_length = int(mask.sum()) 
        
        # Cortamos o tensor para ignorar os [PAD] antes de qualquer cálculo
        input_ids = instance['input_ids'][:actual_length].unsqueeze(0).to(config.device)
        
        # Pegamos os tokens correspondentes a esse corte
        tokens = modelc.tokenizer.convert_ids_to_tokens(input_ids[0])
        tokens = [t for t in tokens if t not in ['[CLS]', '[SEP]']]
        
        id_val = getTextId(tokens).iloc[0]
        
        id = [id_val] * len(tokens)
        
        df_sentence = pd.DataFrame({'text_id': id, 'token': tokens})
        
        for target_idx, class_name in tqdm(enumerate(config.classes), desc="Attributing values"):
            attributions = lig.attribute(inputs=input_ids, target=target_idx, n_steps=50)
            attr_array = attributions.sum(dim=-1).squeeze(0).cpu().detach().numpy()
            
            df_sentence[class_name] = attr_array[1:-1] #remove [cls, sep]
            
        df_sentence.to_csv(f"{config.ig_results_dir}/{modelc.name}.csv")
        
        plotAttributions(df_sentence, i, save_path=f"{config.ig_results_dir}/images/{modelc.name}-{id_val}-ig.png", show=show_graph)
        
        del df_sentence
            
def getTextId(tokens, text_id_csv=f"{config.ig_results_dir}/text-id.csv"):
    
    text = " ".join(tokens)
    
    df = pd.read_csv(text_id_csv)
    
    row = df[df['text'] == text]
    
    if len(row) == 0:
        writeCsvLine(text_id_csv, [len(df), text])
        return len(df)
    
    else:
        return row["id"]
    
    #procurar por texto/id, se não tiver incluir novo e retornar, se existir apenas retornar