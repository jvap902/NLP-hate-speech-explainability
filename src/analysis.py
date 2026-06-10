import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import deque
from . import config
from . import loadDataset
from . import dataVisualization

def compareAttr(model_names: list, instances: pd.DataFrame):
    #função que por instância faz um plot das atribuições dos modelos de acordo com a classe anotada
    #talvez fazer um "detokenize" para ter as tokenizações em comum
    
    indices = instances.index.to_list()
    
    for i in tqdm(indices, desc="Comparing model attributions"):
        
        instance = instances.loc[[i]]
        target_class = instance["class"].values[0]
        text = instance["text"].values[0]
        
        words = re.sub(r'([.,!?])(?! )', r'\1 ', text)
        
        words = re.split(r" |\n", text)
        words = [x for x in words if x != ""]
        
        df = pd.DataFrame()
        df["word"] = words
        
        for name in model_names:
            model_data = loadAttributions(name, words, instance)
            df[f"{name}_avg"] = model_data["avg"]
            df[f"{name}_max"] = model_data["max"]
        
        #print("\n")    
        #print(i, df)
        #print("\n")

        #dataVisualization.plotModelComparison(df, save_path=f'analysis/{target_class}-{i}.png', show=False)


def loadAttributions(model_name, words, instance) -> pd.DataFrame:
    """
    Carrega e detokeniza as atribuições de um modelo para uma instância.
    Retorna um DataFrame com colunas: model, word, <classes>
    """
    path = f"{config.ig_dir}/{model_name}.csv"
    
    target_class = instance["class"].values[0]
    text_id = instance.index.values[0]
    
    df = pd.read_csv(path, index_col=0)
    df = df[df['text_id'] == text_id].reset_index(drop=True)

    tokens = df['token'].tolist()
    attr = df[target_class].values

    avg_attrs, max_attrs = detokenize(tokens, attr, words, model_name)
    
    data = {'max': max_attrs, 'avg': avg_attrs}

    return data


# tokens that always start a new word regardless of ▁
WORD_START_TOKENS = {'@USER', 'HTTPURL', 'URL'}

def detokenize(tokens: list[str], attributions: np.ndarray, words: list[str], model_name: str):
    
    #iterar sobre tokens atribuidos: (1) tirar char especiais e (2) "subtrair" da frase o token
    
    is_bert_style = 'bertimbau' in model_name.lower()

    max_weights = np.empty(len(words))
    avg_weights = np.empty(len(words))
    
    remaining_token_attr = deque(zip(tokens, attributions))

    
    for i, w in enumerate(words):
        
        w_weights = []
        
        while w != "":
            
            tok, attr = remaining_token_attr.popleft()

            clean_token = tok.removeprefix('##') if is_bert_style else tok.removeprefix('▁').strip()
            
            after = removePrefixInsensitive(w, clean_token)
            
            if after == w and clean_token != "":
                
                print(f"Aviso: Problema de sincronia entre token e palavras: word: {w} - token: {clean_token}")
                print(f"Pulando palavra da frase {words}\n")
                
                remaining_token_attr.appendleft((tok, attr))
                w_weights.append(float(0))
                
                break
            
            w = after
               
            w_weights.append(float(attr))
        
        avg_weights[i] = np.array(w_weights).mean()
        max_weights[i] = np.array(w_weights).max()
    
    return avg_weights, max_weights

def removePrefixInsensitive(s: str, prefix: str) -> str:
    """removeprefix case-insensitive."""
    if s.lower().startswith(prefix.lower()):
        return s[len(prefix):]
    return s

if __name__ == "__main__":
    
    model_names = config.models.keys()
    
    instances = pd.read_csv(f'{config.ig_dir}/instances.csv', index_col='id')
    
    _, test = loadDataset.loadTuPyE(-1, -1)
    ig_dataset, indices = loadDataset.igDataset(test)
    
    instances_dataset = ig_dataset.to_pandas()
    instances_dataset.index = indices
    
    instances_dataset = instances_dataset[['text']].copy()
    instances_dataset['class'] = instances['class']
    
    compareAttr(model_names, instances_dataset)