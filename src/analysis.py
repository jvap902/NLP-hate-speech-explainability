import numpy as np
import pandas as pd
from tqdm import tqdm
from . import config
from . import loadDataset
from . import dataVisualization

def compareAttr(model_names: list, instances: pd.DataFrame):
    #função que por instância faz um plot das atribuições dos modelos de acordo com a classe anotada
    #talvez fazer um "detokenize" para ter as tokenizações em comum
    
    indices = instances.index.to_list()
    
    for i in tqdm(indices, desc="Comparing model attributions"):
        
        inst_dfs = []
        instance = instances.loc[[i]]
        target_class = instance["class"].values[0]
        text = instance["text"].values[0]
        
        df = pd.DataFrame()
        df["word"] = text.split(' ')
        
        for name in model_names:
            df[name] = loadAttributions(name, instance)["attribution"]
        
        print(df)
        
        dataVisualization.plotModelComparison(inst_dfs, target_class, model_names, save_path=f'analysis/{target_class}-{i}.png', show=False)
        
        raise


def loadAttributions(model_name, instance) -> pd.DataFrame:
    """
    Carrega e detokeniza as atribuições de um modelo para uma instância.
    Retorna um DataFrame com colunas: model, word, <classes>
    """
    path = f"{config.ig_dir}/{model_name}.csv"
    
    target_class = instance["class"].values[0]
    text = instance["text"].values[0]
    text_id = instance.index.values[0]
    
    df = pd.read_csv(path, index_col=0)
    df = df[df['text_id'] == text_id].reset_index(drop=True)

    tokens = df['token'].tolist()
    attr = df[target_class].values

    avg_attrs = detokenize(tokens, attr, text, model_name)
    
    data = {'model': model_name, 'word': text, 'attribution': avg_attrs}

    return data


# tokens that always start a new word regardless of ▁
WORD_START_TOKENS = {'@USER', 'HTTPURL', 'URL'}

def detokenize(tokens: list[str], attributions: np.ndarray, text: str, model_name: str):
    
    #iterar sobre tokens atribuidos: (1) tirar char especiais e (2) "subtrair" da frase o token
    
    is_bert_style = 'bertimbau' in model_name.lower()

    words = text.split(' ')

    avg_weights = np.empty(len(words))
    
    current_word_idx = 0
    current_word = words[current_word_idx]

    w_weights = []
    
    for token, attr in zip(tokens, attributions):
            
        if is_bert_style:
            is_continuation = token.startswith('##')
        else:
            # placeholders Bernice e tokens com @ sempre iniciam nova palavra
            is_word_start   = token.startswith('▁') or token in WORD_START_TOKENS or token.startswith('@')
            is_continuation = not is_word_start

        #possivelmente achar um jeito de ir subtraindo string original

        if is_continuation:
            clean_token = token[2:] if is_bert_style else token
        else:
            clean_token = token.replace('▁', '').strip()
            
        current_word = current_word.removeprefix(clean_token)
        
        w_weights.append(attr)
        
        if current_word == "":
            avg_weights[current_word_idx] = np.array(w_weights).mean()
            current_word_idx += 1 if current_word_idx < len(words)-1 else -1
            current_word = words[current_word_idx]
            w_weights = []
    
    return avg_weights


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