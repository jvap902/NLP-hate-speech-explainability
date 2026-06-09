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
        target_class = instances.loc[i]["class"]
        
        for name in model_names:
            inst_dfs.append(loadAttributions(name, instances.loc[[i]])) #mudar para cada linha ser uma palavra e cada coluna um modelo
        
        inst_dfs = pd.concat(inst_dfs, ignore_index=True)
        
        dataVisualization.plotModelComparison(inst_dfs, target_class, model_names, save_path=f'analysis/{target_class}-{i}.png', show=False)


def loadAttributions(model_name, instance) -> pd.DataFrame:
    """
    Carrega e detokeniza as atribuições de um modelo para uma instância.
    Retorna um DataFrame com colunas: model, word, <classes>
    """
    path = f"{config.ig_dir}/{model_name}.csv"
    
    target_class = instance["class"].values
    words = instance["text"].values
    text_id = instance.index
    
    print(instance)
    raise
    
    df = pd.read_csv(path, index_col=0)
    df = df[df['text_id'] == text_id].reset_index(drop=True)

    if df.empty:
        raise(f"text_id {text_id} not found in {model_name}.csv")

    tokens = df['token'].tolist()
    attr = df[target_class].values

    avg_attrs = detokenize(tokens, attr, model_name)
    
    data = {'model': model_name, 'word': words, 'attribution': avg_attrs}

    return pd.DataFrame(data)


# tokens that always start a new word regardless of ▁
WORD_START_TOKENS = {'@USER', 'HTTPURL', 'URL'}

def detokenize(tokens: list[str], attributions: np.ndarray, model_name: str):
    
    #iterar sobre tokens atribuidos: (1) tirar char especiais e (2) "subtrair" da frase o token
    
    is_bert_style = 'bertimbau' in model_name.lower()

    words   = []
    weights = []

    for token, attr in zip(tokens, attributions):
        if is_bert_style:
            is_continuation = token.startswith('##')
        else:
            # placeholders Bernice e tokens com @ sempre iniciam nova palavra
            is_word_start   = token.startswith('▁') or token in WORD_START_TOKENS or token.startswith('@')
            is_continuation = not is_word_start and len(words) > 0

        if is_continuation and words:
            clean = token[2:] if is_bert_style else token
            words[-1] += clean
            weights[-1].append(float(attr))
        
        else:
            clean = token.replace('▁', '').strip()
            words.append(clean or token)
            weights.append([float(attr)])

    avg_weights = np.array([np.mean(w) for w in weights])
    return words, avg_weights


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