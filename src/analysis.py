import re
import string
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
            #name = "Bernice"
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

    avg_attrs, max_attrs = wordAttributes(tokens, attr, words, model_name)
    
    data = {'max': max_attrs, 'avg': avg_attrs}

    return data


# tokens that always start a new word regardless of ▁
WORD_START_TOKENS = {'@USER', 'HTTPURL', 'URL'}

def wordAttributes(tokens: list[str], attributions: np.ndarray, words: list[str], model_name: str):
    
    #ainda tratar diferentes separações de pontuação
    
    bert_style = 'bertimbau' in model_name.lower()
    
    model_words = modelWords(tokens, attributions, bert_style)
    model_words = model_words.to_dict(orient='records')
    
    #fazer mapeamento palavras reais e atributos
    
    remaining_words = deque(words)
    remaining_model_words = deque(model_words)
    
    attrs_rows = [{'word': w, 'avg': 0, 'max': 0} for w in words] # preenche com zero caso haja mais palavras que modelo excluiu
    
    while remaining_words:
        w = remaining_words.popleft()
        current_word = ""
        current_attr = []
        
        while remaining_model_words:
            
            row = remaining_model_words[0]
        
            current_word = current_word + row['word']
            current_attr = current_attr + row['attributions']
        
            if w.startswith(current_word):
                
                _ = remaining_model_words.popleft()
                
                if w == current_word:
                    attrs_rows.append({'word': w, 'avg': np.mean(current_attr), 'max': np.max(current_attr)})
            
            else:
                break #pula palavra, já tem zeros onde precisa
                
    print(pd.DataFrame(attrs_rows))
                
            
    
    raise NotImplementedError
    return 


def modelWords(tokens, attributions, bert_style):

    rows_list = []
    ant = ""
    
    curr_word = ""
    word_attrs = []
    
    for tok, attr in zip(tokens, attributions):
        
        if bert_style:
            is_new_word = not tok.startswith("##") and not ant != '@' and tok not in list(string.punctuation.remove('@'))
        else:
            is_new_word = tok.startswith('▁') or tok in WORD_START_TOKENS or tok.startswith('@')
        
        clean_token = tok.removeprefix('##') if bert_style else tok.removeprefix('▁').strip()
        
        if is_new_word: #adiciona nova palavra
            
            if curr_word != "": #necessário por conta da primeira iteração
                rows_list.append({"word": curr_word, "attributions": word_attrs}) #adiciona palavra finalizada
            
            curr_word = clean_token #começa nova palavra
            word_attrs = [float(attr)]
        
        else:
            curr_word = curr_word + clean_token #concatena token limpo
            word_attrs.append(float(attr)) #adiciona atribuição a ele
        
        ant = tok
        
    df = pd.DataFrame(rows_list)
    #print(df)
    raise
    
    return df

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