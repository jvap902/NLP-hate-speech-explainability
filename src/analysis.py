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
        
        # 1. Remove qualquer espaço que tente separar pontuações vizinhas (ex: "? !" vira "?!")
        text = re.sub(r'([.,!?])\s+(?=[.,!?])', r'\1', text)

        # 2. Cola todo o bloco de pontuações na palavra anterior e bota um espaço depois Ex: "amigo    ?!?!?!" -> "amigo?!?!?!" -> "amigo?!?!?! "
        text = re.sub(r'\s*([.,!?]+)(?!\s|$)', r'\1 ', text)
        text = re.sub(r'\s+([.,!?]+)', r'\1', text)

        # 3. Limpeza final de espaçamentos
        text = re.sub(r'\s+', ' ', text).strip()

        # 4. Cria a lista de tokens
        words = text.split()
        
        df = pd.DataFrame()
        df["word"] = words
        
        for name in model_names:
            model_data = loadAttributions(name, words, instance)
            df[f"{name}_avg"] = model_data["avg"]
            df[f"{name}_max"] = model_data["max"]
            
        print("\n", i, df)
        
        dataVisualization.plotModelComparison(df, save_path=f'analysis/{target_class}-{i}.png', show=False)


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

    words_df = wordAttributes(tokens, attr, words, model_name)
    
    data = {'max': words_df['max'].to_list(), 'avg': words_df['avg'].to_list()}

    return data

def wordAttributes(tokens: list[str], attributions: np.ndarray, words: list[str], model_name: str):
    
    #ainda tratar diferentes separações de pontuação
    
    pos_word = []
    for i, w in enumerate(words): #faz lista com posição por conta da repetição de palavras
        
        if 'º' in w: #modelos não conseguem tratar isso, traduz para eles
            if model_name.lower() == 'bernice': w = w.replace('º', 'o')
            elif model_name.lower() == 'albertina': w = w.replace('º', '') #bertimbau exclui a coisa
            
        pos_word.append((i, w.lower()))
    
    words = pos_word
    
    model_words = modelWords(tokens, attributions, model_name)
    
    # mapeamento palavras reais e atributos
    
    remaining_words = deque(words)
    remaining_model_words = deque(model_words)
    
    attrs_rows = [{'id': i, 'word': w, 'avg': 0.0, 'max': 0.0} for i, w in words] # preenche com zero caso haja mais palavras que modelo excluiu
    attrs_df = pd.DataFrame(attrs_rows).set_index('id')
    
    untreated_names = ['@editorahumanas', '@maathbz']
    
    while remaining_words:
        i, w = remaining_words.popleft()
        current_word = ""
        current_attr = []
        
        if model_name.lower() == 'bernice' and any(name in w for name in untreated_names): # bernice é a única que faz todo @usuario virar @USER e como existem 2 nomes não tratados, faz-se uma adaptação
            w = w.replace('@editorahumanas', '@user')
            w = w.replace('@maathbz', '@user')
        
        while remaining_model_words:
            
            row = remaining_model_words[0]
            candidate = current_word + row['word']
        
            if w.startswith(candidate):
                
                current_word = current_word + row['word']
                current_attr = current_attr + row['attributions']
                
                _ = remaining_model_words.popleft() # consome após confirmar match
                
                if w == current_word:
                    attrs_df.loc[i, 'avg'] = np.mean(current_attr)
                    attrs_df.loc[i, 'max'] = np.max(current_attr)
                    break # se palavra está completa passa para a próxima
            
            else:
                break # modelo pulou esta palavra, mantém zeros

    return attrs_df

# tokens that always start a new word regardless of ▁
WORD_START_TOKENS = {'@', '@USER', 'HTTPURL', 'URL'}
def modelWords(tokens, attributions, model_name):

    style = 'bertimbau' if 'bertimbau' in model_name.lower() else model_name.lower()
    
    rows_list = []
    
    ant = "-$+!a!b&^" # apenas algo que não começaria uma frase para não estar em nenhum grupo
    curr_word = ""
    word_attrs = []
    
    for tok, attr in zip(tokens, attributions):
        
        puncts = ".,?!"
        tok_is_punct = tok in puncts
        ant_is_punct = ant in puncts
        new_word_by_punct = ant_is_punct and not tok_is_punct # pontuação deve ser nova palavra se o token anterior não é pontuação
        
        if style == 'bertimbau':
            is_new_word = (not tok.startswith("##") and ant != '@') or new_word_by_punct
            clean_token = tok.removeprefix("##")
        
        elif style == 'bernice':
            is_new_word = (tok.startswith('▁') or tok in WORD_START_TOKENS) or new_word_by_punct
            clean_token = tok.removeprefix('▁').strip()
        
        else:
            is_new_word = (tok.startswith('▁') or tok in WORD_START_TOKENS) or new_word_by_punct
            clean_token = tok.removeprefix('▁').strip()
        
        if is_new_word: #adiciona nova palavra
            
            if curr_word != "": #necessário por conta da primeira iteração
                rows_list.append({"word": curr_word.lower(), "attributions": word_attrs}) #adiciona palavra finalizada
            
            curr_word = clean_token #começa nova palavra
            word_attrs = [float(attr)]
        
        else:
            curr_word = curr_word + clean_token #concatena token limpo
            word_attrs.append(float(attr)) #adiciona atribuição a ele
        
        ant = clean_token
    
    # adiciona último token que não foi adicionado por não ter nada depois
    rows_list.append({"word": curr_word.lower(), "attributions": word_attrs})
    
    return rows_list

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