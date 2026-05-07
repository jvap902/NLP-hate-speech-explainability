from huggingface_hub import login
from pathlib import Path
from datasets import load_dataset, load_from_disk
import numpy as np
from . import config
from skmultilearn.model_selection import iterative_train_test_split
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()

def loadToken(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            return content
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        
def stratifiedIndices(dataset, subset_size):
    
    # 1. Extract labels
    y = np.array([dataset[col] for col in config.class_cols]).T
    
    # 2. Create indices
    indices = np.arange(len(y)).reshape(-1, 1)
    
    # 3. Use the split function to discard the unwanted portion
    # We want 'subset_size' of the data, so test_size here is (1 - subset_size)
    # This keeps 'subset_size' in the 'train' return variables
    idx_subset, _, _, _ = iterative_train_test_split(
        indices, y, test_size = 1 - subset_size
    )
    
    return idx_subset.flatten().tolist()

def loadTuPyE(train_size=-1, test_size=-1): #-1 quer dizer carregar inteiro
    
    dataset_link = "Silly-Machine/TuPyE-Dataset" #https://huggingface.co/datasets/Silly-Machine/TuPyE-Dataset
    
    console.print(Markdown("\n ## Loading dataset via Hugging Face \n"))
            
    try:
        
        hf_token = loadToken('token.txt')
        
        login(token=hf_token, add_to_git_credential=False)
        
        dir_name = "TuPyE"
                    
        if(Path(f'./data/{dir_name}').is_dir()):
            
            console.print(Markdown("###Loading already downloaded dataset"))
            
            train = load_from_disk(f'./data/{dir_name}/train')
            test = load_from_disk(f'./data/{dir_name}/test')
        
        #caso seja necessário baixar
        else:
            console.print(Markdown("\n### Downloading huggingface dataset"))
            
            train = load_dataset(dataset_link, "multilabel", split="train")
            test = load_dataset(dataset_link, "multilabel", split="test")
            
            console.print("Selecionando índices para os subsets")
            if train_size != -1: train = train.select(stratifiedIndices(train, train_size))
            if test_size != -1: test = test.select(stratifiedIndices(test, test_size))
            
            console.print("Salvando subsets em disco")
            train.save_to_disk(f'./data/{dir_name}/train')
            test.save_to_disk(f'./data/{dir_name}/test')
            
        
        #depois de carregados, coloca em formato do pytorch
        train = train.with_format("torch", columns=config.class_cols)
        test = test.with_format("torch", columns=config.class_cols)
        
        console.print(Markdown(f"\n### Datasets loaded"))
        
    except Exception as e:
         raise RuntimeError(f"Failed to load Hugging Face Dataset: {e}")
     
    return train, test