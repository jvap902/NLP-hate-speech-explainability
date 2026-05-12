from huggingface_hub import login
from pathlib import Path
from datasets import load_dataset, load_from_disk, Dataset
import numpy as np
import pandas as pd
from skmultilearn.model_selection import iterative_train_test_split
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from . import config
from .logging import LivePanel

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
    y = np.array([dataset[col] for col in config.classes]).T
    
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
    
    dataset_panel = LivePanel("Loading dataset via [bold blue]Hugging Face[/]")
    
    dataset_panel.start()
            
    try:
        
        hf_token = loadToken('token.txt')
        
        login(token=hf_token, add_to_git_credential=False)
        
        dir_name = "TuPyE"
                    
        if(Path(f'./data/{dir_name}').is_dir()):
            
            dataset_panel.addMessage("## Loading already downloaded dataset")
            
            train = load_from_disk(f'./data/{dir_name}/train')
            test = load_from_disk(f'./data/{dir_name}/test')
        
        #caso seja necessário baixar
        else:
            dataset_panel.addMessage("\n## Downloading huggingface dataset")
            
            train = load_dataset(dataset_link, "multilabel", split="train")
            test = load_dataset(dataset_link, "multilabel", split="test")
            
            dataset_panel.addMessage("Selecionando índices para os subsets")
            if train_size != -1: train = train.select(stratifiedIndices(train, train_size))
            if test_size != -1: test = test.select(stratifiedIndices(test, test_size))
            
            dataset_panel.addMessage("Salvando subsets em disco")
            train.save_to_disk(f'./data/{dir_name}/train')
            test.save_to_disk(f'./data/{dir_name}/test')
            
        
        #depois de carregados, coloca em formato do pytorch
        train = train.with_format("torch", columns=config.classes)
        test = test.with_format("torch", columns=config.classes)
        
        dataset_panel.addMessage(f"\n### Datasets loaded")
        
    except Exception as e:
         raise RuntimeError(f"Failed to load Hugging Face Dataset: {e}")
    
    dataset_panel.stop()
    
    return train, test


def igDataset():    
    dataset = [
        ["Eu queria dar um soco nele", 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ["Texto neutro de exemplo",     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    
    dataset = pd.DataFrame(dataset, columns=['text'] + config.classes)
    
    return Dataset.from_pandas(dataset)