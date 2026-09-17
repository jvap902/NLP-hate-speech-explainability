import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque
from rich.panel import Panel
from rich.console import Console
from rich.markdown import Markdown
from huggingface_hub import login
from datasets import load_dataset, load_from_disk, Dataset
from skmultilearn.model_selection import iterative_train_test_split
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
    y = np.array([dataset[col] for col in config.dataset_classesclasses]).T
    
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
        #train = train.with_format("torch", columns=config.dataset_classes)
        #test = test.with_format("torch", columns=config.dataset_classes)
        
        dataset_panel.addMessage(f"\n### Datasets loaded")
        
    except Exception as e:
         raise RuntimeError(f"Failed to load Hugging Face Dataset: {e}")
    
    dataset_panel.stop()
    
    return train, test

def prepareDataset(tokenized_ds):
    """
    Junta as colunas individuais de classe numa coluna 'labels' float32,
    que é o que o Trainer procura.
    """
    def mergeLabels(example):
        example["labels"] = [np.float32(example[c]) for c in config.dataset_classes]
        return example

    ds = tokenized_ds.map(mergeLabels)
    ds = ds.remove_columns(config.dataset_classes)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds

def getPlotIndices(file_path: str = "utils/selected_instances.json") -> deque[int]:
    with open(file_path, 'r') as f:
        data = json.load(f)

    plot_indices = []

    for class_name in data:
        for instance_type in data[class_name]:
            for instance in data[class_name][instance_type]:
                plot_indices.append(instance["id"])

    plot_indices.sort()
    
    return deque(plot_indices)


def igDataset(test_dataset):
   
    # Agora estamos faremos as atribuições para o dataset inteiro, portanto esta função deixará de ser utilizada
    
    instances_df = pd.read_csv(f'{config.ig_dir}/instances.csv')
    
    indices = instances_df['id'].tolist()
    
    ig_dataset = test_dataset.select(indices)
    
    return ig_dataset, indices