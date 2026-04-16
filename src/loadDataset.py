from huggingface_hub import login
import os
from torch.utils.data import Dataset, Subset, dataloader
from datasets import load_dataset, load_from_disk

def loadToken(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            return content
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def newHuggingfaceDataset(dataset_link, ipc, n_classes):
    print("\n--- Loading dataset via Hugging Face ---")
            
    try:
        
        hf_token = loadToken('token.txt')
        
        login(token=hf_token, add_to_git_credential=False)
        
        dir_name = dataset_link.replace('/','-')
                    
        if(os.Path(f'./data/{dir_name}').is_dir()):
            
            print("Loading already downloaded dataset")
            
            hf_train = load_from_disk(f'./data/{dir_name}/train')
            hf_validation = load_from_disk(f'./data/{dir_name}/validation')
        
        #caso seja necessário baixar
        else:
            print("Downloading huggingface dataset")
            
            hf_train = load_dataset(dataset_link, split='train', streaming=False)
            hf_validation = load_dataset(dataset_link, split='validation', streaming=False)
        
    except Exception as e:
         raise RuntimeError(f"Failed to load Hugging Face Dataset: {e}")
     
    raise NotImplementedError