from torch.utils.data import DataLoader
import pandas as pd
from rich.console import Console
from rich.markdown import Markdown
from src import *

if __name__ == "__main__":
    
    print(f"device set to {config.device}")
    
    train, test = loadDataset.loadTuPyE(-1, -1) #definir tamanhos dos splits
    
    models = config.models
    
    for key, value in models.items(): #apenas testando se é possível carregar
        modelc = Model(value, key)
        
        modelc.getLoader(train, test, batch_size=32)
        
        modelc.getTrainer(epochs_fold=5, compute_metrics=classify.compute_metrics)
            
        #modelc = classify.fineTune(modelc, repeat=4)
        
        eval_data = classify.testModel(modelc)
        print(eval_data)
    
        #dataVisualization.plotLosses(losses, save_path=f"losses/{modelc.name}.png", show=False)
                
        ig_dataset = loadDataset.igDataset()
        
        ig_tokenized = modelc.tokenize(ig_dataset).with_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        
        ig_loader = DataLoader(ig_tokenized, batch_size=32, shuffle=False, num_workers=4, collate_fn=modelc.data_collator)
        
        ig_predictions = classify.classifyInputs(modelc, ig_loader)

        print(ig_predictions)
        
        interpret.explainPrediction(modelc, ig_tokenized, show_graph=False)
        
        del modelc
