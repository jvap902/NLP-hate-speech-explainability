from torch.utils.data import DataLoader
import pandas as pd
from src import *

if __name__ == "__main__":
    
    print(f"device set to {config.device}")
    
    train, test = loadDataset.loadTuPyE(-1, -1) #definir tamanhos dos splits
    
    models = config.models
    
    for key, value in models.items(): #apenas testando se é possível carregar
        modelc = Model(value, key)
        
        modelc.getLoader(train, test, batch_size=32)
        
        #classify.evaluateModel(modelc, modelc.test_loader)
    
        modelc, losses = classify.fineTuneModel(modelc, 15, epoch_save=True)
    
        dataVisualization.plotLosses(losses, save_path=f"output-images/{modelc.name}-loss.png", show=False)
    
        classify.evaluateModel(modelc, modelc.test_loader)
    
        modelc.saveModel()
        
        ig_dataset = loadDataset.igDataset()
        
        ig_tokenized = modelc.tokenize(ig_dataset)
        
        ig_loader = DataLoader(ig_tokenized, batch_size=32, shuffle=False, num_workers=4)
        
        ig_predictions = classify.classifyInputs(modelc, ig_loader)

        print(ig_predictions)
        
        interpret.explainPrediction(modelc, ig_tokenized, show_graph=False)
        
        del modelc