from torch.utils.data import DataLoader
import argparse
from src import *

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, required=False, default="BERTimbau-base", help="Specify a model name present in config.models")
parser.add_argument("-nf", "--no_fine_tune", required=False, action='store_true', default=False, help="Disable fine-tuning process")

args = parser.parse_args()

if __name__ == "__main__":
    
    print(f"device set to {config.device}")
    
    train, test = loadDataset.loadTuPyE(-1, -1) #definir tamanhos dos splits
    
    models = config.models
    
    name = args.model
    link = models[name]
    
    modelc = Model(link, name)
    
    modelc.getLoader(train, test, batch_size=32)
    
    modelc.getTrainer(epochs_fold=5, compute_metrics=classify.compute_metrics)
    
    if "tupy" not in modelc.link.lower() and not args.no_fine_tune:
        modelc = classify.fineTune(modelc, repeat=3)
    
    eval_data = classify.testModel(modelc)
    print(eval_data)

    #dataVisualization.plotLosses(losses, save_path=f"losses/{modelc.name}.png", show=False)
            
    ig_dataset, indices = loadDataset.igDataset(test)
    
    ig_tokenized = modelc.tokenize(ig_dataset).with_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    ig_loader = DataLoader(ig_tokenized, batch_size=32, shuffle=False, num_workers=4, collate_fn=modelc.data_collator)
    
    ig_predictions = classify.classifyInputs(modelc, ig_loader)

    print(ig_predictions)
    
    interpret.explainPrediction(modelc, ig_tokenized, indices, show_graph=False)
    
    del modelc
