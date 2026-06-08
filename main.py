import argparse
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from src import *

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, required=False, default="BERTimbau-base", help="Specify a model name present in config.models")
parser.add_argument("-nf", "--no_fine_tune", required=False, action='store_true', default=False, help="Disable fine-tuning process")
parser.add_argument("-nig", "--new_ig", required=False, action='store_true', default=False, help="Force new Integrated Gradients attribution")

args = parser.parse_args()

def tune(modelc):
    if "tupy" not in modelc.link.lower():
        modelc = classify.fineTune(modelc, repeat=3)
    
        eval_data = classify.testModel(modelc)
        print(eval_data)

def attribute(modelc, ig_dataset):
    ig_tokenized = modelc.tokenize(ig_dataset).with_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    ig_loader = DataLoader(ig_tokenized, batch_size=32, shuffle=False, num_workers=4, collate_fn=modelc.data_collator)
    
    ig_predictions = classify.classifyInputs(modelc, ig_loader)
    ig_predictions['model'] = modelc.name
    ig_predictions['text_id'] = indices
    ig_predictions = ig_predictions[["model", "text_id", "text", "labels", "preds"]]
    ig_predictions.to_csv(f'{config.ig_dir}/predictions.csv', mode='a', header=False, index=False)

    print(ig_predictions)
    
    interpret.explainPrediction(modelc, ig_tokenized, indices, show_graph=False)
    
def loadModelc():
    name = args.model
    link = config.models[name]
    
    modelc = Model(link, name)
    
    modelc.getLoader(train, test, batch_size=32)
    
    modelc.getTrainer(epochs_fold=5, compute_metrics=classify.compute_metrics)
    
    return modelc

if __name__ == "__main__":
    
    print(f"device set to {config.device}")
    
    train, test = loadDataset.loadTuPyE(-1, -1) #definir tamanhos dos splits

    modelc = loadModelc()
    
    if not args.no_fine_tune: tune(modelc)

    ig_dataset, indices = loadDataset.igDataset(test)

    if args.new_ig or not Path(f"{config.ig_dir}/{modelc.name}.csv"): attribute(modelc, ig_dataset)
    
    model_attr = pd.read_csv(f'{config.ig_dir}/{modelc.name}.csv')
    human_attr = pd.read_csv(f'{config.ig_dir}/instances.csv')
    
    #interpret.compare(modelc, indices, model_attr, human_attr)
    
    del modelc