import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from src import *

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, required=False, default="BERTimbau-base", help="Specify a model name present in config.models")
parser.add_argument("-nf", "--no_fine_tune", required=False, action='store_true', default=False, help="Disable fine-tuning process")
parser.add_argument("-ncv", "--no_cross_validation", required=False, action='store_true', default=False, help="Disable K-Fold Cross Validation process")
parser.add_argument("-nig", "--new_ig", required=False, action='store_true', default=False, help="Force new Integrated Gradients attribution")

args = parser.parse_args()

def attribute(modelc, ig_dataset):
    ig_tokenized = modelc.tokenize(ig_dataset).with_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    ig_loader = DataLoader(ig_tokenized, batch_size=32, shuffle=False, num_workers=4, collate_fn=modelc.data_collator)
    
    ig_predictions = classify.classifyInputs(modelc, ig_loader)
    ig_predictions['model'] = modelc.name
    ig_predictions['text_id'] = indices
    ig_predictions = ig_predictions[["model", "text_id", "text", "labels", "preds"]]
    ig_predictions.to_csv(f'{config.ig_dir}/predictions.csv', mode='a', header=False, index=False)

    print(ig_predictions)
    
    y_true = np.array(ig_predictions['labels'].tolist())
    y_pred = np.array(ig_predictions['preds'].tolist())

    print(f1_score(y_true, y_pred, average='micro'))
    
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
    
    # Load dataset
    train, test = loadDataset.loadTuPyE(-1, -1)


    # Generate folds once from the raw dataset for consistency across all models
    
    n_splits=5
    
    folds_panel = LivePanel("Gathering Folds", "Purple")
    folds_panel.start()
    if Path('folds.npz').is_file():
        folds_panel.addMessage("## Loading previously generated folds")
        loaded = np.load('folds.npz')
        folds = [(loaded[f'train_{i}'], loaded[f'val_{i}']) for i in range(n_splits)]
    else:
        folds_panel.addMessage("## Generating new folds, this might take a while")
        folds = classify.get_fold_indices(train, n_splits=n_splits, random_state=42)
    folds_panel.stop()
    
    # Load model
    modelc = loadModelc()
    
    if not args.no_cross_validation: 
        cv_results = classify.crossValidate(model_or_name=modelc, train_dataset=train, model_type="bert", n_splits=5, epochs_fold=5, random_state=42, folds=folds)
        modelc.reset()
        
        # Run traditional baselines with the same folds
        classify.crossValidate("SVM", train, model_type="svm", folds=folds)
        classify.crossValidate("Baseline", train, model_type="baseline", folds=folds)

    if not args.no_fine_tune:
        modelc = classify.fineTune(modelc, train_dataset=train)
    
        eval_data = classify.testModel(modelc)
        print(eval_data)
        
    ig_dataset, indices = loadDataset.igDataset(test)

    if args.new_ig or not Path(f"{config.ig_dir}/{modelc.name}.csv"): attribute(modelc, ig_dataset)
    
    model_attr = pd.read_csv(f'{config.ig_dir}/{modelc.name}.csv')
    instances = pd.read_csv(f'{config.ig_dir}/instances.csv')
    
    #interpret.compare(modelc, model_attr, instances)
    
    del modelc