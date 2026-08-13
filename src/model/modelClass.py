from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from .modelUtils import *
from .. import config
from .. import fileHandler
from ..logging import LivePanel

class Model():
    def __init__(self, model_link, model_name):
        self.name = model_name
        self.link = model_link
        
        self.save_dir = f"{config.model_save_dir}/{self.name}"
        
        self.panel = LivePanel("Loading Model", color="green")
        self.panel.start()
        
        if Path(self.save_dir).is_dir():
            self.loadModel()
        else:
            self.newModel()
        
        self.panel.addMessage("### Model loaded")
        self.panel.stop()
        
        self.model.to(config.device)
        
    def getLoader(self, train, test, batch_size):
        self.train_tokenized = self.tokenize(train)
        self.test_tokenized = self.tokenize(test)
        
        self.data_collator = DataCollatorWithPadding(self.tokenizer)
        
        torch_tokenized_train = self.train_tokenized.with_format(type="torch", columns=["input_ids", "attention_mask", "labels"]) #coloca no formato do PyTorch
        torch_tokenized_test = self.test_tokenized.with_format(type="torch", columns=["input_ids", "attention_mask", "labels"]) #coloca no formato do PyTorch
        
        self.train_loader = DataLoader(torch_tokenized_train, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=self.data_collator)
        self.test_loader = DataLoader(torch_tokenized_test, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=self.data_collator)
        
    def newModel(self):
        
        self.panel.addMessage(f"\n ## Creating new {self.name} \n")
        
        if 'gemma' in self.name:
            self.model, self.tokenizer = getGemma(self)
        elif 'bertimbau' in self.name.lower():
            self.model, self.tokenizer = getBERTimbau(self)
        elif 'bernice' in self.name.lower():
            self.model, self.tokenizer = getBernice(self)
        else:
            self.model, self.tokenizer = getGeneric(self)
        
    def loadModel(self):
        
        self.panel.addMessage(f"\n ## Loading saved model at {self.save_dir} \n")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.save_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.save_dir,
            num_labels=len(config.model_classes),
            problem_type="multi_label_classification"
        )
        
    def saveModel(self, trained_epochs=0):
        model_data = fileHandler.getJsonInfo(config.fine_tune_info_path, [self.name])[0]
        
        if "trained_epochs" in model_data:
            model_data["trained_epochs"] += trained_epochs
        else:
            model_data["trained_epochs"] = trained_epochs
        
        fileHandler.updateJson(json_path=config.fine_tune_info_path, fields=[self.name], values=[model_data])
        self.model.save_pretrained(self.save_dir)
        self.tokenizer.save_pretrained(self.save_dir)
        
    def tokenize(self, dataset):
        
        tokenizer = self.tokenizer
        is_bernice = 'bernice' in self.name.lower()
        max_length = min(tokenizer.model_max_length, 512)
        
        def tokenizeInstance(instance):
            
            texts = instance["text"]
            
            if is_bernice:
                texts = [bernicePreprocess(t) for t in texts]
            
            tokens = tokenizer(texts, truncation=True, max_length=max_length)
            
            labels = []
            batch_size=len(instance["text"])
            
            for i in range(batch_size):
                row = [np.float32(instance[c][i]) for c in config.dataset_classes]

                # not_hate = 1 se hate == 0, reproduzindo o que os autores fizeram
                not_hate = 1.0 - np.float32(instance["hate"][i])
                row.append(not_hate)

                labels.append(row)

            tokens["labels"] = labels
            
            return tokens
        
        cols_to_remove = ["text"]
        ds = dataset.map(tokenizeInstance, batched=True, remove_columns=cols_to_remove)
        return ds
    
    def getTrainer(self, epochs_fold, compute_metrics):
        self._epochs_fold = epochs_fold
        self._compute_metrics = compute_metrics
        self._createTrainer()
    
    def _createTrainer(self):
        """Create (or recreate) the Trainer with the current model."""
        training_args = getTrainingArgs(self, self._epochs_fold)
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_tokenized,
            eval_dataset=self.test_tokenized,
            compute_metrics=self._compute_metrics,
            data_collator=self.data_collator
        )
    
    def reset(self):
        """
        Reset model to pretrained weights (from original checkpoint, not fine-tuned).
        Recreates the Trainer with fresh optimizer/scheduler state.
        This prevents data leakage between cross-validation folds.
        """
        # Start panel briefly for newModel's messages
        self.panel = LivePanel(f"Resetting {self.name}", color="yellow")
        self.panel.start()
        
        # Reload model from original pretrained checkpoint (self.link)
        self.newModel()
        self.model.to(config.device)
        
        self.panel.addMessage("### Model reset complete")
        self.panel.stop()
        
        # Recreate trainer with fresh model
        self._createTrainer()