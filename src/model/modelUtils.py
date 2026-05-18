from .. import config
from torch.cuda import is_available
from transformers import GemmaForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, AutoConfig

def getTrainingArgs(modelc, epochs, batch_size=32, lr=3e-05):
        steps_per_epoch = round(len(modelc.train_tokenized) / batch_size)
        print(len(modelc.train_tokenized), len(modelc.train_tokenized), steps_per_epoch)
        
        training_args = TrainingArguments(
            output_dir='fine-tuned-models',
            eval_strategy='no',
            save_strategy ='best',
            per_device_train_batch_size = batch_size,
            per_device_eval_batch_size = batch_size,
            fp16=is_available(),
            logging_steps=20,
            report_to="none",
            learning_rate=lr,
            num_train_epochs = epochs,
            load_best_model_at_end = True
        )
        
        return training_args

def getGemma(modelc):
    tokenizer = AutoTokenizer.from_pretrained(modelc.link)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = GemmaForSequenceClassification.from_pretrained(
        modelc.link,
        num_labels=len(config.model_classes),
        problem_type="multi_label_classification"
    )
    
    model.config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer #achar config

def getBERTimbau(modelc):
    model = AutoModelForSequenceClassification.from_pretrained(modelc.link, problem_type="multi_label_classification")
    tokenizer = AutoTokenizer.from_pretrained(modelc.link)
    config = AutoConfig.from_pretrained(modelc.link)
    
    return model, tokenizer, config