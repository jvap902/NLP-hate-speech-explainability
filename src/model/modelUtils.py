import re
import torch
from ..config import model_classes
from transformers import GemmaForSequenceClassification, AutoModel, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, AutoConfig

def getTrainingArgs(modelc, epochs, batch_size=32, lr=3e-05):
        steps_per_epoch = round(len(modelc.train_tokenized) / batch_size)
        print(len(modelc.train_tokenized), len(modelc.train_tokenized), steps_per_epoch)
        
        supports_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        
        acc_steps=1
        
        if 'albertina' in modelc.name.lower():
            batch_size = 4
            acc_steps=8
        
        training_args = TrainingArguments(
            output_dir='fine-tuned-models',
            eval_strategy='no',
            save_strategy ='no',
            per_device_train_batch_size = batch_size,
            per_device_eval_batch_size = batch_size,
            gradient_accumulation_steps=acc_steps,
            gradient_checkpointing=True if 'albertina' in modelc.name.lower() else False,
            fp16=torch.cuda.is_available() and not supports_bf16,  # ✅ só usa fp16 se bf16 não estiver disponível
            bf16=supports_bf16,
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
        num_labels=len(model_classes),
        problem_type="multi_label_classification"
    )
    
    model.config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer #achar config

def getBERTimbau(modelc):
    model = AutoModelForSequenceClassification.from_pretrained(modelc.link, num_labels=len(model_classes), problem_type="multi_label_classification")
    model.config = AutoConfig.from_pretrained(modelc.link, problem_type="multi_label_classification")
    tokenizer = AutoTokenizer.from_pretrained(modelc.link)
    
    return model, tokenizer

def getGeneric(modelc):
    model = AutoModelForSequenceClassification.from_pretrained(modelc.link, num_labels=len(model_classes), problem_type="multi_label_classification")
    model.config.problem_type = "multi_label_classification"
    
    tokenizer = AutoTokenizer.from_pretrained(modelc.link)
    
    return model, tokenizer

def getBernice(modelc):
    model = AutoModelForSequenceClassification.from_pretrained(modelc.link, num_labels=len(model_classes),ignore_mismatched_sizes=True)
    model.config.problem_type = "multi_label_classification"

    tokenizer = AutoTokenizer.from_pretrained(modelc.link, model_max_length=128)

    return model, tokenizer

BERNICE_URL_RE    = re.compile(r"https?:\/\/[\w\.\/\?\=\d&#%_:/-]+")
BERNICE_HANDLE_RE = re.compile(r"@\w+")

def bernicePreprocess(text):
    text = BERNICE_HANDLE_RE.sub("@USER", text)
    text = BERNICE_URL_RE.sub("HTTPURL", text)
    return text