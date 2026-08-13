import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    classification_report
)
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
import pandas as pd
from . import config
from .model.modelClass import Model
from .fileHandler import getJsonInfo, updateJson, writeCsvLine, createFile

try:
    from google.colab import files
    IN_COLAB = True
except (ImportError, ModuleNotFoundError):
    IN_COLAB = False
    files = None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred):
    """HuggingFace Trainer-compatible compute_metrics for multi-label."""
    logits = torch.Tensor(pred.predictions)
    labels = pred.label_ids.astype(int)

    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= 0.5).astype(int)

    f1_mi = f1_score(labels, preds, average='micro')
    f1_ma = f1_score(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1-macro': f1_ma,
        'f1-micro': f1_mi
    }


def compute_full_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list) -> dict:
    """
    Compute comprehensive metrics: overall and per-class.
    
    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples, n_classes)
    y_pred : np.ndarray of shape (n_samples, n_classes)
    class_names : list of str

    Returns
    -------
    dict with keys:
        - overall: {accuracy, f1_micro, f1_macro, precision_micro, precision_macro, recall_micro, recall_macro}
        - per_class: {class_name: {f1, precision, recall, support}} for each class
    """
    metrics = {}

    # Overall metrics
    metrics["overall"] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }

    # Per-class metrics
    per_class = {}
    for i, cls_name in enumerate(class_names):
        per_class[cls_name] = {
            "f1": f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
            "precision": precision_score(y_true[:, i], y_pred[:, i], zero_division=0),
            "recall": recall_score(y_true[:, i], y_pred[:, i], zero_division=0),
            "support": int(y_true[:, i].sum()),
        }
    metrics["per_class"] = per_class

    return metrics


# ---------------------------------------------------------------------------
# Stratification helper for multilabel
# ---------------------------------------------------------------------------

def _get_stratification_labels(dataset) -> np.ndarray:
    """
    Convert multi-label vectors into a single integer label for StratifiedKFold.
    Uses the label-combination approach: each unique combination of active labels
    gets a unique integer ID. This preserves the distribution of label combos
    across folds.
    
    For rare combinations that appear fewer times than n_splits, they are merged
    into a special "rare" bucket to avoid StratifiedKFold errors.
    """
    # Build the label matrix from the dataset columns
    label_matrix = np.array(
        [[int(dataset[col][i]) for col in config.dataset_classes] for i in range(len(dataset))]
    )
    # Convert each row to a string hash representing the label combination
    combo_strings = ["_".join(map(str, row)) for row in label_matrix]

    # Count occurrences of each combo
    from collections import Counter
    combo_counts = Counter(combo_strings)
    
    # Merge rare combos (fewer than min_samples_per_class for SKF) into one bucket
    # StratifiedKFold needs at least n_splits samples per class
    min_count = 5  # safe minimum for 5-fold
    rare_label = "__RARE__"
    combo_strings_safe = [
        c if combo_counts[c] >= min_count else rare_label for c in combo_strings
    ]

    # Map unique combos to integers
    unique_combos = sorted(set(combo_strings_safe))
    combo_to_id = {c: idx for idx, c in enumerate(unique_combos)}
    strat_labels = np.array([combo_to_id[c] for c in combo_strings_safe])

    return strat_labels


def get_fold_indices(dataset, n_splits=5, random_state=42) -> list:
    """
    Generate consistent StratifiedKFold indices from a HuggingFace dataset.
    Uses label-combination hashing for stratification.
    
    Returns a list of (train_indices, val_indices) tuples.
    Can be reused across different model types to ensure identical folds.
    """
    strat_labels = _get_stratification_labels(dataset)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    indices = np.arange(len(dataset))
    folds = list(skf.split(indices, strat_labels))

    return folds


def get_fold_indices_iterative(dataset, n_splits=5, random_state=42) -> list:
    """
    Generate consistent fold indices using Iterative Stratification for multilabel data.
    
    Unlike the label-combination approach, this considers each label independently
    and ensures that each fold has approximately the same proportion of positive
    samples for every label. It handles rare labels better by prioritizing them
    during the assignment process.
    
    Uses skmultilearn.model_selection.IterativeStratification.
    
    Parameters
    ----------
    dataset : HuggingFace Dataset
        Must contain columns from config.dataset_classes.
    n_splits : int
        Number of folds.
    random_state : int
        Random seed for reproducibility.

    Returns a list of (train_indices, val_indices) tuples.
    Can be reused across different model types to ensure identical folds.
    """
    from skmultilearn.model_selection import IterativeStratification
    
    # Build label matrix from dataset columns
    label_matrix = np.array(
        [[int(dataset[col][i]) for col in config.dataset_classes] for i in range(len(dataset))]
    )
    
    # IterativeStratification does not have a random_state param, but the order
    # of input determines the output. We shuffle indices with a fixed seed first,
    # then map back to original indices after splitting.
    rng = np.random.default_rng(random_state)
    n_samples = len(dataset)
    shuffled_order = rng.permutation(n_samples)
    label_matrix_shuffled = label_matrix[shuffled_order]
    
    stratifier = IterativeStratification(
        n_splits=n_splits,
        order=2  # considers label pairs for better distribution
    )
    
    folds = []
    for train_idx_local, val_idx_local in stratifier.split(
        X=np.zeros((n_samples, 1)),  # X is unused by the stratifier, just needs shape
        y=label_matrix_shuffled
    ):
        # Map back from shuffled indices to original dataset indices
        train_indices = shuffled_order[train_idx_local]
        val_indices = shuffled_order[val_idx_local]
        folds.append((train_indices, val_indices))
    
    return folds


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _extract_texts_and_labels(dataset) -> tuple:
    """Extract raw texts and multi-label matrix from a HuggingFace Dataset."""
    texts = dataset["text"]
    
    # Build label matrix including not_hate
    label_matrix = []
    for i in range(len(dataset)):
        row = [int(dataset[col][i]) for col in config.dataset_classes]
        not_hate = 1 - int(dataset["hate"][i])
        row.append(not_hate)
        label_matrix.append(row)
    
    return texts, np.array(label_matrix)


# ---------------------------------------------------------------------------
# Model-specific train/predict
# ---------------------------------------------------------------------------

def _train_predict_bert(modelc: Model, train_indices, val_indices, labels_val, epochs_fold=5):
    """Train a BERT model on train split and predict on val split.
    
    Resets the model to pretrained weights before training to prevent
    data leakage between folds.
    
    Parameters
    ----------
    modelc : Model
        The BERT model instance.
    train_indices : np.ndarray
        Indices for training split.
    val_indices : np.ndarray
        Indices for validation split.
    labels_val : np.ndarray
        Ground truth labels for validation (from shared _extract_texts_and_labels).
    epochs_fold : int
        Number of training epochs.
    """
    # Reset to pretrained weights — prevents leakage from previous folds
    modelc.reset()
    
    train_split = modelc.train_tokenized.select(train_indices.tolist())
    val_split = modelc.train_tokenized.select(val_indices.tolist())

    modelc.trainer.train_dataset = train_split
    modelc.trainer.train()

    # Predict on validation
    predictions = modelc.trainer.predict(val_split)
    logits = torch.Tensor(predictions.predictions)
    probs = torch.sigmoid(logits).cpu().numpy()
    y_pred = (probs >= 0.5).astype(int)
    y_true = labels_val

    return y_true, y_pred


def _train_predict_svm(texts_train, labels_train, texts_val, labels_val):
    """Train an SVM (OneVsRest with LinearSVC) using TF-IDF features."""
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(texts_train)
    X_val = vectorizer.transform(texts_val)

    clf = OneVsRestClassifier(LinearSVC(max_iter=10000, random_state=42))
    clf.fit(X_train, labels_train)

    y_pred = clf.predict(X_val)
    y_true = labels_val

    return y_true, y_pred


def _train_predict_baseline(texts_train, labels_train, texts_val, labels_val):
    """Baseline: predicts the most frequent label combination (stratified dummy)."""
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train = vectorizer.fit_transform(texts_train)
    X_val = vectorizer.transform(texts_val)

    clf = OneVsRestClassifier(DummyClassifier(strategy="most_frequent", random_state=42))
    clf.fit(X_train, labels_train)

    y_pred = clf.predict(X_val)
    y_true = labels_val

    return y_true, y_pred


# ---------------------------------------------------------------------------
# Main cross-validation function
# ---------------------------------------------------------------------------

def crossValidate(
    model_or_name,
    train_dataset,
    model_type: str = "bert",
    n_splits: int = 5,
    epochs_fold: int = 5,
    random_state: int = 42,
    folds: list = None,
):
    """
    Perform Stratified K-Fold Cross-Validation for any model type.

    Parameters
    ----------
    model_or_name : Model or str
        - For model_type="bert": a Model instance (already loaded with tokenized data and trainer).
        - For model_type="svm" or "baseline": a string name for CSV output.
    train_dataset : HuggingFace Dataset
        The raw (non-tokenized) training dataset with 'text' and class columns.
        Used for stratification and for traditional models (SVM/baseline).
        For BERT models, the actual training uses model_or_name.train_tokenized.
    model_type : str
        One of "bert", "svm", "baseline".
    n_splits : int
        Number of folds.
    epochs_fold : int
        Epochs per fold (only used for BERT models).
    random_state : int
        Random seed for fold reproducibility.
    folds : list of (train_indices, val_indices), optional
        Pre-computed fold indices to ensure consistency across models.
        If None, folds are generated from train_dataset.

    Returns
    -------
    dict with:
        - fold_metrics: list of per-fold metric dicts
        - average_metrics: averaged overall metrics across folds
        - per_class_metrics: averaged per-class metrics across folds
        - folds: the list of (train_idx, val_idx) used (for reuse with other models)
    """
    # Determine model name for output
    if model_type == "bert":
        model_name = model_or_name.name
    else:
        model_name = model_or_name if isinstance(model_or_name, str) else str(model_or_name)

    # Generate folds if not provided
    if folds is None:
        folds = get_fold_indices(train_dataset, n_splits=n_splits, random_state=random_state)

    # Extract raw texts and labels once — used by all model types for consistency
    all_texts, all_labels = _extract_texts_and_labels(train_dataset)

    # Setup CSV output
    csv_path = f"{config.losses_dir}/{model_name}_cv.csv"
    header = "fold,accuracy,f1_micro,f1_macro,precision_micro,precision_macro,recall_micro,recall_macro"
    createFile(csv_path, header + "\n")

    fold_metrics_list = []
    class_names = config.model_classes  # 12 classes including not_hate

    print(f"\n{'='*60}")
    print(f"Cross-Validation: {model_name} ({model_type}) | {n_splits} folds")
    print(f"{'='*60}")

    for fold_idx, (train_indices, val_indices) in enumerate(folds):
        print(f"\n--- Fold {fold_idx + 1}/{n_splits} ---")

        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)

        if model_type == "bert":
            labels_val = all_labels[val_indices]
            y_true, y_pred = _train_predict_bert(
                model_or_name, train_indices, val_indices, labels_val, epochs_fold
            )
        elif model_type == "svm":
            texts_train = [all_texts[i] for i in train_indices]
            texts_val = [all_texts[i] for i in val_indices]
            labels_train = all_labels[train_indices]
            labels_val = all_labels[val_indices]
            y_true, y_pred = _train_predict_svm(texts_train, labels_train, texts_val, labels_val)
        elif model_type == "baseline":
            texts_train = [all_texts[i] for i in train_indices]
            texts_val = [all_texts[i] for i in val_indices]
            labels_train = all_labels[train_indices]
            labels_val = all_labels[val_indices]
            y_true, y_pred = _train_predict_baseline(texts_train, labels_train, texts_val, labels_val)
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'bert', 'svm', or 'baseline'.")

        # Compute full metrics for this fold
        fold_metrics = compute_full_metrics(y_true, y_pred, class_names)
        fold_metrics_list.append(fold_metrics)

        # Write fold summary to CSV
        o = fold_metrics["overall"]
        writeCsvLine(csv_path, [
            fold_idx,
            o["accuracy"], o["f1_micro"], o["f1_macro"],
            o["precision_micro"], o["precision_macro"],
            o["recall_micro"], o["recall_macro"]
        ])

        print(f"  Accuracy: {o['accuracy']:.4f} | F1-micro: {o['f1_micro']:.4f} | F1-macro: {o['f1_macro']:.4f}")

    # Aggregate metrics across folds
    avg_metrics = _average_fold_metrics(fold_metrics_list)

    # Save per-class results to a separate CSV
    _save_per_class_csv(model_name, avg_metrics["per_class"], class_names)

    # Update info.json
    _update_info_json(model_name, avg_metrics)

    print(f"\n{'='*60}")
    print(f"Average Results for {model_name}:")
    print(f"  Accuracy:        {avg_metrics['overall']['accuracy']:.4f}")
    print(f"  F1-micro:        {avg_metrics['overall']['f1_micro']:.4f}")
    print(f"  F1-macro:        {avg_metrics['overall']['f1_macro']:.4f}")
    print(f"  Precision-micro: {avg_metrics['overall']['precision_micro']:.4f}")
    print(f"  Precision-macro: {avg_metrics['overall']['precision_macro']:.4f}")
    print(f"  Recall-micro:    {avg_metrics['overall']['recall_micro']:.4f}")
    print(f"  Recall-macro:    {avg_metrics['overall']['recall_macro']:.4f}")
    print(f"{'='*60}\n")

    return {
        "fold_metrics": fold_metrics_list,
        "average_metrics": avg_metrics,
        "folds": folds,
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _average_fold_metrics(fold_metrics_list: list) -> dict:
    """Average overall and per-class metrics across folds."""
    n_folds = len(fold_metrics_list)

    # Average overall
    overall_keys = fold_metrics_list[0]["overall"].keys()
    avg_overall = {}
    for key in overall_keys:
        avg_overall[key] = np.mean([fm["overall"][key] for fm in fold_metrics_list])

    # Average per-class
    class_names = fold_metrics_list[0]["per_class"].keys()
    metric_keys = ["f1", "precision", "recall", "support"]
    avg_per_class = {}
    for cls in class_names:
        avg_per_class[cls] = {}
        for mk in metric_keys:
            avg_per_class[cls][mk] = np.mean([fm["per_class"][cls][mk] for fm in fold_metrics_list])

    return {"overall": avg_overall, "per_class": avg_per_class}


def _save_per_class_csv(model_name: str, per_class: dict, class_names: list):
    """Save per-class metrics to a CSV file."""
    csv_path = f"{config.losses_dir}/{model_name}_per_class.csv"
    
    # Always overwrite this file with fresh results
    header = "class,f1,precision,recall,support\n"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(header)
        for cls in class_names:
            m = per_class[cls]
            f.write(f"{cls},{m['f1']:.6f},{m['precision']:.6f},{m['recall']:.6f},{m['support']:.1f}\n")
    
    print(f"Per-class metrics saved to: {csv_path}")


def _update_info_json(model_name: str, avg_metrics: dict):
    """Update fine-tune info JSON with cross-validation results."""
    try:
        model_data = getJsonInfo(config.fine_tune_info_path, [model_name])[0]
    except Exception:
        model_data = {}

    model_data["cv_results"] = {
        "overall": {k: float(v) for k, v in avg_metrics["overall"].items()},
        "per_class": {
            cls: {mk: float(mv) for mk, mv in metrics.items()}
            for cls, metrics in avg_metrics["per_class"].items()
        }
    }

    updateJson(json_path=config.fine_tune_info_path, fields=[model_name], values=[model_data])


# ---------------------------------------------------------------------------
# Legacy-compatible wrapper (keeps the old interface working)
# ---------------------------------------------------------------------------

def fineTune(modelc: Model, repeat: int, train_dataset=None, folds=None):
    """
    Legacy wrapper: runs cross-validation using StratifiedKFold.
    Kept for backward compatibility with main.py's tune() function.
    
    Parameters
    ----------
    modelc : Model
        The BERT model instance (with train_tokenized and trainer set up).
    repeat : int
        Not used anymore (kept for API compat). Folds replace repeats.
    train_dataset : HuggingFace Dataset, optional
        The raw dataset (with class columns) for stratification.
        If None, uses modelc.train_tokenized (works only if class columns exist).
    folds : list, optional
        Pre-computed fold indices for consistency across models.
    """
    dataset_for_strat = train_dataset if train_dataset is not None else modelc.train_tokenized
    
    results = crossValidate(
        model_or_name=modelc,
        train_dataset=dataset_for_strat,
        model_type="bert",
        n_splits=5,
        epochs_fold=5,
        random_state=42,
        folds=folds,
    )
    return modelc, results


# ---------------------------------------------------------------------------
# Testing / evaluation (unchanged)
# ---------------------------------------------------------------------------

def testModel(modelc: Model):
    modelc.trainer.eval_dataset = modelc.test_tokenized
    eval_data = modelc.trainer.evaluate()

    model_data = getJsonInfo(config.fine_tune_info_path, [modelc.name])[0]
    model_data["eval"] = eval_data
    updateJson(json_path=config.fine_tune_info_path, fields=[modelc.name], values=[model_data])

    return eval_data


def classifyInputs(modelc, loader):
    all_texts = []
    all_preds = []
    all_labels = []

    modelc.model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="classifying"):
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)

            logits = modelc.model(input_ids, attention_mask).logits
            probs = torch.sigmoid(logits)
            binary_preds = (probs > 0.5).int()
            all_preds.extend(binary_preds.cpu().tolist())

            labels = batch['labels'].cpu().numpy().astype(int)
            all_labels.extend(labels.tolist())

            if 'text' in batch:
                texts = batch['text']
            else:
                texts = modelc.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            all_texts.extend(texts)

    df = pd.DataFrame({
        'text': all_texts,
        'preds': all_preds,
        'labels': all_labels
    })
    print(df)

    return df
