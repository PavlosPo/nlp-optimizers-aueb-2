import torch
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset
# import evaluate
from torchmetrics.text.bert import BERTScore
# from evaluate import load
from torchmetrics.text.rouge import ROUGEScore
import numpy as np
import wandb
import os
from icecream import ic
import gc
import optuna
import argparse

wandb.require("core")

os.environ["WANDB_MODE"] = "offline"

# Argument parser for GPU ID and seed number
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, required=True, help="GPU ID to use for training")
parser.add_argument("--seed", type=int, required=True, help="Seed number for reproducibility")
parser.add_argument("--optim", type=str, required=True, help="Optimizer string for training")
args = parser.parse_args()

# Parameters for the rest of the script
optimizer_name = args.optim
model_name = "google-t5/t5-small"
dataset = "cnn_dailymail"
seed_num = args.seed
max_length = 512
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
wandb_run_name = f"{optimizer_name}-{dataset}-{model_name.split('-')[1].split('/')[0]}"
output_dir = f"{optimizer_name}/{dataset}/{model_name.split('-')[1].split('/')[0]}"
hyper_param_output_name = "hyperparameter_lr_only"  # Where/How to save the hyperparameters
train_range = 15000  # Number of training examples to use
test_range = 1500  # Number of test+val examples to use combined
val_range = 1500  # Number of validation examples to use
epochs = 4
eval_steps = 1000
logging_steps = 1000
n_trials = 30


# Function to load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to load the dataset
def load_datasets(seed_num):
    loaded_dataset = load_dataset(dataset, '3.0.0').shuffle(seed=seed_num)
    train = loaded_dataset['train']  # Train Dataset 80%
    temp = loaded_dataset['test'].train_test_split(test_size=0.5, seed=seed_num, shuffle=True)
    test = temp['test']  # Test Dataset
    val = temp['train']  # Val Dataset
    return train, val, test

# Load evaluation metrics
rouge_score = ROUGEScore(use_stemmer=True)
bert_score = BERTScore()

def clear_cuda_memory():
    gc.collect()
    torch.cuda.empty_cache()

prefix = "summarize: "  # Required so the T5 model knows that we are going to summarize
def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, padding=True, truncation=True, max_length=max_length)
    labels = tokenizer(text_target=examples["highlights"], padding=True, truncation=True, max_length=max_length)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_name)

def safe_select(dataset, range_end):
    available_samples = len(dataset)
    actual_range = min(range_end, available_samples)
    return dataset.select(range(actual_range))

def prepare_datasets(train, val, test):
    tokenized_dataset_train = safe_select(
        train.map(preprocess_function, batched=True).filter(lambda x: len(x['input_ids']) <= max_length),
        train_range
    )
    
    tokenized_dataset_val = safe_select(
        val.map(preprocess_function, batched=True).filter(lambda x: len(x['input_ids']) <= max_length),
        val_range
    )
    
    tokenized_dataset_test = safe_select(
        test.map(preprocess_function, batched=True).filter(lambda x: len(x['input_ids']) <= max_length),
        test_range
    )
    
    return tokenized_dataset_train, tokenized_dataset_val, tokenized_dataset_test

# Print the actual sizes of the datasets
def print_dataset_sizes(train, val, test):
    print(f"Actual train dataset size: {len(train)}")
    print(f"Actual validation dataset size: {len(val)}")
    print(f"Actual test dataset size: {len(test)}")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result_rouge = rouge_score(preds=decoded_preds, target=decoded_labels)
    result_brt = bert_score(preds=decoded_preds, target=decoded_labels)
    result_brt_average_values = {key: tensors.mean().item() for key, tensors in result_brt.items()}
    results = {**result_rouge, **result_brt_average_values}
    return results

def get_optimizer(optimizer_name, model, learning_rate):
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "sgdm":
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9) # Default 0.9 momentum
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def model_init():
    return AutoModelForSeq2SeqLM.from_pretrained(model_name)

class CustomTrainer(Seq2SeqTrainer):
    def create_optimizer(self):
        self.optimizer = get_optimizer(optimizer_name, self.model, self.args.learning_rate)
        print(f"\nOptimizer: {self.optimizer.__class__.__name__} with name: {optimizer_name} was created.\n")
        return self.optimizer

def optuna_hp_space(trial):
    search_space = (1e-7, 1e-3) if optimizer_name.startswith("sgd") else (1e-7, 1e-5)
    print(f"The search space for {optimizer_name} is {search_space}")
    
    return {
        "learning_rate": trial.suggest_float("learning_rate", search_space[0], search_space[1], log=True),
    }

def main(seed_num):
    train, val, test = load_datasets(seed_num)
    tokenized_dataset_train, tokenized_dataset_val, tokenized_dataset_test = prepare_datasets(train, val, test)
    print_dataset_sizes(tokenized_dataset_train, tokenized_dataset_val, tokenized_dataset_test)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        logging_strategy="steps",
        evaluation_strategy="steps",
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        save_total_limit=1,
        num_train_epochs=epochs,
        predict_with_generate=True,
        seed=seed_num,
        data_seed=seed_num,
        fp16=True,
        push_to_hub=False,
        report_to="wandb",
        run_name=wandb_run_name,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        save_strategy="steps",
        save_steps=eval_steps,  # Save checkpoint at each evaluation
    )

    trainer = CustomTrainer(
        model=None,
        args=training_args,
        train_dataset=tokenized_dataset_train,
        eval_dataset=tokenized_dataset_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        model_init=model_init,
    )

    best_run = trainer.hyperparameter_search(
        hp_space=optuna_hp_space,
        direction="minimize",
        backend="optuna",
        n_trials=n_trials,
        compute_objective=lambda metrics: metrics["eval_loss"],
    )
        
    # Save metrics and hyperparameters
    with open(f"{output_dir}/{hyper_param_output_name}_seed_{seed_num}.txt", "w") as f:
        f.write(f"Seed: {seed_num}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Optimizer: {optimizer_name}\n")
        f.write(f'Training range: {train_range}\n')
        f.write(f'Test range: {test_range}\n')
        f.write(f'Validation range: {val_range}\n')
        f.write(f'Actual train dataset size: {len(tokenized_dataset_train)}\n')
        f.write(f'Actual validation dataset size: {len(tokenized_dataset_val)}\n')
        f.write(f'Actual test dataset size: {len(tokenized_dataset_test)}\n')
        f.write("\nBest hyperparameters:\n")
        f.write("\n".join([f"{param} : {value}" for param, value in best_run.hyperparameters.items()]))

if __name__ == "__main__":
    print(f"Running with seed: {seed_num}")
    main(seed_num)
