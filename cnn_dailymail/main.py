import torch
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset
import evaluate
from torchmetrics.text.bert import BERTScore
from evaluate import load
from torchmetrics.text.rouge import ROUGEScore
import numpy as np
import wandb
import os
import gc
from icecream import ic


wandb.require("core")

# Parameters for the rest of the script
optimizer_name = "adamw"
model_name = "google-t5/t5-small"
dataset =   "cnn_dailymail"
seed_num = 1
max_length = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb_run_name = f"{optimizer_name}-{dataset}-{model_name.split('-')[1].split('/')[0]}"
output_dir = f"{optimizer_name}/{dataset}/{model_name.split('-')[1].split('/')[0]}"
hyper_param_output_name = "hyperparameter_lr_only"  # Where/How to save the hyperparameters
train_range = 150  # Number of training examples to use
test_range = 150 # Number of test+val examples to use combined
val_range = 150  # Number of validation examples to use
epochs = 1
eval_steps = 10
logging_steps = 10
n_trails = 2


# Main
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Dataset
loaded_dataset = load_dataset(dataset, '3.0.0').shuffle(seed=seed_num)
train = loaded_dataset['train'] # Train Dataset 80%
temp = loaded_dataset['test'].train_test_split(test_size=0.5, seed=seed_num, shuffle=True)
test = temp['test'] # Test Dataset
val = temp['train'] # Val Dataset

# Load evaluation
rouge = ROUGEScore(use_stemmer=True)
bert_score = BERTScore(device=device)

def clear_cuda_memory():
    gc.collect()
    torch.cuda.empty_cache()

prefix = "summarize: "  # Required so the T5 model knows that we are going to summarize
def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs)
    labels = tokenizer(text_target=examples["highlights"])
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_name)
def safe_select(dataset, range_end):
    available_samples = len(dataset)
    actual_range = min(range_end, available_samples)
    return dataset.select(range(actual_range))

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

# Print the actual sizes of the datasets
print(f"Actual train dataset size: {len(tokenized_dataset_train)}")
print(f"Actual validation dataset size: {len(tokenized_dataset_val)}")
print(f"Actual test dataset size: {len(tokenized_dataset_test)}")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result_rouge = rouge(preds=decoded_preds, target=decoded_labels)
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
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def model_init():
    return AutoModelForSeq2SeqLM.from_pretrained(model_name)

class CustomTrainer(Seq2SeqTrainer):
    def create_optimizer(self):
        self.optimizer = get_optimizer(optimizer_name, self.model, self.args.learning_rate)
        print(f"\nOptimizer: {self.optimizer.__class__.__name__} with name: {optimizer_name} was created.\n")
def main():
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        logging_strategy="steps",
        evaluation_strategy="steps",
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=eval_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
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
        greater_is_better=False
    )

    class CustomTrainer(Seq2SeqTrainer):
        def create_optimizer(self):
            self.optimizer = get_optimizer(optimizer_name, self.model, self.args.learning_rate)
            print(f"\nOptimizer: {self.optimizer.__class__.__name__} with name: {optimizer_name} was created.\n")

    trainer = CustomTrainer(
        model=model_init(),  # We will initialize the model inside the trainer
        args=training_args,
        train_dataset=tokenized_dataset_train,
        eval_dataset=tokenized_dataset_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    try:
        # Train the model
        train_result = trainer.train()
        
        # Get the best model
        best_model_path = trainer.state.best_model_checkpoint
        best_model = AutoModelForSeq2SeqLM.from_pretrained(best_model_path).to(device)
        
        # Evaluate the best model on the test set
        trainer.model = best_model
        test_results = trainer.evaluate(tokenized_dataset_test, metric_key_prefix="test")
        print(f"Test results with best model: {test_results}")
        
        # Log the test results to wandb
        trainer.log_metrics("test", test_results)
        
        # Save metrics and hyperparameters
        with open(f"{output_dir}/results_seed_{seed_num}.txt", "w") as f:
            f.write(f"Seed: {seed_num}\n")
            f.write(f'Training range: {train_range}\n')
            f.write(f'Test range: {test_range}\n')
            f.write(f'Validation range: {val_range}\n')
            f.write(f'Actual train dataset size: {len(tokenized_dataset_train)}\n')
            f.write(f'Actual validation dataset size: {len(tokenized_dataset_val)}\n')
            f.write(f'Actual test dataset size: {len(tokenized_dataset_test)}\n')
            f.write("\nBest hyperparameters:\n")
            f.write("\n".join([f"{param} : {value}" for param, value in hyperparameters.items()]))
            f.write("\n\nTest results:\n")
            f.write("\n".join([f"{metric} : {value}" for metric, value in test_results.items()]))
        
        # Save the best model
        best_model.save_pretrained(output_dir + f"/best_model_seed_{seed_num}")
        tokenizer.save_pretrained(output_dir + f"/best_model_seed_{seed_num}")
        
        return best_model  # Return the best model for further use

    except torch.cuda.OutOfMemoryError as e:
        print(f"Out of memory error: {e}")
        clear_cuda_memory()
        print("Cleared CUDA memory")
        print("Retrying training...")
        raise optuna.TrialPruned()
    except Exception as e:
        print(f"Unknown error: {e}")
        clear_cuda_memory()
        raise optuna.TrialPruned()

if __name__ == "__main__":
    main()
