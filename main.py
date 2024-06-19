import torch
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset
import evaluate
from torchmetrics.text.bert import BERTScore
from torchmetrics.text.rouge import ROUGEScore
import numpy as np
import wandb
import os
wandb.require("core")

# Parameters
optimizer_name = "adam"
model_name = "google-t5/t5-small"
dataset = "billsum"

wandb_run_name = f"{optimizer_name}-{dataset}-{model_name.split('-')[1].split('/')[0]}"
output_dir = f"{optimizer_name}/{dataset}/{model_name.split('-')[1].split('/')[0]}"

# Load the T5 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Dataset
loaded_dataset = load_dataset(dataset, split="ca_test")
loaded_dataset = loaded_dataset.train_test_split(test_size=0.2)
# train = loaded_dataset['train'] # Train Dataset 80%
# temp = loaded_dataset['test'].train_test_split(test_size=0.5)  # Ignore
# test = temp['test'] # Test Dataset
# val = temp['train'] # Val Dataset

# Load evaluation
rouge = ROUGEScore(use_stemmer=True)

prefix = "summarize: "  # Required so the T5 model knows that we are going to summarize
def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_name)
tokenized_dataset = loaded_dataset.map(preprocess_function, batched=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge(preds=decoded_preds, target=decoded_labels)
    return {k: torch.round(v, decimals=4) for k, v in result.items()}

def get_optimizer(optimizer_name, model, learning_rate):
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    eval_strategy="steps",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=4,
    logging_steps = 500,
    eval_steps =500,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False,
    report_to="wandb",
    run_name=wandb_run_name,
    load_best_model_at_end = True,
    metric_for_best_model = 'loss',
)

class CustomTrainer(Seq2SeqTrainer):
    def create_optimizer(self):
        self.optimizer = get_optimizer(optimizer_name, self.model, self.args.learning_rate)
        print(f"\nOptimizer: {self.optimizer.__class__.__name__} with name: {optimizer_name} was created.\n")

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Example usage
if __name__ == "__main__":
    trainer.train()