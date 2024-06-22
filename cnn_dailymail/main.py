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
train_range = 15000  # Number of training examples to use
test_range = 1500  # Number of test+val examples to use combined
val_range = 1500  # Number of validation examples to use
epochs = 4
eval_steps = 500
logging_steps = 500

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
tokenized_dataset_train = train.map(
    preprocess_function, batched=True).filter(
        lambda x: len(x['input_ids']) <= max_length).select(
            range(0, train_range))
tokenized_dataset_val = val.map(
    preprocess_function, batched=True).filter(
        lambda x: len(x['input_ids']) <= max_length).select(
            range(0, val_range))
tokenized_dataset_test = test.map(
    preprocess_function, batched=True).filter(
        lambda x: len(x['input_ids']) <= max_length).select(
            range(0, test_range))

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

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    logging_strategy="steps",
    eval_strategy="steps",
    logging_steps = 500,
    eval_steps = 500,
    save_steps=500,
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    # save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    seed=seed_num,
    data_seed=seed_num,
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
    train_dataset=tokenized_dataset_train,
    eval_dataset=tokenized_dataset_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Test the model and save results in file and wandb
def test_model():
    results = trainer.predict(tokenized_dataset_test)
    metrics = results[2]
    
    # Optionally, save metrics to a file as well
    with open(f"{output_dir}/train_results.txt", "w") as f:
        f.write("\n".join([f"{metric} : {value}" for metric, value in metrics.items()]))

    trainer.log_metrics("test", metrics)    # upload to wandb

# Example usage
if __name__ == "__main__":
    trainer.train()
    test_model()
    trainer.save_model(output_dir + "/saved_trained_model/")
    