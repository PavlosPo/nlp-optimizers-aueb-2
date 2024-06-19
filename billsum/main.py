import torch
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset
import evaluate
from torchmetrics.text.bert import BERTScore
from torchmetrics.text.rouge import ROUGEScore
import numpy as np

# Load the T5 model and tokenizer
checkpoint = "google-t5/t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

# Dataset
billsum = load_dataset("billsum", split="ca_test")
billsum = billsum.train_test_split(test_size=0.2)
train = billsum['train'] # Train Dataset 80%
temp = billsum['test'].train_test_split(test_size=0.5)  # Ignore
test = temp['test'] # Test Dataset
val = temp['train'] # Val Dataset

# Load evaluation
# rouge = evaluate.load("rouge")
rouge = ROUGEScore(use_stemmer=True)

# Example text to summarize
text = train[0]['text']
summary = train[0]['summary']
print(f"\nExample Text: \n{text}\nExample Summary: \n{summary}\n")

print("Preprocessing the Dataset Phase starts...\n")
prefix = "summarize: "  # Required so the T5 model knows that we are going to summarize
def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)
tokenized_billsum = billsum.map(preprocess_function, batched=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    # prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    # result["gen_len"] = np.mean(prediction_lens)

    # Torchmetrics
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
    output_dir="my_awesome_billsum_model",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=4,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False
)

class CustomTrainer(Seq2SeqTrainer):
    def create_optimizer(self):
        optimizer_name = "adamw"  # Dynamically set the optimizer name here
        self.optimizer = get_optimizer(optimizer_name, self.model, self.args.learning_rate)
        print(f"Optimizer: {self.optimizer.__class__.__name__} with name: {optimizer_name} is created.")

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_billsum["train"],
    eval_dataset=tokenized_billsum["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Example usage
if __name__ == "__main__":
    trainer.train()