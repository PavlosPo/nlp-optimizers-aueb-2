import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM
from datasets import load_dataset
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text.bert import BERTScore
import os
import argparse
from torch.utils.data import DataLoader
from icecream import ic

os.environ["TOKENIZERS_PARALLELISM"] = 'false'

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True, help="Seed number for reproducibility")
parser.add_argument("--optim", type=str, required=True, help="Optimizer to use for training")
args = parser.parse_args()

# Parameters
optimizer_name = args.optim
model_name = "google/t5-small"
dataset_name = "cnn_dailymail"
seed_num = args.seed
max_length = 512
output_dir = f"{optimizer_name}/{dataset_name}/best_{model_name.split('/')[-1]}"
train_range = 15000
test_range = 1500
val_range = 1500
epochs = 4
learning_rate = 9.9879589111261e-06

class T5SummarizationModule(pl.LightningModule):
    def __init__(self, model_name, learning_rate, optimizer_name="adamw"):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.rouge_score = ROUGEScore(use_stemmer=True)
        self.bert_score = BERTScore(model_name_or_path='roberta-large')
        self.valid_predictions = []
        self.valid_labels = []
        self.test_predictions = []
        self.test_labels = []

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log("train_loss", outputs.loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log("val_loss", outputs.loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return outputs.loss

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log("test_loss", outputs.loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return outputs.loss

    def configure_optimizers(self):
        if self.optimizer_name == "adamw":
            return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == "sgd":
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

def preprocess_function(examples, tokenizer, max_length):
    prefix = "summarize: "
    inputs = [prefix + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=max_length)
    labels = tokenizer(text_target=examples["highlights"], padding="max_length", truncation=True, max_length=max_length)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def load_and_prepare_data(tokenizer, max_length):
    dataset = load_dataset(dataset_name, '3.0.0').shuffle(seed=seed_num)
    train = dataset['train']
    temp = dataset['test'].train_test_split(test_size=0.5, seed=seed_num, shuffle=True)
    test = temp['test']
    val = temp['train']

    train_dataset = train.map(
        lambda x: preprocess_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=train.column_names
    ).select(range(min(train_range, len(dataset['train']))))
    val_dataset = val.map(
        lambda x: preprocess_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=val.column_names
    ).select(range(min(val_range, len(temp['train']))))
    test_dataset = test.map(
        lambda x: preprocess_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=test.column_names
    ).select(range(min(test_range, len(temp['test']))))

    return train_dataset, val_dataset, test_dataset

def main():
    pl.seed_everything(seed_num)

    model = T5SummarizationModule(model_name, learning_rate, optimizer_name)
    tokenizer = model.tokenizer

    train_dataset, val_dataset, test_dataset = load_and_prepare_data(tokenizer, max_length)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_name)

    train_loader = DataLoader(train_dataset, batch_size=4, collate_fn=data_collator, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, collate_fn=data_collator, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=4, collate_fn=data_collator, num_workers=0)

    logger = TensorBoardLogger("tb_logs", name="my_model")
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='best-checkpoint',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
        val_check_interval=0.25,
        num_sanity_val_steps=0,
        enable_checkpointing=True,
        accelerator='tpu',
        devices=8,
    )

    trainer.fit(model, train_loader, val_loader)
    test_results = trainer.test(model, test_loader)
    
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/results_with_seed_{seed_num}.txt", "w") as f:
        f.write(f"Seed: {seed_num}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Optimizer: {optimizer_name}\n")
        f.write(f'Training range: {train_range}\n')
        f.write(f'Test range: {test_range}\n')
        f.write(f'Validation range: {val_range}\n')
        f.write(f'Actual train dataset size: {len(train_dataset)}\n')
        f.write(f'Actual validation dataset size: {len(val_dataset)}\n')
        f.write(f'Actual test dataset size: {len(test_dataset)}\n')
        f.write(f'Learning rate: {learning_rate}\n')
        f.write("\nBest checkpoint:\n")
        f.write(f"{checkpoint_callback.best_model_path}\n")
        f.write("\nTest results:\n")
        f.write("\n".join([f"{key}: {value}" for key, value in test_results[0].items()]))

if __name__ == "__main__":
    main()