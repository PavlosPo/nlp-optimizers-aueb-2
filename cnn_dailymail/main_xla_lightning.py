import torch
# import pytorch_lightning as pl
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import torch.utils.data as data
from transformers import DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text.bert import BERTScore
import os
import argparse
from icecream import ic
import numpy as np


os.environ["TOKENIZERS_PARALLELISM"] = 'false'

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True, help="Seed number for reproducibility")
parser.add_argument("--optim", type=str, required=True, help="Optimizer to use for training")
parser.add_argument("--batch_size", type=int, required=True, help="Batch size for training")
args = parser.parse_args()

# Parameters
optimizer_name = args.optim
model_name = "google-t5/t5-small"
dataset_name = "cnn_dailymail"
seed_num = args.seed
max_length = 512
output_dir = f"{optimizer_name}/{dataset_name}/best_{model_name.split('/')[-1]}"
train_range = 15000
test_range = 1500
val_range = 1500
epochs = 4
learning_rate = 9.9879589111261e-06
batch_size = args.batch_size

class T5SummarizationModule(pl.LightningModule):
    def __init__(self, model_name, learning_rate, optimizer_name="adamw", max_new_tokens=20):        
        super().__init__()
        # self.save_hyperparameters()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).train()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.register_buffer("sigma", torch.eye(3)) # Access self.sigma anywhere
        # self.train_loader = train_loader
        # self.val_loader = val_loader
        # self.test_loader = test_loader
        self.max_new_tokens = max_new_tokens
        self.rouge_score = ROUGEScore(use_stemmer=True, sync_on_compute=True)
        ic(self.device)
        ic(self.sigma)
        self.bert_score = BERTScore(model_name_or_path='roberta-large', sync_on_compute=True, max_length=self.max_new_tokens)
        self.training_step_outputs = []
        self.valid_step_predictions = []
        self.valid_step_labels = []
        self.test_step_predictions = []
        self.test_step_labels = []

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        loss = outputs['loss']
        self.log("train_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss # Always return the loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            outputs = self(input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                labels=batch["labels"])
            loss = outputs['loss']
            generated_ids = self.model.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], max_new_tokens=self.max_new_tokens)
            self.valid_step_predictions.append(generated_ids)
            self.valid_step_labels.append(batch["labels"])
            self.log("val_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            return loss
        
    def on_validation_epoch_end(self):
        # Compute metrics and log them
        all_preds = torch.cat(self.valid_step_predictions, dim=0)
        all_labels = torch.cat(self.valid_step_labels, dim=0)
        with torch.no_grad():
            results = self._log_metrics(prefix="val", predictions=all_preds, labels=all_labels)
            # self.log_dict("val_" + results, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.valid_step_predictions.clear()
        self.valid_step_labels.clear()
        
    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            outputs = self(input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                labels=batch["labels"])
            loss = outputs['loss']
            generated_ids = self.model.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], max_new_tokens=self.max_new_tokens)
            self.test_step_predictions.append(generated_ids)
            self.test_step_labels.append(batch["labels"])
            self.log("test_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            return loss

    def on_test_epoch_end(self, ):
        # Compute metrics and log them
        all_preds = torch.cat(self.test_step_predictions, dim=0)
        all_labels = torch.cat(self.test_step_labels, dim=0)
        with torch.no_grad():
            self._log_metrics(prefix="test", predictions=all_preds, labels=all_labels)
            # self.log_dict("test_" + results, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.test_step_predictions.clear()
        self.test_step_labels.clear()
            
    # def _compute_metrics(self, preds, labels):
    #     decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
    #     labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
    #     decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
    #     result_rouge = self.rouge_score(preds=decoded_preds, target=decoded_labels)
    #     result_brt = self.bert_score(preds=decoded_preds, target=decoded_labels)
    #     result_brt_average_values = {key: tensors.mean().item() for key, tensors in result_brt.items()}
    #     results = {**result_rouge, **result_brt_average_values}
    #     return results
    
    # def _log_metrics(self, prefix, predictions, labels):
    #     ic(f"Debug information for {prefix}_predictions:")
    #     ic(len(predictions))
    #     ic(type(predictions))
    #     if len(predictions) > 0:
    #         ic(type(predictions[0]))
    #         ic(predictions[0].shape if hasattr(predictions[0], 'shape') else None)
        
    #     ic(f"Debug information for {prefix}_labels:")
    #     ic(len(labels))
    #     ic(type(labels))
    #     if len(labels) > 0:
    #         ic(type(labels[0]))
    #         ic(labels[0].shape if hasattr(labels[0], 'shape') else None)

    #     metrics = self._compute_metrics(predictions, labels)
    #     self.log_dict({f"{prefix}_{k}": v for k, v in metrics.items()}, on_step=False, on_epoch=True, sync_dist=True)

    def _log_metrics(self, prefix, predictions, labels):
        ic(f"Debug information for {prefix}_predictions:")
        ic(len(predictions))
        ic(type(predictions))
        if len(predictions) > 0:
            ic(type(predictions[0]))
            ic(predictions[0].shape if hasattr(predictions[0], 'shape') else None)
        
        ic(f"Debug information for {prefix}_labels:")
        ic(len(labels))
        ic(type(labels))
        if len(labels) > 0:
            ic(type(labels[0]))
            ic(labels[0].shape if hasattr(labels[0], 'shape') else None)

        metrics = self._compute_metrics(predictions, labels)
        self.log_dict({f"{prefix}_{k}": v for k, v in metrics.items()}, on_step=False, on_epoch=True, sync_dist=True)
        
    # def _compute_metrics(self, predictions, labels):
    #     ic("Shapes inside compute_metrics:")
    #     ic(len(predictions), predictions[0].shape if predictions else None)
    #     ic(len(labels), labels[0].shape if labels else None)
        
    #     decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
    #     # Process labels
    #     processed_labels = []
    #     for label in labels:
    #         label = label[label != -100]  # Remove padding
    #         processed_labels.append(label)
        
    #     decoded_labels = self.tokenizer.batch_decode(processed_labels, skip_special_tokens=True)
        
    #     result_rouge = self.rouge(preds=decoded_preds, target=decoded_labels)
    #     result_brt = self.bert_score(preds=decoded_preds, target=decoded_labels)
        
    #     result_brt_average_values = {key: torch.tensor(tensors.mean().item()) for key, tensors in result_brt.items()}
    #     results = {**result_rouge, **result_brt_average_values}
    #     return results
    def _compute_metrics(self, predictions, labels):
        ic("Shapes inside compute_metrics:")
        ic(len(predictions), predictions[0].shape if len(predictions) > 0 else None)
        ic(len(labels), labels[0].shape if len(labels) > 0 else None)
        
        # Ensure predictions and labels are in correct format
        predictions = predictions.cpu().numpy() if torch.is_tensor(predictions) else predictions
        labels = labels.cpu().numpy() if torch.is_tensor(labels) else labels

        # Decode predictions and labels
        # ic(predictions[0])
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        ic(predictions[0])  # Text
        ic(decoded_preds[0]) # Text or not?
        
        # Convert predictions from logits to token IDs
        # predictions = np.argmax(predictions, axis=-1)
        
        # Process labels
        processed_labels = []
        for label in labels:
            label = label[label != -100]  # Remove padding
            processed_labels.append(label)
        
        decoded_labels = self.tokenizer.batch_decode(processed_labels, skip_special_tokens=True)
        ic(processed_labels[0])
        ic(decoded_labels[0]) 
        
        # Calculate metrics
        result_rouge = self.rouge_score(preds=decoded_preds, target=decoded_labels)
        result_brt = self.bert_score(preds=decoded_preds, target=decoded_labels)
        
        result_brt_average_values = {key: torch.tensor(tensors.mean().item()) for key, tensors in result_brt.items()}
        results = {**result_rouge, **result_brt_average_values}
        return results
    
    def configure_optimizers(self):
        optimizer = self._get_optimizer()
        return optimizer

    def _get_optimizer(self):
        if self.optimizer_name == "adamw":
            return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == "sgd":
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
        
class T5SummarizationDataModule(pl.LightningDataModule):
    def __init__(self, model_name, dataset_name, max_length, batch_size, train_range, val_range, test_range, seed_num):
        super().__init__()
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.train_range = train_range
        self.val_range = val_range
        self.test_range = test_range
        self.seed_num = seed_num
        self.tokenizer = None
        self.data_collator = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    def prepare_data(self):
        # download, IO, etc. Useful with shared filesystems
        # only called on 1 GPU/TPU in distributed
        load_dataset(self.dataset_name, '3.0.0').shuffle(seed=self.seed_num)
        AutoTokenizer.from_pretrained(self.model_name)

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model_name)

        # Load and preprocess the dataset
        dataset = load_dataset(self.dataset_name, '3.0.0').shuffle(seed=self.seed_num)
        
        if stage == 'fit' or stage is None:
            train = dataset['train'].select(range(min(self.train_range, len(dataset['train']))))
            self.train_dataset = self._preprocess_dataset(train)

            temp = dataset['test'].train_test_split(test_size=0.5, seed=self.seed_num, shuffle=True)
            val = temp['train'].select(range(min(self.val_range, len(temp['train']))))
            self.val_dataset = self._preprocess_dataset(val)

        if stage == 'test' or stage is None:
            temp = dataset['test'].train_test_split(test_size=0.5, seed=self.seed_num, shuffle=True)
            test = temp['test'].select(range(min(self.test_range, len(temp['test']))))
            self.test_dataset = self._preprocess_dataset(test)
    
    def _preprocess_dataset(self, dataset):
        return dataset.map(
            lambda x: self._preprocess_function(x),
            batched=True,
            remove_columns=dataset.column_names
        )
        
    def _preprocess_function(self, examples):
        prefix = "summarize: "
        inputs = [prefix + doc for doc in examples["article"]]
        model_inputs = self.tokenizer(inputs, padding="max_length", truncation=True, max_length=self.max_length)
        labels = self.tokenizer(text_target=examples["highlights"], padding="max_length", truncation=True, max_length=self.max_length)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.data_collator, shuffle=True)

    def val_dataloader(self):
        return data.DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.data_collator)

    def test_dataloader(self):
        return data.DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.data_collator)

    def on_exception(self, exception):
        # clean up state after the trainer faced an exception\
            pass
        
    def teardown(self, stage: str):
        # clean up state after the trainer stops, delete files...
        # called on every process in DDP
        if stage == 'fit':
            self.train_dataset = None
            self.val_dataset = None
        elif stage == 'test':
            self.test_dataset = None
        
def main():
    pl.seed_everything(seed_num)
    
    
    model = T5SummarizationModule(model_name=model_name, learning_rate=learning_rate, optimizer_name=optimizer_name)
    
    data_module = T5SummarizationDataModule(
        model_name=model_name,
        dataset_name=dataset_name,
        max_length=max_length,
        batch_size=batch_size,
        train_range=train_range,
        val_range=val_range,
        test_range=test_range,
        seed_num=seed_num
    )
    
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
        log_every_n_steps=100,
        enable_checkpointing=True,
        accelerator='auto',
        devices='auto',
        accumulate_grad_batches=16,
        # precision="1"
    )

    trainer.fit(model, datamodule=data_module)
    test_results = trainer.test(model, datamodule=data_module)
    
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/results_with_seed_{seed_num}.txt", "w") as f:
        f.write(f"Seed: {seed_num}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Optimizer: {optimizer_name}\n")
        f.write(f'Training range: {train_range}\n')
        f.write(f'Test range: {test_range}\n')
        f.write(f'Validation range: {val_range}\n')
        f.write(f'Learning rate: {learning_rate}\n')
        f.write("\nBest checkpoint:\n")
        f.write(f"{checkpoint_callback.best_model_path}\n")
        f.write("\nTest results:\n")
        f.write("\n".join([f"{key}: {value}" for key, value in test_results[0].items()]))

if __name__ == "__main__":
    main()