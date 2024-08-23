import torch
import pickle
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from torchmetrics import MeanMetric
import torch_optimizer as t_optim
import optuna
import wandb
from optuna.storages import RDBStorage
import os
import argparse
from dotenv import load_dotenv
import json

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = 'false'

wandb.require("core")

name_of_database_based_on_server_name = os.getenv("SERVER_NAME")
db_url = f"sqlite:///{name_of_database_based_on_server_name}.db"

# Ask the user to choose between small, base and large model
model_names = {
    "1": "google-t5/t5-small",
    "2": "google-t5/t5-base",
    "3": "google-t5/t5-large"
}
max_length = {
    "1": 512,
    "2": 768,
    "3": 1024
}
model_name = "google-t5/t5-small"
max_length = 512
dataset_name = "cnn_dailymail"
train_range = 350
test_range = 35
val_range = 35
epochs = 2

class T5SummarizationModule(pl.LightningModule):
    def __init__(self, model_name, learning_rate, optimizer_name="adamw", **optimizer_params):        
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).train()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.optimizer_params = optimizer_params
        self.val_loss = MeanMetric() # This line to create a metric for tracking validation loss

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs
        
    def training_step(self, batch, batch_idx):
        outputs = self.forward(input_ids=batch["input_ids"], 
                               attention_mask=batch["attention_mask"], 
                               labels=batch["labels"])
        loss = outputs['loss']
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            outputs = self.forward(input_ids=batch["input_ids"],
                                   attention_mask=batch["attention_mask"],
                                   labels=batch["labels"])
            loss = outputs['loss']
            # Update the validation loss metric
            self.val_loss.update(loss)
        return loss
    
    def on_validation_epoch_end(self):
        avg_loss = self.val_loss.compute() # Compute the mean validation loss for the epoch
        # Log the epoch validation loss
        self.log("val_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_loss.reset() # Reset the metric for the next epoch

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            outputs = self.forward(input_ids=batch["input_ids"], 
                           attention_mask=batch["attention_mask"], 
                           labels=batch["labels"])
            loss = outputs['loss']
            self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
        
    def configure_optimizers(self):
        optimizer = self._get_optimizer()
        return optimizer

    def _get_optimizer(self):
        if self.optimizer_name == "adamw":
            return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, **self.optimizer_params)
        elif self.optimizer_name == "sgd":
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate, **self.optimizer_params)
        elif self.optimizer_name == "sgdm": # Momentum will be added later in the '**self.optimizer_params' kwargs
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate, **self.optimizer_params)
        elif self.optimizer_name == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate, **self.optimizer_params)
        elif self.optimizer_name == "adamax":
            return torch.optim.Adamax(self.parameters(), lr=self.learning_rate, **self.optimizer_params)
        elif self.optimizer_name == "nadam":
            return torch.optim.NAdam(self.parameters(), lr=self.learning_rate, **self.optimizer_params)
        elif self.optimizer_name == "adagrad":
            return torch.optim.Adagrad(self.parameters(), lr=self.learning_rate, **self.optimizer_params)
        elif self.optimizer_name == "adadelta":
            return torch.optim.Adadelta(self.parameters(), lr=self.learning_rate, **self.optimizer_params)
        elif self.optimizer_name == "adabound":
            return t_optim.AdaBound(self.parameters(), lr=self.learning_rate, **self.optimizer_params)
        elif self.optimizer_name == "rmsprop":
            return torch.optim.RMSprop(self.parameters(), lr=self.learning_rate, **self.optimizer_params)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
        
class T5SummarizationDataModule(pl.LightningDataModule):
    def __init__(self, model_name, dataset_name, max_length, 
                 batch_size, train_range, val_range, test_range, seed_num):
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
        self.cache_dir = f"./dataset_cache_{self.seed_num}"

    def prepare_data(self):
        # Downloading data, called only once on 1 GPU/TPU in distributed settings
        load_dataset(self.dataset_name, '3.0.0',  trust_remote_code=True).shuffle(seed=self.seed_num)
        AutoTokenizer.from_pretrained(self.model_name)

    def setup(self, stage):
        # Setting up the data, called on every GPU/TPU in DDP
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model_name)
        
        # Load and preprocess the dataset
        if stage == 'fit' or stage is None:
            self.train_dataset = self._get_or_process_dataset('train')
            self.val_dataset = self._get_or_process_dataset('val')
        if stage == 'test' or stage is None:
            self.test_dataset = self._get_or_process_dataset('test')
            
    def _get_or_process_dataset(self, split):
        cache_file = os.path.join(self.cache_dir, f"{split}_{self.seed_num}.pkl")
        
        if os.path.exists(cache_file):
            print(f"Loading cached {split} dataset...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        print(f"Processing {split} dataset...")
        dataset = load_dataset(self.dataset_name, '3.0.0',  trust_remote_code=True).shuffle(seed=self.seed_num)
        
        if split == 'train':
            data = dataset['train'].select(range(min(self.train_range, len(dataset['train']))))
        elif split in ['val', 'test']:
            temp1 = dataset['test']
            temp2 = dataset['validation']
            # concat the two splits
            temp = concatenate_datasets([temp1, temp2]).train_test_split(test_size=0.5, seed=self.seed_num, shuffle=True)
            if split == 'val':
                data = temp['train'].select(range(min(self.val_range, len(temp['train']))))
            else:
                data = temp['test'].select(range(min(self.test_range, len(temp['test']))))
        
        processed_dataset = self._preprocess_dataset(data)
        
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(processed_dataset, f)
        
        return processed_dataset
    
    def _preprocess_dataset(self, dataset):
        return dataset.map(
            lambda x: self._preprocess_function(x),
            batched=True,
            remove_columns=dataset.column_names
        )
        
    def _preprocess_function(self, examples):
        prefix = "summarize: "
        inputs = [prefix + doc for doc in examples["article"]]
        model_inputs = self.tokenizer(inputs, padding="max_length", 
                                      truncation=True, max_length=self.max_length)
        labels = self.tokenizer(text_target=examples["highlights"], 
                                padding="max_length", truncation=True, max_length=self.max_length)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.data_collator, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.data_collator, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.data_collator, drop_last=True)

def main(seed, optimizer_name, batch_size, learning_rate, **optimizer_params):
    print(f"\nTraining with seed {seed}, optimizer {optimizer_name}, batch size {batch_size}, and learning rate {learning_rate}\n")
    for key, value in optimizer_params.items():
        print(f"Using additional hyperparameter: {key} = {value}")
    
    pl.seed_everything(seed)
    model = T5SummarizationModule(
        model_name=model_name,
        learning_rate=learning_rate,
        optimizer_name=optimizer_name,
        **optimizer_params
    )
    
    data_module = T5SummarizationDataModule(
        model_name=model_name,
        dataset_name=dataset_name,
        max_length=max_length,
        batch_size=batch_size,
        train_range=train_range,
        val_range=val_range,
        test_range=test_range,
        seed_num=seed
    )
    
    # Initialize WandbLogger
    # wandb.finish()  # In case the last run crashed, this will close the previous run
    wandb_logger = WandbLogger(project="t5_summarization_project",
                               name=f"{model_name}_{optimizer_name}_seed_{seed}",
                               log_model=True)
    
    checkpoint_callback = ModelCheckpoint(dirpath= f"checkpoints/{model_name}_{optimizer_name}_seed_{seed}", 
                                          monitor="val_loss", 
                                          mode="min",
                                          save_top_k=1,
                                          save_last=True)
    
    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=wandb_logger,  # Use W&B logger here
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
        val_check_interval=0.3,
        num_sanity_val_steps=0,
        accelerator='auto',
        devices='auto',
        enable_checkpointing=True
    )
    
    hyperparameters = dict(learning_rate=learning_rate, 
                           optimizer_name=optimizer_name, 
                           seed_num=seed, 
                           dataset_name=dataset_name, 
                           model_name=model_name, 
                           max_length=max_length, 
                           batch_size=batch_size, 
                           train_range=train_range, 
                           val_range=val_range, 
                           test_range=test_range,
                           **optimizer_params)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=data_module, ckpt_path="last")
    
    trainer.test(datamodule=data_module, ckpt_path="best")
    wandb.finish()
    print(f"\nFinished training with seed {seed}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="Seed number for reproducibility")
    parser.add_argument("--optim", type=str, required=True, help="Optimizer to use for training")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for training")
    args = parser.parse_args()
    main(args.seed, args.optim, args.batch_size, args.learning_rate)