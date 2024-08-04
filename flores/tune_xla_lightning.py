import torch
import pickle
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
# Import PytorchLightningPruning callback
from optuna.integration import PyTorchLightningPruningCallback
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from torchmetrics import MeanMetric
import optuna
from optuna.storages import RDBStorage
import os
import argparse
from dotenv import load_dotenv

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = 'false'

name_of_database_based_on_server_name = os.getenv("SERVER_NAME")
db_url = f"sqlite:///{name_of_database_based_on_server_name}.db"

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True, help="Seed number for reproducibility")
parser.add_argument("--optim", type=str, required=True, help="Optimizer to use for training")
parser.add_argument("--batch_size", type=int, required=True, help="Batch size for training")
args = parser.parse_args()

# Parameters
optimizer_name = args.optim
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
dataset_name = "facebook/flores"

seed_num = args.seed
train_range = 1000
test_range = 1000
val_range = 1000
epochs = 5
batch_size = args.batch_size
n_trials = 30
# https://github.com/facebookresearch/flores/blob/main/flores200/README.md
# Do not put English, already retrieved as the input of the model.
language_to_choose = ["deu_Latn", "fra_Latn", "ron_Latn"] # German, French, Romanian

learning_rate_range = (1e-7, 1e-3)

class T5TranslationModule(pl.LightningModule):
    def __init__(self, model_name, learning_rate, optimizer_name="adamw"):        
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).train()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.val_loss = MeanMetric()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs
        
    def training_step(self, batch, batch_idx):
        outputs = self.forward(input_ids=batch["input_ids"], 
                               attention_mask=batch["attention_mask"], 
                               labels=batch["labels"])
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self.forward(input_ids=batch["input_ids"],
                               attention_mask=batch["attention_mask"],
                               labels=batch["labels"])
        loss = outputs.loss
        self.val_loss.update(loss)
        return loss
    
    def on_validation_epoch_end(self):
        avg_loss = self.val_loss.compute()
        self.log("val_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_loss.reset()

    def test_step(self, batch, batch_idx):
        outputs = self.forward(input_ids=batch["input_ids"], 
                       attention_mask=batch["attention_mask"], 
                       labels=batch["labels"])
        loss = outputs.loss
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
        
    def configure_optimizers(self):
        return self._get_optimizer()

    def _get_optimizer(self):
        if self.optimizer_name == "adamw":
            return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == "sgd":
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == "sgdm":
            # Default Momentum 0.9
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        elif self.optimizer_name == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == "adagrad":
            return torch.optim.Adagrad(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == "adadelta":
            return torch.optim.Adadelta(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == "rmsprop":
            return torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == "rprop":
            return torch.optim.Rprop(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == "adamax":
            return torch.optim.Adamax(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == "adabound":
            return torch.optim.AdaBound(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

class T5TranslationDataModule(pl.LightningDataModule):
    def __init__(self, model_name, dataset_name, max_length, 
                 batch_size, train_range, val_range, test_range, seed_num,
                 languages):
        super().__init__()
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.train_range = train_range
        self.val_range = val_range
        self.test_range = test_range
        self.seed_num = seed_num
        self.languages = languages
        self.tokenizer = None
        self.data_collator = None
        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []
        self.cache_dir = f"./dataset_cache_{self.seed_num}"

    def prepare_data(self):
        load_dataset(self.dataset_name, 'all', trust_remote_code=True).shuffle(seed=self.seed_num)
        AutoTokenizer.from_pretrained(self.model_name)

    def setup(self, stage=None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model_name)
        
        if stage == 'fit' or stage is None:
            self.train_datasets = self._get_or_process_dataset('train')
            self.val_datasets = self._get_or_process_dataset('validation')
        if stage == 'test' or stage is None:
            self.test_datasets = self._get_or_process_dataset('test')
        
        print(f"Setup complete. Datasets sizes: Train: {len(self.train_datasets)}, Val: {len(self.val_datasets)}, Test: {len(self.test_datasets)}")
        # Set global length for train, val, and test datasets, to save in the output file after hyperparameter tuning
        global train_range, val_range, test_range
        train_range = len(self.train_datasets)
        val_range = len(self.val_datasets)
        test_range = len(self.test_datasets)

    def _get_or_process_dataset(self, split):
        # Create combined dataset from all language pairs
        combined_dataset = []
        
        for language in self.languages: 
            """
            This loop runs once per language code.
            Each language code creates a cache file with the same name in the cache directory.
            If the cache file exists, the dataset is loaded from the cache file.
            If the cache file does not exist, the dataset is loaded from the original dataset and saved in the cache file.
            The dataset is then added to the combined dataset in order to return the combined dataset with all the language 
            pairs that have been set in the 'language_to_choose' list.
            """
            cache_file = os.path.join(self.cache_dir, f"{split}_{language}_{self.seed_num}.pkl")
            
            if os.path.exists(cache_file):
                print(f"Loading cached {split} dataset for {language}...")
                with open(cache_file, 'rb') as f:
                    dataset = pickle.load(f)
                print(f"Loaded {split} dataset for {language} with {len(dataset)} samples")
            else:
                print(f"Processing {split} dataset for {language}...")
                dataset = load_dataset(self.dataset_name, 'all',  trust_remote_code=True)['dev'].shuffle(seed=self.seed_num)
                
                if split == 'train':
                    data = dataset.select(range(min(self.train_range, len(dataset))))
                elif split == 'validation':
                    data = dataset.select(range(min(self.val_range, len(dataset))))
                elif split == 'test':
                    data = dataset.select(range(min(self.test_range, len(dataset))))
                
                processed_dataset = self._preprocess_dataset(data, language)
                
                os.makedirs(self.cache_dir, exist_ok=True)
                with open(cache_file, 'wb') as f:
                    pickle.dump(processed_dataset, f)
                
                dataset = processed_dataset
            
            combined_dataset.extend(dataset)
        
        return combined_dataset
    
    def _preprocess_dataset(self, dataset, target_language):
        """
        Preprocess the dataset by mapping the language codes to their corresponding names.
        For example, "deu_Latn" maps to "German" in the German dataset. 
        This is given in the link: 
        https://github.com/facebookresearch/flores/blob/main/flores200/README.md
        
        The function also maps the target language code to the corresponding name.
        The function returns the preprocessed dataset.

        Args:
            dataset (_dataset_): The dataset to preprocess. 
            target_language (str): The target language code.

        Returns:
            _dataset_: The preprocessed dataset.
        """
        mapping = {
            "deu_Latn": "German",
            "fra_Latn": "French",
            "ron_Latn": "Romanian"
        }
        target_lang_name = mapping[target_language]

        def preprocess_function(examples):
            """
            Internal function to preprocess the dataset.
            """
            model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

            for i in range(len(examples['sentence_eng_Latn'])):
                """
                This loop runs once per sentence in the dataset.
                The sentence is mapped to the given target language name and the target text is mapped to the corresponding language code.
                The function returns the model inputs.
                """
                prefix = f"translate English to {target_lang_name}: "
                input_text = prefix + examples['sentence_eng_Latn'][i]
                target_text = examples[f'sentence_{target_language}'][i]
                
                tokenized_input = self.tokenizer(input_text, max_length=self.max_length, padding="max_length", truncation=True)
                tokenized_target = self.tokenizer(target_text, max_length=self.max_length, padding="max_length", truncation=True)
                
                model_inputs["input_ids"].append(tokenized_input["input_ids"])
                model_inputs["attention_mask"].append(tokenized_input["attention_mask"])
                model_inputs["labels"].append(tokenized_target["input_ids"])

            return model_inputs

        return dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names
        )

    def train_dataloader(self):
        return DataLoader(self.train_datasets, batch_size=self.batch_size, collate_fn=self.data_collator, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_datasets, batch_size=self.batch_size, collate_fn=self.data_collator, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_datasets, batch_size=self.batch_size, collate_fn=self.data_collator, drop_last=True)

# Define the objective function for Optuna
def objective(trial):
    # Define hyperparameters to optimize
    learning_rate = trial.suggest_float("learning_rate",learning_rate_range[0],learning_rate_range[1], log=True)
    
    pl.seed_everything(seed_num)
    
    model = T5TranslationModule(
        model_name=model_name,
        learning_rate=learning_rate,
        optimizer_name=optimizer_name
    )
    
    data_module = T5TranslationDataModule(
        model_name=model_name,
        dataset_name=dataset_name,
        max_length=max_length,
        batch_size=batch_size,
        train_range=train_range,
        val_range=val_range,
        test_range=test_range,
        seed_num=seed_num,
        languages=language_to_choose
    )
    
    logger = TensorBoardLogger("tb_logs", 
                               name=f"{model_name}_{optimizer_name}_seed_{seed_num}_trial_{trial.number}")
    
    # checkpoint_callback = ModelCheckpoint(dirpath= f"checkpoints/{model_name}_{optimizer_name}_seed_{seed_num}_trial_{trial.number}", 
    #                                         monitor="val_loss", 
    #                                         mode="min",
    #                                         save_top_k=1)
    
    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=logger,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")],
        log_every_n_steps=1,
        val_check_interval=0.3,
        num_sanity_val_steps=0,
        accelerator='auto',
        devices='auto',
    )
    hyperparameters = dict(learning_rate=learning_rate)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=data_module)
    
    val_loss = trainer.callback_metrics['val_loss'].item()
    
    return val_loss


def main():
    # Set up the SQLite database storage
    storage = RDBStorage(url=db_url)
    
    # Create or load the study
    study = optuna.create_study(
        direction="minimize", 
        storage=storage, 
        study_name=f"{model_name}_{optimizer_name}_with_seed_{seed_num}", 
        load_if_exists=True
    )
    study.optimize(objective, n_trials=n_trials)  # Adjust n_trials as needed
    
    print("Best trial:")
    trial = study.best_trial
    
    # Define the output directory structure
    output_dir = os.path.join(
        "hypertuning_results_lr_tuning",
        model_name.replace("/", "_"),
        optimizer_name,
        f"seed_{seed_num}"
    )
    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, "best_hyperparameters.txt")
    
    with open(result_file, "w") as f:
        f.write(f"Seed: {seed_num}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Optimizer: {optimizer_name}\n")
        f.write(f'Training range: {train_range}\n')
        f.write(f'Test range: {test_range}\n')
        f.write(f'Validation range: {val_range}\n')
        f.write(f"Best Validation Loss: {trial.value}\n")
        f.write("Best Hyperparameters:\n")
        for key, value in trial.params.items():
            f.write(f"{key}: {value}\n")
        f.write("Search Spaces:\n")
        f.write(f" learning_rate: {learning_rate_range}\n")

if __name__ == "__main__":
    main()