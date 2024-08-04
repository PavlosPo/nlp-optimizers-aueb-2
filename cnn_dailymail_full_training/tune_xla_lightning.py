import torch
import pickle
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from torchmetrics import MeanMetric
import optuna
from optuna.storages import RDBStorage
import torch_optimizer as t_optim
import os
import argparse
from dotenv import load_dotenv

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = 'false'

name_of_database_based_on_server_name = os.getenv("SERVER_NAME")
db_url = f"sqlite:///{name_of_database_based_on_server_name}_full_training.db"

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
dataset_name = "cnn_dailymail"
seed_num = args.seed
train_range = 35000
test_range = 3500
val_range = 3500
epochs = 5
n_trials = 30
learning_rate_range = (1e-7, 1e-3)
betas_range = {
            "beta1" : (0.8, 0.95),
            "beta2" : (0.9, 0.99999)
        }
eps_range = (1e-9, 1e-7)
nadam_momentum_range = (1e-4, 1e-2)
sgdm_momentum_range = (0.7, 0.99999)
adabound_gamma = (1e-4, 2e-3)
adabound_final_lr = (1e-2, 1e-1)
adabound_weight_decay = (1e-2, 1e-1)
batch_size = args.batch_size

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
            
        # print(f"Setup complete. Datasets sizes: Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
        # # Set global length for train, val, and test datasets, to save in the output file after hyperparameter tuning
        # global train_range, val_range, test_range
        # train_range = len(self.train_dataset)
        # val_range = len(self.val_dataset)
        # test_range = len(self.test_dataset)
            
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
            temp = dataset['test'].train_test_split(test_size=0.5, seed=self.seed_num, shuffle=True)
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

# Define the objective function for Optuna
def objective(trial):
    # Define hyperparameters to optimize
    learning_rate = trial.suggest_float("learning_rate",learning_rate_range[0],learning_rate_range[1], log=True)
    
    optimizer_params = {}
    if optimizer_name == "adamw":
        optimizer_params["betas"] = (
            trial.suggest_float("adamw_beta1", betas_range['beta1'][0], betas_range['beta1'][1]),
            trial.suggest_float("adamw_beta2", betas_range["beta2"][0], betas_range["beta2"][1])
        )
        optimizer_params["eps"] = trial.suggest_float("adamw_epsilon", eps_range[0], eps_range[1], log=True)
    elif optimizer_name == "sgd":
        pass
    elif optimizer_name == "sgdm":
        optimizer_params["momentum"] = trial.suggest_float("momentum", sgdm_momentum_range[0], sgdm_momentum_range[1])
    elif optimizer_name == "adam":
        optimizer_params["betas"] = (
            trial.suggest_float("adam_beta1", betas_range['beta1'][0], betas_range['beta1'][1]),
            trial.suggest_float("adam_beta2", betas_range["beta2"][0], betas_range["beta2"][1])
        )
        optimizer_params["eps"] = trial.suggest_float("adam_epsilon", eps_range[0], eps_range[1], log=True)
    # elif optimizer_name == "adagrad": # TODO: Fill this correctly
    #     optimizer_params["lr_decay"] = trial.suggest_float("lr_decay", 0, 1)
    #     optimizer_params["weight_decay"] = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    # elif optimizer_name == "adadelta": # TODO: Fill this correctly
    #     optimizer_params["rho"] = trial.suggest_float("rho", 0, 1)
    #     optimizer_params["eps"] = trial.suggest_float("epsilon", 1e-8, 1e-6, log=True)
    # elif optimizer_name == "rmsprop": # TODO: Fill this correctly
    #     optimizer_params["alpha"] = trial.suggest_float("alpha", 0, 1)
    #     optimizer_params["momentum"] = trial.suggest_float("momentum", 0, 1)
    #     optimizer_params["eps"] = trial.suggest_float("epsilon", 1e-8, 1e-6, log=True)
    elif optimizer_name == "adamax": 
        optimizer_params["betas"] = (
            trial.suggest_float("adamax_beta1", betas_range["beta1"][0], betas_range["beta1"][1]),
            trial.suggest_float("adamax_beta2", betas_range['beta2'][0], betas_range['beta2'][1])
        )
        optimizer_params["eps"] = trial.suggest_float("adamax_epsilon", eps_range[0], eps_range[1], log=True)
    elif optimizer_name == "nadam":
        optimizer_params["betas"] = (
            trial.suggest_float("nadam_beta1", betas_range["beta1"][0], betas_range["beta1"][1]),
            trial.suggest_float("nadam_beta2", betas_range["beta2"][0], betas_range["beta2"][1])
        )
        optimizer_params["eps"] = trial.suggest_float("nadam_epsilon", eps_range[0], eps_range[1], log=True)
        optimizer_params["momentum_decay"] = trial.suggest_float("momentum_decay", nadam_momentum_range[0], nadam_momentum_range[1])
    elif optimizer_name == "adabound" :
        optimizer_params["betas"] = (
            trial.suggest_float("adabound_beta1", betas_range["beta1"][0], betas_range["beta1"][1]),
            trial.suggest_float("adabound_beta2", betas_range["beta2"][0], betas_range["beta2"][1])
        )
        optimizer_params['eps'] = trial.suggest_float("eps", eps_range[0], eps_range[1], log=True)
        optimizer_params["gamma"] = trial.suggest_float("gamma", adabound_gamma[0], adabound_gamma[1])
        optimizer_params["final_lr"] = trial.suggest_float("final_lr", adabound_final_lr[0], adabound_final_lr[1])
        optimizer_params["weight_decay"] = trial.suggest_float("weight_decay",adabound_weight_decay[0], adabound_weight_decay[1] , log=True)
    
    pl.seed_everything(seed_num)
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
        seed_num=seed_num
    )
    
    logger = TensorBoardLogger("tb_logs_full_training", 
                               name=f"{model_name}_{optimizer_name}_seed_{seed_num}_trial_{trial.number}")
    
    checkpoint_callback = ModelCheckpoint(dirpath= f"checkpoints_full_training/{model_name}_{optimizer_name}_seed_{seed_num}_trial_{trial.number}", 
                                            monitor="val_loss", 
                                            mode="min",
                                            save_top_k=1)
    
    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
        val_check_interval=0.3,
        num_sanity_val_steps=0,
        accelerator='auto',
        devices='auto',
    )
    hyperparameters = dict(learning_rate=learning_rate,
                           optimizer_name=optimizer_name,
                           **optimizer_params)
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
        study_name=f"full_training_{model_name}_{optimizer_name}_with_seed_{seed_num}", 
        load_if_exists=True
    )
    study.optimize(objective, n_trials=n_trials)  # Adjust n_trials as needed
    
    print("Best trial:")
    trial = study.best_trial
    
    # Define the output directory structure
    output_dir = os.path.join(
        "hypertuning_results_full_training",
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
        f.write(f"  learning_rate: {learning_rate_range}\n")
        f.write(f"  beta1: ({betas_range['beta1'][0]}, {betas_range['beta1'][1]})\n")
        f.write(f"  beta2: ({betas_range['beta2'][0]}, {betas_range['beta2'][1]})\n")
        f.write(f"  eps: {eps_range}\n")
        if optimizer_name == "adabound":
            f.write(f"  gamma: {adabound_gamma}\n")
            f.write(f"  final_lr: {adabound_final_lr}\n")
            f.write(f"  weight_decay: {adabound_weight_decay}\n")
        if optimizer_name == "nadam":
            f.write(f"  momentum: {nadam_momentum_range}\n")
        if optimizer_name == "sgdm":
            f.write(f"  momentum: {sgdm_momentum_range}\n")
        
        

if __name__ == "__main__":
    main()