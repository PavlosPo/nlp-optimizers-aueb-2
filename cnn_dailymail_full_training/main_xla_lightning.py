import torch
import pickle
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from torchmetrics import MeanMetric
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text.bert import BERTScore
import torch_optimizer as t_optim
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
dataset_name = "cnn_dailymail"
seed_num = args.seed
train_range = 35000
test_range = 3500
val_range = 3500
epochs = 5
n_trials = 30
batch_size = args.batch_size

class T5SummarizationModule(pl.LightningModule):
    def __init__(self, model_name, learning_rate, optimizer_name="adamw", generation_max_tokens=20, **optimizer_params):        
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).train()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.optimizer_params = optimizer_params
        self.val_loss = MeanMetric() # This line to create a metric for tracking validation loss
        self.generation_max_tokens = generation_max_tokens
        self.valid_step_outputs = []
        self.test_step_outputs = []
        self.bert_score_model_to_use = 'microsoft/deberta-xlarge-mnli' # Current Best Model closest to Human Evaluation

    def forward(self, input_ids, attention_mask, labels=None, predict_with_generate=False):
        """
        Forward pass of the model
        Args:
            input_ids: Input token ids of shape (batch_size, sequence_length)
            attention_mask: Attention mask of shape (batch_size, sequence_length)
            labels: Labels of shape (batch_size, target_sequence_length)
            predict_with_generate: Whether to use the model to generate predictions
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        if predict_with_generate:
            generated = self.model.generate(input_ids=input_ids, 
                                                       attention_mask=attention_mask, 
                                                       max_length=self.generation_max_tokens + 2)
            outputs['sequences'] = generated.view(input_ids.shape[0], -1)  # Reshape to match input shape
        return outputs
        
    def training_step(self, batch, batch_idx):
        outputs = self.forward(input_ids=batch["input_ids"], 
                               attention_mask=batch["attention_mask"], 
                               labels=batch["labels"])
        loss = outputs['loss']
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        return loss
    
    def on_train_epoch_end(self):
        pass
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            outputs = self.forward(input_ids=batch["input_ids"],
                                   attention_mask=batch["attention_mask"],
                                   labels=batch["labels"],
                                   predict_with_generate=True)
            loss = outputs['loss']
            self.val_loss.update(loss)  # Update the metric for the epoch validation loss
            
            generated_seq = outputs['sequences'].view(-1, outputs['sequences'].size(-1))  # Flatten if needed
            self.valid_step_outputs.append((generated_seq, batch["labels"])) # Store the outputs for evaluation
        return loss
    
    def on_validation_epoch_end(self):
        avg_loss = self.val_loss.compute() # Compute the mean validation loss for the epoch
        # Log the epoch validation loss
        self.log("val_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_loss.reset() # Reset the metric for the next epoch
        
        self._initialize_metrics()
        self._eval_epoch_end(self.valid_step_outputs, "val") # Evaluate the model on the validation set
        self.valid_step_outputs.clear() # Clear the outputs for the next epoch

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            outputs = self.forward(input_ids=batch["input_ids"], 
                           attention_mask=batch["attention_mask"], 
                           labels=batch["labels"], 
                           predict_with_generate=True)
            generated_seq = outputs['sequences'].view(-1, outputs['sequences'].size(-1))  # Flatten if needed
            loss = outputs['loss']
            self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.test_step_outputs.append((generated_seq, batch["labels"])) # Store the outputs for evaluation
        return loss
    
    def on_test_epoch_end(self):
        self._initialize_metrics()
        self._eval_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()
        
    def _initialize_metrics(self):
        """
        Initialize the metrics used for evaluation.
        Because the BERTScore can not pushed to the TPU if it is in the __init__ method, 
        we have to do this in the training loop in order to run in the TPU.
        """
        if not hasattr(self, "rouge_score"):
            self.rouge_score = ROUGEScore(use_stemmer=True, sync_on_compute=True)
        if not hasattr(self, "bert_score"):
            self.bert_score = BERTScore(model_name_or_path=self.bert_score_model_to_use,
                                        sync_on_compute=True, device=self.device)
    
    def _eval_epoch_end(self, outputs, prefix):
        """
        Evaluate the model on the validation set and log the metrics.
        This should be called at the end of the validation/test epoch.
        """
        all_preds = torch.cat([x[0] for x in outputs], dim=0)
        all_labels = torch.cat([x[1] for x in outputs], dim=0)
        with torch.no_grad():
            self._log_metrics(prefix, all_preds, all_labels)
        
    def _log_metrics(self, prefix, predictions, labels):
        """
        Log the metrics for the predictions and labels based on the prefix.
        """
        metrics = self._compute_metrics(predictions, labels)
        self.log_dict({f"{prefix}_{k}": v for k, v in metrics.items()}, 
                      on_step=False, on_epoch=True, sync_dist=True)
        
    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.
        """
        optimizer = self._get_optimizer()
        return optimizer
    
    def _compute_metrics(self, predictions, labels):
        """
        Helper function to Compute the metrics for the predictions and labels.
        
        Args:
            predictions: The predictions from the model.
            labels: The labels for the predictions.
        Returns:
            The metrics for the predictions and labels as a dictionary.
        """
        if isinstance(predictions, list):
           predictions = torch.cat(predictions, dim=0)
        if isinstance(labels, list):
            labels = torch.cat(labels, dim=0)
        
        predictions = predictions.cpu().numpy() if torch.is_tensor(predictions) else predictions
        labels = labels.cpu().numpy() if torch.is_tensor(labels) else labels

        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)        
        processed_labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(processed_labels, skip_special_tokens=True)
        result_rouge = self.rouge_score(preds=decoded_preds, target=decoded_labels)
        result_brt = self.bert_score(preds=decoded_preds, target=decoded_labels)
        result_brt_average_values = {key: torch.tensor(tensors.mean().item()) for key, tensors in result_brt.items()}
        results = {**result_rouge, **result_brt_average_values}
        return results

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
      
      
def read_best_hyperparameters(base_path='./hyper_tuning_results_lr_tuning/google-t5_t5-small/'):
    """
    Reads the best hyperparameters from the hyper_tuning_results directory for each optimizer for each seed.
    Returns a dictionary of the form {optimizer: {seed: learning_rate}}
    """
    hyperparams = {}
    for optimizer in os.listdir(base_path):
        optimizer_path = os.path.join(base_path, optimizer)
        if os.path.isdir(optimizer_path):
            hyperparams[optimizer] = {}
            for seed_dir in os.listdir(optimizer_path):
                seed_path = os.path.join(optimizer_path, seed_dir)
                if os.path.isdir(seed_path):
                    hyperparams_file = os.path.join(seed_path, 'best_hyperparameters.txt')
                    with open(hyperparams_file, 'r') as f:
                        lines = f.readlines()
                    learning_rate = None
                    for line in lines:
                        if "learning_rate" in line:
                            learning_rate = float(line.split(":")[1].strip())
                    if learning_rate:
                        hyperparams[optimizer][seed_dir] = {'learning_rate': learning_rate}
    return hyperparams

def main():
    hyperparams_per_optimizer = read_best_hyperparameters()
    for optimizer_name, seeds_data in hyperparams_per_optimizer.items():
        print(f"\nTraining models for optimizer: {optimizer_name}\n")
        
        for seed_dir, params in seeds_data.items():
            current_learning_rate = params['learning_rate']
            print(f"Training with seed {seed_dir} with learning rate {current_learning_rate}")
            
            pl.seed_everything(int(seed_dir.split('_')[1]))
            model = T5SummarizationModule(
                model_name=model_name,
                learning_rate=current_learning_rate,
                optimizer_name=optimizer_name,
            )
            
            data_module = T5SummarizationDataModule(
                model_name=model_name,
                dataset_name=dataset_name,
                max_length=max_length,
                batch_size=batch_size,
                train_range=train_range,
                val_range=val_range,
                test_range=test_range,
                seed_num=int(seed_dir.split('_')[1])
            )
            
            logger = TensorBoardLogger("tb_logs", 
                                      name=f"{model_name}_{optimizer_name}_seed_{seed_dir}")
            
            checkpoint_callback = ModelCheckpoint(dirpath= f"checkpoints/{model_name}_{optimizer_name}_seed_{seed_dir}", 
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
            
            hyperparameters = dict(learning_rate=current_learning_rate, 
                                   optimizer_name=optimizer_name, 
                                   seed_num=seed_dir, 
                                   dataset_name=dataset_name, 
                                   model_name=model_name, 
                                   max_length=max_length, 
                                   batch_size=batch_size, 
                                   train_range=train_range, 
                                   val_range=val_range, 
                                   test_range=test_range)
            trainer.logger.log_hyperparams(hyperparameters)
            trainer.fit(model, datamodule=data_module)
            
            trainer.test(model, datamodule=data_module)
            # Log test results to TensorBoard
            for key, value in trainer.callback_metrics.items():
                if key.startswith("test_"):
                    trainer.logger.experiment.add_scalar(f"test_{key}", value, global_step=trainer.global_step)
            print(f"Finished training with seed {seed_dir}\n")
        
        print(f"Finished training all seeds for optimizer: {optimizer_name}\n")

if __name__ == "__main__":
    main()