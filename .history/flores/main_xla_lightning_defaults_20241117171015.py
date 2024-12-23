import torch
import pickle
import os
import wandb
import numpy as np
import nltk
import torch_optimizer as t_optim
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text.bert import BERTScore
from transformers import DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from torchmetrics import MeanMetric
import argparse
from dotenv import load_dotenv
from icecream import ic

load_dotenv()           # This is required for the .env file
nltk.download('punkt_tab')  # This is required for BERTScore to run.
os.environ["TOKENIZERS_PARALLELISM"] = 'false'  # This is required in order not to have Race conditions in TPUs.
wandb.require("core")   # This is required for W&B to work in future versions.

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
bert_score_model_to_use = "microsoft/deberta-large-mnli"
max_length = 512
dataset_name = "facebook/flores"
train_range = 1000
test_range = 1000
val_range = 1000
epochs = 5

class T5TranslationModule(pl.LightningModule):
    def __init__(self, model_name, optimizer_name="adamw", generation_max_tokens=20, bert_score_model_to_use="microsoft/deberta-large-mnli"):        
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).train()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        # self.optimizer_params = optimizer_params
        self.val_loss = MeanMetric() # This line to create a metric for tracking validation loss
        self.generation_max_tokens = generation_max_tokens
        self.valid_step_outputs = []
        self.test_step_outputs = []
        self.bert_score_model_to_use = bert_score_model_to_use

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
            ic(generated_seq)
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
        we have to do this in the training loop in order to run in the TPU, which means, it runs faster.
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
        ic(all_preds)
        ic(all_labels)
        with torch.no_grad():
            self._log_metrics(prefix, all_preds, all_labels)
        
    def _log_metrics(self, prefix, predictions, labels):
        """
        Log the metrics for the predictions and labels based on the prefix.
        """
        metrics = self._compute_metrics(predictions, labels)
        ic(metrics)
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
            return torch.optim.AdamW(self.parameters())
        elif self.optimizer_name == "sgd":
            return torch.optim.SGD(self.parameters())
        elif self.optimizer_name == "sgdm": # Momentum will be added later in the '**self.optimizer_params' kwargs
            # default momentum is 0.9 for sgdm
            return torch.optim.SGD(self.parameters(), momentum=0.9)
        elif self.optimizer_name == "adam":
            return torch.optim.Adam(self.parameters())
        elif self.optimizer_name == "adamax":
            return torch.optim.Adamax(self.parameters())
        elif self.optimizer_name == "nadam":
            return torch.optim.NAdam(self.parameters())
        elif self.optimizer_name == "adagrad":
            return torch.optim.Adagrad(self.parameters())
        elif self.optimizer_name == "adadelta":
            return torch.optim.Adadelta(self.parameters())
        elif self.optimizer_name == "adabound":
            return t_optim.AdaBound(self.parameters())
        elif self.optimizer_name == "rmsprop":
            return torch.optim.RMSprop(self.parameters())
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
        
class T5TranslationDataModule(pl.LightningDataModule):
    def __init__(self, model_name, dataset_name, max_length, 
                 batch_size, train_range, val_range, test_range, seed_num,
                 languages = ["deu_Latn", "fra_Latn", "ron_Latn"]):
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
                # TODO: Check if this is correct, to use temp1, temp2
                print(f"Processing {split} dataset for {language}...")
                dataset = load_dataset(self.dataset_name, 'all',  trust_remote_code=True)
                temp1 = dataset['dev']
                temp2 = dataset['devtest']
                # concat the two splits, and get 80% train and 10% test and 10% validation
                # total dataset
                dataset = concatenate_datasets([temp1, temp2]).train_test_split(test_size=0.2, seed=self.seed_num, shuffle=True)
                
                if split == 'train':
                    data = dataset['train']
                elif split == 'validation':
                    # Split again into train and test, and select train as the validation set
                    # 10% of the total dataset is selected as the validation set
                    selected_data = dataset['test'].train_test_split(test_size=0.5, seed=self.seed_num, shuffle=True)['train']
                    data = selected_data.select(range(min(self.test_range, len(selected_data))))
                    
                elif split == 'test':
                    # Split again into train and test, and select test as the test set
                    # 10% of the total dataset is selected as the test set
                    selected_data = dataset['test'].train_test_split(test_size=0.5, seed=self.seed_num, shuffle=True)['test']
                    data = selected_data.select(range(min(self.test_range, len(selected_data))))
                
                
                processed_dataset = self._preprocess_dataset(data, language)
                
                os.makedirs(self.cache_dir, exist_ok=True)
                with open(cache_file, 'wb') as f:
                    pickle.dump(processed_dataset, f)
                
                dataset = processed_dataset
            
            combined_dataset.extend(dataset)
        
        return combined_dataset
    
    def _preprocess_dataset(self, dataset, target_language_code):
        """
        Preprocess the dataset by mapping the language codes to their corresponding names.
        For example, "deu_Latn" maps to "German" in the German dataset. 
        This is given in the link: 
        https://github.com/facebookresearch/flores/blob/main/flores200/README.md
        
        The function also maps the target language code to the corresponding name.
        The function returns the preprocessed dataset.

        Args:
            dataset (_dataset_): The dataset to preprocess. 
            target_language_code (str): The target language code.

        Returns:
            _dataset_: The preprocessed dataset.
        """
        mapping = {
            "deu_Latn": "German",
            "fra_Latn": "French",
            "ron_Latn": "Romanian"
        }
        target_lang_name = mapping[target_language_code]

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
                target_text = examples[f'sentence_{target_language_code}'][i]
                
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

def main(seed, optimizer_name, batch_size, training_mode="Default Hyperparams"):
    ic.disable()
    print(f"Training with seed {seed}, optimizer {optimizer_name}, batch size {batch_size}")
    print(f"Training mode: {training_mode}")

    # Optimizer-specific hyperparameter filtering
    optimizer_name_lower = optimizer_name.lower()
    # filtered_params = {}

    # if optimizer_name_lower in ['adam', 'adamw', 'adamax', 'nadam']:
    #     # Common parameters for Adam-like optimizers
    #     if "betas" in optimizer_params:
    #         filtered_params["betas"] = optimizer_params["betas"]
    #     if "eps" in optimizer_params:
    #         filtered_params["eps"] = optimizer_params["eps"]
    # if optimizer_name_lower in ['sgd', 'sgdm']:
    #     # Parameters specific to SGD and SGDM
    #     if "momentum" in optimizer_params:
    #         filtered_params["momentum"] = optimizer_params["momentum"]
    # if optimizer_name_lower == 'rmsprop':
    #     # Parameters specific to RMSprop
    #     if "alpha" in optimizer_params:
    #         filtered_params["alpha"] = optimizer_params["alpha"]
    #     if "momentum" in optimizer_params:
    #         filtered_params["momentum"] = optimizer_params["momentum"]
    # if optimizer_name_lower == 'nadam':
    #     # NAdam-specific parameter
    #     if "momentum_decay" in optimizer_params:
    #         filtered_params["momentum_decay"] = optimizer_params["momentum_decay"]

    # # Add weight_decay if provided (common to all optimizers)
    # if "weight_decay" in optimizer_params:
    #     filtered_params["weight_decay"] = optimizer_params["weight_decay"]

    # Log filtered parameters
    # for key, value in filtered_params.items():
    #     print(f"Using hyperparameter: {key} = {value}")

    pl.seed_everything(seed)
    model = T5TranslationModule(
        model_name=model_name,
        # learning_rate=learning_rate,
        optimizer_name=optimizer_name,
        # **optimizer_params
    )

    data_module = T5TranslationDataModule(
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
    wandb.finish()  # In case the last run crashed, this will close the previous run
    wandb_logger = WandbLogger(project="t5_translation_project",
                               name=f"{model_name}_{optimizer_name}_seed_{seed}_defaults")

    checkpoint_callback = ModelCheckpoint(dirpath= f"checkpoints/{model_name}_{optimizer_name}_seed_{seed}_defaults", 
                                          monitor="val_loss", 
                                          mode="min",
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

    hyperparameters = dict(# learning_rate=learning_rate, 
                           optimizer_name=optimizer_name, 
                           seed_num=seed, 
                           dataset_name=dataset_name, 
                           model_name=model_name, 
                           max_length=max_length, 
                           batch_size=batch_size, 
                           train_range=train_range, 
                           val_range=val_range, 
                           test_range=test_range,
                           bert_score_model_used=bert_score_model_to_use,
                           training_mode=training_mode,)
                           # **optimizer_params)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=data_module)

    trainer.test(datamodule=data_module, ckpt_path="best")
    wandb.finish()
    print(f"\nFinished training with seed {seed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="Seed number for reproducibility")
    parser.add_argument("--optim", type=str, required=True, help="Optimizer to use for training")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for training")
    # parser.add_argument("--learning_rate", type=float, required=True, help="Learning rate for training")
    parser.add_argument("--training_mode", type=str, default="Default Hyperparams", help="Training mode")
    
    # Add arguments for all possible optimizer parameters
    # parser.add_argument("--betas", nargs=2, type=float, help="Beta parameters for Adam-like optimizers")
    # parser.add_argument("--eps", type=float, help="Epsilon parameter for optimizers")
    # parser.add_argument("--momentum", type=float, help="Momentum parameter for SGD")
    # parser.add_argument("--alpha", type=float, help="Alpha parameter for RMSprop")
    # # parser.add_argument("--weight_decay", type=float, help="Weight decay parameter")
    # parser.add_argument("--momentum_decay", type=float, help="Momentum decay for NAdam")
    
    args = parser.parse_args()
    
    # Convert args to dictionary and remove None values
    # optimizer_params = {k: v for k, v in vars(args).items() if k not in ["seed", "optim", "batch_size", "learning_rate", "training_mode"] and v is not None}
    
    # Convert betas tuple to list if it exists
    # if "betas" in optimizer_params:
    #     optimizer_params["betas"] = tuple(optimizer_params["betas"])
    
    main(args.seed, args.optim, args.batch_size, args.training_mode)