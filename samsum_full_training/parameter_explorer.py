import os
from main_xla_lightning import main as train_model  # Import the main function from main.py

def read_best_hyperparameters(base_path='./hypertuning_results_full_training/google-t5_t5-small/'):
    hyperparams = {}
    training_mode = "full_training_mode"
    
    # Mapping optimizer-specific names to generic parameter names to be used in the training with the correct argument names
    param_mapping = {
        'adam': {'beta1': 'betas[0]', 'beta2': 'betas[1]', 'epsilon': 'eps'},
        'adamax': {'beta1': 'betas[0]', 'beta2': 'betas[1]', 'epsilon': 'eps'},
        'adamw': {'beta1': 'betas[0]', 'beta2': 'betas[1]', 'epsilon': 'eps'},
        'nadam': {'beta1': 'betas[0]', 'beta2': 'betas[1]', 'epsilon': 'eps', 'momentum_decay': 'momentum_decay'},
        'sgdm': {'momentum': 'momentum'},
        # Add mappings for other optimizers if needed
    }
    
    if not os.path.exists(base_path):   # Depends on the directory structure
        base_path = "./hypertuning_results_lr_tuning/google-t5_t5-small/"
        training_mode = "lr_only_training_mode"

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
                    
                    hyperparam_dict = {}
                    in_best_hyperparameters = False
                    for line in lines:
                        line = line.strip()
                        if line == "Best Hyperparameters:":
                            in_best_hyperparameters = True
                        elif in_best_hyperparameters:
                            if line == "Search Spaces:":
                                break  # Stop if we reach the Search Spaces section
                            if ":" in line:
                                key, value = line.split(":")
                                key = key.strip()
                                value = float(value.strip())
                                
                                # Extract the base key name (e.g., 'adam_beta1' -> 'beta1')
                                base_key = key.split('_')[-1]
                                
                                # Map to the correct generic parameters if the optimizer is in the mapping
                                if optimizer in param_mapping and base_key in param_mapping[optimizer]:
                                    mapped_key = param_mapping[optimizer][base_key]
                                    if 'betas' in mapped_key:
                                        # Handle tuple for betas
                                        if mapped_key == 'betas[0]':
                                            hyperparam_dict['betas'] = (value, hyperparam_dict.get('betas', (None, None))[1])
                                        elif mapped_key == 'betas[1]':
                                            hyperparam_dict['betas'] = (hyperparam_dict.get('betas', (None, None))[0], value)
                                    else:
                                        hyperparam_dict[mapped_key] = value
                                else:
                                    hyperparam_dict[key] = value
                    
                    if hyperparam_dict:
                        hyperparams[optimizer][seed_dir] = hyperparam_dict
    return hyperparams

def clean_checkpoints():
    print("Cleaning checkpoints...")
    if os.path.exists("./checkpoints"):
        os.system("rm -r ./checkpoints/*")
    if os.path.exists("./checkpoints_full_training"):
        os.system("rm -r ./checkpoints_full_training/*")
    print("Finished cleaning checkpoints")

def main():
    hyperparams_per_optimizer, training_mode = read_best_hyperparameters()
    batch_size = 16  # Add or modify batch sizes as needed
    
    for optimizer_name, seeds_data in hyperparams_per_optimizer.items():
        print(f"\nExploring models for optimizer: {optimizer_name}\n")
        
        for seed_dir, params in seeds_data.items():
            seed = int(seed_dir.split('_')[1])
            learning_rate = params.pop('learning_rate', None)
            # Debugging: Check if 'learning_rate' is still in params
            if 'learning_rate' in params:
                print(f"Error: 'learning_rate' is still in params for seed {seed} and optimizer {optimizer_name}")
            if learning_rate is None:
                print(f"Skipping seed {seed} due to missing learning rate.")
                continue
            # Directly call the main function from main.py, explicitly passing the learning rate and other params
            train_model(seed, optimizer_name, batch_size, learning_rate, training_mode=training_mode, **params)
            clean_checkpoints()
        
        print(f"Finished exploring all configurations for optimizer: {optimizer_name}\n")

    print("Finished exploring all optimizers")

if __name__ == "__main__":
    main()

