import os
from main_xla_lightning import main as train_model  # Import the main function from main.py

def read_best_hyperparameters(base_path='./hypertuning_results_full_training/google-t5_t5-small/'):
    """
    The read_best_hyperparameters function now reads all hyperparameters under the "Best Hyperparameters" section and adds them to a dictionary.
    Each key-value pair represents a hyperparameter and its corresponding value.
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
                                value = value.strip()
                                if key and value:
                                    hyperparam_dict[key] = float(value)
                    
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
    hyperparams_per_optimizer = read_best_hyperparameters()
    batch_size = 4  # Add or modify batch sizes as needed
    
    for optimizer_name, seeds_data in hyperparams_per_optimizer.items():
        print(f"\nExploring models for optimizer: {optimizer_name}\n")
        
        for seed_dir, params in seeds_data.items():
            seed = int(seed_dir.split('_')[1])
            learning_rate = params.pop('learning_rate', None)
            
            if learning_rate is None:
                print(f"Skipping seed {seed} due to missing learning rate.")
                continue
            
            print(f"Running training with seed {seed}, optimizer {optimizer_name}, batch size {batch_size}, and hyperparameters: {params}")
            print("Additional params: ", params)
            
            # Directly call the main function from main.py
            train_model(seed, optimizer_name, batch_size, learning_rate, **params)
            
            clean_checkpoints()
        
        print(f"Finished exploring all configurations for optimizer: {optimizer_name}\n")

    print("Finished exploring all optimizers")

if __name__ == "__main__":
    main()
