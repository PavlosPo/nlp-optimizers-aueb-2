import os
from main_xla_lightning import main as train_model  # Import the main function from main.py

def read_best_hyperparameters(base_path='./hypertuning_results_full_training/google-t5_t5-small/'):
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
                    in_best_hyperparameters = False
                    for line in lines:
                        if line.strip() == "Best Hyperparameters:":
                            in_best_hyperparameters = True
                        elif in_best_hyperparameters and "learning_rate:" in line:
                            learning_rate = float(line.split(":")[1].strip())
                            break  # Stop after finding the correct learning rate
                        elif line.strip() == "Search Spaces:":
                            break  # Stop if we reach the Search Spaces section
                    
                    if learning_rate is not None:
                        hyperparams[optimizer][seed_dir] = {'learning_rate': learning_rate}
    return hyperparams

def main():
    hyperparams_per_optimizer = read_best_hyperparameters()
    batch_size = 4  # Add or modify batch sizes as needed
    
    for optimizer_name, seeds_data in hyperparams_per_optimizer.items():
        print(f"\nExploring models for optimizer: {optimizer_name}\n")
        
        for seed_dir, params in seeds_data.items():
            seed = int(seed_dir.split('_')[1])
            learning_rate = params['learning_rate']
            
            print(f"Running training with seed {seed}, optimizer {optimizer_name}, batch size {batch_size}, and learning rate {learning_rate}")
            
            # Directly call the main function from main.py
            train_model(seed, optimizer_name, batch_size, learning_rate)
        
        print(f"Finished exploring all configurations for optimizer: {optimizer_name}\n")

if __name__ == "__main__":
    main()