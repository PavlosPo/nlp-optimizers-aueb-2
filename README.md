# NLP Optimizers in Deep Learning

This repository contains the code and experiments conducted for my MSc thesis in Computer Science at AUEB University (2024). The thesis explores optimization algorithms in deep learning, focusing on Natural Language Processing (NLP) tasks.

## Motivation and Problem Statement

In recent years, deep learning has introduced various optimization algorithms, with Adam (Adaptive Moment Optimization algorithm) emerging as a popular choice since Kingma and Ba introduced it in 2014. Hundreds of optimization methods are now available, each claiming to offer improvements for specific tasks or architectures. This plethora of options makes selecting an optimizer a crucial decision in deep learning projects.

While Gkouti et al. (2024) compared optimizer performance for encoder-only models in classification tasks, questions remain regarding the generalizability of their findings across different models and tasks. Their research suggested that tuning the learning rate alone can yield similar results to tuning all hyperparameters, with SGDM outperforming SGD. However, these conclusions were drawn from a limited set of deep learning applications.

This thesis expands on Gkouti et al.’s work by broadening the analysis in several key areas:

1. **Model Architectures**: We extend the analysis to include encoder-decoder models, essential for sequence-to-sequence tasks.
2. **Task Diversity**: Beyond classification tasks, we explore text summarization to see if optimization dynamics hold true for generative tasks.
3. **Dataset Variability**: We use multiple datasets across different domains to avoid biases from domain-specific characteristics.
4. **Optimizer Comparison**: The comparison between adaptive optimizers (Adam and variants) and non-adaptive methods (SGD, SGDM) is revisited under these expanded conditions.

The thesis aims to critically assess whether Gkouti et al.‘s principles—particularly regarding learning rate tuning and SGDM’s superiority over SGD—apply across a broader spectrum of applications. The main research questions are:

- Can tuning just the learning rate be nearly as effective as tuning all hyperparameters in more complex models and tasks?
- Do adaptive optimizers like Adam provide significant performance advantages over SGDM in tasks such as text summarization?
- How do optimization dynamics vary across different datasets and domains?

By answering these questions, this research provides insights into optimizer selection and hyperparameter tuning strategies. It may simplify the model development process by identifying general optimization principles across various deep learning scenarios.
## Further Explanations

Each dataset has its own folder. Additionally, there are `**_full_training` folders (where `**` represents the dataset's name). These folders contain code optimized for hyperparameter tuning of more than one parameter, unlike the simpler folders that only focus on learning rate tuning. Inside each folder, you will find three Python scripts:

- **`main_xla_lightning.py`**: This script handles the training, validation, testing, and reporting of the model. It also saves and uploads the results for presentation.
  
- **`tune_xla_lightning.py`**: This script is responsible for hyperparameter tuning. It trains and validates the model with different hyperparameter configurations using the Optuna framework. The best hyperparameters are then saved in `.txt` files, which include information such as the dataset range, optimizer used, and additional details.

- **`parameter_explorer_creation_of_bash_script.py`**: This script automatically generates a bash script called `run_experiments.sh`. The bash script uses the best hyperparameters found by `tune_xla_lightning.py` to run and report the final experiments (uploading the results to WandB). It streamlines the process of running the best results with minimal manual effort.

### Extra Bash Scripts

In addition to the main scripts, there are extra bash scripts such as `run_script_adam.sh` and `run_script_sgdm.sh`. These scripts automate the process of running hyperparameter searches across different seeds, optimizers (e.g., Adam, SGDM), batch sizes, and other variables. They allow for more efficient and organized tuning across various configurations.

### Optuna Database Files

If you find any `.db` files in the folders, these are created by the Optuna framework during hyperparameter tuning. While they aren't essential for running the experiments, they are retained in some folders for reference.

### Reported Results

The best hyperparameters are saved in either the `hypertuning_results_lr_tuning` folders (for learning rate tuning) or the `hypertuning_results_full_training` folders (where more than just the learning rate was tuned). These results reflect the optimal configurations discovered through the experiments.