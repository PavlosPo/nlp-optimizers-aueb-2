import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
import matplotlib.lines as mlines
import shutil

def sample_data(data, data_type, sample_rate=0.1 / 5.0):
    """
    Sample data points based on data type:
    - For training data: Take every Nth point to reduce density
    - For validation data: Keep all points
    """
    if data_type == 'train':
        n = max(1, int(1/sample_rate))
        return data.iloc[::n].copy()
    return data

def create_legend_seperately(plt, ax, optimizers, styles, output_legend_filename="optimizers_styles_legend_20epochs.png"):
    # Create a standalone legend
    fig, ax = plt.subplots(figsize=(7, 0.5))
    ax.axis('off')

    legend_handles = [
        mlines.Line2D(
            [], [], 
            color=styles[optimizer]["color"], 
            linestyle=styles[optimizer]["linestyle"], 
            marker=styles[optimizer]["marker"], 
            label=optimizer.upper()
        )
        for optimizer in optimizers
    ]

    legend = ax.legend(
        handles=legend_handles, 
        loc='center', 
        frameon=False, 
        ncol=len(optimizers),
    )

    output_filename = "optimizers_styles_legend_20epochs_horizontal.png"
    fig.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Legend saved as {output_filename}")

def plot_csv(file_path, output_path, optimizers, styles):
    data = pd.read_csv(file_path)
    
    filename = os.path.basename(file_path)
    parts = filename.split("_")
    if len(parts) < 4:
        print(f"Filename {filename} does not match expected format. Skipping.")
        return
    
    # Handle both Rouge metrics and other metrics
    if len(parts) == 6:  # For files like dataset_val_rouge_1_basic_20.csv
        dataset_name, data_type, metric, metric_num, mode, epochs = parts[:6]
        metric = metric + metric_num + "_fmeasure"
    else:  # For files like dataset_val_loss_full_20.csv
        dataset_name, data_type, metric, mode, epochs = parts[:5]

    y_axis_label = metric.capitalize().replace("F1", "F1 Score").replace("loss", "Loss").replace("Rouge2_fmeasure", "ROUGE-2").replace("Rouge1_fmeasure", "ROUGE-1")
    
    sampled_data = sample_data(data, data_type)

    width, height = figaspect(0.5)
    plt.figure(figsize=(width, height))
    plt.rcParams['axes.spines.left'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.bottom'] = False

    optimizer_label_mapping = {
        "adamw": "AdamW",
        "nadam": "NAdam",
        "adam": "Adam",
        "sgdm": "SGDM",
        "sgd": "SGD",
    }
    
    all_y_values = []
    for optimizer in optimizers:
        mean_col = f"optimizer_name: {optimizer} - {data_type}_{metric}"
        min_col = f"optimizer_name: {optimizer} - {data_type}_{metric}__MIN"
        max_col = f"optimizer_name: {optimizer} - {data_type}_{metric}__MAX"
        
        if mean_col not in sampled_data.columns:
            print(f"Missing column {mean_col} in {file_path}. Skipping {optimizer}.")
            continue

        mean_values = sampled_data[mean_col]
        std_dev = (sampled_data[max_col] - sampled_data[min_col]) / 2 if max_col in sampled_data.columns else 0
        all_y_values.extend(mean_values)

        plt.errorbar(
            sampled_data["Step"],
            mean_values,
            color=styles[optimizer]["color"],
            linestyle=styles[optimizer]["linestyle"],
            marker=styles[optimizer]["marker"],
            capsize=3,
            label=optimizer_label_mapping[optimizer.lower()]
        )
        plt.fill_between(sampled_data["Step"], mean_values - std_dev, mean_values + std_dev, 
                        alpha=0.2, color=styles[optimizer]["color"])
    
    plt.xlabel("Step", fontsize=24)
    plt.ylabel(y_axis_label, fontsize=24)
    if data_type == "train":
        plt.ylim(0, 2)
        if dataset_name == "flores":
            plt.ylim(0, 20)
            
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)    
    plt.legend(fontsize=12, loc='best')
    plt.tight_layout()
    
    output_file = os.path.join(output_path, filename.replace(".csv", ".pdf"))
    plt.savefig(output_file, dpi=300, format='pdf', bbox_inches='tight')
    print(f"Plot saved as: {output_file}")
    plt.close()

def process_20epochs_files(input_folder, output_folder, optimizers, styles):
    """
    Process only files ending with '20.csv' and organize them into '20epochs' subfolders
    """
    for root, _, files in os.walk(input_folder):
        # Filter for files ending with '20.csv'
        epoch20_files = [f for f in files if f.endswith('20.csv')]
        
        if epoch20_files:
            dataset_name = os.path.basename(root)
            # Create 20epochs subfolder in the dataset's output folder
            output_path = os.path.join(output_folder, dataset_name, '20epochs')
            os.makedirs(output_path, exist_ok=True)
            
            # Copy the 20-epoch CSV files to the new subfolder
            for file in epoch20_files:
                file_path = os.path.join(root, file)
                # Copy the CSV file to the 20epochs subfolder
                dest_path = os.path.join(output_path, file)
                shutil.copy2(file_path, dest_path)
                print(f"Copied {file} to {dest_path}")
                
                # Generate plot
                plot_csv(file_path, output_path, optimizers, styles)

if __name__ == "__main__":
    folder_of_model = "t5-small"
    input_folders = ["cnn", "flores", "xsum", "samsum", "iwslt"]
    input_folders = [os.path.join(folder_of_model, folder) for folder in input_folders]
    output_folder = os.path.join(folder_of_model, "plots")

    optimizers = ["adamw", "nadam", "adam", "sgdm", "sgd"]
    styles = {
        "adamw": {"color": "#3D9E3E", "linestyle": "-", "marker": "o"},
        "nadam": {"color": "#338DD8", "linestyle": "--", "marker": "s"},
        "adam": {"color": "#ECBB33", "linestyle": ":", "marker": "^"},
        "sgdm": {"color": "#C1433C", "linestyle": "-.", "marker": "v"},
        "sgd": {"color": "#DF672A", "linestyle": "-", "marker": "d"},
    }

    for input_folder in input_folders:
        print(f"Processing 20-epoch files in folder: {input_folder}")
        process_20epochs_files(input_folder, output_folder, optimizers, styles)

    create_legend_seperately(plt, plt.gca(), optimizers, styles)