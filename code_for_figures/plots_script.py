import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect

def sample_data(data, data_type, sample_rate=0.1 / 5.0):
    """
    Sample data points based on data type:
    - For training data: Take every Nth point to reduce density
    - For validation data: Keep all points
    """
    if data_type == 'train':
        n = max(1, int(1/sample_rate))
        return data.iloc[::n].copy()
    return data  # Return all data for validation

# def adjustFigAspect(fig,aspect=1):
#     '''
#     Adjust the subplot parameters so that the figure has the correct
#     aspect ratio.
#     '''
#     xsize,ysize = fig.get_size_inches()
#     minsize = min(xsize,ysize)
#     xlim = .4*minsize/xsize
#     ylim = .4*minsize/ysize
#     if aspect < 1:
#         xlim *= aspect
#     else:
#         ylim /= aspect
#     fig.subplots_adjust(left=.5-xlim,
#                         right=.5+xlim,
#                         bottom=.5-ylim,
#                         top=.5+ylim)

def plot_csv(file_path, output_path):
    # Load data
    data = pd.read_csv(file_path)
    
    # Infer data type and metric from filename
    filename = os.path.basename(file_path)
    parts = filename.split("_")
    if len(parts) < 4:
        print(f"Filename {filename} does not match expected format. Skipping.")
        return
    dataset_name, data_type, metric, mode = parts[:4]

    # Prepare plot labels
    y_axis_label = metric.capitalize().replace("f1", "F1 Score").replace("loss", "Loss")
    
    # Sample data
    sampled_data = sample_data(data, data_type)

    # Optimizers, colors, and styles
    optimizers = ["adamw", "nadam", "adam", "sgdm", "sgd"]
    styles = {
        "adamw": {"color": "#3D9E3E", "linestyle": "-", "marker": "o"},
        "nadam": {"color": "#338DD8", "linestyle": "--", "marker": "s"},
        "adam": {"color": "#ECBB33", "linestyle": ":", "marker": "^"},
        "sgdm": {"color": "#C1433C", "linestyle": "-.", "marker": "v"},
        "sgd": {"color": "#DF672A", "linestyle": "-", "marker": "d"},
    }

    # Create plot
    width, height = figaspect(0.5)
    plt.figure(figsize=(width, height))
    #remove frame from each side of plot
    plt.rcParams['axes.spines.left'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.bottom'] = False
    
    # Initialize y-values for percentile calculation
    all_y_values = []
    for optimizer in optimizers:
        mean_col = f"optimizer_name: {optimizer} - {data_type}_{metric}"
        min_col = f"optimizer_name: {optimizer} - {data_type}_{metric}__MIN"
        max_col = f"optimizer_name: {optimizer} - {data_type}_{metric}__MAX"

        # Check if columns exist
        if mean_col not in sampled_data.columns:
            print(f"Missing column {mean_col} in {file_path}. Skipping {optimizer}.")
            continue

        # Extract data
        mean_values = sampled_data[mean_col]
        std_dev = (sampled_data[max_col] - sampled_data[min_col]) / 2 if max_col in sampled_data.columns else 0
        all_y_values.extend(mean_values)

        # Plot
        plt.errorbar(
            sampled_data["Step"],
            mean_values,
            # yerr=std_dev,
            label=optimizer.upper(),
            color=styles[optimizer]["color"],
            linestyle=styles[optimizer]["linestyle"],
            marker=styles[optimizer]["marker"],
            capsize=3,
        )
        plt.fill_between(sampled_data["Step"], mean_values - std_dev, mean_values + std_dev, alpha=0.2, color=styles[optimizer]["color"])
    
    # Calculate quartiles for the y-axis
    quartiles = np.percentile(all_y_values, [25, 50, 75, 100])
    # print(quartiles)
    
    
    # Customize plot
    plt.xlabel("Step", fontsize=24)
    plt.ylabel(y_axis_label, fontsize=24, )
    if data_type == "train":
        plt.ylim(0, 2)  # Limit y-axis for training data
        # If this is the flores dataset
        if dataset_name == "flores":
            plt.ylim(0, 20)
            
    plt.yticks(fontsize=24)  # Set quartile values as y-axis ticks
    plt.xticks(fontsize=24)
    
    # ratio = 1.0
    # xleft, xright = ax.get_xlim()
    # ybottom, ytop = ax.get_ylim()
    # plt.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    if dataset_name == "cnn":
        # plt.ylim(0, 20)
        plt.legend(title="Optimizers", fontsize=18)
    
    plt.tight_layout()

    # Save plot
    output_file = os.path.join(output_path, filename.replace(".csv", ".pdf"))
    plt.savefig(output_file, dpi=300, format='pdf')
    print(f"Plot saved as: {output_file}")
    plt.close()

def process_folder(input_folder, output_folder):
    """
    Traverse the input folder hierarchy, process CSV files, and save plots.
    """
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                dataset_name = os.path.basename(root)  # Folder name as dataset name
                output_path = os.path.join(output_folder, dataset_name)

                # Create output directory if it doesn't exist
                os.makedirs(output_path, exist_ok=True)

                # Generate plot
                plot_csv(file_path, output_path)

# Input and output folder paths
input_folders = ["cnn", "flores", "xsum", "samsum", "iwslt"]  # Replace with your datasets folder
output_folder = "plots"    # Replace with your desired output folder

for input_folder in input_folders:
    print(f"Processing folder: {input_folder}")
    # Process the folder
    process_folder(input_folder, output_folder)
    print("")
