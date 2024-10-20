import json
import matplotlib.pyplot as plt
import numpy as np
import math

def load_json_data(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def create_bar_plots(data):
    # Collect all unique optimizers from both parts of the data
    all_optimizers = list({optimizer for table in data for optimizer in table['header'][1:]})
    
    # Define a color map and assign a unique color to each optimizer
    optimizer_color_map = {"Adam": "#ECBB33",
                           "AdamW": "#3D9E3E",
                           "SGD": "#DF672A",
                           "SGDM": "#C1433C",
                           "NAdam": "#338DD8"}  # Dictionary to map each optimizer to a color
    
    # Collect all unique metrics
    all_metrics = list({row[0] for table in data for row in table['data']})
    
    # Initialize a dictionary to store the values and variances for each optimizer/metric
    combined_data = {metric: {optimizer: {'value': None, 'variance': None} for optimizer in all_optimizers} for metric in all_metrics}

    # Populate combined_data with values from both parts
    for table in data:
        optimizers = table['header'][1:]
        for row in table['data']:
            metric = row[0]
            for i, optimizer in enumerate(optimizers):
                combined_data[metric][optimizer] = {
                    'value': row[i+1]['value'],
                    'variance': row[i+1]['variance']
                }
    
    # Create a single plot for each metric with all optimizers
    for metric in all_metrics:
        values = [combined_data[metric][optimizer]['value'] for optimizer in all_optimizers]
        errors = [combined_data[metric][optimizer]['variance'] if combined_data[metric][optimizer]['variance'] is not None else np.nan for optimizer in all_optimizers]
        
        x = np.arange(len(all_optimizers))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(x, values, width, yerr=errors, capsize=5, color=[optimizer_color_map[optimizer] for optimizer in all_optimizers])
        ax.set_ylabel(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(all_optimizers, rotation=45)
        ax.set_title(f'{metric} by Optimizer')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            if not math.isnan(height):  # Check to avoid placing text on NaN bars
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'barplot_{metric.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close()

# Main execution
data = load_json_data('table_data.json')
create_bar_plots(data)
print("All bar plots have been created and saved as PNG files.")
