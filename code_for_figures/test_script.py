import os
import pandas as pd

# Directory containing CSV files
folder_of_model = "t5-small/" # "t5_v1_1-small" or "t5-small"
csv_dir = folder_of_model + "/tests"
output_file = folder_of_model + "/summary_table_tests.csv"

# Initialize list for summary
summary = []

# Iterate over all CSV files
for filename in os.listdir(csv_dir):
    if filename.endswith(".csv"):
        # Parse filename to extract metadata
        parts = filename.split("_")
        dataset = parts[0]  # Dataset name
        metric = parts[2]  # Metric type (e.g., 'test_f1', 'test_loss')
        if metric == "rouge":
            metric += parts[3]
        mode = parts[-1].replace(".csv", "")  # Training mode (e.g., 'train', 'test')

        # Load CSV
        filepath = os.path.join(csv_dir, filename)
        data = pd.read_csv(filepath)

        # Extract optimizers from column names
        for column in data.columns:
            if "optimizer_name" in column and not column.endswith(("__MIN", "__MAX")):
                # Extract optimizer name and relevant columns
                optimizer = column.split(": ")[1].split(" - ")[0]
                metric_name = column.split(" - ")[1]
                mean_col = column
                min_col = f"{column}__MIN"
                max_col = f"{column}__MAX"

                # Ensure required columns exist
                if mean_col in data.columns and min_col in data.columns and max_col in data.columns:
                    # Extract final step values
                    final_row = data.iloc[-1]
                    mean = final_row[mean_col]
                    std = (final_row[max_col] - final_row[min_col]) / 2

                    # Append to summary
                    summary.append([dataset, mode, optimizer, metric_name, final_row["Step"], mean, std])

# Create a DataFrame for summary
summary_df = pd.DataFrame(summary, columns=["Dataset Name", "Training Mode", "Optimizer", "Metric", "Step", "Mean", "Std"])

print(summary_df['Metric'].unique())

# Save summary to CSV
summary_df.to_csv(output_file, index=False)