import pandas as pd
import os 

# Load the CSV dataset
foler_of_model = "t5-small"
folder = os.path.join(foler_of_model, 'summary_table_tests.csv')
data = pd.read_csv(folder)

# Define the template for the LaTeX table
def generate_latex_table(mode, grouped_data):
    table = """\\begin{table*}[htbp]
    {\\small
    \\centering
    \\begin{tabular}{|l|c|c|c|c|c|}
        \\hline
        & CNN/DailyMail & XSum & SAMSum & IWSLT & Flores \\\\
         Optimizer & ROUGE 2 & ROUGE 2 & ROUGE 2 & ROUGE 2  & ROUGE 2 \\\\
        \\hline
    """
    
    optimizers = ['adam', 'nadam', 'adamw', 'sgdm', 'sgd']
    relevant_combinations = {
        'cnn': 'test_rouge2_fmeasure',
        'xsum': 'test_rouge2_fmeasure',
        'samsum': 'test_rouge2_fmeasure',
        'iwslt': 'test_rouge2_fmeasure',
        'flores': 'test_rouge2_fmeasure'
    }

    for optimizer in optimizers:
        row = f"{optimizer.capitalize()}"
        for dataset, metric in relevant_combinations.items():
            # Get the filtered value for the specific dataset, optimizer, and metric
            filtered = grouped_data.get((dataset, optimizer, metric), None)
            if filtered is not None:
                mean, std = filtered
                row += f" & {mean:.3f} $\\pm$ {std:.3f}"
            else:
                row += " & "
        table += row + " \\\\\n"  # Add a newline and escape characters for LaTeX

    table += """\\hline
    \\end{tabular}
    }
    \\caption{Evaluation scores on test data for the {mode.capitalize()} mode.}
    \\label{{tab:{mode}_mode}}
\\end{table*}
"""
    return table


# Process data for each training mode
latex_tables = {}
for mode in ['basic', 'defaults', 'full']:
    mode_data = data[data['Training Mode'] == mode]
    grouped = mode_data.groupby(['Dataset Name', 'Optimizer', 'Metric'])[['Mean', 'Std']].mean()
    grouped_data = {
        (dataset, optimizer, metric): (row['Mean'], row['Std'])
        for (dataset, optimizer, metric), row in grouped.iterrows()
    }
    latex_tables[mode] = generate_latex_table(mode, grouped_data)

# Save LaTeX tables to separate text files
for mode, table in latex_tables.items():
    with open(f"{mode}_mode_table.txt", "w") as file:
        file.write(table)

print("LaTeX tables generated and saved.")
