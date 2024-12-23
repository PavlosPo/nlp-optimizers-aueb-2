import pandas as pd

def generate_latex_table(csv_files, output_file, dataset_name, mode):
    """
    Generate LaTeX tables based on CSV files and save them to a .txt file.

    Args:
        csv_files (dict): Dictionary of metric names and their corresponding CSV file paths.
        output_file (str): Name of the output .txt file.
        dataset_name (str): Name of the dataset (e.g., IWSLT, Flores).
        mode (str): Mode description (e.g., basic mode, learning rate tuning mode only).
    """
    optimizers = ["nadam", "sgd", "sgdm", "adamw", "adam"]

    header_template = r"""
        \begin{{table}}[H]
            \centering
            \caption{{This table presents the mean values of each optimizer's performance, averaged over four different seeds, for the {dataset} dataset in {mode} mode.}}
            \begin{{tabular}}{{>{{\raggedright\arraybackslash}}p{{0.3\textwidth}} *{{{columns}}}{{>{{\centering\arraybackslash}}p{{0.2\textwidth}}}}}}
            \toprule
            \textbf{{Metrics}} & {optimizers} \\
            \midrule
        """

    row_template = "    {metric} & {values} \\\\ \n"

    footer_template = (
        "    \\bottomrule\n"
        "    \\end{{tabular}}\n"
        "    \\label{{label}}\n"
        "\\end{{table}}\n"
    )

    output_content = ""
    metric_tables = {}
    for metric, csv_file in csv_files.items():
        df = pd.read_csv(csv_file)
        metric_tables[metric] = {}
        for optimizer in optimizers:
            mean_col = f"optimizer_name: {optimizer} - {metric}"
            min_col = f"optimizer_name: {optimizer} - {metric}__MIN"
            max_col = f"optimizer_name: {optimizer} - {metric}__MAX"
            if mean_col in df.columns:
                mean = df[mean_col].iloc[0]
                min_val = df[min_col].iloc[0]
                max_val = df[max_col].iloc[0]
                metric_tables[metric][optimizer] = f"${mean:.4f}$ ($\\pm${(max_val - min_val) / 2:.4f})"
            else:
                metric_tables[metric][optimizer] = "--"

    part = 1
    for i in range(0, len(optimizers), 3):
        current_optimizers = optimizers[i:i+3]
        columns = len(current_optimizers)
        optimizer_headers = " & ".join([opt.capitalize() for opt in current_optimizers])

        table_header = header_template.format(
            dataset=dataset_name,
            mode=mode,
            columns=columns,
            optimizers=optimizer_headers,
        )

        table_rows = ""
        for metric, data in metric_tables.items():
            values = " & ".join([data[opt] for opt in current_optimizers])
            table_rows += row_template.format(metric=metric.replace("_", " ").capitalize(), values=values)

        label = f"{dataset_name.lower()}_test_results_{part}"
        table_footer = footer_template.format(label=label)

        output_content += table_header + table_rows + table_footer + "\n\n"
        part += 1

    with open(output_file, "w") as file:
        file.write(output_content)


# Example usage
csv_files = {
    "test_loss": "./flores/flores_test_loss_full.csv",
    "test_f1": "./flores/flores_test_f1_full.csv",
    "test_rouge1_fmeasure": "./flores/flores_test_rouge_1_full.csv",
    "test_rouge2_fmeasure": "./flores/flores_test_rouge_2_full.csv",
}

generate_latex_table(csv_files, "flores_test_full.txt", "Flores", "Full Mode")
