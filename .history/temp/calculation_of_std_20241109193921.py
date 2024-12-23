import numpy as np

def calculate_mean_and_std_error(values):
    mean = np.mean(values)
    std_error = np.std(values, ddof=1) / np.sqrt(len(values))
    return mean, std_error

def format_for_latex(mean, std_error):
    mean_str = "{:.1e}".format(mean)
    std_error_str = "{:.1e}".format(std_error)
    return f"{mean_str} \\pm {std_error_str}"

def main():
    values = []
    print("Enter numerical values (float32). Type 'done' to finish:")
    while True:
        user_input = input()
        if user_input.lower() == 'done':
            break
        try:
            value = np.float32(user_input)
            values.append(value)
        except ValueError:
            print("Invalid input. Please enter a numerical value or 'done' to finish.")
    
    if values:
        mean, std_error = calculate_mean_and_std_error(values)
        formatted_output = format_for_latex(mean, std_error)
        print(f"LaTeX format: {formatted_output}")
    else:
        print("No values entered.")

if __name__ == "__main__":
    main()
