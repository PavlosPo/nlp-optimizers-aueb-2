import numpy as np

def calculate_mean_and_std_error(values):
    mean = np.mean(values)
    std_error = np.std(values, ddof=1) / np.sqrt(len(values))
    return mean, std_error

def format_for_siunitx(mean, std_error):
    # Format values to scientific notation compatible with siunitx LaTeX formatting
    mean_str = "{:.1e}".format(mean)
    std_error_str = "{:.1e}".format(std_error)
    # Final format for LaTeX with siunitx syntax
    return f"\\num{{{mean_str}}}(\\pm\\num{{{std_error_str}}})"

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
        formatted_output = format_for_siunitx(mean, std_error)
        print(f"LaTeX siunitx format: {formatted_output}")
    else:
        print("No values entered.")

if __name__ == "__main__":
    main()
