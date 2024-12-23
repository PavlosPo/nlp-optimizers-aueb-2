import re
import numpy as np

def calculate_mean_and_std_error(values):
    mean = np.mean(values)
    std_error = np.std(values, ddof=1) / np.sqrt(len(values))
    return mean, std_error

def format_for_siunitx(mean, std_error):
    mean_str = "{:.1e}".format(mean)
    std_error_str = "{:.1e}".format(std_error)
    return f"\\num{{{mean_str}}}(\\pm\\num{{{std_error_str}}})"

def main():
    values = []
    print("Enter numerical values (float32). Type 'done' to finish:")
    while True:
        user_input = input()
        if user_input.lower() == 'done':
            break
        # Insert "e" if the input looks like `3.7-09`
        user_input = re.sub(r"(\d)(-)", r"\1e-", user_input)
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
