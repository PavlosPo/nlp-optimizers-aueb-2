import re
import json

def extract_table_data(latex_content):
    tables = []
    table_pattern = r'\\begin{table}.*?\\end{table}'
    table_matches = re.findall(table_pattern, latex_content, re.DOTALL)
    
    for table_match in table_matches:
        caption_pattern = r'\\caption{(.*?)}'
        caption_match = re.search(caption_pattern, table_match)
        caption = caption_match.group(1) if caption_match else ""
        
        rows = []
        row_pattern = r'(.*?) \\\\'
        row_matches = re.findall(row_pattern, table_match)
        
        for row_match in row_matches:
            cells = re.split(r'\s*&\s*', row_match.strip())
            processed_cells = []
            for cell in cells:
                # Extract numerical value and variance
                value_match = re.search(r'\{([\d.]+)\}', cell)
                variance_match = re.search(r'\(\$\\pm\$\$\{([\d.]+)\}\$\)', cell)
                
                if value_match:
                    value = float(value_match.group(1))
                    variance = float(variance_match.group(1).strip('\\pm$')) if variance_match else None
                    processed_cells.append({"value": value, "variance": variance})
                else:
                    processed_cells.append(cell.strip('$'))
            
            rows.append(processed_cells)
        
        if rows:
            header = rows[0]
            data = rows[1:]
            tables.append({"caption": caption, "header": header, "data": data})
    
    return tables

def save_to_json(tables, filename):
    with open(filename, 'w') as f:
        json.dump(tables, f, indent=2)

# Main execution
with open('text.txt', 'r') as f:
    latex_content = f.read()

tables = extract_table_data(latex_content)
save_to_json(tables, 'table_data.json')
print("Data has been extracted and saved to table_data.json")