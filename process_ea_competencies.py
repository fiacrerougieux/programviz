import pandas as pd
import sys

# Define the input file name and the specific sheet to read
input_file = 'SOLABHT2_with_assessment_categories.xlsx'
sheet_name = 'assessment-ea-matrix' # Specify the sheet name
output_file = 'SOLABHT2_ea_competency_column.xlsx'

# Define the EA competency columns based on the actual names found in the sheet.
ea_competency_columns = [
    '1.1 - Science Fundamentals',
    '1.2 - Math Foundations',
    '1.3 - Specialist Knowledge',
    '1.4 - Research Directions',
    '1.5 - Design Practice',
    '1.6 - Sustainable Practice',
    '2.1 - Problem Solving',
    '2.2 - Tech Application',
    '2.3 - Design Processes',
    '2.4 - Project Management',
    '3.1 - Ethical Conduct',
    '3.2 - Effective Communication',
    '3.3 - Creative Mindset',
    '3.4 - Info Management',
    '3.5 - Self Management',
    '3.6 - Team Leadership'
]

try:
    # Read the specific sheet from the Excel file.
    try:
        # Use sheet_name parameter
        df = pd.read_excel(input_file, sheet_name=sheet_name, engine='openpyxl', header=0)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    except ValueError as e:
        # More specific error if sheet doesn't exist
        if f"Worksheet named '{sheet_name}' not found" in str(e):
             print(f"Error: Sheet named '{sheet_name}' not found in '{input_file}'.")
             # Optionally list available sheets if needed
             # xl = pd.ExcelFile(input_file)
             # print(f"Available sheets: {xl.sheet_names}")
        else:
             print(f"Error reading sheet '{sheet_name}' from Excel file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        sys.exit(1)

    # Identify the actual EA competency columns present in the sheet
    valid_ea_columns = [col for col in ea_competency_columns if col in df.columns]

    if not valid_ea_columns:
        print(f"Error: None of the expected EA competency columns were found in the sheet '{sheet_name}'.")
        print(f"Expected columns based on script: {ea_competency_columns}")
        print(f"Columns found in sheet: {df.columns.tolist()}")
        print("Please check the 'ea_competency_columns' list in the script and update it to match your file.")
        sys.exit(1)
    else:
        print(f"Found the following EA competency columns to process: {valid_ea_columns}")


    # Identify columns to keep (item, description, course code - assuming these are the first 3)
    # Adjust if the number of initial columns is different.
    initial_columns_to_keep = ['item', 'description', 'course code'] # Adjust if needed
    columns_to_keep = [col for col in initial_columns_to_keep if col in df.columns]
    if len(columns_to_keep) != len(initial_columns_to_keep):
        print(f"Warning: Could not find all expected initial columns {initial_columns_to_keep}. Found: {columns_to_keep}")


    # Function to find all marked EA competencies for a row
    def find_ea_competencies(row):
        marked_competencies = []
        for col in valid_ea_columns:
            cell_value = row[col]
            # Check if the cell is non-empty (adjust logic if marker is different, e.g., check for 'x')
            if pd.notna(cell_value) and str(cell_value).strip() != '':
                marked_competencies.append(col)
        # Join the list of competencies into a comma-separated string
        # Return None if the list is empty, otherwise return the joined string
        return ', '.join(marked_competencies) if marked_competencies else None

    # Apply the function to each row to create the new 'EA Competency' column
    df['EA Competency'] = df.apply(find_ea_competencies, axis=1)


    # Create the new DataFrame with the desired columns
    output_columns = columns_to_keep + ['EA Competency']
    df_output = df[output_columns]

    # Write the new DataFrame to a new Excel file
    try:
        df_output.to_excel(output_file, index=False, engine='openpyxl')
        print(f"Successfully created '{output_file}' with the EA competency column.")
    except Exception as e:
        print(f"Error writing output file '{output_file}': {e}")
        sys.exit(1)

except Exception as e:
    print(f"An unexpected error occurred: {e}")
    sys.exit(1)
