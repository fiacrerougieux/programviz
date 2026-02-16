import pandas as pd
import sys

# Define the input and output file names
input_file = 'SOLABHT2_with_assessment_categories.xlsx'
output_file = 'SOLABHT2_assessment_type_column.xlsx'

# Define the assessment category columns based on the visual structure in the image.
# IMPORTANT: Verify these names exactly match the column headers in your Excel file.
assessment_columns = [
    'Quizzes',
    'Research Paper',
    'Presentation',
    'Projects',
    'Practical Exam',
    'Problem Set',
    'Reports',
    'Labs',
    'Demonstrative',
    'Journal/Assignment',
    'Participation/Views (Assessment)', # Adjust if the '(Assessment)' part is different
    'Exams',
    'Other',
    'Thesis/Dissertation',
    'Group Work/Performance'
]

try:
    # Read the Excel file. Adjust 'header' if your headers are not on the first row (index 0).
    try:
        df = pd.read_excel(input_file, engine='openpyxl', header=0)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found in the current directory.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        sys.exit(1)


    # Identify the actual assessment columns present in the file
    valid_assessment_columns = [col for col in assessment_columns if col in df.columns]

    if not valid_assessment_columns:
        print(f"Error: None of the expected assessment columns were found in the file.")
        print(f"Expected columns based on script: {assessment_columns}")
        print(f"Columns found in file: {df.columns.tolist()}")
        print("Please check the 'assessment_columns' list in the script and update it to match your file.")
        sys.exit(1)

    # Identify columns to keep (assuming they are before the first assessment column found)
    first_assessment_col_index = df.columns.get_loc(valid_assessment_columns[0])
    columns_to_keep = df.columns[:first_assessment_col_index].tolist()

    # Function to find the assessment type for a row
    def find_assessment_type(row):
        for col in valid_assessment_columns:
            cell_value = row[col]
            # Check if the cell contains 'x' (case-insensitive, ignoring whitespace)
            if pd.notna(cell_value) and str(cell_value).strip().lower() == 'x':
                return col
        return None # Return None (which becomes NaN/blank in Excel) if no 'x' is found

    # Apply the function to each row to create the new 'Assessment Type' column
    df['Assessment Type'] = df.apply(find_assessment_type, axis=1)

    # Create the new DataFrame with the desired columns
    output_columns = columns_to_keep + ['Assessment Type']
    df_output = df[output_columns]

    # Write the new DataFrame to a new Excel file
    try:
        df_output.to_excel(output_file, index=False, engine='openpyxl')
        print(f"Successfully created '{output_file}' with the assessment type column.")
    except Exception as e:
        print(f"Error writing output file '{output_file}': {e}")
        sys.exit(1)


except Exception as e:
    print(f"An unexpected error occurred: {e}")
    sys.exit(1)
