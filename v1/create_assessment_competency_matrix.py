import pandas as pd
import sys

# Define input and output file names
assessment_type_file = 'SOLABHT2_assessment_type_column.xlsx'
ea_competency_file = 'SOLABHT2_ea_competency_column.xlsx'
output_matrix_file = 'assessment_type_vs_ea_competency_matrix.xlsx'

# Define the columns to merge on (adjust if needed)
merge_columns = ['item', 'description', 'course code']

try:
    # --- Read Input Files ---
    try:
        df_assessment = pd.read_excel(assessment_type_file, engine='openpyxl')
        print(f"Successfully read {assessment_type_file}")
    except FileNotFoundError:
        print(f"Error: Input file '{assessment_type_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading {assessment_type_file}: {e}")
        sys.exit(1)

    try:
        df_competency = pd.read_excel(ea_competency_file, engine='openpyxl')
        print(f"Successfully read {ea_competency_file}")
    except FileNotFoundError:
        print(f"Error: Input file '{ea_competency_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading {ea_competency_file}: {e}")
        sys.exit(1)

    # --- Merge DataFrames ---
    # Check if merge columns exist in both dataframes
    missing_cols_assess = [col for col in merge_columns if col not in df_assessment.columns]
    missing_cols_comp = [col for col in merge_columns if col not in df_competency.columns]

    if missing_cols_assess:
        print(f"Error: Merge columns missing in {assessment_type_file}: {missing_cols_assess}")
        sys.exit(1)
    if missing_cols_comp:
        print(f"Error: Merge columns missing in {ea_competency_file}: {missing_cols_comp}")
        sys.exit(1)

    try:
        # Perform an inner merge to keep only rows present in both files
        df_merged = pd.merge(df_assessment, df_competency, on=merge_columns, how='inner')
        print(f"Successfully merged data. Merged shape: {df_merged.shape}")
        if df_merged.empty:
             print("Warning: Merged DataFrame is empty. Check if the merge columns match between the files.")
             # Optional: Check for differences in key columns if merge fails unexpectedly
             # print("Sample keys from assessment file:", df_assessment[merge_columns].head())
             # print("Sample keys from competency file:", df_competency[merge_columns].head())

    except Exception as e:
        print(f"Error merging dataframes: {e}")
        sys.exit(1)

    # --- Process Merged Data ---
    # Drop rows where either Assessment Type or EA Competency is missing
    df_processed = df_merged.dropna(subset=['Assessment Type', 'EA Competency'])
    print(f"Shape after dropping NA in key columns: {df_processed.shape}")


    # Split comma-separated EA Competencies and explode the DataFrame
    # Ensure 'EA Competency' is string type before splitting
    df_processed['EA Competency'] = df_processed['EA Competency'].astype(str)
    df_processed['EA Competency'] = df_processed['EA Competency'].str.split(', ')
    df_exploded = df_processed.explode('EA Competency')

    # Trim whitespace from exploded competencies
    df_exploded['EA Competency'] = df_exploded['EA Competency'].str.strip()
    print(f"Shape after exploding EA Competencies: {df_exploded.shape}")


    # --- Create the Matrix ---
    if not df_exploded.empty:
        # Group by Assessment Type and EA Competency, then count occurrences
        matrix_data = df_exploded.groupby(['Assessment Type', 'EA Competency']).size().unstack(fill_value=0)

        # Optional: Sort rows/columns alphabetically for consistency
        matrix_data = matrix_data.sort_index(axis=0) # Sort Assessment Types (rows)
        matrix_data = matrix_data.sort_index(axis=1) # Sort EA Competencies (columns)

        print("Successfully created the frequency matrix.")
    else:
        print("Cannot create matrix because the processed data is empty after merging/exploding.")
        # Create an empty DataFrame if needed, or exit
        matrix_data = pd.DataFrame() # Or sys.exit(1) if an empty file is not desired


    # --- Write Output File ---
    try:
        matrix_data.to_excel(output_matrix_file, engine='openpyxl')
        if not matrix_data.empty:
            print(f"Successfully wrote matrix to '{output_matrix_file}'")
        else:
            print(f"Wrote an empty matrix to '{output_matrix_file}' as no valid pairings were found.")

    except Exception as e:
        print(f"Error writing output matrix file '{output_matrix_file}': {e}")
        sys.exit(1)

except Exception as e:
    print(f"An unexpected error occurred: {e}")
    sys.exit(1)
