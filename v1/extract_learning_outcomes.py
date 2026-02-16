#!/usr/bin/env python3
import os
import re
import pandas as pd
from PyPDF2 import PdfReader
from datetime import date
import logging
import glob
import argparse
from pathlib import Path
import json

# Setup basic logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

###############################################################################
# PDF Processing & Regex Extraction Logic (Adapted from original script)
###############################################################################

# Regex to find individual CLO entries
outcome_regex = re.compile(
    r"^\s*(CLO\d+)\s*:\s*(.*?)(?=\s*CLO\d+\s*:|Assessment Item|Course Learning Outcomes|Workshop assessment|In-term test|\s{3,}|\t+|\Z)",
    re.MULTILINE | re.IGNORECASE | re.DOTALL
)

def remove_illegal_chars(text):
    """Removes characters illegal in XML 1.0 (relevant for Excel .xlsx)."""
    illegal_xml_chars_re = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]')
    return illegal_xml_chars_re.sub('', text)

def detect_column_structure(text):
    """
    Detect potential column structure in the text based on whitespace patterns.
    Returns a list of potential column boundary positions.
    """
    lines = text.split('\n')
    potential_boundaries = []
    for line in lines:
        matches = list(re.finditer(r'(\s{3,}|\t+)', line))
        for match in matches:
            potential_boundaries.append(match.start())
    
    boundary_counts = {}
    for pos in potential_boundaries:
        boundary_counts[pos] = boundary_counts.get(pos, 0) + 1
    
    common_boundaries = [pos for pos, count in boundary_counts.items() 
                         if count >= max(2, len(lines) * 0.1)]
    
    return sorted(common_boundaries)

def detect_table_structure(text):
    """
    Detect if the text contains a table structure and identify the column containing learning outcomes.
    Returns a tuple of (is_table, learning_outcome_column_index, header_row_index).
    """
    lines = text.split('\n')
    header_candidates = []
    for i, line in enumerate(lines):
        if re.search(r'course\s+learning\s+outcomes', line, re.IGNORECASE) and \
           re.search(r'assessment\s+item', line, re.IGNORECASE):
            header_candidates.append(i)
    
    if not header_candidates: return (False, -1, -1)
    
    header_row = header_candidates[0]
    header_line = lines[header_row]
    lo_match = re.search(r'course\s+learning\s+outcomes', header_line, re.IGNORECASE)
    ai_match = re.search(r'assessment\s+item', header_line, re.IGNORECASE)
    
    if not lo_match or not ai_match: return (False, -1, -1)
    
    learning_outcome_column = 0 if lo_match.start() < ai_match.start() else 1
    return (True, learning_outcome_column, header_row)

def clean_outcome_text(text, column_boundaries=None, is_table=False, lo_column=0):
    """Clean up the learning outcome text."""
    # Initial cleanup
    text = re.sub(r'^\W+|\W+$', '', text) # Remove leading/trailing special chars

    # Cut off at next CLO
    clo_match = re.search(r'CLO\d+\s*:', text)
    if clo_match and clo_match.start() > 0:
        text = text[:clo_match.start()].strip()
        
    # Remove everything after the bullet point (•)
    bullet_match = text.find('•')
    if bullet_match > 0:
        text = text[:bullet_match].strip()

    # Remove common trailing sections/indicators
    indicators = [
        'Detailed Assessment Description', 'Workshop assessment', 'In-term test',
        'Assessment Item', 'Assessment Length', 'Assignment submission', 'Submission notes',
        'Weight:', 'Due:', 'Weighting:', 'Due date:', 'Submission method:',
        'Group assessment', 'Individual assessment', 'Generative AI', 
        'Course Learning Outcomes', 'Learning and Teaching Technologies', 'Printed:',
        r'\(PLOs?.*?\)', # PLO references
        r'\s+[A-Z]{4}\d{4}.*?(?=\Z)' # Course code at end
    ]
    for indicator in indicators:
        # Use find for simple strings, search for regex patterns
        try:
            match_pos = text.find(indicator) if not indicator.startswith('\\') else -1
            if match_pos == -1 and indicator.startswith('\\'):
                 match = re.search(indicator, text)
                 match_pos = match.start() if match else -1
        except re.error: # Handle potential invalid regex patterns if any are added
             match_pos = text.find(indicator)

        if match_pos > 0:
            text = text[:match_pos].strip()

    # Use column boundaries if available
    if column_boundaries:
        for boundary in column_boundaries:
            if boundary < len(text):
                text = text[:boundary].strip()
                break
    else:
        # Fallback: Look for multiple spaces/tabs
        column_boundary = re.search(r'(\s{3,}|\t+)', text)
        if column_boundary:
            text = text[:column_boundary.start()].strip()

    # Table-specific cleaning
    if is_table and lo_column == 0:
        assessment_match = re.search(r'(workshop|in-term test|assessment)', text, re.IGNORECASE)
        if assessment_match:
            text = text[:assessment_match.start()].strip()

    # Final whitespace cleanup and illegal char removal
    text = re.sub(r'\s+', ' ', text).strip()
    text = remove_illegal_chars(text)
    
    return text

def extract_outcomes_from_pdf(pdf_path):
    """
    Extracts learning outcomes from a single PDF file using regex.
    
    Args:
        pdf_path (str): Path to the PDF file.
        
    Returns:
        list: A list of dictionaries, each containing 'CLO Number' and 'Learning Outcome'.
              Returns an empty list if extraction fails or no outcomes are found.
    """
    logging.debug(f"Attempting to extract outcomes from: {pdf_path}")
    extracted_outcomes = []
    
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    # Basic text normalization
                    page_text = page_text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
                    text += page_text + "\n"
            except Exception as page_e:
                logging.warning(f"  Error extracting text from page {i+1} in {Path(pdf_path).name}: {page_e}")

        if not text:
            logging.warning(f"  No text extracted from {Path(pdf_path).name}.")
            return []

        # Detect structure for better cleaning
        is_table, lo_column, header_row = detect_table_structure(text)
        column_boundaries = detect_column_structure(text)
        if is_table: logging.debug("  Detected table structure.")
        if column_boundaries: logging.debug(f"  Detected column boundaries: {column_boundaries}")

        matches = outcome_regex.finditer(text)
        temp_outcomes = {} # Use dict to handle duplicates/updates

        for match in matches:
            clo_number = match.group(1).upper()
            outcome_raw = match.group(2)
            
            cleaned_outcome = clean_outcome_text(outcome_raw, column_boundaries, is_table, lo_column)

            # Only store if meaningful content remains
            if cleaned_outcome and len(cleaned_outcome) > 10 and not re.search(r'CLO\d+', cleaned_outcome[4:]): # Check beyond initial CLO
                # If duplicate CLO, keep the longer version
                if clo_number not in temp_outcomes or len(cleaned_outcome) > len(temp_outcomes[clo_number]):
                     temp_outcomes[clo_number] = cleaned_outcome

        # Convert dict to list of dicts
        for clo_num, outcome_text in temp_outcomes.items():
             extracted_outcomes.append({
                 'CLO Number': clo_num,
                 'Learning Outcome': outcome_text
             })
             
        logging.debug(f"  Extracted {len(extracted_outcomes)} outcomes for {Path(pdf_path).name}.")

    except Exception as e:
        logging.error(f"  Failed processing {Path(pdf_path).name}: {e}", exc_info=False) # Set exc_info=True for full traceback if needed

    return extracted_outcomes

###############################################################################
# Excel Processing & File Handling Logic (Adapted from assessment_reporter.py)
###############################################################################

def find_course_outline_pdf(course_code, pdf_dir="CO"):
    """
    Find the PDF file for a given course code in the specified directory.
    """
    logging.debug(f"Searching for PDF file for course: {course_code} in {pdf_dir}")
    pattern = os.path.join(pdf_dir, f"CO_{course_code}_*.pdf")
    matching_files = glob.glob(pattern)
    logging.debug(f"Pattern used: {pattern} - Matching files: {matching_files}")
    
    if matching_files:
        # Prefer shorter filenames if multiple matches exist (less likely to be drafts)
        best_match = min(matching_files, key=len)
        logging.info(f"Found PDF for course {course_code}: {best_match}")
        return best_match
    
    logging.warning(f"No PDF found for course {course_code} in {pdf_dir}")
    return None

def read_excel_into_dataframe(excel_path, sheet_name="Schedule"):
    """
    Read a specific sheet from an Excel file into a DataFrame.
    """
    logging.info(f"Reading courses from Excel file: {excel_path}, sheet: {sheet_name}")
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        # Ensure 'course code' column exists
        if 'course code' not in df.columns:
             logging.error(f"'course code' column not found in sheet '{sheet_name}' of {excel_path}")
             return pd.DataFrame()
        return df
    except FileNotFoundError:
        logging.error(f"Excel file not found: {excel_path}")
        return pd.DataFrame()
    except Exception as e:
        logging.exception(f"Error processing Excel file {excel_path}, sheet {sheet_name}: {e}")
        return pd.DataFrame()

def process_excel_file(excel_path, pdf_dir="CO"):
    """
    Process a single Excel file to extract learning outcomes and expand the sheet.
    
    Args:
        excel_path (str): Path to the Excel file to process.
        pdf_dir (str): Directory containing the course outline PDFs.
    """
    logging.info(f"Processing Excel file: {excel_path}")
    
    schedule_df = read_excel_into_dataframe(excel_path, "Schedule")
    if schedule_df.empty:
        logging.error(f"Could not read 'Schedule' sheet or 'course code' column missing in {excel_path}. Skipping.")
        return
    
    output_rows = []
    processed_courses = set() # Track courses processed in this run

    # Iterate over each row in the input DataFrame
    for idx, row in schedule_df.iterrows():
        try:
            # Attempt to convert course code to string and strip whitespace
            course_code = str(row['course code']).strip()
        except Exception:
            logging.warning(f"Could not read course code from row {idx}. Skipping row.")
            output_rows.append(row) # Keep the original row if course code is invalid/missing
            continue

        # Check if this is a valid standard course code (e.g., "BABS1201")
        if re.match(r'^[A-Z]{4}\d{4}$', course_code):
            logging.info(f"Processing course: {course_code}")
            
            # Find the PDF for this course
            pdf_path = find_course_outline_pdf(course_code, pdf_dir)
            
            if pdf_path:
                # Extract learning outcomes using the regex method
                learning_outcomes = extract_outcomes_from_pdf(pdf_path)
                
                if learning_outcomes:
                    logging.info(f"  Found {len(learning_outcomes)} outcomes for {course_code}.")
                    # Create a new row for each learning outcome
                    for outcome in learning_outcomes:
                        new_row = row.copy()
                        new_row['CLO Number'] = outcome.get('CLO Number', '')
                        new_row['Learning Outcome'] = outcome.get('Learning Outcome', '')
                        output_rows.append(new_row)
                else:
                    # No outcomes found, add the original row with empty outcome columns
                    logging.warning(f"  No learning outcomes extracted for {course_code} from {pdf_path}.")
                    new_row = row.copy()
                    new_row['CLO Number'] = ''
                    new_row['Learning Outcome'] = ''
                    output_rows.append(new_row)
            else:
                # No PDF found, add the original row with empty outcome columns
                logging.warning(f"  No PDF found for {course_code}.")
                new_row = row.copy()
                new_row['CLO Number'] = ''
                new_row['Learning Outcome'] = ''
                output_rows.append(new_row)
                
            processed_courses.add(course_code)
        else:
            # If not a valid course code format, just add the original row
            logging.debug(f"Skipping row {idx}: '{course_code}' is not a standard course code format.")
            output_rows.append(row)
    
    # Convert the list of rows to a DataFrame
    if not output_rows:
        logging.warning(f"No rows generated for output file from {excel_path}.")
        return
        
    output_df = pd.DataFrame(output_rows)
    
    # Define output path
    output_filename = Path(excel_path).stem + '_with_learning_outcomes.xlsx'
    output_path = Path(excel_path).parent / output_filename
    
    logging.info(f"Saving learning outcome results to: {output_path}")
    try:
        # Ensure the output directory exists (though it should, being the same as input)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_excel(output_path, sheet_name="Schedule", index=False, engine='openpyxl')
        logging.info(f"Successfully saved learning outcome results to: {output_path}")
    except Exception as e:
        logging.exception(f"Error saving Excel file {output_path}: {e}")

def extract_and_add_clo_descriptions_to_competencies(competencies_file="competencies_SPREE.json", pdf_dir="CO"):
    """
    Extracts CLO descriptions from PDFs for courses listed in the competencies file
    and adds them to the competencies file structure.
    """
    logging.info(f"Starting extraction of CLO descriptions for courses in {competencies_file}")

    try:
        with open(competencies_file, 'r') as f:
            competencies_data = json.load(f)
    except FileNotFoundError:
        logging.error(f"Competencies file not found: {competencies_file}")
        return
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {competencies_file}")
        return

    updated_competencies_data = {}
    clo_descriptions = {} # Store extracted descriptions by course and CLO ID

    # First pass: Collect all unique course codes from the competencies file
    all_course_codes = set()
    for comp_area, sub_comps in competencies_data.items():
        for sub_comp, courses in sub_comps.items():
            for course_code in courses.keys():
                all_course_codes.add(course_code)

    # Second pass: Extract CLO descriptions from PDFs for each unique course code
    for course_code in all_course_codes:
        logging.info(f"Extracting CLO descriptions for course: {course_code}")
        pdf_path = find_course_outline_pdf(course_code, pdf_dir)
        if pdf_path:
            outcomes = extract_outcomes_from_pdf(pdf_path)
            if outcomes:
                clo_descriptions[course_code] = {item['CLO Number']: item['Learning Outcome'] for item in outcomes}
                logging.info(f"  Extracted {len(outcomes)} CLO descriptions for {course_code}.")
            else:
                logging.warning(f"  No CLO descriptions extracted for {course_code} from PDF.")
        else:
            logging.warning(f"  No PDF found for course {course_code} to extract CLO descriptions.")

    # Third pass: Update the competencies data structure with CLO descriptions
    for comp_area, sub_comps in competencies_data.items():
        updated_sub_comps = {}
        for sub_comp, courses in sub_comps.items():
            updated_courses = {}
            for course_code, clo_ids in courses.items():
                updated_clos = {}
                for clo_id in clo_ids:
                    # Find the description for this CLO
                    description = clo_descriptions.get(course_code, {}).get(clo_id, "Description not found")
                    # Update the structure to include the description
                    updated_clos[clo_id] = {
                        "description": description,
                        "relevant_assessments": [] # Initialize empty list for assessments
                    }
                updated_courses[course_code] = updated_clos
            updated_sub_comps[sub_comp] = updated_courses
        updated_competencies_data[comp_area] = updated_sub_comps

    # Save the updated competencies data back to the JSON file
    try:
        with open(competencies_file, 'w') as f:
            json.dump(updated_competencies_data, f, indent=4)
        logging.info(f"Successfully updated {competencies_file} with CLO descriptions.")
    except Exception as e:
        logging.exception(f"Error saving updated {competencies_file}: {e}")


###############################################################################
# Main Execution
###############################################################################

def main():
    logging.info("Starting script to process learning outcomes...")
    
    parser = argparse.ArgumentParser(description='Extract Learning Outcomes from PDFs or update competencies JSON with descriptions.')
    parser.add_argument('--input', type=str, default=None,
                        help='Path to a specific input Excel file for standard processing (e.g., data/originals/my_schedule.xlsx)')
    parser.add_argument('--directory', type=str, default="data/originals",
                        help='Directory containing input Excel files for standard processing (default: data/originals)')
    parser.add_argument('--pdf-dir', type=str, default="CO",
                        help='Directory containing the Course Outline PDF files (default: CO)')
    parser.add_argument('--update-competencies', action='store_true',
                        help='Extract CLO descriptions from PDFs for courses in competencies_SPREE.json and update the JSON file.')
    parser.add_argument('--competencies-file', type=str, default="competencies_SPREE.json",
                        help='Path to the competencies JSON file (default: competencies_SPREE.json)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Debug logging enabled.")

    if args.update_competencies:
        extract_and_add_clo_descriptions_to_competencies(args.competencies_file, args.pdf_dir)
    elif args.input:
        if not os.path.exists(args.input):
             logging.error(f"Input file not found: {args.input}")
             return
        # Process a single file
        logging.info(f"Processing single file: {args.input}")
        process_excel_file(args.input, args.pdf_dir)
    else:
        # Process all files in the specified directory
        if not os.path.isdir(args.directory):
             logging.error(f"Input directory not found: {args.directory}")
             return
        process_all_files_in_directory(args.directory, args.pdf_dir)
    
    logging.info("Script finished.")

if __name__ == "__main__":
    main()
