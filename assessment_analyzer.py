#!/usr/bin/env python3
import os
import csv
import json
import subprocess
import pandas as pd
import glob
import re
from pathlib import Path
import logging
import argparse
import requests
# Attempt to import TEMPLATES from the 'course-mapping' directory
# This import is only needed for framework analysis, not for linking assessments to CLOs
import sys
# Add the parent directory of the parent directory (project root) to the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from course_mapping.course_analyzer import TEMPLATES
except ImportError:
    # Log an error but allow the script to continue if only linking assessments to CLOs
    logging.error("Failed to import TEMPLATES from course-mapping/course_analyzer.py. This is required for framework analysis but not for linking assessments to CLOs.")
    TEMPLATES = {} # Provide an empty dict as a fallback

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# python3 assessment_analyzer.py --input to_analyse/SOLABHT1_with_assessments.xlsx --framework ea

###############################################################################
# Utility Functions
###############################################################################

def get_assessment_json_path(course_code, assessment_name, framework_name="ai"):
    """
    Get the path where the assessment framework analysis JSON would be stored.
    
    Args:
        course_code (str): The course code
        assessment_name (str): The assessment name
        framework_name (str): The framework name
        
    Returns:
        str: The path to the JSON file where analysis should be checked and saved.
    """
    # Define the directory where analysis files should be checked/saved
    # Note: This assumes a naming convention of assessment_<framework_name>_framework
    output_dir = os.path.join("json", "assessments", f"assessment_{framework_name}_framework")

    # Create a safe filename from the assessment name
    safe_assessment_name = re.sub(r'[^\w\s-]', '', assessment_name).strip().replace(' ', '_')
    
    # Create the output filename
    output_filename = f"{course_code}_{safe_assessment_name}.json"
    output_path = os.path.join(output_dir, output_filename)
    
    return output_path

def check_existing_assessment_analysis(course_code, assessment_name, framework_name="ai"):
    """
    Check if an assessment has already been analyzed and saved.
    
    Args:
        course_code (str): The course code
        assessment_name (str): The assessment name
        framework_name (str): The framework name
        
    Returns:
        dict or None: The analysis data if found, None otherwise
    """
    output_path = get_assessment_json_path(course_code, assessment_name, framework_name)
    
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                data = json.load(f)
            logging.info(f"Existing assessment analysis found for {course_code} - {assessment_name}")
            return data
        except json.JSONDecodeError:
            logging.error(f"Existing assessment analysis for {course_code} - {assessment_name} is corrupted, will reanalyze.")
            return None
    
    return None

def save_assessment_framework_json(data, course_code, assessment_name, framework_name="ai"):
    """
    Save the assessment framework analysis as a JSON file.
    
    Args:
        data (dict): The framework analysis data
        course_code (str): The course code
        assessment_name (str): The assessment name
        framework_name (str): The framework name
        
    Returns:
        str: The path to the saved file
    """
    # Get the output path (which includes the correct directory)
    output_path = get_assessment_json_path(course_code, assessment_name, framework_name)
    # Ensure the directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Get the output path
    output_path = get_assessment_json_path(course_code, assessment_name, framework_name)
    
    # Save the data
    try:
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        logging.info(f"Saved assessment framework analysis to: {output_path}")
        return output_path
    except Exception as e:
        logging.exception(f"Error saving assessment framework analysis: {e}")
        return None

def generate_framework_for_assessment(course_code, assessment_name, assessment_description, framework_name="ai"):
    """
    Call the LLM to analyze an assessment against the specified framework.
    
    Args:
        course_code (str): The course code
        assessment_name (str): Name of the assessment item
        assessment_description (str): Description of the assessment
        framework_name (str): Name of the framework to use (default: "ai")
        
    Returns:
        dict or None: The generated framework data, or None if generation failed
    """
    # Check if the framework exists
    if framework_name not in TEMPLATES:
        logging.error(f"Framework '{framework_name}' not found. Available frameworks: {', '.join(TEMPLATES.keys())}")
        return None
    
    # Get the framework template
    framework_template = TEMPLATES[framework_name]
    
    # Create a context from the assessment information
    assessment_context = f"Course Code: {course_code}\nAssessment Name: {assessment_name}\nAssessment Description: {assessment_description}"
    
    # Call the LLM to analyze the assessment against the framework
    framework_data = generate_framework_json(assessment_context, framework_template)
    
    if framework_data:
        try:
            data = json.loads(framework_data)
            
            # Save the framework data to a JSON file
            save_assessment_framework_json(data, course_code, assessment_name, framework_name)
            
            return data
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing JSON response: {e}")
            return None
    
    return None

def generate_framework_json(assessment_context, template_framework):
    """
    Generates a JSON representation of assessment alignments according to
    the provided template framework using the Fireworks API.
    
    Args:
        assessment_context (str): Assessment name and description.
        template_framework (list): A list (or JSON structure) representing
                                   the template alignment.
    
    Returns:
        str or None: A valid JSON string if successful, or None on failure.
    """
    system_message = (
        "You are an assistant that outputs valid JSON only. "
        "Below is an assessment description. Use it to find what elements of the framework are present in this assessment."
        "If they are not present or there is not enough information to say they are, just return null."
        "Your final message must be valid JSON with no extra text."
    )

    # Create a prompt that includes the assessment context and the example template.
    user_prompt = (
        f"CONTEXT about the assessment:\n\n{assessment_context}\n\n"
        f"Here is an example JSON for the framework alignment:\n"
        f"```json\n{json.dumps(template_framework, indent=2)}\n```\n"
        "Please produce a JSON array of one or more assessment entries in the same key structure. "
        "For each framework element, indicate if and how this specific assessment addresses that element. "
        "If the assessment does not address a particular element, use null for that value. "
        "No extra commentary outside JSON."
    )

    payload = {
        "model": "accounts/fireworks/models/deepseek-r1",
        "temperature": 0.2,
        "max_tokens": 16384,
        "top_p": 0.7,
        "top_k": 50,
        "repetition_penalty": 1,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ]
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('FIREWORKS_API_KEY')}"
    }

    try:
        logging.info("Calling Fireworks API to analyze assessment")
        response = requests.post(
            "https://api.fireworks.ai/inference/v1/chat/completions",
            headers=headers, 
            json=payload
        )
        response.raise_for_status()
        raw_output = response.json()['choices'][0]['message']['content']

        # Try to locate a JSON object in the response.
        json_match = re.search(r'(\[.*\]|\{.*\})', raw_output, re.DOTALL)
        if not json_match:
            logging.error("No JSON object found in the response.")
            return None

        potential_json = json_match.group(1)
        try:
            data = json.loads(potential_json)
            return json.dumps(data, indent=2)
        except json.JSONDecodeError as e:
            logging.error(f"Could not decode JSON from response: {e}")
            return None

    except requests.exceptions.RequestException as e:
        logging.error(f"An error occurred while communicating with the Fireworks API: {e}")
        return None
    except Exception as e:
        logging.exception(f"Unexpected error: {e}")
        return None

# Engineers Australia (EA) Competency Definitions provided by the user
EA_COMPETENCY_DEFINITIONS = {
    "1.1": "Understanding of natural and physical sciences relevant to engineering.",
    "1.2": "Application of established engineering methods to complex problem-solving.",
    "1.3": "In-depth understanding of specialist bodies of knowledge within their field and the ability to discern knowledge development and research directions.",
    "1.4": "Introduction to current research trends and technological advancements.",
    "1.5": "Knowledge of engineering design practices and contextual factors that impact the engineering discipline.",
    "1.6": "Familiarity with professional ethics and standards.",
    "2.1": "Foundation in mathematics, physics, and chemistry.",
    "2.2": "Fluent application of engineering techniques, tools, and resources.",
    "2.3": "Exposure to specialized fields such as thermodynamics and fluid mechanics.",
    "2.4": "Demonstrate strategic leadership in engineering projects and teams.",
    "3.1": "Exhibit exemplary ethical standards and professional judgment.",
    "3.2": "Effective oral and written communication in professional and lay domains.",
    "3.3": "Drive innovation through advanced research and development initiatives.",
    "3.4": "Professional use and management of information.",
    "3.5": "Orderly management of self and professional conduct.",
    "3.6": "Influence and develop policies impacting engineering practice."
}


def generate_assessment_clo_link(ea_competency_description, clo_description, assessment_description):
    """
    Calls the LLM to determine if an assessment is relevant to a CLO, considering the EA competency.

    Args:
        ea_competency_description (str): The description of the relevant EA competency.
        clo_description (str): The description of the Course Learning Outcome.
        assessment_description (str): The description of the assessment item.

    Returns:
        bool or None: True if the assessment is relevant to the CLO in the context of the EA competency,
                      False otherwise, or None if the LLM call failed.
    """
    system_message = (
        "You are an assistant that determines if a given assessment description "
        "is relevant to a specific Course Learning Outcome (CLO) description, "
        "considering the context of an Engineers Australia (EA) competency. "
        "Respond with a JSON object containing a single boolean key 'is_relevant'. "
        "If the assessment description strongly aligns with or helps achieve the CLO, "
        "and is relevant in the context of the provided EA competency, "
        "set 'is_relevant' to true. Otherwise, set it to false. "
        "Your final message must be valid JSON with no extra text."
    )

    user_prompt = (
        f"Engineers Australia (EA) Competency Description:\n{ea_competency_description}\n\n"
        f"Course Learning Outcome (CLO) Description:\n{clo_description}\n\n"
        f"Assessment Description:\n{assessment_description}\n\n"
        "Based on these descriptions, is the assessment relevant to the CLO in the context of the EA competency? "
        "Respond with a JSON object like: ```json\n{{\"is_relevant\": true/false}}\n```"
    )

    payload = {
        "model": "accounts/fireworks/models/deepseek-r1", # Using the same model as generate_framework_json
        "temperature": 0.2,
        "max_tokens": 16384,
        "top_p": 0.7,
        "top_k": 50,
        "repetition_penalty": 1,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ]
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('FIREWORKS_API_KEY')}"
    }

    try:
        logging.info("Calling Fireworks API to link assessment to CLO and EA competency")
        response = requests.post(
            "https://api.fireworks.ai/inference/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        raw_output = response.json()['choices'][0]['message']['content']

        # Try to locate a JSON object in the response.
        json_match = re.search(r'\{.*?\}', raw_output, re.DOTALL)
        if not json_match:
            logging.error("No JSON object found in the response for assessment-CLO-EA link.")
            return None

        potential_json = json_match.group(0)
        try:
            data = json.loads(potential_json)
            return data.get('is_relevant', False) # Default to False if key is missing
        except json.JSONDecodeError as e:
            logging.error(f"Could not decode JSON from response for assessment-CLO-EA link: {e}")
            return None

    except requests.exceptions.RequestException as e:
        logging.error(f"An error occurred while communicating with the Fireworks API for assessment-CLO-EA link: {e}")
        return None
    except Exception as e:
        logging.exception(f"Unexpected error during assessment-CLO-EA link: {e}")
        return None


###############################################################################
# Main Logic
###############################################################################

def read_excel_into_dataframe(excel_path, sheet_name="Schedule"):
    """
    Read a specific sheet from an Excel file into a DataFrame.
    """
    logging.info(f"Reading data from Excel file: {excel_path}, sheet: {sheet_name}")
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        return df
    except Exception as e:
        logging.exception(f"Error processing Excel file {excel_path}, sheet {sheet_name}: {e}")
        return pd.DataFrame()

def process_assessment_level_spreadsheet(excel_path, framework_name="ai"):
    """
    Process a single Excel file that has assessment-level data.
    This function will analyze each assessment against the specified framework.
    
    Args:
        excel_path (str): Path to the Excel file to process
        framework_name (str): Name of the framework to use (default: "ai")
    """
    logging.info(f"Processing assessment-level spreadsheet: {excel_path} using framework: {framework_name}")
    
    # Check if the framework exists
    if framework_name not in TEMPLATES:
        logging.error(f"Framework '{framework_name}' not found. Available frameworks: {', '.join(TEMPLATES.keys())}")
        return
    
    # Get the framework template
    framework_template = TEMPLATES[framework_name]
    
    # Extract categories dynamically from the first item in the framework template
    if not framework_template or not isinstance(framework_template, list) or not framework_template[0]:
        logging.error(f"Invalid framework template for '{framework_name}'")
        return
    
    # Get all keys except "Course code" or "Course Code" as categories
    framework_categories = []
    for key in framework_template[0].keys():
        if key.lower() not in ["course code", "course_code"]:
            framework_categories.append(key)
    
    # Read the spreadsheet
    df = read_excel_into_dataframe(excel_path)
    if df.empty:
        logging.error(f"Failed to read data from {excel_path}")
        return
    
    # Check if required columns exist
    required_columns = ['course code', 'item', 'description']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logging.error(f"Missing required columns in {excel_path}: {missing_columns}")
        return
    
    # Create a copy of the dataframe to modify
    output_df = df.copy()
    
    # Add columns for each framework category if they don't exist
    for category in framework_categories:
        if category not in output_df.columns:
            output_df[category] = ""
    
    # Process each assessment row
    for idx, row in output_df.iterrows():
        course_code = str(row['course code']) if pd.notna(row['course code']) else ""
        assessment_name = str(row['item']) if pd.notna(row['item']) else ""
        assessment_description = str(row['description']) if pd.notna(row['description']) else ""
        
        if not course_code or not assessment_name or not assessment_description:
            logging.warning(f"Skipping row {idx} due to missing course code, assessment name, or description")
            continue
        
        logging.info(f"Processing assessment: {course_code} - {assessment_name}")
        
        # Check if this assessment has already been analyzed
        existing_data = check_existing_assessment_analysis(course_code, assessment_name, framework_name)
        
        if existing_data:
            framework_data = existing_data
            logging.info(f"Using existing analysis for {course_code} - {assessment_name}")
        else:
            # Generate framework data for this assessment
            framework_data = generate_framework_for_assessment(
                course_code,
                assessment_name, 
                assessment_description, 
                framework_name
            )
        
        if framework_data:
            # Usually the generated JSON might be a list with one dict
            # or just a dict. Adjust as needed:
            if isinstance(framework_data, list) and framework_data:
                framework_data = framework_data[0]
            
            # Fill framework columns if present in the analysis data
            for category in framework_categories:
                if category in framework_data:
                    output_df.at[idx, category] = framework_data[category]
    
    # Save the updated DataFrame back to Excel in data/analysed folder
    output_dir = os.path.join("data", "analysed")
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract just the filename without path
    base_filename = os.path.basename(excel_path)
    output_path = os.path.join(output_dir, os.path.splitext(base_filename)[0] + f'_{framework_name}_analysis.xlsx')
    
    logging.info(f"Saving analysis results to: {output_path}")
    try:
        output_df.to_excel(output_path, sheet_name="assessment-level", index=False)
        logging.info(f"Successfully saved analysis results to: {output_path}")
    except Exception as e:
        logging.exception(f"Error saving Excel file {output_path}: {e}")

def process_all_files_in_directory(directory="assessment_data", framework_name="ai"):
    """
    Process all Excel files in the specified directory.
    
    Args:
        directory (str): Directory containing Excel files to process (default: assessment_data)
        framework_name (str): Name of the framework to use (default: "ai")
    """
    logging.info(f"Processing all Excel files in directory: {directory} using framework: {framework_name}")

    # Get all Excel files in the directory that have "with_assessments" in the name
    excel_files = glob.glob(os.path.join(directory, "*_with_assessments.xlsx")) # Corrected glob pattern

    if not excel_files:
        logging.warning(f"No assessment-level Excel files found in directory: {directory}")
        return
    
    logging.info(f"Found {len(excel_files)} assessment-level Excel files to process")
    
    # Process each Excel file
    for excel_file in excel_files:
        logging.info(f"Processing file: {excel_file}")
        process_assessment_level_spreadsheet(excel_file, framework_name)
    
    logging.info(f"Completed processing all {len(excel_files)} Excel files")

def link_assessments_to_clos(competencies_file="competencies_SPREE.json", assessment_data_dir="assessment_data"):
    """
    Links assessments to CLOs based on LLM analysis and updates the competencies file.
    Skips CLOs that already have relevant assessments.
    """
    logging.info(f"Starting linking of assessments to CLOs for {competencies_file}")

    # Check if FIREWORKS_API_KEY is set
    if not os.environ.get('FIREWORKS_API_KEY'):
        logging.error("FIREWORKS_API_KEY environment variable is not set. Please set it before running this script.")
        return

    try:
        with open(competencies_file, 'r') as f:
            competencies_data = json.load(f)
        logging.info(f"Successfully loaded competencies data from {competencies_file}")
    except FileNotFoundError:
        logging.error(f"Competencies file not found: {competencies_file}")
        return
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {competencies_file}")
        return

    updated_competencies_data = competencies_data.copy() # Work on a copy
    logging.info(f"Found {len(updated_competencies_data)} competency areas in the competencies file")

    # Iterate through the competencies structure
    for comp_area, sub_comps in updated_competencies_data.items():
        logging.info(f"Processing competency area {comp_area} with {len(sub_comps)} sub-competencies")
        for sub_comp_key, courses in sub_comps.items():
            # Use the sub_comp_key directly as the EA competency key
            ea_competency_key = sub_comp_key
            ea_competency_description = EA_COMPETENCY_DEFINITIONS.get(ea_competency_key, "Description not found")

            if ea_competency_description == "Description not found":
                 logging.warning(f"  EA Competency description not found for key: {ea_competency_key}. Skipping courses under this competency.")
                 continue

            logging.info(f"  Processing sub-competency {sub_comp_key} with {len(courses)} courses")
            for course_code, clos in courses.items():
                logging.info(f"    Processing course {course_code} with {len(clos)} CLOs")

                # Read the assessment JSON for this course
                assessment_json_path = os.path.join(assessment_data_dir, f"{course_code}_assessments.json")
                assessments_data = []
                try:
                    with open(assessment_json_path, 'r') as f:
                        assessments_data = json.load(f)
                    logging.info(f"      Loaded {len(assessments_data)} assessments for {course_code}.")
                except FileNotFoundError:
                    logging.warning(f"      Assessment JSON not found for {course_code} at {assessment_json_path}. Skipping.")
                    continue
                except json.JSONDecodeError:
                    logging.error(f"      Error decoding assessment JSON for {course_code} at {assessment_json_path}. Skipping.")
                    continue

                # Iterate through each CLO for the current course
                for clo_id, clo_info in clos.items():
                    # Check if this CLO already has relevant assessments
                    if "relevant_assessments" in clo_info and clo_info["relevant_assessments"]:
                        logging.info(f"      CLO {clo_id} already has {len(clo_info['relevant_assessments'])} relevant assessments. Skipping.")
                        continue

                    clo_description = clo_info.get("description", "Description not found")
                    logging.info(f"      Processing CLO {clo_id}: {clo_description}")

                    if clo_description == "Description not found":
                        logging.warning(f"        Skipping CLO {clo_id} for {course_code} due to missing description.")
                        continue

                    relevant_assessments = []
                    # Iterate through each assessment for the current course
                    for assessment in assessments_data:
                        assessment_name = assessment.get("item", "Unknown Assessment")
                        assessment_description = assessment.get("description", "No description provided")

                        if assessment_name == "Unknown Assessment":
                            logging.warning(f"        Skipping assessment with missing name in {course_code}_assessments.json.")
                            continue

                        logging.info(f"        Checking relevance of assessment '{assessment_name}' to CLO {clo_id} and EA competency {ea_competency_key}")

                        # Call LLM to check relevance, including EA competency description
                        is_relevant = generate_assessment_clo_link(ea_competency_description, clo_description, assessment_description)

                        if is_relevant is True:
                            logging.info(f"          Assessment '{assessment_name}' is relevant.")
                            relevant_assessments.append(assessment_name)
                        elif is_relevant is False:
                             logging.info(f"          Assessment '{assessment_name}' is NOT relevant.")
                        else:
                            logging.error(f"          LLM call failed for assessment '{assessment_name}', CLO {clo_id}, EA {ea_competency_key}. Skipping.")


                    # Update the relevant_assessments list for the current CLO
                    updated_competencies_data[comp_area][sub_comp_key][course_code][clo_id]["relevant_assessments"] = relevant_assessments

                    # Save the updated competencies data after each CLO is processed
                    # This ensures that if the script is interrupted, progress is not lost
                    try:
                        with open(competencies_file, 'w') as f:
                            json.dump(updated_competencies_data, f, indent=4)
                        logging.info(f"      Saved progress for CLO {clo_id} in {course_code}.")
                    except Exception as e:
                        logging.exception(f"      Error saving progress for CLO {clo_id} in {course_code}: {e}")

    # Final save of the updated competencies data
    try:
        with open(competencies_file, 'w') as f:
            json.dump(updated_competencies_data, f, indent=4)
        logging.info(f"Successfully updated {competencies_file} with relevant assessments.")
    except Exception as e:
        logging.exception(f"Error saving updated {competencies_file}: {e}")


def get_framework_descriptions():
    """
    Generate descriptions for each framework based on their content.
    
    Returns:
        dict: Dictionary mapping framework names to their descriptions
    """
    descriptions = {
        "wellbeing": "Student wellbeing framework focusing on relationships, competencies, and learning environment",
        "sdg": "Sustainable Development Goals framework covering 17 global goals",
        "ea": "Engineering Australia framework for engineering education and practice",
        "amc": "Australian Medical Council framework for medical education",
        "nmba": "Nursing and Midwifery Board of Australia standards framework",
        "teacher": "Teacher standards framework for education professionals",
        "aps": "Australian Psychological Society framework for psychology education",
        "ca": "Chartered Accountants framework for accounting education and practice",
        "ai": "Artificial Intelligence framework for AI literacy and skills",
        "architect": "Architecture framework for professional practice and sustainability",
        "biomedical": "Biomedical Science Program Learning Outcomes framework"
    }
    return descriptions

def list_available_frameworks():
    """
    Print a summary of all available frameworks with descriptions.
    """
    descriptions = get_framework_descriptions()
    
    print("\nAvailable Frameworks:\n")
    print(f"{'Framework':<15} | Description")
    print("-" * 80)
    
    for name in sorted(TEMPLATES.keys()):
        desc = descriptions.get(name, "No description available")
        print(f"{name:<15} | {desc}")
    
    print("\nUse with: python assessment_analyzer.py --framework <framework_name>\n")

###############################################################################
# Main Execution
###############################################################################

def main():
    logging.info("Starting assessment analysis script...")
    
    parser = argparse.ArgumentParser(description='Analyze assessments against a framework or link assessments to CLOs using LLM.')
    parser.add_argument('--input', type=str, default=None,
                        help='Path to a specific input Excel file for framework analysis (e.g., data/analysed/my_schedule_with_assessments.xlsx)')
    parser.add_argument('--directory', type=str, default=os.path.join("data", "analysed"),
                        help='Directory containing *_with_assessments.xlsx files for framework analysis (default: data/analysed)')
    parser.add_argument('--framework', type=str, default="ai",
                        help=f'Framework to use for analysis (default: ai). Available frameworks: {", ".join(TEMPLATES.keys())}')
    parser.add_argument('--list-frameworks', action='store_true',
                        help='List all available frameworks with descriptions')
    parser.add_argument('--link-assessments-to-clos', action='store_true',
                        help='Link assessments to CLOs in competencies_SPREE.json using LLM analysis.')
    parser.add_argument('--competencies-file', type=str, default="competencies_SPREE.json",
                        help='Path to the competencies JSON file (default: competencies_SPREE.json)')
    parser.add_argument('--assessment-data-dir', type=str, default="assessment_data",
                        help='Directory containing COURSECODE_assessments.json files (default: assessment_data)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Debug logging enabled.")

    if args.list_frameworks:
        list_available_frameworks()
    elif args.link_assessments_to_clos:
        # Explicitly set logging level for this mode
        logging.getLogger().setLevel(logging.INFO)
        link_assessments_to_clos(args.competencies_file, args.assessment_data_dir)
    elif args.input:
        # Process a single file for framework analysis
        if not os.path.exists(args.input):
             logging.error(f"Input file not found: {args.input}")
             return
        logging.info(f"Processing single file for framework analysis: {args.input} using framework: {args.framework}")
        process_assessment_level_spreadsheet(args.input, args.framework)
    else:
                # Process all files in the specified directory
        process_all_files_in_directory(args.directory, args.framework)
    
    logging.info("Done processing assessment-level spreadsheets.")

if __name__ == "__main__":
    main()