#!/usr/bin/env python3
import os
import json
import logging
import pandas as pd
import re
import requests
from pathlib import Path
import PyPDF2 # Added for PDF reading
import glob # Added for finding PDF files
import time # Added for potential rate limiting

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Constants
COMPETENCIES_FILE = "competencies_SPREE.json"
COURSE_OUTLINE_DIR = "CO"
INTERMEDIATE_RESULTS_DIR = "intermediate_results"
FINAL_OUTPUT_FILE = "dreyfus_competency_analysis.json"
FIREWORKS_API_KEY = os.environ.get('FIREWORKS_API_KEY')

###############################################################################
# PDF and File Utilities
###############################################################################

def find_course_outline_pdf(course_code):
    """Find the PDF course outline file for a given course code."""
    # Ensure course_code is a string
    course_code_str = str(course_code).strip()
    search_pattern = os.path.join(COURSE_OUTLINE_DIR, f"CO_{course_code_str}_*.pdf")
    files = glob.glob(search_pattern)
    if files:
        # Prefer files that don't have '-1' or similar suffixes if multiple exist
        preferred_files = [f for f in files if not re.search(r'-\d+\.pdf$', f)]
        if preferred_files:
            # Sort preferred files to be deterministic (e.g., shortest name first)
            preferred_files.sort(key=len)
            logging.debug(f"Found preferred PDF for {course_code_str}: {preferred_files[0]}")
            return preferred_files[0]
        # Sort all files to be deterministic if no preferred file found
        files.sort(key=len)
        logging.debug(f"Found PDF for {course_code_str} (no preferred): {files[0]}")
        return files[0] # Return the first match (shortest name) if no preferred file found
    logging.warning(f"No PDF found for course code {course_code_str} using pattern {search_pattern}")
    return None

def extract_text_from_pdf(pdf_path):
    """Extract text content from a PDF file."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            logging.debug(f"Reading {num_pages} pages from {pdf_path}")
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    else:
                        logging.warning(f"No text extracted from page {i+1} of {pdf_path}")
                except Exception as page_e:
                    logging.error(f"Error extracting text from page {i+1} of {pdf_path}: {page_e}")
            if text:
                logging.info(f"Successfully extracted text from {pdf_path} (length: {len(text)})")
            else:
                logging.warning(f"Extracted text is empty for {pdf_path}")
            # Basic cleaning: replace multiple newlines/spaces
            text = re.sub(r'\s+', ' ', text).strip()
            return text
    except FileNotFoundError:
        logging.error(f"PDF file not found: {pdf_path}")
        return None
    except PyPDF2.errors.PdfReadError as pdf_err:
         logging.error(f"Error reading PDF structure in {pdf_path}: {pdf_err}. File might be corrupted or password-protected.")
         return None
    except Exception as e:
        logging.error(f"Unexpected error reading PDF {pdf_path}: {e}")
        return None


###############################################################################
# Dreyfus Model Framework and Utilities (Adapted for Competencies/CLOs)
###############################################################################

# Dreyfus Model Levels
DREYFUS_LEVELS = {
    5: "5. Expert",
    4: "4. Proficient",
    3: "3. Competent",
    2: "2. Advanced Beginner",
    1: "1. Novice",
    0: "0. Not Applicable" # For cases where CLOs don't map well or data is missing
}

# Template adapted for competency/CLO analysis
# This serves as an example for the LLM's output format.
dreyfus_competency_template = [
    {
        "Competency ID": "1.1", # Example
        "Course code": "COURSE101", # Example
        "CLOs Analyzed": ["CLO1", "CLO2"], # Example
        "Dreyfus model": {
            "level": 2, # Example level
            "description": "Advanced Beginner", # Example description
            "explanation": "Based on the full course outline, the specified CLOs (e.g., CLO1, CLO2) primarily require students to apply learned guidelines and recognize basic situational components, characteristic of the Advanced Beginner stage."
        },
        "Justification": "The text related to CLO1 and CLO2 emphasizes following procedures and identifying key aspects as taught, rather than independent problem-solving or holistic understanding.",
        "Recommendations": "To elevate to 'Competent', CLOs could involve tasks requiring students to plan sequences of actions or adapt procedures to moderately complex scenarios described in the outline."
    }
]

def get_intermediate_save_path(competency_id):
    """Get the path for saving intermediate results for a competency."""
    # Sanitize competency_id for filename
    safe_competency_id = re.sub(r'[^\w\-.]', '_', str(competency_id))
    os.makedirs(INTERMEDIATE_RESULTS_DIR, exist_ok=True)
    return os.path.join(INTERMEDIATE_RESULTS_DIR, f"competency_{safe_competency_id}.json")

def load_intermediate_results(competency_id):
    """Load previously saved intermediate results for a competency."""
    path = get_intermediate_save_path(competency_id)
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            logging.info(f"Loaded intermediate results from {path}")
            return data
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from intermediate file {path}. Starting fresh for this competency.")
            return {}
        except Exception as e:
            logging.error(f"Error loading intermediate file {path}: {e}. Starting fresh.")
            return {}
    return {}

def save_intermediate_results(competency_id, data):
    """Save intermediate results for a competency."""
    path = get_intermediate_save_path(competency_id)
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logging.info(f"Saved intermediate results to {path}")
    except Exception as e:
        logging.error(f"Error saving intermediate results to {path}: {e}")


def generate_dreyfus_model_for_competency(competency_id, course_code, clo_ids, full_pdf_text):
    """
    Generate Dreyfus model analysis for a specific competency within a course,
    based on the full text of the course outline PDF.

    Args:
        competency_id (str): The ID of the competency being analyzed (e.g., "1.1").
        course_code (str): The course code (e.g., "SOLA1070").
        clo_ids (list): List of CLO IDs associated with this competency in this course (e.g., ["CLO1", "CLO2"]).
        full_pdf_text (str): The entire text extracted from the course outline PDF.

    Returns:
        dict or None: The generated Dreyfus model analysis data (single object), or None if generation failed.
    """
    if not full_pdf_text:
        logging.warning(f"No PDF text provided for {course_code} - Competency {competency_id}. Skipping analysis.")
        # Return a placeholder indicating missing data
        return {
            "Competency ID": competency_id,
            "Course code": course_code,
            "CLOs Analyzed": clo_ids,
            "Dreyfus model": {
                "level": 0,
                "description": DREYFUS_LEVELS[0],
                "explanation": "Analysis skipped due to missing or empty PDF text content."
            },
            "Justification": "Full course outline text could not be obtained or was empty.",
            "Recommendations": "Verify the PDF file exists, is readable, and text extraction was successful."
        }

    # Call the LLM to analyze the CLOs within the full PDF context
    logging.info(f"Analyzing Competency {competency_id} for Course {course_code} using CLOs: {', '.join(clo_ids)} within full PDF text.")
    dreyfus_json_str = generate_dreyfus_json_for_competency(full_pdf_text, competency_id, course_code, clo_ids)

    if dreyfus_json_str:
        try:
            # The LLM should return a list containing one analysis object
            analysis_list = json.loads(dreyfus_json_str)
            if isinstance(analysis_list, list) and len(analysis_list) > 0:
                 # Add/overwrite key info just in case LLM missed it
                 analysis_data = analysis_list[0] # Get the first (and should be only) object
                 analysis_data["Competency ID"] = competency_id
                 analysis_data["Course code"] = course_code
                 analysis_data["CLOs Analyzed"] = clo_ids

                 # Validate/Ensure Dreyfus model structure and level
                 if "Dreyfus model" not in analysis_data or not isinstance(analysis_data.get("Dreyfus model"), dict):
                      logging.warning("LLM response missing 'Dreyfus model' object. Setting to level 0.")
                      analysis_data["Dreyfus model"] = {"level": 0, "description": DREYFUS_LEVELS[0], "explanation": "Dreyfus model object missing in LLM response."}
                 else:
                     level = analysis_data["Dreyfus model"].get("level")
                     if level not in DREYFUS_LEVELS:
                         logging.warning(f"LLM returned invalid Dreyfus level '{level}'. Setting to 0.")
                         analysis_data["Dreyfus model"]["level"] = 0
                         analysis_data["Dreyfus model"]["description"] = DREYFUS_LEVELS[0]
                     else:
                         # Ensure description matches the level
                         analysis_data["Dreyfus model"]["description"] = DREYFUS_LEVELS[level]

                 # Ensure other fields exist
                 analysis_data.setdefault("Justification", "Justification missing in LLM response.")
                 analysis_data.setdefault("Recommendations", "Recommendations missing in LLM response.")

                 return analysis_data # Return the single analysis object
            else:
                 logging.error(f"LLM response was not a list with at least one element: {dreyfus_json_str[:500]}...")
                 return None # Indicate failure
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing JSON response from LLM: {e}. Response: {dreyfus_json_str[:500]}...")
            return None # Indicate failure
        except Exception as e:
             logging.error(f"Unexpected error processing LLM response: {e}. Response: {dreyfus_json_str[:500]}...")
             return None
    else:
        logging.error("LLM did not return data.")
        return None # Indicate failure


def generate_dreyfus_json_for_competency(full_pdf_text, competency_id, course_code, clo_ids):
    """
    Generates a JSON representation of Dreyfus model analysis by sending the full PDF text
    to the Fireworks API and asking it to focus on specific CLOs for a competency.

    Args:
        full_pdf_text (str): The entire text extracted from the course outline PDF.
        competency_id (str): Competency ID (e.g., "1.1").
        course_code (str): Course code (e.g., "SOLA1070").
        clo_ids (list): List of CLO IDs to focus on (e.g., ["CLO1", "CLO2"]).

    Returns:
        str or None: A valid JSON string (representing a list with one analysis object) if successful, or None on failure.
    """
    if not FIREWORKS_API_KEY:
        logging.error("FIREWORKS_API_KEY environment variable not set.")
        return None

    # Truncate very long PDF text if necessary, though Deepseek R1 has a large context window
    # max_context_chars = 30000 # Example limit, adjust as needed based on model/API limits
    # if len(full_pdf_text) > max_context_chars:
    #     logging.warning(f"PDF text for {course_code} is very long ({len(full_pdf_text)} chars), truncating to {max_context_chars}.")
    #     full_pdf_text = full_pdf_text[:max_context_chars]

    system_message = (
        "You are an expert educational analyst specializing in the Dreyfus model of skill acquisition. "
        "You will be given the full text of a university course outline PDF. "
        "Your task is to analyze this text, specifically focusing on the Course Learning Outcomes (CLOs) mentioned in the prompt (e.g., CLO1, CLO3), "
        "to determine the Dreyfus level (1-Novice, 2-Advanced Beginner, 3-Competent, 4-Proficient, 5-Expert) "
        "that best reflects the skills students are expected to demonstrate for a specific competency associated with those CLOs. "
        "If the specified CLOs cannot be clearly identified in the text, or if their description doesn't align with Dreyfus skill levels (e.g., purely factual recall, stating attitudes), use level 0 (Not Applicable). "
        "You MUST output **only valid JSON** formatted as a list containing a single object, precisely matching the structure shown in the example. No other text, commentary, or explanations outside the JSON structure are allowed."
        "\nDreyfus Levels Summary:\n"
        "1. Novice: Follows rules rigidly, needs step-by-step instructions.\n"
        "2. Advanced Beginner: Starts applying guidelines to specific situations, recognizes some patterns based on explicit teaching.\n"
        "3. Competent: Can plan actions, troubleshoot known issues, consciously chooses approaches based on goals, manages complexity.\n"
        "4. Proficient: Sees situations holistically, understands nuances, prioritizes effectively, learns from experience, adapts procedures.\n"
        "5. Expert: Acts intuitively based on deep understanding, fluid performance, handles novel situations effectively, reflects and improves.\n"
        "0. Not Applicable: CLOs focus on non-skill aspects or are not found/interpretable in the provided text."
    )

    # Construct the user prompt carefully
    clo_list_str = ", ".join(clo_ids)
    user_prompt = (
        f"Analyze the following course outline text for Course '{course_code}'. Your goal is to determine the Dreyfus level (0-5) "
        f"associated with Competency '{competency_id}', focusing specifically on the descriptions and implications of Course Learning Outcomes (CLOs): **{clo_list_str}**. "
        f"Read the entire text provided below to understand the context, but base your Dreyfus level assessment primarily on what is stated or implied about these specific CLOs.\n\n"
        f"Full Course Outline Text:\n```\n{full_pdf_text}\n```\n\n"
        f"Now, produce a JSON list containing exactly one object structured precisely like this example:\n"
        f"```json\n{json.dumps(dreyfus_competency_template, indent=2)}\n```\n"
        f"In your JSON output, ensure the 'Competency ID' is '{competency_id}', 'Course code' is '{course_code}', and 'CLOs Analyzed' is {json.dumps(clo_ids)}. "
        f"Provide a concise 'explanation' connecting the identified CLO descriptions (from the text) to the chosen Dreyfus level. "
        f"Include a 'Justification' summarizing the evidence from the text supporting your level assessment. "
        f"Add 'Recommendations' if appropriate (e.g., how CLOs might be modified for a different level). "
        f"Output **only** the JSON list. Do not include any text before or after the JSON."
    )

    payload = {
        "model": "accounts/fireworks/models/deepseek-r1", # Using Deepseek R1 as specified
        "temperature": 0.1, # Low temperature for consistent JSON output
        "max_tokens": 2048, # Max tokens for the response JSON (adjust if needed)
        "top_p": 0.7,
        "top_k": 50,
        "repetition_penalty": 1.05, # Slightly discourage repetition
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ]
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {FIREWORKS_API_KEY}"
    }

    max_retries = 2
    retry_delay = 5 # seconds
    for attempt in range(max_retries + 1):
        try:
            logging.info(f"Calling Fireworks API for Competency {competency_id}, Course {course_code} (Attempt {attempt + 1})")
            response = requests.post(
                "https://api.fireworks.ai/inference/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=120 # Increased timeout for potentially long processing
            )
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

            raw_output = response.json()['choices'][0]['message']['content']
            logging.debug(f"Raw LLM output received: {raw_output[:500]}...") # Log beginning of response

            # --- New JSON Extraction Logic ---
            potential_json = None
            think_tag_end = "</think>"
            think_tag_pos = raw_output.find(think_tag_end)

            if think_tag_pos != -1:
                # Found the tag, try extracting content after it
                json_part = raw_output[think_tag_pos + len(think_tag_end):].strip()
                logging.debug(f"Found '</think>' tag. Extracted content after tag: {json_part[:500]}...")
                if json_part.startswith('[') and json_part.endswith(']'):
                    potential_json = json_part
                else:
                    logging.warning("Content after '</think>' tag doesn't look like a JSON list.")
            else:
                # No '</think>' tag found, assume the whole response might be the JSON
                logging.debug("No '</think>' tag found in response. Attempting to parse entire response.")
                if raw_output.strip().startswith('[') and raw_output.strip().endswith(']'):
                    potential_json = raw_output.strip()
                else:
                    logging.error(f"LLM response does not appear to be a JSON list and no '</think>' tag found: {raw_output[:500]}...")

            # --- Try Parsing the Extracted JSON ---
            if potential_json:
                try:
                    # Validate JSON structure before returning
                    data = json.loads(potential_json)
                    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                        logging.info(f"Successfully extracted and parsed valid JSON structure from LLM for {competency_id}, {course_code}")
                        return potential_json # Return the raw JSON string
                    else:
                        logging.error(f"Extracted JSON is valid but not the expected list of objects: {potential_json[:500]}...")
                        # Fall through to retry or return None
                except json.JSONDecodeError as e:
                    logging.error(f"Could not decode extracted JSON from LLM response: {e}. Extracted part: {potential_json[:500]}...")
                    # Fall through to retry or return None
            else:
                # If potential_json is None after checks, log error and fall through to retry/fail
                logging.error("Failed to identify a potential JSON list in the response.")
                # Fall through to retry or return None


        except requests.exceptions.Timeout:
            logging.error(f"API request timed out for {competency_id}, {course_code} (Attempt {attempt + 1})")
            if attempt == max_retries: return None
            time.sleep(retry_delay)
        except requests.exceptions.RequestException as e:
            logging.error(f"API request error for {competency_id}, {course_code} (Attempt {attempt + 1}): {e}")
            # Check for rate limit errors (e.g., 429) - might need specific handling
            if hasattr(e, 'response') and e.response is not None and e.response.status_code == 429:
                 logging.warning("Rate limit likely hit. Waiting longer...")
                 time.sleep(retry_delay * 5) # Wait longer for rate limits
            elif attempt == max_retries:
                 return None # Return None after max retries for other request errors
            else:
                 time.sleep(retry_delay) # Wait before retrying for other errors
        except Exception as e:
            logging.exception(f"Unexpected error during API call for {competency_id}, {course_code} (Attempt {attempt + 1}): {e}")
            if attempt == max_retries: return None
            time.sleep(retry_delay)

    logging.error(f"Failed to get valid response from LLM after {max_retries + 1} attempts for {competency_id}, {course_code}.")
    return None # Return None if all retries fail


###############################################################################
# Main Processing Logic
###############################################################################

def main():
    """Main function to orchestrate the competency analysis."""
    logging.info("Starting Dreyfus model analysis for competencies...")

    # Load competencies data
    try:
        with open(COMPETENCIES_FILE, 'r') as f:
            competencies_data = json.load(f)
        logging.info(f"Loaded competencies data from {COMPETENCIES_FILE}")
    except FileNotFoundError:
        logging.error(f"Competencies file not found: {COMPETENCIES_FILE}")
        return
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {COMPETENCIES_FILE}")
        return
    except Exception as e:
        logging.error(f"Error loading {COMPETENCIES_FILE}: {e}")
        return

    # Check for API Key
    if not FIREWORKS_API_KEY:
        logging.error("FIREWORKS_API_KEY environment variable is not set. Cannot proceed.")
        print("\nError: FIREWORKS_API_KEY environment variable is not set.")
        print("Please set the API key before running the script.")
        print("Example (Linux/macOS): export FIREWORKS_API_KEY='your_api_key_here'")
        print("Example (Windows CMD): set FIREWORKS_API_KEY=your_api_key_here")
        print("Example (Windows PowerShell): $env:FIREWORKS_API_KEY='your_api_key_here'")
        return

    final_results = {} # Dictionary to hold all results: {competency_id: {course_code: analysis}}

    # Iterate through each competency group (e.g., "1", "2", "3")
    for group_id, competencies in competencies_data.items():
        logging.info(f"Processing Competency Group: {group_id}")
        # Iterate through each specific competency (e.g., "1.1", "1.2")
        for competency_id, courses in competencies.items():
            logging.info(f"--- Processing Competency: {competency_id} ---")

            # Load intermediate results if they exist
            intermediate_data = load_intermediate_results(competency_id)
            if competency_id not in final_results:
                 final_results[competency_id] = {}

            # Iterate through each course listed for this competency
            course_count = len(courses)
            for i, (course_code, clo_ids) in enumerate(courses.items()):
                logging.debug(f"Processing Course {i+1}/{course_count}: {course_code} for Competency {competency_id} (CLOs: {', '.join(clo_ids)})")

                # Check if already processed in intermediate results
                if course_code in intermediate_data:
                    logging.info(f"Skipping {course_code} for Competency {competency_id} - found in intermediate results.")
                    final_results[competency_id][course_code] = intermediate_data[course_code]
                    continue # Move to the next course

                analysis_result = None # Reset analysis result for the current course

                # Find the course outline PDF
                pdf_path = find_course_outline_pdf(course_code)
                if not pdf_path:
                    logging.warning(f"Could not find PDF for {course_code}. Skipping analysis.")
                    analysis_result = { # Placeholder for missing PDF
                        "Competency ID": competency_id, "Course code": course_code, "CLOs Analyzed": clo_ids,
                        "Dreyfus model": {"level": 0, "description": DREYFUS_LEVELS[0], "explanation": "Analysis skipped - Course Outline PDF not found."},
                        "Justification": "PDF file missing.", "Recommendations": "Ensure PDF exists in CO directory with correct naming convention."}
                else:
                    # Extract FULL text from PDF
                    full_pdf_text = extract_text_from_pdf(pdf_path)
                    if not full_pdf_text:
                         logging.warning(f"Could not extract text from PDF {pdf_path}. Skipping analysis for {course_code}.")
                         analysis_result = { # Placeholder for extraction failure
                             "Competency ID": competency_id, "Course code": course_code, "CLOs Analyzed": clo_ids,
                             "Dreyfus model": {"level": 0, "description": DREYFUS_LEVELS[0], "explanation": "Analysis skipped - Failed to extract text from PDF or text was empty."},
                             "Justification": "PDF text extraction error or empty content.", "Recommendations": "Check PDF readability, format, or extraction logic."}
                    else:
                        # Generate Dreyfus model analysis using LLM with FULL PDF TEXT
                        analysis_result = generate_dreyfus_model_for_competency(
                            competency_id, course_code, clo_ids, full_pdf_text
                        )
                        if analysis_result is None:
                             # Placeholder for LLM failure after retries
                             analysis_result = {
                                 "Competency ID": competency_id, "Course code": course_code, "CLOs Analyzed": clo_ids,
                                 "Dreyfus model": {"level": 0, "description": DREYFUS_LEVELS[0], "explanation": "Analysis failed - Error during LLM call or response processing after retries."},
                                 "Justification": "LLM API error, timeout, or invalid/unparseable response.", "Recommendations": "Check API key, connection, model availability, prompt, and LLM response format compliance."}


                # Store the result (even if it's a placeholder for failure)
                if analysis_result:
                    final_results[competency_id][course_code] = analysis_result
                    intermediate_data[course_code] = analysis_result # Update intermediate data for saving
                    # Save intermediate results immediately after processing a course successfully
                    save_intermediate_results(competency_id, intermediate_data)
                else:
                    # This case should ideally not be reached if placeholders are used for all failures
                    logging.error(f"Analysis result was unexpectedly None for {course_code}, Competency {competency_id}. Storing error placeholder.")
                    # Store placeholder but still save intermediate state if needed
                    analysis_result = {
                         "Competency ID": competency_id, "Course code": course_code, "CLOs Analyzed": clo_ids,
                         "Dreyfus model": {"level": 0, "description": DREYFUS_LEVELS[0], "explanation": "Internal error - analysis function returned None unexpectedly."},
                         "Justification": "Script logic error.", "Recommendations": "Review script execution flow and error handling."}
                    final_results[competency_id][course_code] = analysis_result
                    intermediate_data[course_code] = analysis_result
                    # Save intermediate results even if there was an internal error placeholder
                    save_intermediate_results(competency_id, intermediate_data)


            # Logging moved outside the inner loop
            logging.info(f"--- Finished processing all courses for Competency: {competency_id} ---")

    # Save the final combined results
    try:
        with open(FINAL_OUTPUT_FILE, 'w') as f:
            json.dump(final_results, f, indent=2)
        logging.info(f"Saved final combined Dreyfus competency analysis to: {FINAL_OUTPUT_FILE}")
    except Exception as e:
        logging.error(f"Error saving final results to {FINAL_OUTPUT_FILE}: {e}")

    logging.info("Dreyfus model analysis for competencies finished.")


if __name__ == "__main__":
    main()
