import json
import pandas as pd

def debug_labels():
    with open('bachelor_RE_RE_Systems_Strand.json', 'r') as f:
        course_schedule = json.load(f)

    # Replicate process_data logic for courses and labels
    courses = []
    # Sort years keys
    years = sorted(course_schedule.keys(), key=lambda x: int(x))
    for year in years:
        # Sort term keys
        terms = sorted(course_schedule[year].keys(), key=lambda x: int(x))
        for term in terms:
            courses.extend(course_schedule[year][term])

    unwanted_courses = {"DISP", "GEN ED", "ELECTIVE", "ELEC3115", "ELEC4122"}
    filtered_courses = [course for course in courses if course not in unwanted_courses]

    category_counters = {"Math": 0, "Physics": 0, "Specialisation": 0, "Design": 0, "Other": 0}
    x_labels = []
    for course in filtered_courses:
        if course.startswith("MATH"):
            category = "Math"
        elif course.startswith("PHYS"):
            category = "Physics"
        elif (course.startswith("ENGG") or course.startswith("SOLA") or 
              course.startswith("MMAN") or course.startswith("ELEC")):
            category = "Specialisation"
        elif course.startswith("DESN"):
            category = "Design"
        else:
            category = "Other"
        
        category_counters[category] += 1
        label = f"{category} ({category_counters[category]})"
        x_labels.append(label)
        print(f"Course: {course}, Label: {label}")

    print("\n--- Check for missing labels ---")
    targets = {
        "PHYS1221": "Physics (2)", 
        "MATH2069": "Math (4)", 
        "MMAN2700": "Specialisation (9)"
    }
    
    found_labels = set(x_labels)
    for course_code in targets:
        # Find expected label for course
        expected_category = ""
        if course_code.startswith("MATH"): expected_category = "Math"
        elif course_code.startswith("PHYS"): expected_category = "Physics"
        elif course_code.startswith("DESN"): expected_category = "Design"
        else: expected_category = "Specialisation"
        
        # We can't easily guess the number without re-running the loop logic, 
        # but let's check if the specific label text exists in our generated list roughly.
        # Actually, let's just reverse check: which label corresponds to this course in our loop?
        pass

    # Re-run loop to map course -> label
    course_to_label = {}
    
    # Reset counters
    category_counters = {"Math": 0, "Physics": 0, "Specialisation": 0, "Design": 0, "Other": 0}
    for course in filtered_courses:
        if course.startswith("MATH"): category = "Math"
        elif course.startswith("PHYS"): category = "Physics"
        elif (course.startswith("ENGG") or course.startswith("SOLA") or 
              course.startswith("MMAN") or course.startswith("ELEC")): category = "Specialisation"
        elif course.startswith("DESN"): category = "Design"
        else: category = "Other"
        category_counters[category] += 1
        label = f"{category} ({category_counters[category]})"
        course_to_label[course] = label

    print("\nTarget Course verification:")
    for code, expected_desc in targets.items():
        if code in course_to_label:
            print(f"{code} -> Generated Label: '{course_to_label[code]}'. (User expects '{expected_desc}')")
        else:
            print(f"{code} -> NOT FOUND in separate filtered_courses list.")

    print("\n--- Checking Data Contribution ---")
    try:
        with open('competencies_SPREE.json', 'r') as f:
            comp_data = json.load(f)
            
        competencies_list = [
            "1.1", "1.2", "1.3", "1.4", "1.5", "1.6",
            "2.1", "2.2", "2.3", "2.4",
            "3.1", "3.2", "3.3", "3.4", "3.5", "3.6"
        ]
        
        for course in targets:
            print(f"\nChecking contributions for {course}:")
            found_any = False
            for section in comp_data.values(): # Top level "1", "2" etc
                for comp_id, methods in section.items():
                    if comp_id in competencies_list:
                        if course in methods:
                            print(f"  - Contributes to {comp_id} (CLOs: {len(methods[course])})")
                            found_any = True
            if not found_any:
                print("  - NO CONTRIBUTIONS FOUND in detailed_competencies!")
                
    except Exception as e:
        print(f"Error checking competencies: {e}")


if __name__ == "__main__":
    debug_labels()
