import json
import pandas as pd # Added pandas import
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Global settings for visualizations
FONT_SIZE = 26  # Font size in points
FONT_FAMILY = "Times New Roman"
EXPORT_SCALE = 1  # Scale factor for image export

# Define competencies dictionary
competencies = {
    "1.1": "Science Fundamentals",
    "1.2": "Math Foundations",
    "1.3": "Specialist Knowledge",
    "1.4": "Research Directions",
    "1.5": "Design Practice",
    "1.6": "Sustainable Practice",
    "2.1": "Problem Solving",
    "2.2": "Tech Application",
    "2.3": "Design Processes",
    "2.4": "Project Management",
    "3.1": "Ethical Conduct",
    "3.2": "Effective Communication",
    "3.3": "Creative Mindset",
    "3.4": "Info Management",
    "3.5": "Self Management",
    "3.6": "Team Leadership"
}

def process_data(course_competencies, course_schedule):
    """Process the data to create matrices for visualization"""
    courses = []
    for year in sorted(course_schedule.keys()):
        for term in sorted(course_schedule[year].keys()):
            courses.extend(course_schedule[year][term])

    unwanted_courses = {"DISP", "GEN ED", "ELECTIVE", "ELEC3115", "ELEC4122"}
    filtered_courses = [course for course in courses if course not in unwanted_courses]

    detailed_competency_ids = [comp_id for comp_id in competencies.keys() if '.' in comp_id]
    matrix = [[0 for _ in range(len(filtered_courses))] for _ in range(len(detailed_competency_ids))]

    for comp_id, sub_comps in course_competencies.items():
        for sub_comp, courses_dict in sub_comps.items():
            if sub_comp in detailed_competency_ids:
                for course, clos in courses_dict.items():
                    if course in filtered_courses:
                        row_index = detailed_competency_ids.index(sub_comp)
                        col_index = filtered_courses.index(course)
                        matrix[row_index][col_index] += len(clos)

    # Calculate total CLOs per course
    total_clos_per_course = [0] * len(filtered_courses)
    for row in matrix:
        for i, val in enumerate(row):
            total_clos_per_course[i] += val

    # Calculate percentage matrix
    percentage_matrix = []
    for row in matrix:
        percentage_row = []
        for i, val in enumerate(row):
            if total_clos_per_course[i]:
                percentage_row.append((val / total_clos_per_course[i]) * 100)
            else:
                percentage_row.append(0)
        percentage_matrix.append(percentage_row)

    # Calculate cumulative percentage matrix
    cumulative_percentage_matrix = []
    for row in percentage_matrix:
        cumulative_row = []
        sum_val = 0
        for val in row:
            sum_val += val
            cumulative_row.append(min(sum_val, 100))
        cumulative_percentage_matrix.append(cumulative_row)

    # Create shapes and annotations for year divisions
    shapes = []
    annotations = []
    bar_courses = ["ELEC1111", "MATH2069", "ELEC2911"]
    years = ["Year 1", "Year 2", "Year 3", "Year 4"]

    bar_positions = []
    for bar_course in bar_courses:
        if bar_course in filtered_courses:
            idx = filtered_courses.index(bar_course)
            bar_positions.append(idx + 0.5)
            shapes.append({
                "type": "line",
                "x0": idx + 0.5,
                "y0": -0.5,
                "x1": idx + 0.5,
                "y1": len(detailed_competency_ids) - 0.5,
                "line": {"color": "black", "width": 2}
            })

    all_positions = [-0.5] + bar_positions + [len(filtered_courses) - 0.5]
    for i, year in enumerate(years):
        annotations.append({
            "x": (all_positions[i] + all_positions[i + 1]) / 2,
            "y": 1.10,
            "yref": "paper",
            "text": year,
            "showarrow": False,
            "font": {"family": FONT_FAMILY, "size": FONT_SIZE, "color": "black"}
        })

    # Create combined competency labels
    combined_competency_labels = [f"{comp_id} - {competencies[comp_id]}" for comp_id in detailed_competency_ids]

    # Create x-axis labels
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
        x_labels.append(f"{category} ({category_counters[category]})")

    return {
        "filtered_courses": filtered_courses,
        "percentage_matrix": percentage_matrix,
        "cumulative_percentage_matrix": cumulative_percentage_matrix,
        "x_labels": x_labels,
        "combined_competency_labels": combined_competency_labels,
        "shapes": shapes,
        "annotations": annotations,
        "detailed_competency_ids": detailed_competency_ids
    }

def create_heatmap(data, is_cumulative=False):
    """Create a heatmap visualization"""
    matrix = data["cumulative_percentage_matrix"] if is_cumulative else data["percentage_matrix"]
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=data["x_labels"],
        y=data["combined_competency_labels"],
        colorscale='YlOrRd',
        reversescale=False,
        zmin=0,
        zmax=100,
        colorbar=dict(
            title=dict(
                text="Cumulative CLOs<br>(%)" if is_cumulative else "Percentage of<br>CLOs",
                font=dict(family=FONT_FAMILY, size=FONT_SIZE)
            ),
            tickfont=dict(family=FONT_FAMILY, size=FONT_SIZE)
        )
    ))

    fig.update_layout(
        xaxis=dict(
            title=dict(
                text="Courses",
                font=dict(family=FONT_FAMILY, size=FONT_SIZE)
            ),
            tickmode='array',
            tickvals=list(range(len(data["x_labels"]))),
            ticktext=data["x_labels"],
            tickangle=-90,
            tickfont=dict(family=FONT_FAMILY, size=FONT_SIZE)
        ),
        yaxis=dict(
            title=dict(
                text="Competencies",
                font=dict(family=FONT_FAMILY, size=FONT_SIZE)
            ),
            tickmode='array',
            tickvals=list(range(len(data["combined_competency_labels"]))),
            ticktext=data["combined_competency_labels"],
            tickfont=dict(family=FONT_FAMILY, size=FONT_SIZE)
        ),
        # Set figure size to span two columns in A4 paper
        height=700,
        width=1200,
        font=dict(family=FONT_FAMILY, size=FONT_SIZE),
        shapes=data["shapes"],
        annotations=data["annotations"],
        margin=dict(l=200, r=50, t=80, b=0)
    )

    return fig

def create_radar_plot(data):
    """Create a radar plot visualization"""
    bar_courses = ["ELEC1111", "MATH2069", "ELEC2911", "SOLA4953"]
    course_labels = ["Year 1", "Year 2", "Year 3", "Year 4"]
    colors = [
        'rgba(254, 204, 92, 0.30)',
        'rgba(253, 141, 60, 0.25)',
        'rgba(240, 59, 32, 0.20)',
        'rgba(189, 0, 38, 0.15)'
    ]
    
    fig = go.Figure()
    
    for i, course in enumerate(reversed(bar_courses)):
        if course in data["filtered_courses"]:
            idx = data["filtered_courses"].index(course)
            r_values = [row[idx] for row in data["cumulative_percentage_matrix"]]
            r_values.append(r_values[0])  # Close the loop
            theta_values = data["combined_competency_labels"] + [data["combined_competency_labels"][0]]
            
            fig.add_trace(go.Scatterpolar(
                r=r_values,
                theta=theta_values,
                fill='toself',
                fillcolor=colors[i],
                name=list(reversed(course_labels))[i],
                line=dict(
                    shape='spline',
                    color=colors[i].replace('0.', '1.')
                )
            ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=False,
                tickvals=[0, 25, 50, 75, 100],
                gridcolor='lightgrey',
                linecolor='grey'
            ),
            angularaxis=dict(
                tickfont=dict(family=FONT_FAMILY, size=FONT_SIZE, color='black'),
                gridcolor='lightgrey',
                linecolor='black'
            ),
            bgcolor='white'
        ),
        showlegend=True,
        legend=dict(
            font=dict(family=FONT_FAMILY, size=FONT_SIZE),
            bgcolor='white',
            bordercolor='white',
            borderwidth=0,
            x=1.1,
            y=1,
            xanchor='left',
            yanchor='top'
        ),
        # Set figure size to span two columns in A4 paper
        width=1200,
        height=600,
        font=dict(family=FONT_FAMILY, size=FONT_SIZE)
    )
    
    return fig

def create_assessment_competency_heatmap(matrix_file='assessment_type_vs_ea_competency_matrix.xlsx'):
    """Create a heatmap for Assessment Type vs EA Competency frequency."""
    try:
        # Read the matrix, using the first column as the index
        df_matrix = pd.read_excel(matrix_file, index_col=0, engine='openpyxl')
    except FileNotFoundError:
        print(f"Error: Matrix file '{matrix_file}' not found.")
        return None
    except Exception as e:
        print(f"Error reading matrix file '{matrix_file}': {e}")
        return None

    # --- Normalize the data ---
    # Calculate row sums (total counts per assessment type)
    row_sums = df_matrix.sum(axis=1)
    # Divide each element by its row sum, multiply by 100. Handle division by zero.
    df_normalized = df_matrix.apply(lambda x: (x / row_sums[x.name] * 100) if row_sums[x.name] > 0 else 0, axis=1)

    # Prepare data for heatmap
    z_values = df_normalized.values
    y_labels = df_normalized.index.tolist() # Assessment Types
    x_labels = df_normalized.columns.tolist() # EA Competencies

    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=x_labels,
        y=y_labels,
        colorscale='YlOrRd', # Match existing style
        reversescale=False,
        # zmin=0, # Removed fixed zmin
        # zmax=100, # Removed fixed zmax - Plotly will now auto-scale
        colorbar=dict(
            title=dict(
                text="Percentage within<br>Assessment Type (%)", # Updated colorbar title
                font=dict(family=FONT_FAMILY, size=FONT_SIZE)
            ),
            tickfont=dict(family=FONT_FAMILY, size=FONT_SIZE)
        )
    ))

    fig.update_layout(
        # title=dict( # Removed title
        #      text="Assessment Type vs. EA Competency Frequency",
        #      font=dict(family=FONT_FAMILY, size=FONT_SIZE + 2), # Slightly larger title
        #      x=0.5, # Center title
        #      xanchor='center'
        # ),
        xaxis=dict(
            title=dict(
                text="EA Competency", # Updated x-axis title
                font=dict(family=FONT_FAMILY, size=FONT_SIZE)
            ),
            tickmode='array',
            tickvals=list(range(len(x_labels))),
            ticktext=x_labels,
            tickangle=-45, # Adjusted angle for potentially long competency names
            tickfont=dict(family=FONT_FAMILY, size=FONT_SIZE * 0.8) # Slightly smaller ticks if needed
        ),
        yaxis=dict(
            title=dict(
                text="Assessment Type", # Updated y-axis title
                font=dict(family=FONT_FAMILY, size=FONT_SIZE)
            ),
            tickmode='array',
            tickvals=list(range(len(y_labels))),
            ticktext=y_labels,
            tickfont=dict(family=FONT_FAMILY, size=FONT_SIZE),
            autorange="reversed" # Often makes sense for matrices read from files
        ),
        # Adjust size and margins as needed for readability
        height=800, # Increased height for potentially more rows
        width=1400, # Increased width for potentially more columns
        font=dict(family=FONT_FAMILY, size=FONT_SIZE),
        # shapes=data["shapes"], # No year shapes needed here
        # annotations=data["annotations"], # No year annotations needed here
        margin=dict(l=300, r=50, t=30, b=250) # Reduced top margin (t) from 100 to 30
    )

    return fig


def main():
    # Load data for original figures
    with open('competencies_SPREE.json', 'r') as f:
        course_competencies = json.load(f)
    
    with open('bachelor_RE_RE_Systems_Strand.json', 'r') as f:
        course_schedule = json.load(f)
    
    # Process data
    data = process_data(course_competencies, course_schedule)
    
    # Create and save visualizations
    heatmap_non_cum = create_heatmap(data, is_cumulative=False)
    heatmap_cum = create_heatmap(data, is_cumulative=True)
    radar_plot = create_radar_plot(data)

    # Create the new assessment vs competency heatmap
    assessment_competency_heatmap = create_assessment_competency_heatmap()

    # Save as HTML files
    heatmap_non_cum.write_html("heatmap_non_cum.html")
    heatmap_cum.write_html("heatmap_cum.html")
    radar_plot.write_html("radar_plot.html")
    if assessment_competency_heatmap:
        assessment_competency_heatmap.write_html("assessment_competency_heatmap.html")

    # Save as static image files
    heatmap_non_cum.write_image("heatmap_non_cum.png", scale=EXPORT_SCALE)
    heatmap_cum.write_image("heatmap_cum.png", scale=EXPORT_SCALE)
    radar_plot.write_image("radar_plot.png", scale=EXPORT_SCALE)
    if assessment_competency_heatmap:
        assessment_competency_heatmap.write_image("assessment_competency_heatmap.png", scale=EXPORT_SCALE)

    print("Visualizations created successfully!")

if __name__ == "__main__":
    main()
