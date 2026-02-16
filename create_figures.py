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

    # --- NEW: Program-share percentage matrix (Option A) ---
    total_program_links = sum(sum(row) for row in matrix)  # shared denominator T
    if total_program_links == 0:
        raise ValueError("Total program CLO links is 0. Check your JSON mapping.")

    program_share_matrix = []
    num_competencies = len(detailed_competency_ids)
    for row in matrix:
        # Scale by number of competencies so 100% = 1/N share (e.g. 1/16th)
        program_share_matrix.append([(val / total_program_links) * num_competencies * 100 for val in row])

    # --- NEW: Cumulative program-share matrix (no cap needed) ---
    cumulative_program_share_matrix = []
    for row in program_share_matrix:
        cumulative_row = []
        running = 0.0
        for val in row:
            running += val
            cumulative_row.append(running)
        cumulative_program_share_matrix.append(cumulative_row)

    # --- Existing cumulative (within-course share cumulated + capped) ---
    # You can keep or remove this. It is NOT Option A.
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
        "program_share_matrix": program_share_matrix,
        "cumulative_program_share_matrix": cumulative_program_share_matrix,
        "total_program_links": total_program_links,
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

def create_heatmap_program_share(data, cumulative=True, zmax=None):
    """Heatmap for Option A: share of ALL mapped CLO links in the program (%)"""
    matrix = data["cumulative_program_share_matrix"] if cumulative else data["program_share_matrix"]

    # Let Plotly auto-scale; Option A values may be small (e.g., 0–10%)
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=data["x_labels"],
        y=data["combined_competency_labels"],
        colorscale='YlOrRd',
        reversescale=False,
        zmin=0 if zmax else None,
        zmax=zmax,
        colorbar=dict(
            title=dict(
                text="Cumulative share of<br>CLO-competency links" if cumulative else "Relative Program<br>Share (%)",
                font=dict(family=FONT_FAMILY, size=FONT_SIZE)
            ),
            tickfont=dict(family=FONT_FAMILY, size=FONT_SIZE)
        )
    ))

    fig.update_layout(
        xaxis=dict(
            title=dict(text="Courses", font=dict(family=FONT_FAMILY, size=FONT_SIZE)),
            tickmode='array',
            tickvals=list(range(len(data["x_labels"]))),
            ticktext=data["x_labels"],
            tickangle=-90,
            tickfont=dict(family=FONT_FAMILY, size=FONT_SIZE)
        ),
        yaxis=dict(
            title=dict(text="Competencies", font=dict(family=FONT_FAMILY, size=FONT_SIZE)),
            tickmode='array',
            tickvals=list(range(len(data["combined_competency_labels"]))),
            ticktext=data["combined_competency_labels"],
            tickfont=dict(family=FONT_FAMILY, size=FONT_SIZE)
        ),
        height=700,
        width=1200,
        font=dict(family=FONT_FAMILY, size=FONT_SIZE),
        shapes=data["shapes"],
        annotations=data["annotations"],
        margin=dict(l=200, r=50, t=80, b=0)
    )

    return fig

def create_side_by_side_heatmaps(data):
    """Create side-by-side heatmaps: (a) Saturated at 100%, (b) Auto-scaled."""
    
    # Create subplots with 2 columns
    fig = make_subplots(
        rows=1, cols=2,
        # subplot_titles=("(a) Saturated at 100%", "(b) Saturated at Max"), # Removed titles as requested
        horizontal_spacing=0.35 # Increase spacing for colorbars to avoid overlap
    )

    # Heatmap (a): Saturated at Max (Free Scale)
    fig.add_trace(go.Heatmap(
        z=data["cumulative_program_share_matrix"],
        x=data["x_labels"],
        y=data["combined_competency_labels"],
        colorscale='YlOrRd',
        reversescale=False,
        # zmin/zmax auto
        colorbar=dict(
            title=dict(
                text="(%)",
                font=dict(family=FONT_FAMILY, size=FONT_SIZE)
            ),
            tickfont=dict(family=FONT_FAMILY, size=FONT_SIZE),
            x=0.36, # Position first colorbar closer to first plot
            len=1
        )
    ), row=1, col=1)

    # Heatmap (b): Saturated at 100%
    fig.add_trace(go.Heatmap(
        z=data["cumulative_program_share_matrix"],
        x=data["x_labels"],
        y=data["combined_competency_labels"],
        colorscale='YlOrRd',
        reversescale=False,
        zmin=0,
        zmax=100,
        colorbar=dict(
            title=dict(
                text="(%)",
                font=dict(family=FONT_FAMILY, size=FONT_SIZE)
            ),
            tickfont=dict(family=FONT_FAMILY, size=FONT_SIZE),
            x=1.02, # Position second colorbar
            len=1
        )
    ), row=1, col=2)

    # Update axis properties for both subplots
    for col in [1, 2]:
        fig.update_xaxes(
            title_text="Courses",
            title_font=dict(family=FONT_FAMILY, size=FONT_SIZE),
            tickmode='array',
            tickvals=list(range(len(data["x_labels"]))),
            ticktext=data["x_labels"],
            tickangle=-90,
            tickfont=dict(family=FONT_FAMILY, size=FONT_SIZE),
            row=1, col=col
        )
        
    # Y-axis for left plot
    fig.update_yaxes(
        title_text="Competencies",
        title_font=dict(family=FONT_FAMILY, size=FONT_SIZE),
        tickmode='array',
        tickvals=list(range(len(data["combined_competency_labels"]))),
        ticktext=data["combined_competency_labels"],
        tickfont=dict(family=FONT_FAMILY, size=FONT_SIZE),
        row=1, col=1
    )

    # Y-axis for right plot (hide labels if desired, or keep them)
    # Keeping them for clarity as requested "side by side" usually implies full plots
    fig.update_yaxes(
         title_text="Competencies",
         title_font=dict(family=FONT_FAMILY, size=FONT_SIZE),
         tickmode='array',
         tickvals=list(range(len(data["combined_competency_labels"]))),
         ticktext=data["combined_competency_labels"],
         tickfont=dict(family=FONT_FAMILY, size=FONT_SIZE),
         row=1, col=2
    )
    
    # Add year lines to both subplots
    for shape in data["shapes"]:
        # Subplot 1 (xref='x', yref='y')
        new_shape_1 = shape.copy()
        new_shape_1['xref'] = 'x'
        new_shape_1['yref'] = 'y'
        fig.add_shape(new_shape_1, row=1, col=1)
        
        # Subplot 2 (xref='x2', yref='y2')
        new_shape_2 = shape.copy()
        new_shape_2['xref'] = 'x2'
        new_shape_2['yref'] = 'y2'
        fig.add_shape(new_shape_2, row=1, col=2)

    # Add annotations (Year labels) - tricky with subplots, use specific x/y
    # Or just add them to layout.annotations which takes xref/yref
    # We need to duplicate annotations for x2
    
    # We will use the layout update to set global font and size
    fig.update_layout(
        height=800,
        width=2200, # Wide figure
        font=dict(family=FONT_FAMILY, size=FONT_SIZE),
        margin=dict(l=200, r=50, t=100, b=0),
        showlegend=False
    )
    
    # Add (a) and (b) labels
    fig.add_annotation(
        x=0.0, y=1.05, xref="paper", yref="paper", # Moved (a) into visible range
        text="(a)", showarrow=False,
        font=dict(family=FONT_FAMILY, size=FONT_SIZE+10, color="black")
    )
    fig.add_annotation(
        x=0.52, y=1.05, xref="paper", yref="paper",
        text="(b)", showarrow=False,
        font=dict(family=FONT_FAMILY, size=FONT_SIZE+10, color="black")
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

def create_side_by_side_radar(data):
    """Create side-by-side radar plots: (a) Free Scale (With ref line), (b) Capped at 100% (No ref line)."""
    bar_courses = ["ELEC1111", "MATH2069", "ELEC2911", "SOLA4953"]
    course_labels = ["Year 1", "Year 2", "Year 3", "Year 4"]
    colors = [
        'rgba(254, 204, 92, 0.30)',
        'rgba(253, 141, 60, 0.25)',
        'rgba(240, 59, 32, 0.20)',
        'rgba(189, 0, 38, 0.15)'
    ]

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'polar'}, {'type': 'polar'}]],
        horizontal_spacing=0.15
    )

    # Common radar data setup
    theta_values = data["combined_competency_labels"] + [data["combined_competency_labels"][0]]
    
    # 1. Left Plot: Free Scale, With Ref Line (was Right Plot)
    for i, course in enumerate(reversed(bar_courses)):
        if course in data["filtered_courses"]:
            idx = data["filtered_courses"].index(course)
            raw_r_values = [row[idx] for row in data["cumulative_program_share_matrix"]]
            r_values_free = raw_r_values + [raw_r_values[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=r_values_free,
                theta=theta_values,
                fill='toself',
                fillcolor=colors[i],
                mode='lines',
                line=dict(shape='spline', color=colors[i].replace('0.', '1.')),
                name=list(reversed(course_labels))[i],
                showlegend=True, # Show legend entries here
                legendgroup=list(reversed(course_labels))[i], # Group by year
            ), row=1, col=1)

    # Left plot reference line
    r_circle = [100] * len(theta_values)
    fig.add_trace(go.Scatterpolar(
        r=r_circle,
        theta=theta_values,
        mode='lines',
        line=dict(color='black', dash='dash', width=2),
        name='100% Reference',
        showlegend=True, # Show in legend
        hoverinfo='skip'
    ), row=1, col=1)

    # 2. Right Plot: Capped at 100, No Ref Line (was Left Plot)
    max_val_right = 100
    for i, course in enumerate(reversed(bar_courses)):
        if course in data["filtered_courses"]:
            idx = data["filtered_courses"].index(course)
            raw_r_values = [row[idx] for row in data["cumulative_program_share_matrix"]]
            r_values = [min(val, max_val_right) for val in raw_r_values]
            r_values.append(r_values[0])
            
            fig.add_trace(go.Scatterpolar(
                r=r_values,
                theta=theta_values,
                fill='toself',
                fillcolor=colors[i],
                mode='lines', # Fill only
                line=dict(shape='spline', color=colors[i].replace('0.', '1.')),
                name=list(reversed(course_labels))[i],
                showlegend=False, # Already shown in left
                legendgroup=list(reversed(course_labels))[i], # Link to left plot legend item
            ), row=1, col=2)

    # Layout updates
    # Left Polar (Free Scale)
    fig.update_layout(
         polar=dict(
            radialaxis=dict(visible=True, showticklabels=True, gridcolor='lightgrey', linecolor='grey'), # Free range
            angularaxis=dict(tickfont=dict(family=FONT_FAMILY, size=FONT_SIZE-8, color='black'), gridcolor='lightgrey', linecolor='black'),
            bgcolor='white',
            domain = dict(x=[0, 0.45])
        ),
        # Right Polar (Capped)
        polar2=dict(
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, gridcolor='lightgrey', linecolor='grey'),
            angularaxis=dict(tickfont=dict(family=FONT_FAMILY, size=FONT_SIZE-8, color='black'), gridcolor='lightgrey', linecolor='black'), # Smaller font for dense radar
            bgcolor='white',
            domain = dict(x=[0.55, 1])
        ),
        font=dict(family=FONT_FAMILY, size=FONT_SIZE),
        width=2200,
        height=800,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.15, # Move even higher to avoid ANY overlap with (b)
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=50, r=50, t=150, b=50) # Added explicit margins to prevent clipping
    )

    # Add (a) and (b) labels
    fig.add_annotation(x=0.0, y=1.08, xref="paper", yref="paper", text="(a)", showarrow=False, font=dict(family=FONT_FAMILY, size=FONT_SIZE+10))
    fig.add_annotation(x=0.55, y=1.08, xref="paper", yref="paper", text="(b)", showarrow=False, font=dict(family=FONT_FAMILY, size=FONT_SIZE+10))

    return fig

def create_radar_plot_program_share(data, max_value=None, show_reference_line=True):
    """Radar plot for Option A: cumulative share of ALL program CLO links (%). max_value sets fixed cap (e.g. 100)."""
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
            raw_r_values = [row[idx] for row in data["cumulative_program_share_matrix"]]
            if max_value is not None:
                r_values = [min(val, max_value) for val in raw_r_values]
            else:
                r_values = raw_r_values
            r_values.append(r_values[0])
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

    # Radial max: Option A values won’t approach 100; pick a sensible range.
    # Use a dynamic max (e.g., 10% or 1.1×max observed).
    if max_value is not None:
        radial_max = max_value
    else:
        max_r = 0.0
        for row in data["cumulative_program_share_matrix"]:
            max_r = max(max_r, max(row))
        radial_max = max(110, 1.1 * max_r) # Ensure it covers 100

    # Add 100% reference line (using categorical theta to match axes)
    if show_reference_line:
        theta_circle = data["combined_competency_labels"] + [data["combined_competency_labels"][0]]
        r_circle = [100] * len(theta_circle)
        
        fig.add_trace(go.Scatterpolar(
            r=r_circle,
            theta=theta_circle,
            mode='lines',
            line=dict(color='black', dash='dash', width=2),
            name='100% Reference',
            showlegend=False,
            hoverinfo='skip'
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, radial_max],
                showticklabels=True,
                tickfont=dict(family=FONT_FAMILY, size=FONT_SIZE),
                gridcolor='lightgrey',
                linecolor='grey',
                # Title removed as requested
                # title=dict(text="Cumulative share of program CLO links (%)")
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
            x=1.1, y=1,
            xanchor='left',
            yanchor='top'
        ),
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


def export_to_excel(data, filename="heatmap_data.xlsx"):
    """Export cumulative matrices to Excel."""
    try:
        # Create DataFrames
        df_cum_share = pd.DataFrame(
            data["cumulative_program_share_matrix"],
            index=data["combined_competency_labels"],
            columns=data["x_labels"]
        )
        
        df_cum_percent = pd.DataFrame(
            data["cumulative_percentage_matrix"],
            index=data["combined_competency_labels"],
            columns=data["x_labels"]
        )

        with pd.ExcelWriter(filename) as writer:
            df_cum_share.to_excel(writer, sheet_name='Program Share (Opt A)')
            df_cum_percent.to_excel(writer, sheet_name='Within Course (Orig)')
            
        print(f"Data exported to {filename}")
        
    except Exception as e:
        print(f"Error exporting to Excel: {e}")

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

    # Option A figures (program-share based)
    # 1. Saturated at Max (auto-scaled)
    heatmap_program_share_cum_max = create_heatmap_program_share(data, cumulative=True, zmax=None)
    # 2. Saturated at 100%
    heatmap_program_share_cum_100 = create_heatmap_program_share(data, cumulative=True, zmax=100)
    
    # Radar plots (Program Share)
    # 1. Capped at 100%
    radar_program_share_100 = create_radar_plot_program_share(data, max_value=100, show_reference_line=False)
    # 2. Free Scaling (Max)
    radar_program_share_max = create_radar_plot_program_share(data, max_value=None, show_reference_line=True)
    # 3. Combined Side-by-Side Radar
    radar_program_share_combined = create_side_by_side_radar(data)
    
    # 3. Side-by-side Combined Heatmap
    heatmap_program_share_combined = create_side_by_side_heatmaps(data)

    # Create the new assessment vs competency heatmap
    assessment_competency_heatmap = create_assessment_competency_heatmap()

    # Save as HTML files
    heatmap_non_cum.write_html("heatmap_non_cum.html")
    heatmap_cum.write_html("heatmap_cum.html")
    radar_plot.write_html("radar_plot.html")
    heatmap_program_share_cum_max.write_html("heatmap_program_share_cum_max.html")
    heatmap_program_share_cum_100.write_html("heatmap_program_share_cum_100.html")
    heatmap_program_share_combined.write_html("heatmap_program_share_combined.html")
    radar_program_share_100.write_html("radar_program_share_100.html")
    radar_program_share_max.write_html("radar_program_share_max.html")
    radar_program_share_combined.write_html("radar_program_share_combined.html")
    
    if assessment_competency_heatmap:
        assessment_competency_heatmap.write_html("assessment_competency_heatmap.html")

    # Save as static image files
    heatmap_non_cum.write_image("heatmap_non_cum.png", scale=EXPORT_SCALE)
    heatmap_cum.write_image("heatmap_cum.png", scale=EXPORT_SCALE)
    radar_plot.write_image("radar_plot.png", scale=EXPORT_SCALE)
    heatmap_program_share_cum_max.write_image("heatmap_program_share_cum_max.png", scale=EXPORT_SCALE)
    heatmap_program_share_cum_100.write_image("heatmap_program_share_cum_100.png", scale=EXPORT_SCALE)
    heatmap_program_share_combined.write_image("heatmap_program_share_combined.png", scale=EXPORT_SCALE)
    radar_program_share_100.write_image("radar_program_share_100.png", scale=EXPORT_SCALE)
    radar_program_share_max.write_image("radar_program_share_max.png", scale=EXPORT_SCALE)
    radar_program_share_combined.write_image("radar_program_share_combined.png", scale=EXPORT_SCALE)
    
    if assessment_competency_heatmap:
        assessment_competency_heatmap.write_image("assessment_competency_heatmap.png", scale=EXPORT_SCALE)

    # Export data to Excel
    export_to_excel(data, "heatmap_cumulative_data.xlsx")

    print("Visualizations created successfully!")

if __name__ == "__main__":
    main()
