
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# Page Config
# ==========================================
st.set_page_config(
    page_title="Simplified YoY Analysis", 
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Simplified Year-over-Year Analysis")
st.markdown("**Upload farmdata_datasources.csv to analyze corporation-filtered percent points and significant changes**")

# ==========================================
# Analysis Functions
# ==========================================

@st.cache_data
def load_and_analyze_data(df, selected_corps=None, significance_thresholds=None):
    """Load data and perform YoY analysis with corporation filtering"""
    
    # Filter for allowed breeds first
    ALLOWED_BREEDS = {"fleckvieh", "braunvieh", "schwarzbunt"}
    if 'breed' in df.columns:
        
        df = df[df['breed'].str.lower().isin(ALLOWED_BREEDS)].copy()
    
    # Filter by selected corporations if specified
    if selected_corps and 'corporation' in df.columns:
        df = df[df['corporation'].isin(selected_corps)].copy()
    
    # Parameters to analyze
    COLUMNS_TO_ANALYZE = [
        'calf_loss_rate', 'loss_rate', 'attrition_rate', 'herd_size', 
        'intercalving_period', 'slaughter_weight', 'weaning_age', 'fat_content', 
        'farming_method', 'protein_content', 'milk_yield', 'gutting_rate', 
        'manure_type', 'live_weight', 'grazing_days', 'first_calving_age', 'breed'
    ]
    
    BINARY_COLUMNS = ['farming_method', 'manure_type', 'breed']
    
    # Use provided significance thresholds or default values
    SIGNIFICANCE_THRESHOLDS = significance_thresholds or {
        'milk_yield': 0.20,
        'protein_content': 0.08,
        'fat_content': 0.08,
        'attrition_rate': 0.08,  # absolute
        'loss_rate': 0.02,  # absolute
        'live_weight': 0.10,
        'slaughter_weight': 0.10,
        'gutting_rate': 0.10,
        'grazing_days': 30,  # absolute
        'intercalving_period': 0.10,
        'first_calving_age': 0.10,
        'weaning_age': 2,  # absolute (weeks)
        'calf_loss_rate': 0.02,  # absolute
    }
    
    # Parameters that are in percentage format (for percent points analysis)
    PERCENTAGE_PARAMETERS = {
        'attrition_rate', 'loss_rate', 'calf_loss_rate', 'fat_content', 'protein_content'
    }
    
    # Filter for available columns
    available_columns = [col for col in COLUMNS_TO_ANALYZE if col in df.columns]
    
    changes_data = []
    id_column = 'farm_id' if 'farm_id' in df.columns else 'farm_name'
    
    # Calculate YoY changes
    for farm_id, farm_data in df.groupby(id_column):
        farm_data = farm_data.sort_values('year')
        years = farm_data['year'].unique()
        
        for i in range(len(years)-1):
            current_year = years[i+1]
            previous_year = years[i]
            
            current_data = farm_data[farm_data['year'] == current_year].iloc[0]
            previous_data = farm_data[farm_data['year'] == previous_year].iloc[0]
            
            for column in available_columns:
                if column in BINARY_COLUMNS:
                    # Binary columns
                    current_val = current_data[column]
                    previous_val = previous_data[column]
                    
                    if pd.notna(current_val) and pd.notna(previous_val):
                        change_occurred = current_val != previous_val
                        changes_data.append({
                            'farm_id': farm_id,
                            'parameter': column,
                            'year_comparison': f"{previous_year}-{current_year}",
                            'current_value': current_val,
                            'previous_value': previous_val,
                            'change_type': 'binary',
                            'changed': change_occurred,
                            'is_significant': change_occurred,
                            'assessment_id': current_data.get('assessment_id', f"{farm_id}_{current_year}"),
                            'absolute_change': None,
                            'percent_change': None,
                            'is_percentage_param': False
                        })
                else:
                    # Numerical columns
                    current_val = pd.to_numeric(current_data[column], errors='coerce')
                    previous_val = pd.to_numeric(previous_data[column], errors='coerce')
                    
                    if pd.notna(current_val) and pd.notna(previous_val) and previous_val != 0:
                        absolute_change = current_val - previous_val
                        percent_change = (current_val - previous_val) / previous_val
                        
                        # Check significance
                        is_significant = False
                        if column in SIGNIFICANCE_THRESHOLDS:
                            threshold = SIGNIFICANCE_THRESHOLDS[column]
                            
                            if column in ['attrition_rate', 'loss_rate', 'calf_loss_rate', 'grazing_days', 'weaning_age']:
                                # Absolute thresholds
                                is_significant = abs(absolute_change) > threshold
                            elif column == 'herd_size':
                                # Special formula: 0.06 + (0.50 * Math.exp(-n/50))
                                dynamic_threshold = 0.06 + (0.50 * np.exp(-previous_val/50))
                                is_significant = abs(percent_change) > dynamic_threshold
                            else:
                                # Percentage thresholds
                                is_significant = abs(percent_change) > threshold
                        
                        changes_data.append({
                            'farm_id': farm_id,
                            'parameter': column,
                            'year_comparison': f"{previous_year}-{current_year}",
                            'current_value': current_val,
                            'previous_value': previous_val,
                            'absolute_change': absolute_change,
                            'percent_change': percent_change,
                            'change_type': 'numerical',
                            'changed': absolute_change != 0,
                            'is_significant': is_significant,
                            'assessment_id': current_data.get('assessment_id', f"{farm_id}_{current_year}"),
                            'is_percentage_param': column in PERCENTAGE_PARAMETERS
                        })
    
    return pd.DataFrame(changes_data)

def calculate_avg_changes_summary(changes_df):
    """Calculate average changes for percentage parameters"""
    
    # Filter for percentage parameters only
    percentage_changes = changes_df[
        (changes_df['is_percentage_param'] == True) & 
        (changes_df['change_type'] == 'numerical')
    ].copy()
    
    if percentage_changes.empty:
        return pd.DataFrame()
    
    summary_data = []
    
    for param in percentage_changes['parameter'].unique():
        param_data = percentage_changes[percentage_changes['parameter'] == param]
        
        if len(param_data) > 0:
            # Calculate statistics for all changes - use absolute values
            avg_pct_change = abs(param_data['percent_change'].mean()) * 100
            avg_pp_change = abs(param_data['absolute_change'].mean()) * 100  # Convert from decimal to percentage points (0.02 -> 2 pp)
            significant_count = param_data['is_significant'].sum()
            total_count = len(param_data)
            
            # Calculate statistics for significant changes only
            significant_data = param_data[param_data['is_significant'] == True]
            if len(significant_data) > 0:
                avg_pp_change_significant = abs(significant_data['absolute_change'].mean()) * 100  # Convert to percentage points
            else:
                avg_pp_change_significant = None
            
            summary_data.append({
                'Parameter': param,
                'Avg % Change': avg_pct_change,
                'Avg Percentage Points Change': avg_pp_change,
                'Avg PP Change (Significant Only)': avg_pp_change_significant,
                'Significant Changes': significant_count,
                'Total Changes': total_count,
                'Significance Rate': (significant_count / total_count * 100) if total_count > 0 else 0
            })
    
    return pd.DataFrame(summary_data)

def create_change_distribution_plots(changes_df, selected_parameters=None):
    """Create distribution plots for parameter changes"""
    
    # Filter out binary parameters
    BINARY_COLUMNS = ['farming_method', 'manure_type', 'breed']
    numerical_changes = changes_df[
        (changes_df['change_type'] == 'numerical') & 
        (~changes_df['parameter'].isin(BINARY_COLUMNS))
    ].copy()
    
    if numerical_changes.empty:
        return None
        
    # Filter by selected parameters if specified
    if selected_parameters:
        numerical_changes = numerical_changes[numerical_changes['parameter'].isin(selected_parameters)]
    
    return numerical_changes

def plot_parameter_distribution(changes_df, parameter, plot_type='histogram', nbins=30, use_absolute_values=True, show_each_value=False):
    """Create a distribution plot for a specific parameter"""
    
    param_data = changes_df[changes_df['parameter'] == parameter].copy()
    
    if param_data.empty:
        return None
    
    # Define parameter display preferences
    PERCENTAGE_POINTS_PARAMS = {'attrition_rate', 'calf_loss_rate', 'loss_rate'}
    PERCENT_CHANGE_PARAMS = {'live_weight', 'slaughter_weight', 'gutting_rate', 'first_calving_age', 
                           'intercalving_period', 'milk_yield', 'protein_content', 'fat_content'}
    DAYS_PARAMS = {'grazing_days'}
    WEEKS_PARAMS = {'weaning_age'}
    
    # Determine display type for this parameter
    if parameter in PERCENTAGE_POINTS_PARAMS:
        display_type = 'percentage_points'
    elif parameter in PERCENT_CHANGE_PARAMS:
        display_type = 'percent_change'
    elif parameter in DAYS_PARAMS:
        display_type = 'days'
    elif parameter in WEEKS_PARAMS:
        display_type = 'weeks'
    else:
        # Default behavior based on original parameter classification
        is_percentage_param = param_data['is_percentage_param'].iloc[0] if not param_data.empty else False
        display_type = 'percentage_points' if is_percentage_param else 'percent_change'
    
    # Create the plot based on type
    if plot_type == 'histogram':
        # Determine data and labels based on display type
        if display_type == 'percentage_points':
            x_data = (param_data['absolute_change'] * 100).copy().abs()  # Convert decimal to percentage points
            title_suffix = " (Percentage Points)"
            x_label = "Change (Percentage Points)"
            tick_format = None
        elif display_type == 'percent_change':
            x_data = (param_data['percent_change'] * 100).copy().abs()  # Convert to % and absolute
            title_suffix = " (Percent Change)"
            x_label = "Change (%)"
            tick_format = None
        elif display_type == 'days':
            x_data = param_data['absolute_change'].copy().abs()  # Always absolute
            title_suffix = " (Days)"
            x_label = "Change (Days)"
            tick_format = None
        elif display_type == 'weeks':
            x_data = param_data['absolute_change'].copy().abs()  # Always absolute
            title_suffix = " (Weeks)"
            x_label = "Change (Weeks)"
            tick_format = None
        else:
            # Fallback
            x_data = param_data['percent_change'].copy().abs()
            title_suffix = " (Percent Change)"
            x_label = "Change (%)"
            tick_format = '.1%'
        
        # Determine number of bins - fix granularity issue
        if show_each_value:
            # For maximum granularity, let plotly determine optimal bins based on unique values
            fig = px.histogram(
                x=x_data,
                title=f'{parameter.replace("_", " ").title()} - Distribution of Changes{title_suffix}',
                labels={'x': x_label, 'count': 'Number of Datapoints'}
            )
            # Force bins to be based on unique values
            unique_values = sorted(x_data.dropna().unique())
            if len(unique_values) <= 200:  # Only if reasonable number
                bin_edges = unique_values + [unique_values[-1] + (unique_values[1] - unique_values[0]) if len(unique_values) > 1 else unique_values[0] + 0.001]
                fig.update_traces(xbins=dict(start=min(unique_values), end=max(unique_values), size=(max(unique_values) - min(unique_values)) / len(unique_values) if len(unique_values) > 1 else 0.001))
        else:
            # Calculate proper bin width for the specified number of bins
            data_range = x_data.max() - x_data.min()
            if data_range > 0:
                bin_width = data_range / nbins
                fig = px.histogram(
                    x=x_data,
                    title=f'{parameter.replace("_", " ").title()} - Distribution of Changes{title_suffix}',
                    labels={'x': x_label, 'count': 'Number of Datapoints'}
                )
                fig.update_traces(xbins=dict(start=x_data.min(), end=x_data.max(), size=bin_width))
            else:
                fig = px.histogram(
                    x=x_data,
                    title=f'{parameter.replace("_", " ").title()} - Distribution of Changes{title_suffix}',
                    labels={'x': x_label, 'count': 'Number of Datapoints'},
                    nbins=nbins
                )
        
        # Update layout
        layout_update = {
            'xaxis_title': x_label,
            'yaxis_title': "Number of Datapoints",
            'showlegend': False
        }
        
        if tick_format:
            layout_update['xaxis_tickformat'] = tick_format
            
        fig.update_layout(**layout_update)
    
    elif plot_type == 'box':
        # Determine data and labels based on display type
        if display_type == 'percentage_points':
            y_data = (param_data['absolute_change'] * 100).copy().abs()  # Convert decimal to percentage points
            title_suffix = " (Percentage Points)"
            y_label = "Change (Percentage Points)"
            tick_format = None
        elif display_type == 'percent_change':
            y_data = (param_data['percent_change'] * 100).copy().abs()  # Convert to % and absolute
            title_suffix = " (Percent Change)"
            y_label = "Change (%)"
            tick_format = None
        elif display_type == 'days':
            y_data = param_data['absolute_change'].copy().abs()  # Always absolute
            title_suffix = " (Days)"
            y_label = "Change (Days)"
            tick_format = None
        elif display_type == 'weeks':
            y_data = param_data['absolute_change'].copy().abs()  # Always absolute
            title_suffix = " (Weeks)"
            y_label = "Change (Weeks)"
            tick_format = None
        else:
            # Fallback
            y_data = param_data['percent_change'].copy().abs()
            title_suffix = " (Percent Change)"
            y_label = "Change (%)"
            tick_format = '.1%'
            
        fig = px.box(
            y=y_data,
            title=f'{parameter.replace("_", " ").title()} - Box Plot of Changes{title_suffix}',
            labels={'y': y_label}
        )
        
        if tick_format:
            fig.update_layout(yaxis_tickformat=tick_format)
    
    elif plot_type == 'scatter':
        # Scatter plot showing relationship between absolute and percent change
        # Always use absolute values for both axes
        
        if display_type == 'percentage_points':
            x_data = (param_data['absolute_change'] * 100).copy().abs()  # Convert decimal to percentage points
            x_label = 'Change (Percentage Points)'
        elif display_type == 'days':
            x_data = param_data['absolute_change'].copy().abs()
            x_label = 'Change (Days)'
        elif display_type == 'weeks':
            x_data = param_data['absolute_change'].copy().abs()
            x_label = 'Change (Weeks)'
        else:
            x_data = param_data['absolute_change'].copy().abs()
            x_label = 'Absolute Change'
            
        # Y-axis always shows absolute percent change
        y_data = (param_data['percent_change'] * 100).copy().abs()
        
        fig = px.scatter(
            x=x_data,
            y=y_data,
            color=param_data['is_significant'],
            title=f'{parameter.replace("_", " ").title()} - Absolute Change vs Percent Change',
            labels={
                'x': x_label,
                'y': 'Percent Change (%)',
                'color': 'Significant Change'
            },
            hover_data=[param_data['farm_id'], param_data['year_comparison']]
        )
    
    return fig

def create_distribution_table(data, display_type, table_granularity=0.01):
    """Create a detailed table showing the distribution of change values"""
    
    if data.empty:
        return pd.DataFrame()
    
    # Remove NaN values
    clean_data = data.dropna()
    
    if len(clean_data) == 0:
        return pd.DataFrame()
    
    data_min = clean_data.min()
    data_max = clean_data.max()
    
    # Create bins based on granularity
    if data_max == data_min:
        # All values are the same
        return pd.DataFrame({
            'Range': [f"{data_min:.3f}"],
            'Count': [len(clean_data)],
            'Percentage': [100.0]
        })
    
    # Calculate number of bins based on range and granularity
    num_bins = max(1, int((data_max - data_min) / table_granularity))
    # Cap at reasonable number for performance
    num_bins = min(num_bins, 1000)
    
    # Create bin edges
    bin_edges = pd.cut(clean_data, bins=num_bins, include_lowest=True, duplicates='drop')
    value_counts = bin_edges.value_counts().sort_index()
    
    # Create table data
    table_data = []
    total_count = len(clean_data)
    
    for interval, count in value_counts.items():
        if count > 0:
            left = interval.left
            right = interval.right
            percentage = (count / total_count) * 100
            
            # Format the range based on display type
            if display_type == 'percentage_points':
                range_str = f"{left:.3f} - {right:.3f} pp"
            elif display_type == 'percent_change':
                range_str = f"{left:.2f}% - {right:.2f}%"
            elif display_type in ['days', 'weeks']:
                unit = display_type.replace('s', '')  # Remove 's' to get singular
                range_str = f"{left:.1f} - {right:.1f} {unit}s"
            else:
                range_str = f"{left:.3f} - {right:.3f}"
            
            table_data.append({
                'Range': range_str,
                'Count': count,
                'Percentage': f"{percentage:.1f}%",
                'Sort_Key': left
            })
    
    if not table_data:
        return pd.DataFrame()
    
    # Convert to DataFrame and sort by value
    table_df = pd.DataFrame(table_data)
    table_df = table_df.sort_values('Sort_Key').drop('Sort_Key', axis=1)
    
    return table_df

def create_comparison_matrix(changes_df, selected_parameters, nbins=20):
    """Create a matrix showing change distributions for multiple parameters"""
    
    if not selected_parameters or len(selected_parameters) < 2:
        return None
    
    # Create subplots
    n_params = len(selected_parameters)
    cols = min(3, n_params)
    rows = (n_params + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[param.replace("_", " ").title() for param in selected_parameters],
        vertical_spacing=0.15
    )
    
    for i, param in enumerate(selected_parameters):
        row = (i // cols) + 1
        col = (i % cols) + 1
        
        param_data = changes_df[changes_df['parameter'] == param]
        
        if not param_data.empty:
            is_percentage_param = param_data['is_percentage_param'].iloc[0]
            
            if is_percentage_param:
                x_data = param_data['absolute_change']
            else:
                x_data = param_data['percent_change']
            
            fig.add_trace(
                go.Histogram(x=x_data, nbinsx=nbins, name=param, showlegend=False),
                row=row, col=col
            )
    
    fig.update_layout(height=300*rows, title_text="Parameter Change Distributions Comparison")
    return fig


# ==========================================
# Streamlit App Interface
# ==========================================

# Sidebar for data upload and filtering
with st.sidebar:
    st.header("📁 Data Upload")
    
    # Upload farmdata_datasources.csv file
    st.info("🎯 Please upload the **farmdata_datasources.csv** file to begin analysis.")
    uploaded_file = st.file_uploader("Upload farmdata_datasources.csv", type=['csv'])
    
    if uploaded_file is not None:
        original_df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded file: {original_df.shape[0]:,} rows")
    else:
        original_df = None
    
    if original_df is not None:
        # Apply breed filtering first
        ALLOWED_BREEDS = {"fleckvieh", "braunvieh", "schwarzbunt"}
        if 'breed' in original_df.columns:
            breed_filtered_df = original_df[original_df['breed'].str.lower().isin(ALLOWED_BREEDS)].copy()
            st.write(f"**Farms (breed filtered):** {breed_filtered_df['farm_id'].nunique() if 'farm_id' in breed_filtered_df.columns else 'N/A'}")
            st.write("**Breeds:** fleckvieh, braunvieh, schwarzbunt")
        else:
            breed_filtered_df = original_df
            st.write(f"**Farms:** {breed_filtered_df['farm_id'].nunique() if 'farm_id' in breed_filtered_df.columns else 'N/A'}")
        
        st.header("🎛️ Filters")
        
        # Corporation filter
        if 'corporation' in breed_filtered_df.columns:
            available_corps = sorted(breed_filtered_df['corporation'].dropna().unique().tolist())
            selected_corps = st.multiselect(
                "Select Corporations:",
                available_corps,
                default=available_corps,
                help="Choose which corporations to include in the analysis"
            )
        else:
            selected_corps = None
        
        # Significance Rules Editor
        st.header("⚙️ Significance Rules")
        
        with st.expander("Edit Significance Thresholds", expanded=False):
            st.markdown("**Adjust the thresholds for determining significant changes:**")
            
            # Default thresholds
            default_thresholds = {
                'milk_yield': 0.20,
                'protein_content': 0.08,
                'fat_content': 0.08,
                'attrition_rate': 0.08,
                'loss_rate': 0.02,
                'live_weight': 0.10,
                'slaughter_weight': 0.10,
                'gutting_rate': 0.10,
                'grazing_days': 30,
                'intercalving_period': 0.10,
                'first_calving_age': 0.10,
                'weaning_age': 2,
                'calf_loss_rate': 0.02,
            }
            
            # Create editable thresholds
            custom_thresholds = {}
            
            st.markdown("**Percentage-based thresholds (relative change):**")
            col1, col2 = st.columns(2)
            
            with col1:
                custom_thresholds['milk_yield'] = st.number_input(
                    "Milk Yield", 
                    min_value=0.001, max_value=1.0, 
                    value=default_thresholds['milk_yield'], 
                    step=0.001,
                    format="%.3f",
                    help="Threshold for milk yield % change"
                )
                custom_thresholds['protein_content'] = st.number_input(
                    "Protein Content", 
                    min_value=0.001, max_value=1.0, 
                    value=default_thresholds['protein_content'], 
                    step=0.001,
                    format="%.3f",
                    help="Threshold for protein content % change"
                )
                custom_thresholds['fat_content'] = st.number_input(
                    "Fat Content", 
                    min_value=0.001, max_value=1.0, 
                    value=default_thresholds['fat_content'], 
                    step=0.001,
                    format="%.3f",
                    help="Threshold for fat content % change"
                )
                custom_thresholds['live_weight'] = st.number_input(
                    "Live Weight", 
                    min_value=0.001, max_value=1.0, 
                    value=default_thresholds['live_weight'], 
                    step=0.001,
                    format="%.3f",
                    help="Threshold for live weight % change"
                )
            
            with col2:
                custom_thresholds['slaughter_weight'] = st.number_input(
                    "Slaughter Weight", 
                    min_value=0.001, max_value=1.0, 
                    value=default_thresholds['slaughter_weight'], 
                    step=0.001,
                    format="%.3f",
                    help="Threshold for slaughter weight % change"
                )
                custom_thresholds['gutting_rate'] = st.number_input(
                    "Gutting Rate", 
                    min_value=0.001, max_value=1.0, 
                    value=default_thresholds['gutting_rate'], 
                    step=0.001,
                    format="%.3f",
                    help="Threshold for gutting rate % change"
                )
                custom_thresholds['intercalving_period'] = st.number_input(
                    "Intercalving Period", 
                    min_value=0.001, max_value=1.0, 
                    value=default_thresholds['intercalving_period'], 
                    step=0.001,
                    format="%.3f",
                    help="Threshold for intercalving period % change"
                )
                custom_thresholds['first_calving_age'] = st.number_input(
                    "First Calving Age", 
                    min_value=0.001, max_value=1.0, 
                    value=default_thresholds['first_calving_age'], 
                    step=0.001,
                    format="%.3f",
                    help="Threshold for first calving age % change"
                )
            
            st.markdown("**Absolute-value thresholds (fixed amounts):**")
            col3, col4 = st.columns(2)
            
            with col3:
                custom_thresholds['attrition_rate'] = st.number_input(
                    "Attrition Rate (pp)", 
                    min_value=0.001, max_value=1.0, 
                    value=default_thresholds['attrition_rate'], 
                    step=0.001,
                    format="%.3f",
                    help="Threshold for attrition rate in percentage points"
                )
                custom_thresholds['loss_rate'] = st.number_input(
                    "Loss Rate (pp)", 
                    min_value=0.001, max_value=0.5, 
                    value=default_thresholds['loss_rate'], 
                    step=0.001,
                    format="%.3f",
                    help="Threshold for loss rate in percentage points"
                )
                custom_thresholds['calf_loss_rate'] = st.number_input(
                    "Calf Loss Rate (pp)", 
                    min_value=0.001, max_value=0.5, 
                    value=default_thresholds['calf_loss_rate'], 
                    step=0.001,
                    format="%.3f",
                    help="Threshold for calf loss rate in percentage points"
                )
            
            with col4:
                custom_thresholds['grazing_days'] = st.number_input(
                    "Grazing Days", 
                    min_value=1, max_value=100, 
                    value=int(default_thresholds['grazing_days']), 
                    step=1,
                    help="Threshold for grazing days (absolute change)"
                )
                custom_thresholds['weaning_age'] = st.number_input(
                    "Weaning Age (weeks)", 
                    min_value=0.1, max_value=10.0, 
                    value=float(default_thresholds['weaning_age']), 
                    step=0.1,
                    format="%.2f",
                    help="Threshold for weaning age in weeks"
                )
            
            st.info("💡 **Note:** Herd size uses a dynamic formula: 0.06 + (0.50 × e^(-n/50))")
            
            # Store custom thresholds in session state for use in analysis
            if 'custom_significance_thresholds' not in st.session_state:
                st.session_state.custom_significance_thresholds = custom_thresholds
            else:
                st.session_state.custom_significance_thresholds.update(custom_thresholds)

# Main content area
if original_df is not None:
    # Run analysis with filters
    with st.spinner('🔄 Analyzing year-over-year changes...'):
        # Get custom thresholds from session state
        custom_thresholds = st.session_state.get('custom_significance_thresholds', None)
        changes_df = load_and_analyze_data(original_df, selected_corps=selected_corps, significance_thresholds=custom_thresholds)
        
    if changes_df.empty:
        st.warning("⚠️ No year-over-year changes found in the data.")
        st.stop()
    
    # Calculate summary for percentage parameters
    avg_summary = calculate_avg_changes_summary(changes_df)
    
    # Display key metrics
    st.header("📊 Analysis Summary")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Changes", f"{len(changes_df):,}")
    with col2:
        significant_count = changes_df['is_significant'].sum()
        st.metric("Significant Changes", f"{significant_count:,}", 
                 f"{(significant_count/len(changes_df)*100):.1f}%")
    with col3:
        st.metric("Farms Analyzed", f"{changes_df['farm_id'].nunique():,}")
    
    # Show average changes for percentage parameters
    if not avg_summary.empty:
        st.header("📈 Average Changes for Percentage Parameters")
        st.markdown("*These parameters are naturally in percentage format, so percent points changes are most meaningful*")
        
        # Format the summary for display
        display_summary = avg_summary.copy()
        display_summary['Avg % Change'] = display_summary['Avg % Change'].apply(lambda x: f"{x:.2f}%")
        display_summary['Avg Percentage Points Change'] = display_summary['Avg Percentage Points Change'].apply(lambda x: f"avg percentage points change {x:.1f} pp")
        display_summary['Avg PP Change (Significant Only)'] = display_summary['Avg PP Change (Significant Only)'].apply(
            lambda x: f"avg percentage points change {x:.1f} pp" if pd.notna(x) else "No significant changes"
        )
        display_summary['Significance Rate'] = display_summary['Significance Rate'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(display_summary, width='stretch', hide_index=True)
    
    # Significant changes analysis
    st.header("⚠️ All Assessments with Significant Changes")
    
    # Filter for significant changes only
    significant_changes = changes_df[changes_df['is_significant'] == True].copy()
    
    if not significant_changes.empty:
        st.info(f"Found **{len(significant_changes):,}** significant changes across **{significant_changes['assessment_id'].nunique():,}** assessments")
        
        # Create significance rules reference with current thresholds
        with st.expander("📋 Significance Rules Reference"):
            # Get current thresholds (custom or default)
            current_thresholds = custom_thresholds if custom_thresholds else {
                'milk_yield': 0.20, 'protein_content': 0.08, 'fat_content': 0.08,
                'attrition_rate': 0.08, 'loss_rate': 0.02, 'live_weight': 0.10,
                'slaughter_weight': 0.10, 'gutting_rate': 0.10, 'grazing_days': 30,
                'intercalving_period': 0.10, 'first_calving_age': 0.10, 'weaning_age': 2,
                'calf_loss_rate': 0.02
            }
            
            rules_data = [
                {"Datapoint": "breed", "Rule": "Binary change"},
                {"Datapoint": "farming_method", "Rule": "Binary change"},
                {"Datapoint": "milk_yield", "Rule": f"±{current_thresholds.get('milk_yield', 0.20)*100:.0f}%"},
                {"Datapoint": "protein_content", "Rule": f"±{current_thresholds.get('protein_content', 0.08)*100:.0f}%"},
                {"Datapoint": "fat_content", "Rule": f"±{current_thresholds.get('fat_content', 0.08)*100:.0f}%"},
                {"Datapoint": "herd_size", "Rule": "Formula: 0.06 + (0.50 × e^(-n/50))"},
                {"Datapoint": "attrition_rate", "Rule": f">{current_thresholds.get('attrition_rate', 0.08)*100:.0f} percentage points"},
                {"Datapoint": "loss_rate", "Rule": f">{current_thresholds.get('loss_rate', 0.02)*100:.1f} percentage points"},
                {"Datapoint": "live_weight", "Rule": f"±{current_thresholds.get('live_weight', 0.10)*100:.0f}%"},
                {"Datapoint": "slaughter_weight", "Rule": f"±{current_thresholds.get('slaughter_weight', 0.10)*100:.0f}%"},
                {"Datapoint": "gutting_rate", "Rule": f"±{current_thresholds.get('gutting_rate', 0.10)*100:.0f}%"},
                {"Datapoint": "grazing_days", "Rule": f">{current_thresholds.get('grazing_days', 30):.0f} days"},
                {"Datapoint": "intercalving_period", "Rule": f"±{current_thresholds.get('intercalving_period', 0.10)*100:.0f}%"},
                {"Datapoint": "first_calving_age", "Rule": f"±{current_thresholds.get('first_calving_age', 0.10)*100:.0f}%"},
                {"Datapoint": "weaning_age", "Rule": f">{current_thresholds.get('weaning_age', 2):.1f} weeks"},
                {"Datapoint": "calf_loss_rate", "Rule": f">{current_thresholds.get('calf_loss_rate', 0.02)*100:.1f} percentage points"},
                {"Datapoint": "manure_type", "Rule": "Binary change"}
            ]
            st.dataframe(pd.DataFrame(rules_data), width='stretch', hide_index=True)
        
        # Add filters for significant changes
        col1, col2, col3 = st.columns(3)
        
        with col1:
            param_filter_sig = st.selectbox(
                "Filter by parameter:",
                ["All"] + sorted(significant_changes['parameter'].unique().tolist()),
                key="sig_param_filter"
            )
        
        with col2:
            year_filter_sig = st.selectbox(
                "Filter by year comparison:",
                ["All"] + sorted(significant_changes['year_comparison'].unique().tolist()),
                key="sig_year_filter"
            )
        
        with col3:
            change_type_filter = st.selectbox(
                "Filter by change type:",
                ["All", "Numerical", "Binary"],
                key="sig_type_filter"
            )
        
        # Apply filters
        filtered_sig = significant_changes.copy()
        
        if param_filter_sig != "All":
            filtered_sig = filtered_sig[filtered_sig['parameter'] == param_filter_sig]
        
        if year_filter_sig != "All":
            filtered_sig = filtered_sig[filtered_sig['year_comparison'] == year_filter_sig]
        
        if change_type_filter != "All":
            if change_type_filter == "Numerical":
                filtered_sig = filtered_sig[filtered_sig['change_type'] == 'numerical']
            else:
                filtered_sig = filtered_sig[filtered_sig['change_type'] == 'binary']
        
        # Prepare display data with significance explanation
        display_sig = filtered_sig.copy()
        
        # Add significance explanation column
        def explain_significance(row):
            if row['change_type'] == 'binary':
                return f"Changed from '{row['previous_value']}' to '{row['current_value']}'"
            else:
                pct_change = abs(row['percent_change'] * 100) if pd.notna(row['percent_change']) else 0
                abs_change = abs(row['absolute_change']) if pd.notna(row['absolute_change']) else 0
                
                # Format based on parameter type
                if row['is_percentage_param']:
                    # Convert decimal to percentage points: 0.02 -> 2 pp
                    pp_change = abs_change * 100
                    return f"{pp_change:.2f} percentage points ({pct_change:.1f}%)"
                elif row['parameter'] in ['grazing_days', 'weaning_age']:
                    unit = "days" if row['parameter'] == 'grazing_days' else "weeks"
                    return f"{abs_change:.1f} {unit} ({pct_change:.1f}%)"
                else:
                    return f"{pct_change:.1f}% ({abs_change:.3f} units)"
        
        display_sig['Change Explanation'] = display_sig.apply(explain_significance, axis=1)
        display_sig['From → To'] = display_sig.apply(lambda x: f"{x['previous_value']} → {x['current_value']}", axis=1)
        
        # Select columns to display
        display_cols = ['assessment_id', 'farm_id', 'parameter', 'year_comparison', 'From → To', 'Change Explanation']
        
        st.write(f"**Showing {len(filtered_sig):,} significant changes**")
        st.dataframe(
            display_sig[display_cols].sort_values(['parameter', 'assessment_id']), 
            width='stretch',
            hide_index=True,
            height=400
        )
        
        # Export significant changes
        export_sig = filtered_sig.copy()
        export_sig['change_explanation'] = export_sig.apply(explain_significance, axis=1)
        
        csv_buffer = BytesIO()
        export_sig[['assessment_id', 'farm_id', 'parameter', 'year_comparison', 'current_value', 'previous_value', 'change_explanation', 'absolute_change', 'percent_change']].to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        st.download_button(
            label="⬇️ Download Significant Changes (CSV)",
            data=csv_buffer,
            file_name="significant_changes_detailed.csv",
            mime="text/csv"
        )
        
    else:
        st.info("No significant changes found with current filters.")
    
    # ==========================================
    # Parameter Change Distribution Visualization
    # ==========================================
    
    st.header("📈 Parameter Change Distribution Visualizations")
    st.markdown("*Visualize the distribution of changes for each parameter (excluding binary parameters)*")
    
    # Get numerical parameters (exclude binary ones)
    BINARY_COLUMNS = ['farming_method', 'manure_type', 'breed']
    
    # Debug: show all unique parameters
    st.info(f"📊 **All parameters found**: {sorted(changes_df['parameter'].unique().tolist())}")
    
    # Get numerical parameters with more robust filtering
    numerical_params = []
    for param in changes_df['parameter'].unique():
        if param not in BINARY_COLUMNS:
            param_data = changes_df[changes_df['parameter'] == param]
            # Check if any rows have numerical data
            if not param_data.empty and (param_data['change_type'] == 'numerical').any():
                numerical_params.append(param)
    
    numerical_params = sorted(numerical_params)
    
    st.info(f"📈 **Numerical parameters available for visualization**: {numerical_params}")
    
    # Show debug information in an expander
    with st.expander("🔍 Debug Information - Parameter Detection"):
        st.write("**Parameter Analysis:**")
        
        debug_data = []
        for param in sorted(changes_df['parameter'].unique()):
            param_data = changes_df[changes_df['parameter'] == param]
            change_types = param_data['change_type'].unique()
            total_count = len(param_data)
            is_binary = param in BINARY_COLUMNS
            has_numerical = (param_data['change_type'] == 'numerical').any()
            
            debug_data.append({
                'Parameter': param,
                'Total Records': total_count,
                'Change Types': ', '.join(change_types),
                'Is Binary': is_binary,
                'Has Numerical Data': has_numerical,
                'Will Show in Viz': param in numerical_params
            })
        
        debug_df = pd.DataFrame(debug_data)
        st.dataframe(debug_df, hide_index=True, use_container_width=True)
        
        if debug_data:
            missing_params = [d['Parameter'] for d in debug_data if not d['Will Show in Viz'] and not d['Is Binary']]
            if missing_params:
                st.warning(f"⚠️ **Parameters not showing in visualization**: {missing_params}")
                st.write("**Possible reasons:** No numerical change data, all values are NaN, or filtering issue")
    
    if numerical_params:
        # Create tabs for different visualization types
        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📊 Individual Parameters", "🔍 Parameter Comparison", "📋 Summary Statistics"])
        
        with viz_tab1:
            st.subheader("Individual Parameter Analysis")
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                selected_param = st.selectbox(
                    "Select parameter to visualize:",
                    numerical_params,
                    help="Choose a parameter to see its change distribution"
                )
            
            with col2:
                plot_type = st.selectbox(
                    "Plot type:",
                    ["histogram", "box", "scatter"],
                    help="Histogram: distribution of changes, Box: quartiles and outliers, Scatter: absolute vs percent change"
                )
            
            with col3:
                if plot_type == "histogram":
                    nbins = st.slider(
                        "Number of bins:",
                        min_value=10,
                        max_value=200,
                        value=50,
                        step=10,
                        help="Control the granularity of the histogram - higher values show more detail"
                    )
                else:
                    nbins = 50
            
            # Additional options for histogram
            if plot_type == "histogram" and selected_param:
                st.write("**Histogram Options:**")
                col1, col2 = st.columns(2)
                
                with col1:
                    show_each_value = st.checkbox(
                        "Show each value as separate bin",
                        help="Maximum granularity: each unique change value gets its own bin"
                    )
                
                with col2:
                    show_table = st.checkbox(
                        "Show distribution table",
                        value=True,
                        help="Display detailed table with value ranges and counts"
                    )
                
                # Table granularity control
                if show_table:
                    st.write("**Table Granularity:**")
                    
                    # Determine default granularity based on parameter type
                    PERCENTAGE_POINTS_PARAMS = {'attrition_rate', 'calf_loss_rate', 'loss_rate'}
                    PERCENT_CHANGE_PARAMS = {'live_weight', 'slaughter_weight', 'gutting_rate', 'first_calving_age', 
                                           'intercalving_period', 'milk_yield', 'protein_content', 'fat_content'}
                    DAYS_PARAMS = {'grazing_days'}
                    WEEKS_PARAMS = {'weaning_age'}
                    
                    if selected_param in PERCENTAGE_POINTS_PARAMS:
                        default_granularity = 0.01
                        min_val, max_val = 0.001, 1.0
                        step_val = 0.001
                        help_text = "Bin width in percentage points (e.g., 0.01 = groups like 0.01-0.02 pp)"
                    elif selected_param in PERCENT_CHANGE_PARAMS:
                        default_granularity = 1.0
                        min_val, max_val = 0.1, 50.0
                        step_val = 0.1
                        help_text = "Bin width in percent (e.g., 1.0 = groups like 1.0-2.0%)"
                    elif selected_param in DAYS_PARAMS:
                        default_granularity = 5.0
                        min_val, max_val = 0.5, 100.0
                        step_val = 0.5
                        help_text = "Bin width in days (e.g., 5.0 = groups like 5-10 days)"
                    elif selected_param in WEEKS_PARAMS:
                        default_granularity = 0.1
                        min_val, max_val = 0.1, 10.0
                        step_val = 0.1
                        help_text = "Bin width in weeks (e.g., 0.1 = groups like 0.1-0.2 weeks)"
                    else:
                        default_granularity = 0.01
                        min_val, max_val = 0.001, 1.0
                        step_val = 0.001
                        help_text = "Bin width for grouping values"
                    
                    table_granularity = st.slider(
                        "Table bin width:",
                        min_value=min_val,
                        max_value=max_val,
                        value=default_granularity,
                        step=step_val,
                        help=help_text
                    )
                else:
                    table_granularity = 0.01
                
                st.info("ℹ️ **Note**: All values are displayed as positive (absolute values) for clarity.")
            else:
                show_each_value = False
                show_table = False
                table_granularity = 0.01
                
            # Always use absolute values
            use_absolute_values = True
            
            if selected_param:
                fig = plot_parameter_distribution(changes_df, selected_param, plot_type, nbins, use_absolute_values, show_each_value)
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show detailed distribution table if requested
                    if show_table and plot_type == "histogram":
                        st.subheader("📊 Detailed Distribution Table")
                        
                        param_data = changes_df[changes_df['parameter'] == selected_param]
                        
                        # Determine display type and get data
                        PERCENTAGE_POINTS_PARAMS = {'attrition_rate', 'calf_loss_rate', 'loss_rate'}
                        PERCENT_CHANGE_PARAMS = {'live_weight', 'slaughter_weight', 'gutting_rate', 'first_calving_age', 
                                               'intercalving_period', 'milk_yield', 'protein_content', 'fat_content'}
                        DAYS_PARAMS = {'grazing_days'}
                        WEEKS_PARAMS = {'weaning_age'}
                        
                        if selected_param in PERCENTAGE_POINTS_PARAMS:
                            table_data = (param_data['absolute_change'] * 100).abs()  # Convert decimal to percentage points
                            display_type = 'percentage_points'
                        elif selected_param in PERCENT_CHANGE_PARAMS:
                            table_data = (param_data['percent_change'] * 100).abs()
                            display_type = 'percent_change'
                        elif selected_param in DAYS_PARAMS:
                            table_data = param_data['absolute_change'].abs()
                            display_type = 'days'
                        elif selected_param in WEEKS_PARAMS:
                            table_data = param_data['absolute_change'].abs()
                            display_type = 'weeks'
                        else:
                            table_data = (param_data['percent_change'] * 100).abs()
                            display_type = 'percent_change'
                        
                        # Create and display the table
                        dist_table = create_distribution_table(table_data, display_type, table_granularity)
                        
                        if not dist_table.empty:
                            st.dataframe(dist_table, hide_index=True, use_container_width=True, height=400)
                            
                            # Show summary info
                            total_ranges = len(dist_table)
                            total_datapoints = dist_table['Count'].sum()
                            st.caption(f"📈 **{total_ranges}** value ranges • **{total_datapoints}** total datapoints • Bin width: **{table_granularity}**")
                            
                            # Export option for table
                            csv_buffer = BytesIO()
                            dist_table.to_csv(csv_buffer, index=False)
                            csv_buffer.seek(0)
                            st.download_button(
                                label="⬇️ Download Distribution Table (CSV)",
                                data=csv_buffer,
                                file_name=f"{selected_param}_distribution_table.csv",
                                mime="text/csv",
                                key=f"download_{selected_param}_table"
                            )
                        else:
                            st.warning("No data available for the distribution table.")
                    
                    # Show statistics for the selected parameter
                    param_data = changes_df[changes_df['parameter'] == selected_param]
                    
                    # Determine display type for statistics
                    PERCENTAGE_POINTS_PARAMS = {'attrition_rate', 'calf_loss_rate', 'loss_rate'}
                    PERCENT_CHANGE_PARAMS = {'live_weight', 'slaughter_weight', 'gutting_rate', 'first_calving_age', 
                                           'intercalving_period', 'milk_yield', 'protein_content', 'fat_content'}
                    DAYS_PARAMS = {'grazing_days'}
                    WEEKS_PARAMS = {'weaning_age'}
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Changes", len(param_data))
                    
                    with col2:
                        significant_count = param_data['is_significant'].sum()
                        st.metric("Significant Changes", significant_count, 
                                 f"{(significant_count/len(param_data)*100):.1f}%")
                    
                    # Display statistics based on parameter type
                    if selected_param in PERCENTAGE_POINTS_PARAMS:
                        with col3:
                            avg_change = param_data['absolute_change'].abs().mean() * 100  # Convert decimal to percentage points
                            st.metric("Avg Change (PP)", f"{avg_change:.2f}")
                        with col4:
                            median_change = param_data['absolute_change'].abs().median() * 100  # Convert decimal to percentage points
                            st.metric("Median Change (PP)", f"{median_change:.2f}")
                    elif selected_param in PERCENT_CHANGE_PARAMS:
                        with col3:
                            avg_change = (param_data['percent_change'].abs() * 100).mean()
                            st.metric("Avg Change (%)", f"{avg_change:.2f}%")
                        with col4:
                            median_change = (param_data['percent_change'].abs() * 100).median()
                            st.metric("Median Change (%)", f"{median_change:.2f}%")
                    elif selected_param in DAYS_PARAMS:
                        with col3:
                            avg_change = param_data['absolute_change'].abs().mean()
                            st.metric("Avg Change (Days)", f"{avg_change:.1f}")
                        with col4:
                            median_change = param_data['absolute_change'].abs().median()
                            st.metric("Median Change (Days)", f"{median_change:.1f}")
                    elif selected_param in WEEKS_PARAMS:
                        with col3:
                            avg_change = param_data['absolute_change'].abs().mean()
                            st.metric("Avg Change (Weeks)", f"{avg_change:.1f}")
                        with col4:
                            median_change = param_data['absolute_change'].abs().median()
                            st.metric("Median Change (Weeks)", f"{median_change:.1f}")
                    else:
                        # Default to percent change
                        with col3:
                            avg_change = (param_data['percent_change'].abs() * 100).mean()
                            st.metric("Avg Change (%)", f"{avg_change:.2f}%")
                        with col4:
                            median_change = (param_data['percent_change'].abs() * 100).median()
                            st.metric("Median Change (%)", f"{median_change:.2f}%")
                else:
                    st.warning("No data available for the selected parameter.")
        
        with viz_tab2:
            st.subheader("Compare Multiple Parameters")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                selected_params_multi = st.multiselect(
                    "Select parameters to compare:",
                    numerical_params,
                    default=numerical_params[:4] if len(numerical_params) >= 4 else numerical_params,
                    help="Select 2-9 parameters to compare their change distributions"
                )
            
            with col2:
                nbins_multi = st.slider(
                    "Bins per histogram:",
                    min_value=10,
                    max_value=100,
                    value=30,
                    step=10,
                    help="Control granularity for comparison histograms - higher values show more detail",
                    key="comparison_bins"
                )
            
            if len(selected_params_multi) >= 2:
                comparison_fig = create_comparison_matrix(changes_df, selected_params_multi, nbins_multi)
                
                if comparison_fig:
                    st.plotly_chart(comparison_fig, use_container_width=True)
                    
                    # Summary table for selected parameters
                    summary_data = []
                    for param in selected_params_multi:
                        param_data = changes_df[changes_df['parameter'] == param]
                        is_percentage_param = param_data['is_percentage_param'].iloc[0] if not param_data.empty else False
                        
                        significant_count = param_data['is_significant'].sum()
                        total_count = len(param_data)
                        
                        if is_percentage_param:
                            avg_change = abs(param_data['absolute_change'].mean()) * 100  # Convert decimal to percentage points
                            avg_change_str = f"{avg_change:.2f} pp"
                        else:
                            avg_change = abs(param_data['percent_change'].mean()) * 100
                            avg_change_str = f"{avg_change:.2f}%"
                        
                        summary_data.append({
                            'Parameter': param.replace("_", " ").title(),
                            'Total Changes': total_count,
                            'Significant Changes': significant_count,
                            'Significance Rate': f"{(significant_count/total_count*100):.1f}%",
                            'Average Change': avg_change_str
                        })
                    
                    st.subheader("Comparison Summary")
                    st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)
                
            elif len(selected_params_multi) == 1:
                st.info("Please select at least 2 parameters for comparison.")
            else:
                st.info("Please select parameters to compare.")
        
        with viz_tab3:
            st.subheader("Detailed Statistics for All Parameters")
            
            # Create comprehensive statistics table
            detailed_stats = []
            
            for param in numerical_params:
                param_data = changes_df[changes_df['parameter'] == param]
                is_percentage_param = param_data['is_percentage_param'].iloc[0] if not param_data.empty else False
                
                if not param_data.empty:
                    significant_count = param_data['is_significant'].sum()
                    total_count = len(param_data)
                    
                    if is_percentage_param:
                        change_col = 'absolute_change'
                        unit = 'pp'
                    else:
                        change_col = 'percent_change'
                        param_data = param_data.copy()
                        param_data[change_col] = param_data[change_col] * 100
                        unit = '%'
                    
                    stats = {
                        'Parameter': param.replace("_", " ").title(),
                        'Type': 'Percentage Points' if is_percentage_param else 'Percent Change',
                        'Total Changes': total_count,
                        'Significant Changes': f"{significant_count} ({(significant_count/total_count*100):.1f}%)",
                        f'Mean Change ({unit})': f"{param_data[change_col].mean():+.3f}",
                        f'Median Change ({unit})': f"{param_data[change_col].median():+.3f}",
                        f'Std Dev ({unit})': f"{param_data[change_col].std():.3f}",
                        f'Min Change ({unit})': f"{param_data[change_col].min():+.3f}",
                        f'Max Change ({unit})': f"{param_data[change_col].max():+.3f}",
                        f'25th Percentile ({unit})': f"{param_data[change_col].quantile(0.25):+.3f}",
                        f'75th Percentile ({unit})': f"{param_data[change_col].quantile(0.75):+.3f}"
                    }
                    detailed_stats.append(stats)
            
            if detailed_stats:
                detailed_df = pd.DataFrame(detailed_stats)
                st.dataframe(detailed_df, hide_index=True, use_container_width=True, height=400)
                
                # Export option
                csv_buffer = BytesIO()
                detailed_df.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)
                st.download_button(
                    label="⬇️ Download Parameter Statistics (CSV)",
                    data=csv_buffer,
                    file_name="parameter_change_statistics.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No numerical parameters found for statistics.")
    
    else:
        st.warning("No numerical parameters available for visualization after filtering out binary parameters.")

else:
    st.info("👆 Please upload the **farmdata_datasources.csv** file to begin analysis.")
    
    st.markdown("""
    ### 📋 This Simplified App Focuses On:
    
    1. **🏢 Corporation Filtering** - Choose specific corporations to analyze
    2. **📊 Average Changes for Percentage Parameters** - Shows mean change and percent points for rate-based metrics
    3. **📋 Detailed Percent Points Analysis** - Compare percentage vs absolute changes
    4. **⚠️ Significant Changes List** - All assessments with changes exceeding thresholds
    
    ### 🎯 Key Features:
    - **Auto-filters** for fleckvieh, braunvieh, schwarzbunt breeds only
    - **Highlights percentage parameters** where percent points are most meaningful
    - **Assessment-level detail** with specific IDs and explanations
    - **Export capabilities** for further analysis
    """)

# Footer
st.markdown("---")
st.markdown("*Simplified YoY Analysis App • Focus on Percent Points & Significant Changes*")
