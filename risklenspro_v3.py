# Part 1: Imports, Setup, Helper Functions

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
    f1_score,
    precision_score,
    recall_score,
    make_scorer # For cross-validation scoring
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import scipy.stats as stats # For Monte Carlo simulation if needed
import io # For download/upload
import pickle # For saving/loading session state (models)
import base64 # For download links
import lime # For LIME explanations
import lime.lime_tabular
import copy # For deep copying data in scenario planning
import matplotlib.pyplot as plt # For LIME plots (can be hidden)
import time # For timing operations
import json # For saving/loading simple state

# --- Constants & Configuration ---

# Arcadis Branding
ARCADIS_ORANGE = "#E67300"
ARCADIS_LIGHT_GREY = "#F5F5F5"
ARCADIS_DARK_GREY = "#646469"
ARCADIS_BLACK = "#000000"
ARCADIS_SECONDARY_PALETTE = ["#00A3A1", ARCADIS_DARK_GREY, "#D6D6D8", ARCADIS_LIGHT_GREY] # Teal, Dark Grey, Mid Grey, Light Grey
# Placeholder for Arcadis Logo URL (replace with actual URL if available)
ARCADIS_LOGO_URL = "https://placehold.co/200x60/ffffff/E67300?text=Arcadis+Logo" # Placeholder

# Features eligible for Scenario Planning
SCENARIO_FEATURES_NUMERIC = ['InitialCostEstimate', 'InitialScheduleDays', 'ScopeChanges']
SCENARIO_FEATURES_CATEGORICAL = ['ResourceAvailability', 'TechnicalComplexity', 'VendorReliability', 'ClientEngagement']

# Monte Carlo Simulation Settings
N_MONTE_CARLO_RUNS = 1000

# --- Page Configuration ---
st.set_page_config(
    page_title="RiskLens Pro - Project Risk",
    page_icon="📊", # Consider an Arcadis favicon if available
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for Enhanced Arcadis Branding & Styling ---
st.markdown(f"""
<style>
    /* Basic Arcadis Theme Elements */
    .stApp {{
        background-color: {ARCADIS_LIGHT_GREY};
    }}
    .stButton>button {{
        border-radius: 8px; border: 1px solid {ARCADIS_ORANGE}; background-color: {ARCADIS_ORANGE}; color: white;
        transition: background-color 0.3s ease, border-color 0.3s ease; font-weight: bold; padding: 0.5rem 1rem;
        width: auto; display: inline-block; margin-right: 10px; margin-top: 10px; /* Added margin-top */
    }}
    .stButton>button:hover {{ background-color: #D06300; border-color: #D06300; color: white; }}
    .stButton>button:active {{ background-color: #B85A00; border-color: #B85A00; }}

    /* Specific button styling for Train Model in sidebar */
    [data-testid="stSidebar"] .stButton>button {{
        width: 95%; /* Make sidebar buttons wider */
        margin-bottom: 10px;
    }}

    /* Style download buttons */
    .stDownloadButton>button {{
        background-color: {ARCADIS_DARK_GREY}; border-color: {ARCADIS_DARK_GREY}; color: white; font-weight: bold;
        padding: 0.5rem 1rem; border-radius: 8px; width: auto; display: inline-block; margin-right: 10px; margin-bottom: 10px;
    }}
    .stDownloadButton>button:hover {{ background-color: #505055; border-color: #505055; }}

    /* Sidebar */
    .stSidebar {{ background-color: #FFFFFF; border-right: 1px solid #D6D6D8; }}
    [data-testid="stSidebarNav"] {{ padding-top: 0rem; }} /* Reduce top padding in sidebar nav area */
    [data-testid="stSidebarUserContent"] {{ padding-top: 1rem; }} /* Adjust padding for content below logo */

    /* Headings */
    h1, h2 {{ color: {ARCADIS_BLACK}; font-weight: bold; }}
    h3 {{ color: {ARCADIS_ORANGE}; font-weight: bold; border-bottom: 2px solid {ARCADIS_ORANGE}; padding-bottom: 5px; margin-bottom: 15px; }}
    h4, h5, h6 {{ color: {ARCADIS_DARK_GREY}; }}

    /* Metrics (Tiles for Executive Summary) */
    .stMetric {{
        background-color: #FFFFFF; border: 1px solid #D6D6D8; border-left: 5px solid {ARCADIS_ORANGE}; /* Accent border */
        border-radius: 8px; padding: 15px 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 10px; /* Ensure spacing between metrics */
    }}
    .stMetric > label {{ font-weight: bold; color: {ARCADIS_DARK_GREY}; }} /* Style metric label */
    .stMetric > div[data-testid="stMetricValue"] {{ font-size: 2em; font-weight: bold; color: {ARCADIS_BLACK}; }} /* Style metric value */
    .stMetric > div[data-testid="stMetricDelta"] {{ font-size: 0.9em; }} /* Style metric delta */


    /* Tabs */
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
        background-color: {ARCADIS_ORANGE}; color: white; border-radius: 8px 8px 0 0; font-weight: bold; border-bottom: none;
    }}
    .stTabs [data-baseweb="tab-list"] button {{
        border-radius: 8px 8px 0 0; color: {ARCADIS_DARK_GREY}; background-color: #E0E0E0; border-bottom: none;
    }}
    .stTabs [data-baseweb="tab-list"] {{ border-bottom: 2px solid {ARCADIS_ORANGE}; padding-bottom: 0; }}
    .stTabs [data-baseweb="tab-panel"] {{ background-color: {ARCADIS_LIGHT_GREY}; padding-top: 25px; border: none; }}

    /* Containers */
    .stVerticalBlock {{ padding-bottom: 1rem; }}
    div[data-testid="stVerticalBlock"]>div[style*="flex-direction: column;"]>div[data-testid="stVerticalBlock"],
    div[data-testid="stVerticalBlock"]>div[style*="flex-direction: column;"]>div[data-testid="stHorizontalBlock"] {{
        /* Target nested vertical blocks and horizontal blocks within vertical blocks */
        border-radius: 8px !important; border: 1px solid #D6D6D8 !important; padding: 20px !important;
        margin-bottom: 20px !important; background-color: #FFFFFF !important; box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
    }}
    /* Welcome page specific container */
    .welcome-section {{
        background-color: #FFFFFF; border: 1px solid #D6D6D8; border-radius: 8px;
        padding: 25px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}
     .welcome-section h3 {{ border: none; margin-bottom: 10px; }} /* Override default h3 style */

    /* Arcadis Logo in Sidebar */
    [data-testid="stSidebarNav"]::before {{
        content: ""; display: block; background-image: url({ARCADIS_LOGO_URL});
        background-size: contain; background-repeat: no-repeat; background-position: center 10px; /* Adjust vertical position */
        height: 60px; /* Adjust height */ margin-bottom: 10px; /* Space below logo */
    }}

    /* Expander in Sidebar */
    [data-testid="stSidebar"] .stExpander {{
        border: none !important; border-radius: 0px !important; background-color: transparent !important; margin-bottom: 0px;
        border-top: 1px solid #eee !important; /* Separator line */
    }}
    [data-testid="stSidebar"] .stExpander header {{
        font-weight: bold; color: {ARCADIS_BLACK}; background-color: transparent !important; border-radius: 0 !important;
        padding: 10px 0px !important; /* Adjust padding */
    }}
     [data-testid="stSidebar"] .stExpander div[data-testid="stExpanderDetails"] {{
         padding-left: 10px !important; /* Indent expander content */
     }}


    /* Dataframes */
    .stDataFrame {{ border-radius: 8px; overflow: hidden; }}

    /* Markdown links */
    a {{ color: {ARCADIS_ORANGE}; }}

    /* Main area padding */
    .main .block-container {{ padding: 2rem; }}

    /* Lists for capabilities/questions */
    .styled-list li {{ margin-bottom: 10px; line-height: 1.6; color: {ARCADIS_DARK_GREY}; }}
    .styled-list li b {{ color: {ARCADIS_BLACK}; }}
    .styled-list li i {{ color: {ARCADIS_ORANGE}; font-style: normal; font-weight: bold; }} /* Tab names */

</style>
""", unsafe_allow_html=True)


# --- Helper Functions (Keep essential ones) ---

@st.cache_data # Cache the data generation
def generate_mock_data(num_projects=250):
    """Generates enhanced mock project data for demonstration."""
    np.random.seed(42) # for reproducibility
    data = {
        'ProjectID': [f'PROJ{i:04}' for i in range(1, num_projects + 1)],
        'ProjectName': [f'Project Alpha {i}' if i%3==0 else f'Project Beta {i}' if i%3==1 else f'Project Gamma {i}' for i in range(1, num_projects + 1)],
        'Region': np.random.choice(['North America', 'Europe', 'APAC', 'MEA', 'LATAM'], num_projects, p=[0.35, 0.25, 0.15, 0.1, 0.15]),
        'ProjectType': np.random.choice(['Infrastructure', 'Building', 'Water', 'Environment', 'Digital'], num_projects, p=[0.3, 0.3, 0.15, 0.1, 0.15]),
        'InitialCostEstimate': np.random.lognormal(mean=14, sigma=1.5, size=num_projects).clip(50000, 20000000),
        'InitialScheduleDays': np.random.randint(60, 1500, num_projects),
        'ActualCost': 0.0,
        'ActualScheduleDays': 0.0,
        'ScopeChanges': np.random.poisson(1.5, num_projects),
        'ResourceAvailability': np.random.choice(['High', 'Medium', 'Low'], num_projects, p=[0.35, 0.45, 0.2]),
        'TechnicalComplexity': np.random.choice(['Low', 'Medium', 'High', 'Very High'], num_projects, p=[0.25, 0.4, 0.25, 0.1]),
        'VendorReliability': np.random.choice(['Good', 'Average', 'Poor'], num_projects, p=[0.45, 0.4, 0.15]),
        'ClientEngagement': np.random.choice(['High', 'Medium', 'Low'], num_projects, p=[0.5, 0.35, 0.15]),
        'PermittingDelays': np.random.choice([0, 1], num_projects, p=[0.75, 0.25]),
        'RiskEventOccurred': np.random.choice([0, 1], num_projects, p=[0.7, 0.3])
    }
    df = pd.DataFrame(data)

    # Simulate actual cost and schedule
    cost_variance_noise = np.random.normal(0, 0.05, num_projects)
    schedule_variance_noise = np.random.normal(0, 0.04, num_projects)
    df['CostVarianceFactor'] = (
        1.0 + cost_variance_noise +
        df['ScopeChanges'] * np.random.uniform(0.015, 0.035, num_projects) +
        df['ResourceAvailability'].map({'High': -0.02, 'Medium': 0.015, 'Low': 0.05}) +
        df['TechnicalComplexity'].map({'Low': -0.015, 'Medium': 0.025, 'High': 0.06, 'Very High': 0.10}) +
        df['VendorReliability'].map({'Good': -0.015, 'Average': 0.02, 'Poor': 0.055}) +
        df['ClientEngagement'].map({'High': -0.015, 'Medium': 0.01, 'Low': 0.03}) +
        df['PermittingDelays'] * np.random.uniform(0.025, 0.07, num_projects) +
        df['RiskEventOccurred'] * np.random.uniform(0.05, 0.15, num_projects)
    )
    df['ScheduleVarianceFactor'] = (
        1.0 + schedule_variance_noise +
        df['ScopeChanges'] * np.random.uniform(0.02, 0.05, num_projects) +
        df['ResourceAvailability'].map({'High': -0.025, 'Medium': 0.025, 'Low': 0.06}) +
        df['TechnicalComplexity'].map({'Low': -0.015, 'Medium': 0.04, 'High': 0.08, 'Very High': 0.12}) +
        df['VendorReliability'].map({'Good': -0.015, 'Average': 0.03, 'Poor': 0.07}) +
        df['ClientEngagement'].map({'High': -0.02, 'Medium': 0.015, 'Low': 0.04}) +
        df['PermittingDelays'] * np.random.uniform(0.04, 0.10, num_projects) +
        df['RiskEventOccurred'] * np.random.uniform(0.06, 0.20, num_projects)
    )
    df['ActualCost'] = df['InitialCostEstimate'] * df['CostVarianceFactor']
    df['ActualScheduleDays'] = df['InitialScheduleDays'] * df['ScheduleVarianceFactor']
    df['CostVariancePerc'] = ((df['ActualCost'] - df['InitialCostEstimate']) / df['InitialCostEstimate']) * 100
    df['ScheduleVariancePerc'] = ((df['ActualScheduleDays'] - df['InitialScheduleDays']) / df['InitialScheduleDays']) * 100

    # Define 'DerailmentRisk' (Target Variable)
    cost_threshold = 10
    schedule_threshold = 15
    df['DerailmentRisk_Actual'] = ((df['CostVariancePerc'] > cost_threshold) | (df['ScheduleVariancePerc'] > schedule_threshold)).astype(int)

    # Initialize prediction columns
    df['DerailmentRisk_Predicted_Prob'] = np.nan
    df['DerailmentRisk_Predicted'] = pd.NA # Use pandas NA for integer type

    # Add mock completion date
    start_date = pd.Timestamp('2022-01-01')
    df['CompletionDate'] = [start_date + pd.Timedelta(days=np.random.randint(0, 3*365)) for _ in range(num_projects)]
    df = df.sort_values('CompletionDate').reset_index(drop=True)

    return df

@st.cache_data
def generate_mock_risk_register(project_ids):
    """Generates a more detailed mock risk register."""
    risks = ["Scope Creep", "Resource Shortage", "Technical Debt", "Vendor Delay", "Budget Cuts", "Stakeholder Conflict", "Integration Issues", "Testing Failure", "Regulatory Changes", "Site Access Issues", "Material Price Volatility", "Design Errors", "Cybersecurity Threat", "Weather Impact", "Subcontractor Default"]
    impact_areas = ["Cost", "Schedule", "Quality", "Reputation", "Safety"]
    likelihood_levels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    impact_levels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    status_levels = ['Planned', 'In Progress', 'Complete', 'Not Started', 'Overdue', 'On Hold']
    owners = ['PM', 'Engineer', 'Client', 'Vendor', 'HSE Lead', 'IT Lead']

    data = []
    for pid in project_ids:
        num_risks = np.random.randint(1, 8)
        project_risks = np.random.choice(risks, num_risks, replace=False)
        for risk in project_risks:
            likelihood = np.random.choice(likelihood_levels, p=[0.1, 0.2, 0.4, 0.2, 0.1])
            impact = np.random.choice(impact_levels, p=[0.1, 0.3, 0.35, 0.15, 0.1])
            data.append({
                'ProjectID': pid, 'RiskID': f"RISK-{np.random.randint(1000, 9999)}", 'RiskDescription': risk,
                'Likelihood': likelihood, 'Impact': impact, 'ImpactArea': np.random.choice(impact_areas),
                'MitigationStatus': np.random.choice(status_levels, p=[0.2, 0.15, 0.1, 0.35, 0.1, 0.1]),
                'Owner': np.random.choice(owners),
                'DueDate': pd.Timestamp.now() + pd.Timedelta(days=np.random.randint(10, 120)) if np.random.rand() > 0.4 else pd.NaT,
                'MitigationAction': f"Action plan for {risk}"
            })
    df = pd.DataFrame(data)
    likelihood_map = {'Very Low': 1, 'Low': 2, 'Medium': 3, 'High': 4, 'Very High': 5}
    impact_map = {'Very Low': 1, 'Low': 2, 'Medium': 3, 'High': 4, 'Very High': 5}
    df['LikelihoodScore'] = df['Likelihood'].map(likelihood_map).fillna(0).astype(int)
    df['ImpactScore'] = df['Impact'].map(impact_map).fillna(0).astype(int)
    df['RiskScore'] = df['LikelihoodScore'] * df['ImpactScore']
    def assign_urgency(row):
        if row['MitigationStatus'] in ['Overdue', 'Not Started'] and row['RiskScore'] >= 12: return 'Urgent'
        if row['RiskScore'] >= 16: return 'High'
        if row['RiskScore'] >= 9: return 'Medium'
        if row['MitigationStatus'] == 'Overdue': return 'Medium'
        return 'Low'
    df['MitigationUrgency'] = df.apply(assign_urgency, axis=1)
    return df

def plot_confusion_matrix_plotly(cm, labels):
    """Plots a confusion matrix using Plotly with Arcadis colors."""
    fig = go.Figure(data=go.Heatmap(
                   z=cm, x=labels, y=labels, hoverongaps=False,
                   colorscale=[[0, ARCADIS_LIGHT_GREY], [1, ARCADIS_ORANGE]],
                   text=cm, texttemplate="%{text}", textfont={"size":14, "color": ARCADIS_BLACK}))
    fig.update_layout(
        title='Confusion Matrix', xaxis_title="Predicted label", yaxis_title="True label",
        xaxis={'side': 'top'}, margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def plot_precision_recall_curve_plotly(y_true, y_prob, model_name):
    """Plots Precision-Recall curve using Plotly with Arcadis colors."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, y=precision, mode='lines', name=f'{model_name} (AUC = {pr_auc:.3f})',
        line=dict(color=ARCADIS_ORANGE, width=2.5)
    ))
    no_skill = len(y_true[y_true==1]) / len(y_true) if len(y_true) > 0 else 0
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[no_skill, no_skill], mode='lines', name='No Skill Baseline',
        line=dict(color=ARCADIS_DARK_GREY, width=2, dash='dash')
    ))
    fig.update_layout(
        title='Precision-Recall Curve', xaxis_title='Recall', yaxis_title='Precision',
        xaxis=dict(range=[0.0, 1.0]), yaxis=dict(range=[0.0, 1.05]),
        legend=dict(x=0.4, y=0.1, bgcolor='rgba(255,255,255,0.7)'),
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def plot_roc_curve_plotly(y_true, y_prob, model_name):
    """Plots ROC curve using Plotly."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode='lines', name=f'{model_name} (AUC = {roc_auc:.3f})',
        line=dict(color=ARCADIS_ORANGE, width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines', name='Random Guess',
        line=dict(color=ARCADIS_DARK_GREY, width=2, dash='dash')
    ))
    fig.update_layout(
        title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
        xaxis=dict(range=[0.0, 1.0]), yaxis=dict(range=[0.0, 1.05]),
        legend=dict(x=0.4, y=0.1, bgcolor='rgba(255,255,255,0.7)'),
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def df_to_csv(df):
    """Converts DataFrame to CSV bytes for download."""
    output = io.StringIO()
    df.to_csv(output, index=False)
    return output.getvalue().encode('utf-8')

def create_download_link(object_to_download, download_filename, link_text):
    """Generates a link to download object_to_download."""
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = df_to_csv(object_to_download)
        mime = 'text/csv'
    else: mime = 'application/octet-stream'
    b64 = base64.b64encode(object_to_download).decode()
    return f'<a href="data:{mime};base64,{b64}" download="{download_filename}">{link_text}</a>'

@st.cache_resource(show_spinner="Initializing LIME explainer...")
def get_lime_explainer(_X_train_processed, _feature_names, _class_names, _categorical_features_indices):
    """Creates and caches a LIME Tabular Explainer."""
    try:
        # Ensure _X_train_processed is a NumPy array for LIME
        training_data_np = np.array(_X_train_processed)
        valid_cat_indices = [i for i in _categorical_features_indices if i < training_data_np.shape[1]]
        if len(valid_cat_indices) != len(_categorical_features_indices):
            st.warning("Some categorical feature indices were out of bounds for LIME.")

        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=training_data_np, # Use NumPy array
            feature_names=_feature_names,
            class_names=_class_names,
            categorical_features=valid_cat_indices,
            mode='classification',
            verbose=False,
            random_state=42
        )
        return explainer
    except Exception as e:
        st.error(f"❌ Could not initialize LIME: {e}")
        return None

def get_model_pipeline(model_name, numerical_features, categorical_features, X_train_shape, y_train_balance=None):
    """Creates a scikit-learn pipeline for a given model name."""
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer(
        transformers=[('num', numeric_transformer, numerical_features), ('cat', categorical_transformer, categorical_features)],
        remainder='passthrough', # Keep other columns if any (shouldn't be needed if features defined correctly)
        verbose_feature_names_out=False # Keep feature names simpler
    )
    # Ensure preprocessor outputs pandas DataFrame to retain feature names
    preprocessor.set_output(transform="pandas")

    # Define models
    if model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced', n_jobs=-1, max_depth=12, min_samples_leaf=5)
    elif model_name == "Logistic Regression":
        model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=2000, solver='liblinear', C=0.8)
    elif model_name == "XGBoost":
        scale_pos_weight = 1
        if y_train_balance is not None and len(y_train_balance) > 0: # Check if y_train_balance is not empty
             pos_count = y_train_balance.sum(); neg_count = len(y_train_balance) - pos_count
             scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
        model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss',
                                  scale_pos_weight=scale_pos_weight, n_estimators=150, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8)
    else: raise ValueError(f"Unknown model name: {model_name}")

    # Create the full pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    return pipeline

def calculate_feature_importance(pipeline, feature_names):
    """Extracts feature importance from a trained pipeline."""
    importance_df = None
    try:
        final_estimator = pipeline.named_steps['classifier']
        preprocessor = pipeline.named_steps['preprocessor']

        # Get feature names *after* preprocessing
        processed_feature_names = preprocessor.get_feature_names_out()

        if hasattr(final_estimator, 'feature_importances_'):
            importances = final_estimator.feature_importances_
        elif hasattr(final_estimator, 'coef_'):
            importances = np.abs(final_estimator.coef_[0])
        else:
            importances = None

        if importances is not None and len(processed_feature_names) == len(importances):
            importance_df = pd.DataFrame({'Feature': processed_feature_names, 'Importance': importances})
            importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        elif importances is not None:
             st.warning(f"Feature importance length mismatch: {len(processed_feature_names)} names vs {len(importances)} importances.")
             # Attempt to match based on provided feature_names if lengths differ significantly
             if len(feature_names) == len(importances):
                 importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                 importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
                 st.info("Used original feature names for importance due to mismatch.")

    except Exception as e: st.warning(f"Could not calculate feature importance: {e}")
    return importance_df

def plot_waterfall_plotly(feature_contributions, base_value, predicted_value, feature_names):
    """Creates a waterfall chart using Plotly showing feature contributions."""
    # Ensure lengths match
    if len(feature_contributions) != len(feature_names):
        st.error(f"Waterfall plot error: Mismatch between contributions ({len(feature_contributions)}) and feature names ({len(feature_names)}).")
        return go.Figure() # Return empty figure

    measures = ["relative"] * len(feature_contributions)
    values = list(feature_contributions)
    texts = [f"{val:.3f}" for val in values]
    y_values = [base_value] + values
    x_labels = ["Base Value"] + feature_names
    # cumulative = np.cumsum(y_values) # Not directly needed for plotly waterfall

    fig = go.Figure(go.Waterfall(
        name = "Prediction", orientation = "v", measure = ["absolute"] + measures, # Base is absolute
        x = x_labels,
        textposition = "outside",
        text = [f"{base_value:.3f}"] + texts, # Show values on bars
        y = [base_value] + values, # Values for each step/feature
        connector = {"line":{"color":ARCADIS_DARK_GREY}},
        increasing = {"marker":{"color":ARCADIS_ORANGE}}, # Positive contributions
        decreasing = {"marker":{"color":ARCADIS_SECONDARY_PALETTE[0]}}, # Negative contributions (Teal)
        # Totals marker style applied implicitly at the end
    ))

    # Add the final prediction value explicitly using 'total' measure
    # Plotly calculates the total automatically, but we can customize its appearance if needed
    # If we add it manually, ensure it doesn't duplicate the automatic total.
    # Let Plotly handle the total calculation based on the relative steps.

    fig.update_layout(
            title="Prediction Waterfall Chart",
            yaxis_title="Prediction Probability",
            showlegend = False,
            waterfallgap = 0.3, # Space between bars
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
    )
    # Adjust y-axis range slightly for better visibility if needed
    # min_y = min(0, base_value, predicted_value, *cumulative) - 0.05
    # max_y = max(base_value, predicted_value, *cumulative) + 0.05
    # fig.update_yaxes(range=[min_y, max_y]) # Auto range usually works well

    return fig

# --- Training Function (Moved Logic Here) ---
def train_model_and_explainers(df_train, features, numerical_features, categorical_features, target, train_size, random_state, model_name, threshold):
    """Handles data prep, splitting, training, prediction, and explainer init."""
    results = {'success': False, 'message': '', 'pipeline': None, 'X_test_original': None, 'X_test_original_index': None,
               'y_test': None, 'y_pred_prob': None, 'feature_importance': None, 'lime_explainer': None,
               'X_train_processed': None, 'X_test_processed': None, 'features_processed_names': None,
               'categorical_features_indices': None, 'categorical_features_names': None, 'y_train': None, 'preprocessor': None}
    try:
        # Prepare data
        X = df_train[features].copy()
        y = df_train[target].copy()

        # Drop rows where target is NaN
        valid_indices = y.dropna().index
        X = X.loc[valid_indices]
        y = y.loc[valid_indices]

        # Check if data remains after dropping NaN target
        if X.empty or y.empty:
            results['message'] = "❌ Error: No valid data remaining after removing rows with missing target values."
            return results

        # Impute missing values in features (simple mean/mode imputation)
        for col in numerical_features:
            if X[col].isnull().any():
                mean_val = X[col].mean()
                if pd.isna(mean_val): # Handle case where entire column might be NaN
                    mean_val = 0
                X[col] = X[col].fillna(mean_val)
        for col in categorical_features:
            if X[col].isnull().any():
                mode_val = X[col].mode()
                fill_val = mode_val[0] if not mode_val.empty else 'Unknown'
                X[col] = X[col].fillna(fill_val)

        # Check for sufficient data for splitting
        if len(X) < 2: # Need at least 2 samples for train/test split
             results['message'] = f"❌ Error: Insufficient data ({len(X)} samples) for training after cleaning."
             return results
        # Adjust train_size if dataset is very small to ensure test set has at least 1 sample
        min_test_samples = 1
        max_train_size = 1.0 - (min_test_samples / len(X))
        adjusted_train_size = min(train_size, max_train_size)
        if adjusted_train_size <= 0: # Handle edge case where dataset size equals min_test_samples
            results['message'] = f"❌ Error: Dataset too small ({len(X)} samples) to create a train/test split."
            return results

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=adjusted_train_size, random_state=random_state, stratify=y # Stratify if possible
        )

        # Create and train pipeline
        pipeline = get_model_pipeline(model_name, numerical_features, categorical_features, X_train.shape, y_train)
        pipeline.fit(X_train, y_train)

        # Store preprocessor and processed data
        results['preprocessor'] = pipeline.named_steps['preprocessor']
        # Use transform method, ensuring output is DataFrame if set_output='pandas'
        results['X_train_processed'] = pipeline.named_steps['preprocessor'].transform(X_train)
        results['X_test_processed'] = pipeline.named_steps['preprocessor'].transform(X_test)
        results['features_processed_names'] = pipeline.named_steps['preprocessor'].get_feature_names_out().tolist()

        # Get indices of categorical features *in the processed data*
        try:
            cat_transformer = pipeline.named_steps['preprocessor'].named_transformers_['cat']
            onehot_encoder = cat_transformer.named_steps['onehot']
            cat_feature_names_out = onehot_encoder.get_feature_names_out(categorical_features)
            # Map processed names back to indices in the full processed feature list
            cat_indices = [results['features_processed_names'].index(name) for name in cat_feature_names_out if name in results['features_processed_names']]
            results['categorical_features_indices'] = cat_indices
        except Exception as e_cat_idx:
            st.warning(f"Could not determine categorical feature indices for LIME: {e_cat_idx}")
            results['categorical_features_indices'] = [] # Default to empty list

        results['categorical_features_names'] = categorical_features # Original names

        # Store pipeline and other results
        results['pipeline'] = pipeline
        results['X_test_original'] = X_test.copy()
        results['X_test_original_index'] = X_test.index # Store the original index of the test set
        results['y_train'] = y_train # Store for potential future use
        results['y_test'] = y_test
        results['y_pred_prob'] = pipeline.predict_proba(X_test)[:, 1]

        # Calculate feature importance
        results['feature_importance'] = calculate_feature_importance(pipeline, results['features_processed_names'])

        # Initialize LIME explainer (using cached function)
        # Ensure processed data passed to LIME is NumPy array
        results['lime_explainer'] = get_lime_explainer(
            np.array(results['X_train_processed']), # Pass NumPy array
            results['features_processed_names'],
            ['No Derailment', 'Derailment'],
            results['categorical_features_indices']
        )

        results['success'] = True
        results['message'] = f"✅ {model_name} model trained successfully! Predictions and explainers are ready."

    except ValueError as ve: # Catch specific errors like stratification issues
        results['message'] = f"❌ Error during training setup: {ve}. Check data distribution or try without stratification if dataset is small."
        st.exception(ve)
    except Exception as e:
        results['message'] = f"❌ Error during training: {e}"
        st.exception(e) # Log the full traceback for debugging

    return results


# --- Initialize Session State ---
default_state = {
    'project_data': None, 'risk_register': None, 'model_pipeline': None, 'trained_models': {},
    'model_comparison_results': None, 'preprocessor': None, 'numerical_features': [], 'categorical_features': [],
    'features_processed_names': None, 'categorical_features_indices': None, 'categorical_features_names': None,
    'X_train_original': None, 'X_train_processed': None, 'X_test_processed': None, 'X_test_original': None,
    'X_test_original_index': None, 'y_train': None, 'y_test': None, 'y_pred_prob': None,
    'feature_importance': None, 'predictions_made': False, 'lime_explainer': None,
    'monte_carlo_results': None, 'batch_scenario_data': None, 'batch_scenario_results': None,
    'auto_insights': None, 'current_model_choice': "Random Forest", 'current_prediction_threshold': 0.4,
    'train_size': 0.8, 'random_state': 42, 'target_variable': 'DerailmentRisk_Actual',
    'selected_numerical_features': [], # Store user selections separately
    'selected_categorical_features': []
}
# Initialize state variables if they don't exist
for key, value in default_state.items():
    if key not in st.session_state: st.session_state[key] = value

# Load initial data if state is empty
if st.session_state.project_data is None:
    st.session_state.project_data = generate_mock_data()
    # Set default selected features based on mock data
    df_mock = st.session_state.project_data
    target_mock = st.session_state.target_variable
    cols_to_exclude_mock = [
        'ProjectID', 'ProjectName', 'ActualCost', 'ActualScheduleDays', 'CostVariancePerc',
        'ScheduleVariancePerc', target_mock, 'DerailmentRisk_Predicted_Prob', 'DerailmentRisk_Predicted',
        'CostVarianceFactor', 'ScheduleVarianceFactor', 'CompletionDate'
    ]
    features_mock = [col for col in df_mock.columns if col not in cols_to_exclude_mock]
    st.session_state.selected_numerical_features = df_mock[features_mock].select_dtypes(include=np.number).columns.tolist()
    st.session_state.selected_categorical_features = df_mock[features_mock].select_dtypes(include=['object', 'category']).columns.tolist()

if st.session_state.risk_register is None and st.session_state.project_data is not None:
    st.session_state.risk_register = generate_mock_risk_register(st.session_state.project_data['ProjectID'].tolist())

# --- Sidebar ---
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("---")

# --- Data Loading (Moved to Sidebar) ---
st.sidebar.subheader("1. Data Status")
if st.session_state.project_data is not None:
    st.sidebar.success(f"✅ Project Data ({len(st.session_state.project_data)} rows)")
    if st.session_state.risk_register is not None:
        st.sidebar.success(f"✅ Risk Register ({len(st.session_state.risk_register)} rows)")
    else:
        st.sidebar.warning("⚠️ Risk Register not loaded.")
    # Allow quick upload/reload here - simpler than full Data Management tab
    uploaded_sidebar_proj = st.sidebar.file_uploader("Upload/Replace Project Data (CSV)", type=['csv'], key="sidebar_proj_upload")
    if uploaded_sidebar_proj:
        try:
            df_uploaded_proj = pd.read_csv(uploaded_sidebar_proj)
            if 'ProjectID' in df_uploaded_proj.columns and st.session_state.target_variable in df_uploaded_proj.columns:
                st.session_state.project_data = df_uploaded_proj
                # Reset dependent states
                st.session_state.model_pipeline = None; st.session_state.trained_models = {}
                st.session_state.predictions_made = False; st.session_state.lime_explainer = None
                st.session_state.monte_carlo_results = None; st.session_state.batch_scenario_results = None
                st.session_state.auto_insights = None; st.session_state.risk_register = None
                # Regenerate mock register or prompt upload
                try:
                    st.session_state.risk_register = generate_mock_risk_register(st.session_state.project_data['ProjectID'].tolist())
                    st.sidebar.info("Generated mock risk register.")
                except Exception: pass # Ignore if fails
                st.rerun()
            else: st.sidebar.error(f"CSV must have 'ProjectID' & '{st.session_state.target_variable}'")
        except Exception as e: st.sidebar.error(f"Error loading: {e}")

else:
    st.sidebar.error("❌ Project Data not loaded.")
    st.sidebar.info("Go to '💾 Data Management' tab to upload data.")

st.sidebar.markdown("---")

# --- Model Configuration (Moved to Sidebar) ---
st.sidebar.subheader("2. Model Configuration")

# Model Selection
st.session_state.current_model_choice = st.sidebar.selectbox(
    "Select Model:", ["Random Forest", "XGBoost", "Logistic Regression"],
    index=["Random Forest", "XGBoost", "Logistic Regression"].index(st.session_state.current_model_choice),
    key="model_select_sidebar", help="Choose the algorithm."
)

# Prediction Threshold
st.session_state.current_prediction_threshold = st.sidebar.slider(
    "Prediction Threshold:", min_value=0.0, max_value=1.0, value=st.session_state.current_prediction_threshold, step=0.01,
    key="pred_threshold_sidebar", help="Probability threshold for 'High Risk'."
)

# Feature Selection Expander
with st.sidebar.expander("Select Features & Parameters"):
    if st.session_state.project_data is not None:
        df_sidebar_config = st.session_state.project_data
        target_sidebar = st.session_state.target_variable
        cols_to_exclude_sidebar = [
            'ProjectID', 'ProjectName', 'ActualCost', 'ActualScheduleDays', 'CostVariancePerc',
            'ScheduleVariancePerc', target_sidebar, 'DerailmentRisk_Predicted_Prob', 'DerailmentRisk_Predicted',
            'CostVarianceFactor', 'ScheduleVarianceFactor', 'CompletionDate'
        ]
        # Ensure columns exist in the DataFrame before attempting selection
        features_sidebar = [col for col in df_sidebar_config.columns if col not in cols_to_exclude_sidebar]
        potential_numerical_sidebar = df_sidebar_config[features_sidebar].select_dtypes(include=np.number).columns.tolist()
        potential_categorical_sidebar = df_sidebar_config[features_sidebar].select_dtypes(include=['object', 'category']).columns.tolist()

        # Use the separate session state vars for user selection, filter defaults by available columns
        default_num_sidebar = [f for f in st.session_state.selected_numerical_features if f in potential_numerical_sidebar]
        default_cat_sidebar = [f for f in st.session_state.selected_categorical_features if f in potential_categorical_sidebar]

        st.session_state.selected_numerical_features = st.multiselect(
            "Numerical Features:", potential_numerical_sidebar,
            default=default_num_sidebar,
            key="num_features_sidebar_select"
        )
        st.session_state.selected_categorical_features = st.multiselect(
            "Categorical Features:", potential_categorical_sidebar,
            default=default_cat_sidebar,
            key="cat_features_sidebar_select"
        )

        # Train/test split settings
        st.session_state.train_size = st.slider("Training Proportion:", 0.5, 0.9, st.session_state.train_size, 0.05, key="train_split_sidebar_slider")
        st.session_state.random_state = st.number_input("Random Seed:", 0, 1000, st.session_state.random_state, 1, key="random_seed_sidebar_input")
    else:
        st.info("Load project data first.")

st.sidebar.markdown("---")

# --- Train Button (Moved to Sidebar) ---
st.sidebar.subheader("3. Train Model")
train_button_sidebar = st.sidebar.button("🚀 Train Selected Model", key="train_model_sidebar", use_container_width=True)

if train_button_sidebar:
    selected_features_sidebar = st.session_state.selected_numerical_features + st.session_state.selected_categorical_features
    if st.session_state.project_data is None:
        st.sidebar.error("❌ Cannot train: Project data not loaded.")
    elif not selected_features_sidebar:
         st.sidebar.error("❌ Cannot train: No features selected in the expander.")
    elif st.session_state.target_variable not in st.session_state.project_data.columns:
         st.sidebar.error(f"❌ Cannot train: Target '{st.session_state.target_variable}' not found.")
    else:
        with st.spinner(f"Training {st.session_state.current_model_choice} model..."):
            # Ensure features selected actually exist
            valid_selected_features = [f for f in selected_features_sidebar if f in st.session_state.project_data.columns]
            valid_numerical = [f for f in st.session_state.selected_numerical_features if f in st.session_state.project_data.columns]
            valid_categorical = [f for f in st.session_state.selected_categorical_features if f in st.session_state.project_data.columns]

            if not valid_selected_features:
                 st.sidebar.error("❌ None of the selected features exist in the current dataset.")
            else:
                training_results = train_model_and_explainers(
                    df_train=st.session_state.project_data, # Use full data
                    features=valid_selected_features,
                    numerical_features=valid_numerical,
                    categorical_features=valid_categorical,
                    target=st.session_state.target_variable,
                    train_size=st.session_state.train_size,
                    random_state=st.session_state.random_state,
                    model_name=st.session_state.current_model_choice,
                    threshold=st.session_state.current_prediction_threshold
                )

                st.sidebar.info(training_results['message']) # Display message in sidebar

                if training_results['success']:
                    # Update session state
                    st.session_state.model_pipeline = training_results['pipeline']
                    st.session_state.trained_models[st.session_state.current_model_choice] = training_results['pipeline']
                    st.session_state.X_test_original = training_results['X_test_original']
                    st.session_state.X_test_original_index = training_results['X_test_original_index']
                    st.session_state.y_test = training_results['y_test']
                    st.session_state.y_pred_prob = training_results['y_pred_prob']
                    st.session_state.feature_importance = training_results['feature_importance']
                    st.session_state.lime_explainer = training_results['lime_explainer']
                    st.session_state.preprocessor = training_results['preprocessor']
                    st.session_state.X_train_processed = training_results['X_train_processed']
                    st.session_state.X_test_processed = training_results['X_test_processed']
                    st.session_state.features_processed_names = training_results['features_processed_names']
                    st.session_state.categorical_features_indices = training_results['categorical_features_indices']
                    st.session_state.categorical_features_names = training_results['categorical_features_names']
                    st.session_state.y_train = training_results['y_train']
                    # Store selected features used for this training run
                    st.session_state.numerical_features = valid_numerical
                    st.session_state.categorical_features = valid_categorical

                    # Update main dataframe predictions
                    if st.session_state.X_test_original_index is not None and st.session_state.y_pred_prob is not None:
                        if len(st.session_state.X_test_original_index) == len(st.session_state.y_pred_prob):
                            valid_test_indices = st.session_state.X_test_original_index.intersection(st.session_state.project_data.index)
                            st.session_state.project_data['DerailmentRisk_Predicted_Prob'] = np.nan
                            st.session_state.project_data['DerailmentRisk_Predicted'] = pd.NA
                            st.session_state.project_data.loc[valid_test_indices, 'DerailmentRisk_Predicted_Prob'] = st.session_state.y_pred_prob[st.session_state.X_test_original_index.isin(valid_test_indices)]
                            st.session_state.project_data.loc[valid_test_indices, 'DerailmentRisk_Predicted'] = (st.session_state.project_data.loc[valid_test_indices, 'DerailmentRisk_Predicted_Prob'] >= st.session_state.current_prediction_threshold).astype('Int64')
                            st.session_state.predictions_made = True
                            st.session_state.auto_insights = None # Reset insights
                            st.sidebar.success("Training complete. View results in tabs.")
                            # st.rerun() # Rerun to update tabs immediately
                        else:
                            st.sidebar.error(f"Prediction Error: Length mismatch.")
                            st.session_state.predictions_made = False
                            st.session_state.model_pipeline = None
                    else:
                        st.sidebar.error("Prediction Error: Missing test index or probabilities.")
                        st.session_state.predictions_made = False
                        st.session_state.model_pipeline = None
                else:
                    st.session_state.model_pipeline = None
                    st.session_state.predictions_made = False


# --- Session Management (Sidebar) ---
st.sidebar.markdown("---")
st.sidebar.subheader("💾 Session Management")
save_state_placeholder = st.sidebar.empty()
if st.sidebar.button("Save Current Session", key="save_session_sidebar"):
    try:
        with st.spinner("Saving session state..."):
            state_to_save = { # Save user selections and data
                'project_data': st.session_state.project_data.to_json(orient='split', date_format='iso') if st.session_state.project_data is not None else None,
                'risk_register': st.session_state.risk_register.to_json(orient='split', date_format='iso') if st.session_state.risk_register is not None else None,
                'current_model_choice': st.session_state.current_model_choice,
                'current_prediction_threshold': st.session_state.current_prediction_threshold,
                'selected_numerical_features': st.session_state.selected_numerical_features,
                'selected_categorical_features': st.session_state.selected_categorical_features,
                'train_size': st.session_state.train_size,
                'random_state': st.session_state.random_state,
            }
            state_json = json.dumps(state_to_save, indent=2); state_bytes = state_json.encode('utf-8')
            b64 = base64.b64encode(state_bytes).decode()
            href = f'<a href="data:application/json;base64,{b64}" download="risklens_session.json">📥 Download Session File</a>'
            save_state_placeholder.markdown(href, unsafe_allow_html=True)
            st.sidebar.success("Session state ready for download.")
    except Exception as e: st.sidebar.error(f"Error saving session: {e}")

uploaded_session_file_sidebar = st.sidebar.file_uploader("Load Session File (.json)", type=['json'], key="session_uploader_sidebar")
if uploaded_session_file_sidebar is not None:
    try:
        with st.spinner("Loading session state..."):
            state_loaded = json.load(uploaded_session_file_sidebar)
            restored_keys = []
            # Load data first
            if 'project_data' in state_loaded and state_loaded['project_data'] is not None:
                try:
                    df_loaded = pd.read_json(state_loaded['project_data'], orient='split')
                    for col in df_loaded.select_dtypes(include='float').columns:
                         if df_loaded[col].dropna().mod(1).eq(0).all(): df_loaded[col] = df_loaded[col].astype(pd.Int64Dtype())
                    st.session_state.project_data = df_loaded
                    restored_keys.append('project_data')
                except Exception as e_load_proj: st.sidebar.error(f"Error parsing project data: {e_load_proj}")
            if 'risk_register' in state_loaded and state_loaded['risk_register'] is not None:
                try:
                    st.session_state.risk_register = pd.read_json(state_loaded['risk_register'], orient='split')
                    restored_keys.append('risk_register')
                except Exception as e_load_risk: st.sidebar.error(f"Error parsing risk register: {e_load_risk}")

            # Load other config state
            for key, value in state_loaded.items():
                if key not in ['project_data', 'risk_register'] and key in st.session_state:
                    st.session_state[key] = value
                    restored_keys.append(key)

            st.sidebar.success(f"Session state loaded for: {', '.join(restored_keys)}.")
            st.sidebar.warning("⚠️ Models were not saved. Please retrain.")
            # Reset model/prediction states
            st.session_state.model_pipeline = None; st.session_state.trained_models = {}
            st.session_state.predictions_made = False; st.session_state.lime_explainer = None
            st.session_state.monte_carlo_results = None; st.session_state.batch_scenario_results = None
            st.session_state.auto_insights = None
            st.rerun()
    except Exception as e: st.sidebar.error(f"Error loading session file: {e}")


# --- Main Application Area ---
st.title("RiskLens Pro – Project Risk Prediction")
st.markdown(f"_Leveraging data and advanced analytics to anticipate and mitigate project derailment._")
if st.session_state.predictions_made:
    st.success(f"Model Trained: **{st.session_state.current_model_choice}** (Threshold: {st.session_state.current_prediction_threshold:.2f}). Explore the tabs below.")
else:
    st.info("Configure settings and click **'Train Selected Model'** in the sidebar to begin analysis.")
st.markdown("---")


# --- Define Tabs (New Structure) ---
tab_titles_new = [
    "👋 Welcome",
    "📊 Executive Summary",
    "🔍 Portfolio Deep Dive",
    "🔬 Model Analysis & Explainability",
    "🎲 Simulation & Scenarios",
    "💾 Data Management"
]
tabs = st.tabs(tab_titles_new)

# --- Welcome Tab (Narrative Focus) ---
with tabs[0]:
    st.markdown("<div class='welcome-section'>", unsafe_allow_html=True)
    st.header("The Challenge: Navigating Project Complexity")
    st.markdown("""
    Delivering complex projects on time and within budget is a significant challenge. Factors like scope changes, resource constraints, technical hurdles, and external dependencies can introduce risks, leading to costly overruns and delays. Proactively identifying and understanding these risks is crucial for successful project delivery and maintaining client satisfaction.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='welcome-section'>", unsafe_allow_html=True)
    st.header("The Solution: RiskLens Pro")
    st.markdown(f"""
    **RiskLens Pro**, powered by Arcadis expertise, leverages your project data and machine learning to provide early warnings about potential project derailment. By analyzing historical patterns and current project characteristics, it predicts the likelihood of significant cost or schedule overruns, enabling you to focus attention where it's needed most.
    """)
    # Placeholder for a relevant visual - e.g., a conceptual diagram
    # st.image("path/to/conceptual_diagram.png", caption="Predictive Risk Identification Flow")
    st.markdown("</div>", unsafe_allow_html=True)


    st.markdown("<div class='welcome-section'>", unsafe_allow_html=True)
    st.header("Key Capabilities")
    col1_cap, col2_cap = st.columns(2)
    with col1_cap:
        st.markdown("""
        <ul class="styled-list">
            <li><b>Predictive Modeling:</b> Utilizes algorithms like Random Forest, XGBoost, and Logistic Regression to forecast risk probabilities.</li>
            <li><b>Portfolio Overview:</b> Provides a high-level summary of risk across all projects (<i>Executive Summary</i> tab).</li>
            <li><b>Detailed Analysis:</b> Allows filtering and sorting projects to identify specific areas of concern (<i>Portfolio Deep Dive</i> tab).</li>
            <li><b>Dynamic Risk Register:</b> Integrates project predictions with your registered risks for focused mitigation.</li>
        </ul>
        """, unsafe_allow_html=True)
    with col2_cap:
        st.markdown("""
        <ul class="styled-list">
            <li><b>Explainability (LIME):</b> Reveals the key factors driving the risk prediction for individual projects (<i>Model Analysis</i> tab).</li>
            <li><b>Model Evaluation:</b> Assesses the accuracy and reliability of the predictive model (<i>Model Analysis</i> tab).</li>
            <li><b>Uncertainty Quantification:</b> Uses Monte Carlo simulation to understand the confidence in predictions (<i>Simulation</i> tab).</li>
            <li><b>Scenario Planning:</b> Explores the potential impact of changes on project risk ('What-If' analysis) (<i>Simulation</i> tab).</li>
        </ul>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


    st.markdown("<div class='welcome-section'>", unsafe_allow_html=True)
    st.header("Answering Your Questions")
    st.markdown("""
    RiskLens Pro is designed to help answer critical project management questions:
    <ul class="styled-list">
        <li>❓ <b>Which projects need my immediate attention?</b> (See <i>Executive Summary</i> & <i>Portfolio Deep Dive</i>)</li>
        <li>❓ <b>Why is a specific project flagged as high-risk?</b> (Use LIME in <i>Model Analysis</i>)</li>
        <li>❓ <b>How reliable is this prediction?</b> (Check <i>Model Analysis</i> metrics & <i>Simulation</i> results)</li>
        <li>❓ <b>What happens if resources become scarce or scope increases?</b> (Explore <i>Simulation & Scenarios</i>)</li>
        <li>❓ <b>Which factors generally drive risk in our portfolio?</b> (View Feature Importance in <i>Model Analysis</i>)</li>
    </ul>
    <b>Get started by configuring your analysis and training a model using the options in the sidebar.</b>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Executive Summary Tab ---
with tabs[1]:
    st.header("📊 Executive Summary")
    st.markdown("_High-level overview of the portfolio's risk landscape based on the trained model._")

    if not st.session_state.predictions_made:
        st.info("ℹ️ Train a model using the sidebar configuration to view the Executive Summary.")
    elif st.session_state.project_data is None:
        st.warning("⚠️ Project data not loaded. Go to '💾 Data Management'.")
    else:
        df_summary = st.session_state.project_data.copy()
        current_threshold = st.session_state.current_prediction_threshold
        model_name_display = st.session_state.current_model_choice

        # Ensure predictions reflect current threshold
        if 'DerailmentRisk_Predicted_Prob' in df_summary.columns and df_summary['DerailmentRisk_Predicted_Prob'].notna().any():
            mask = df_summary['DerailmentRisk_Predicted_Prob'].notna()
            df_summary.loc[mask, 'DerailmentRisk_Predicted'] = (df_summary.loc[mask, 'DerailmentRisk_Predicted_Prob'] >= current_threshold).astype('Int64')
            df_summary.loc[~mask, 'DerailmentRisk_Predicted'] = pd.NA
            kpi_source = f"(Model: {model_name_display}, Threshold: {current_threshold:.2f})"
        else:
             df_summary['DerailmentRisk_Predicted'] = pd.NA # No predictions available
             kpi_source = "(No predictions available)"

        # --- Key Metrics (Tiles) ---
        st.subheader("📈 Portfolio Risk Snapshot")
        kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
        total_projects_summ = len(df_summary)
        predicted_count_summ = df_summary['DerailmentRisk_Predicted'].notna().sum()
        high_risk_count_summ = int(df_summary['DerailmentRisk_Predicted'].eq(1).sum())
        high_risk_rate_summ = (high_risk_count_summ / predicted_count_summ) * 100 if predicted_count_summ > 0 else 0

        with kpi_col1:
            st.metric(label="Total Projects", value=f"{total_projects_summ}")
        with kpi_col2:
            st.metric(label="🚨 High-Risk Projects", value=f"{high_risk_count_summ}", help=f"Count of projects predicted as high risk {kpi_source}.")
        with kpi_col3:
            st.metric(label="📈 High-Risk Rate", value=f"{high_risk_rate_summ:.1f}%", help=f"Percentage of projects with predictions that are high risk {kpi_source}.")

        # Add average overrun metrics if actuals are available
        if 'CostVariancePerc' in df_summary.columns and 'ScheduleVariancePerc' in df_summary.columns:
            kpi_col4, kpi_col5 = st.columns(2)
            predicted_high_risk_df_summ = df_summary[df_summary['DerailmentRisk_Predicted'] == 1]
            avg_cost_overrun_hr_summ = predicted_high_risk_df_summ[predicted_high_risk_df_summ['CostVariancePerc'] > 0]['CostVariancePerc'].mean()
            avg_schedule_overrun_hr_summ = predicted_high_risk_df_summ[predicted_high_risk_df_summ['ScheduleVariancePerc'] > 0]['ScheduleVariancePerc'].mean()
            avg_cost_overrun_hr_summ = 0 if pd.isna(avg_cost_overrun_hr_summ) else avg_cost_overrun_hr_summ
            avg_schedule_overrun_hr_summ = 0 if pd.isna(avg_schedule_overrun_hr_summ) else avg_schedule_overrun_hr_summ
            with kpi_col4:
                 st.metric(label="💰 Avg. Cost Overrun % (High-Risk)", value=f"{avg_cost_overrun_hr_summ:.1f}%", help=f"Avg historical cost overrun for projects predicted as high risk {kpi_source}.")
            with kpi_col5:
                 st.metric(label="⏳ Avg. Schedule Overrun % (High-Risk)", value=f"{avg_schedule_overrun_hr_summ:.1f}%", help=f"Avg historical schedule overrun for projects predicted as high risk {kpi_source}.")


        st.markdown("---")

        # --- AI-Generated Summary ---
        st.subheader("💡 AI-Generated Narrative Summary")
        with st.container():
            # Generate or display insights
            if st.session_state.auto_insights:
                st.markdown(st.session_state.auto_insights)
            elif st.session_state.feature_importance is not None and not st.session_state.feature_importance.empty:
                 st.info("Generating narrative summary...")
                 try:
                     top_features_summ = st.session_state.feature_importance.head(3)['Feature'].tolist()
                     insight_text = f"The **{model_name_display}** model predicts **{high_risk_count_summ}** projects ({high_risk_rate_summ:.1f}% of those with predictions) are at high risk of derailment, using a threshold of {current_threshold:.2f}.\n\n"
                     insight_text += f"**Key Insights:**\n"
                     insight_text += f"* **Primary Risk Drivers:** Across the portfolio, the factors most strongly correlated with increased risk appear to be: **{', '.join(top_features_summ)}**.\n"

                     # Add risk by type/region insights if available
                     highest_risk_type_text_summ = "N/A"
                     if 'ProjectType' in df_summary.columns:
                         risk_by_type_summ = df_summary.dropna(subset=['DerailmentRisk_Predicted']).groupby('ProjectType')['DerailmentRisk_Predicted'].mean().sort_values(ascending=False)
                         if not risk_by_type_summ.empty:
                             highest_risk_type_summ = risk_by_type_summ.index[0]; highest_risk_rate_t = risk_by_type_summ.iloc[0]
                             highest_risk_type_text_summ = f"'{highest_risk_type_summ}' ({highest_risk_rate_t:.1%})"
                             insight_text += f"* **Highest Risk Project Type:** Projects classified as {highest_risk_type_text_summ} show the highest average predicted risk rate.\n"

                     highest_risk_region_text_summ = "N/A"
                     if 'Region' in df_summary.columns:
                         risk_by_region_summ = df_summary.dropna(subset=['DerailmentRisk_Predicted']).groupby('Region')['DerailmentRisk_Predicted'].mean().sort_values(ascending=False)
                         if not risk_by_region_summ.empty:
                             highest_risk_region_summ = risk_by_region_summ.index[0]; highest_risk_rate_r = risk_by_region_summ.iloc[0]
                             highest_risk_region_text_summ = f"'{highest_risk_region_summ}' ({highest_risk_rate_r:.1%})"
                             insight_text += f"* **Highest Risk Region:** The '{highest_risk_region_summ}' region currently exhibits the highest average predicted risk rate at {highest_risk_rate_r:.1%}.\n"

                     # Add prediction distribution insight
                     if 'DerailmentRisk_Predicted_Prob' in df_summary.columns and df_summary['DerailmentRisk_Predicted_Prob'].notna().any():
                         avg_prob_summ = df_summary['DerailmentRisk_Predicted_Prob'].mean(); median_prob_summ = df_summary['DerailmentRisk_Predicted_Prob'].median()
                         insight_text += f"* **Prediction Certainty:** The average predicted risk probability across projects is **{avg_prob_summ:.1%}** (median: **{median_prob_summ:.1%}**). A wider spread might indicate greater uncertainty overall.\n"

                     insight_text += f"\n**Recommendation:** Prioritize investigation and potential mitigation actions for the identified high-risk projects, paying close attention to the top risk drivers ({', '.join(top_features_summ[:2])}...). Consider focusing efforts on projects within the {highest_risk_type_text_summ} type or {highest_risk_region_text_summ} region if applicable. Use the 'Portfolio Deep Dive' and 'Model Analysis' tabs for more detailed investigation."
                     st.session_state.auto_insights = insight_text
                     st.markdown(insight_text)
                 except Exception as e_insight_summ:
                     st.warning(f"Could not generate full narrative summary: {e_insight_summ}")
                     if 'insight_text' in locals(): st.markdown(insight_text) # Show partial if possible
            else:
                st.info("Train the model to generate the narrative summary.")

        st.markdown("---")

        # --- Key Visuals ---
        st.subheader("Visual Overview")
        viz_col_summ1, viz_col_summ2 = st.columns(2)

        with viz_col_summ1:
            # Risk Distribution Histogram
            st.markdown("**Predicted Risk Distribution**")
            if df_summary['DerailmentRisk_Predicted_Prob'].notna().any():
                fig_risk_dist_summ = px.histogram(df_summary.dropna(subset=['DerailmentRisk_Predicted_Prob']),
                                                x="DerailmentRisk_Predicted_Prob", nbins=30,
                                                title="Distribution of Predicted Risk Probabilities",
                                                labels={'DerailmentRisk_Predicted_Prob': 'Predicted Probability'},
                                                color_discrete_sequence=[ARCADIS_ORANGE])
                fig_risk_dist_summ.add_vline(x=current_threshold, line_dash="dash", line_color=ARCADIS_BLACK, annotation_text=f"Threshold ({current_threshold:.2f})")
                fig_risk_dist_summ.update_layout(bargap=0.1, title_x=0.5, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=30, b=10)) # Adjust margins
                st.plotly_chart(fig_risk_dist_summ, use_container_width=True)
            else:
                st.info("No prediction probabilities available.")

        with viz_col_summ2:
             # Top 5 Risky Projects (Simpler View)
            st.markdown(f"**⚠️ Top 5 Highest Risk Projects**")
            if df_summary['DerailmentRisk_Predicted_Prob'].notna().any():
                top_5_risky = df_summary.sort_values(by='DerailmentRisk_Predicted_Prob', ascending=False, na_position='last').head(5)
                top_5_risky['Risk Probability'] = top_5_risky['DerailmentRisk_Predicted_Prob'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                st.dataframe(top_5_risky[['ProjectID', 'ProjectName', 'Risk Probability']], use_container_width=True, hide_index=True)
            else:
                st.info("No predictions available.")

# --- Portfolio Deep Dive Tab ---
with tabs[2]:
    st.header("🔍 Portfolio Deep Dive")
    st.markdown("_Explore detailed project data, predictions, and registered risks._")

    if not st.session_state.predictions_made:
        st.info("ℹ️ Train a model using the sidebar configuration to view the Portfolio Deep Dive.")
    elif st.session_state.project_data is None:
        st.warning("⚠️ Project data not loaded. Go to '💾 Data Management'.")
    else:
        df_portfolio = st.session_state.project_data.copy()
        df_risk_reg_portfolio = st.session_state.risk_register

        # --- Full Project Table with Filters ---
        st.subheader("Project Data & Predictions")
        with st.container():
            # Add filters for Project Type, Region, Predicted Risk
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            with filter_col1:
                # Check if 'ProjectType' column exists before creating filter
                if 'ProjectType' in df_portfolio.columns:
                    types = ["All"] + sorted(df_portfolio['ProjectType'].dropna().unique())
                    sel_type = st.selectbox("Filter by Project Type:", types, key="filter_type_deepdive")
                else:
                    sel_type = "All"
                    st.caption("'ProjectType' column not found.")
            with filter_col2:
                 # Check if 'Region' column exists
                if 'Region' in df_portfolio.columns:
                    regions = ["All"] + sorted(df_portfolio['Region'].dropna().unique())
                    sel_region = st.selectbox("Filter by Region:", regions, key="filter_region_deepdive")
                else:
                    sel_region = "All"
                    st.caption("'Region' column not found.")
            with filter_col3:
                # Check if predictions exist
                if 'DerailmentRisk_Predicted' in df_portfolio.columns and df_portfolio['DerailmentRisk_Predicted'].notna().any():
                    risk_filter = ["All", "High Risk", "Low Risk"]
                    sel_risk = st.selectbox("Filter by Predicted Risk:", risk_filter, key="filter_risk_deepdive",
                                            help="Filters based on current threshold in sidebar.")
                else:
                    sel_risk = "All"
                    st.caption("No predictions available.")


            # Apply filters
            df_filtered_portfolio = df_portfolio.copy()
            if sel_type != "All" and 'ProjectType' in df_filtered_portfolio.columns:
                df_filtered_portfolio = df_filtered_portfolio[df_filtered_portfolio['ProjectType'] == sel_type]
            if sel_region != "All" and 'Region' in df_filtered_portfolio.columns:
                df_filtered_portfolio = df_filtered_portfolio[df_filtered_portfolio['Region'] == sel_region]
            if sel_risk != "All" and 'DerailmentRisk_Predicted' in df_filtered_portfolio.columns:
                risk_val = 1 if sel_risk == "High Risk" else 0
                # Handle potential pd.NA in comparison
                df_filtered_portfolio = df_filtered_portfolio[df_filtered_portfolio['DerailmentRisk_Predicted'].eq(risk_val)]

            # --- FIX 3: Create formatted column BEFOREHAND ---
            df_display_portfolio = df_filtered_portfolio.copy() # Create a copy for display modifications
            if 'InitialCostEstimate' in df_display_portfolio.columns:
                # Convert to numeric, coercing errors and filling NaNs
                cost_numeric = pd.to_numeric(df_display_portfolio['InitialCostEstimate'], errors='coerce').fillna(0)
                # Create the display column by dividing
                df_display_portfolio['InitialCostEstimate_Display'] = cost_numeric / 1000000

            # Define columns to display (including the new formatted one)
            display_columns = df_display_portfolio.columns.tolist()
            if 'InitialCostEstimate_Display' in display_columns:
                 # Optionally remove the original if desired, or keep both
                 if 'InitialCostEstimate' in display_columns:
                     display_columns.remove('InitialCostEstimate')

            # Display filtered data using the modified dataframe
            st.dataframe(df_display_portfolio[display_columns], # Select columns for display
                         use_container_width=True,
                         hide_index=True,
                         column_config={ # Add formatting for key columns
                             "DerailmentRisk_Predicted_Prob": st.column_config.ProgressColumn(
                                 "Risk Probability", format="%.1f%%", min_value=0, max_value=1,
                                 help="Predicted probability of project derailment"
                             ),
                             "DerailmentRisk_Predicted": st.column_config.NumberColumn(
                                 "Predicted Risk (1=High)", help="1 = High Risk, 0 = Low Risk (based on threshold)"
                             ),
                             "InitialCostEstimate_Display": st.column_config.NumberColumn( # Format the new column
                                 "Initial Cost ($ M)",
                                 format="$ %.2fM", # Apply format, NO factor needed
                                 help="Initial cost estimate in millions of dollars"
                                 ),
                             "CostVariancePerc": st.column_config.NumberColumn("Cost Var %", format="%.1f%%"),
                             "ScheduleVariancePerc": st.column_config.NumberColumn("Sched Var %", format="%.1f%%"),
                         })
            st.caption(f"Displaying {len(df_filtered_portfolio)} of {len(df_portfolio)} projects.")

        st.markdown("---")

        # --- Risk Breakdowns ---
        st.subheader("Risk Breakdowns")
        breakdown_col1, breakdown_col2 = st.columns(2)
        with breakdown_col1:
            # Risk by Project Type Bar Chart
            st.markdown("**Risk by Project Type**")
            if 'ProjectType' in df_portfolio.columns and df_portfolio['DerailmentRisk_Predicted'].notna().any():
                 df_pred_valid_type = df_portfolio.dropna(subset=['DerailmentRisk_Predicted', 'ProjectType'])
                 # Ensure prediction column is numeric for mean calculation
                 df_pred_valid_type['DerailmentRisk_Predicted_Num'] = pd.to_numeric(df_pred_valid_type['DerailmentRisk_Predicted'], errors='coerce')
                 risk_by_type_deep = df_pred_valid_type.groupby('ProjectType')['DerailmentRisk_Predicted_Num'].mean().reset_index()
                 risk_by_type_deep.rename(columns={'DerailmentRisk_Predicted_Num': 'HighRiskRate'}, inplace=True)
                 risk_by_type_deep['HighRiskRate'] *= 100 # As percentage

                 fig_type_deep = px.bar(risk_by_type_deep.sort_values('HighRiskRate', ascending=False),
                                   x='ProjectType', y='HighRiskRate',
                                   title="Avg. Predicted High-Risk Rate by Project Type",
                                   labels={'HighRiskRate': 'High-Risk Rate (%)', 'ProjectType': 'Project Type'},
                                   color_discrete_sequence=[ARCADIS_SECONDARY_PALETTE[0]]) # Teal
                 fig_type_deep.update_layout(title_x=0.5, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=30, b=10))
                 st.plotly_chart(fig_type_deep, use_container_width=True)
            else: st.info("Risk by Project Type requires predictions and 'ProjectType' column.")

        with breakdown_col2:
             # Risk by Region Bar Chart
            st.markdown("**Risk by Region**")
            if 'Region' in df_portfolio.columns and df_portfolio['DerailmentRisk_Predicted'].notna().any():
                 df_pred_valid_region = df_portfolio.dropna(subset=['DerailmentRisk_Predicted', 'Region'])
                 df_pred_valid_region['DerailmentRisk_Predicted_Num'] = pd.to_numeric(df_pred_valid_region['DerailmentRisk_Predicted'], errors='coerce')
                 risk_by_region_deep = df_pred_valid_region.groupby('Region')['DerailmentRisk_Predicted_Num'].mean().reset_index()
                 risk_by_region_deep.rename(columns={'DerailmentRisk_Predicted_Num': 'HighRiskRate'}, inplace=True)
                 risk_by_region_deep['HighRiskRate'] *= 100 # As percentage

                 fig_region_deep = px.bar(risk_by_region_deep.sort_values('HighRiskRate', ascending=False),
                                   x='Region', y='HighRiskRate',
                                   title="Avg. Predicted High-Risk Rate by Region",
                                   labels={'HighRiskRate': 'High-Risk Rate (%)', 'Region': 'Region'},
                                   color_discrete_sequence=[ARCADIS_SECONDARY_PALETTE[1]]) # Dark Grey
                 fig_region_deep.update_layout(title_x=0.5, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=30, b=10))
                 st.plotly_chart(fig_region_deep, use_container_width=True)
            else: st.info("Risk by Region requires predictions and 'Region' column.")

        st.markdown("---")

        # --- Dynamic Risk Register ---
        st.subheader("🚨 Dynamic Risk Register (Predicted High-Risk Projects)")
        if df_risk_reg_portfolio is None:
            st.warning("⚠️ Risk Register data not loaded.")
        else:
            st.markdown("_Focuses on projects currently predicted as high-risk and their associated registered risks._")
            high_risk_project_ids_deep = df_portfolio[df_portfolio['DerailmentRisk_Predicted'].eq(1)]['ProjectID'].tolist()

            if high_risk_project_ids_deep:
                df_register_filtered_deep = df_risk_reg_portfolio[df_risk_reg_portfolio['ProjectID'].isin(high_risk_project_ids_deep)].copy()
                if not df_register_filtered_deep.empty:
                    project_probs_deep = df_portfolio.loc[df_portfolio['ProjectID'].isin(high_risk_project_ids_deep), ['ProjectID', 'DerailmentRisk_Predicted_Prob', 'ProjectName']].set_index('ProjectID')
                    df_register_filtered_deep = df_register_filtered_deep.merge(project_probs_deep.reset_index(), on='ProjectID', how='left')

                    # Filtering and Display Logic
                    filter_cols_deep = st.columns([2, 2, 2, 1])
                    with filter_cols_deep[0]: selected_projects_deep = st.multiselect("Filter by ProjectID:", options=sorted(high_risk_project_ids_deep), key="reg_proj_filter_deep")
                    with filter_cols_deep[1]: selected_urgencies_deep = st.multiselect("Filter by Mitigation Urgency:", options=sorted(df_register_filtered_deep['MitigationUrgency'].unique()), key="reg_urg_filter_deep")
                    with filter_cols_deep[2]: selected_statuses_deep = st.multiselect("Filter by Mitigation Status:", options=sorted(df_register_filtered_deep['MitigationStatus'].unique()), key="reg_stat_filter_deep")
                    with filter_cols_deep[3]: sort_by_deep = st.selectbox("Sort by:", ["RiskScore", "DerailmentRisk_Predicted_Prob", "DueDate"], index=0, key="reg_sort_deep"); sort_asc_deep = st.checkbox("Ascending", False, key="reg_asc_deep")

                    df_display_deep = df_register_filtered_deep.copy()
                    if selected_projects_deep: df_display_deep = df_display_deep[df_display_deep['ProjectID'].isin(selected_projects_deep)]
                    if selected_urgencies_deep: df_display_deep = df_display_deep[df_display_deep['MitigationUrgency'].isin(selected_urgencies_deep)]
                    if selected_statuses_deep: df_display_deep = df_display_deep[df_display_deep['MitigationStatus'].isin(selected_statuses_deep)]
                    df_display_deep = df_display_deep.sort_values(by=sort_by_deep, ascending=sort_asc_deep, na_position='last')

                    st.markdown(f"**Displaying {len(df_display_deep)} risks for {len(df_display_deep['ProjectID'].unique())} high-risk projects.**")
                    st.dataframe(df_display_deep[['ProjectID', 'ProjectName', 'DerailmentRisk_Predicted_Prob', 'RiskID', 'RiskDescription', 'Likelihood', 'Impact', 'RiskScore', 'MitigationUrgency', 'MitigationStatus', 'Owner', 'DueDate']].rename(columns={'DerailmentRisk_Predicted_Prob': 'Proj. Risk Prob.'}),
                        use_container_width=True,
                        column_config={
                             "Proj. Risk Prob.": st.column_config.NumberColumn("Proj. Risk Prob.", format="%.3f"),
                             "RiskScore": st.column_config.NumberColumn(format="%d"), "DueDate": st.column_config.DateColumn(format="YYYY-MM-DD"),
                        }
                    )

                    # Risk Register Heatmap
                    st.markdown("**Risk Register Heatmap (Filtered Projects)**")
                    if not df_display_deep.empty and 'LikelihoodScore' in df_display_deep.columns and 'ImpactScore' in df_display_deep.columns:
                         heatmap_df_deep = df_display_deep.dropna(subset=['LikelihoodScore', 'ImpactScore'])
                         if not heatmap_df_deep.empty:
                             heatmap_data_deep = heatmap_df_deep.groupby(['LikelihoodScore', 'ImpactScore']).size().unstack(fill_value=0).reindex(index=range(1, 6), columns=range(1, 6), fill_value=0)
                             likelihood_map_rev_deep = {1: 'Very Low', 2: 'Low', 3: 'Medium', 4: 'High', 5: 'Very High'}
                             impact_map_rev_deep = {1: 'Very Low', 2: 'Low', 3: 'Medium', 4: 'High', 5: 'Very High'}
                             fig_heatmap_deep = go.Figure(data=go.Heatmap(
                                 z=heatmap_data_deep.values, x=[impact_map_rev_deep.get(i, i) for i in heatmap_data_deep.columns], y=[likelihood_map_rev_deep.get(i, i) for i in heatmap_data_deep.index],
                                 colorscale=[[0, ARCADIS_LIGHT_GREY], [0.5, "#FFA500"], [1, ARCADIS_ORANGE]], hoverongaps=False,
                                 text=heatmap_data_deep.values, texttemplate="%{text}", textfont={"size":10, "color": ARCADIS_BLACK}
                             ))
                             fig_heatmap_deep.update_layout(title='Risk Count by Likelihood and Impact', xaxis_title="Impact", yaxis_title="Likelihood",
                                 yaxis={'categoryorder':'array', 'categoryarray': [likelihood_map_rev_deep.get(i, i) for i in range(5, 0, -1)]}, xaxis={'side': 'top'},
                                 margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                             st.plotly_chart(fig_heatmap_deep, use_container_width=True)
                         else: st.info("No risks with valid scores found for heatmap.")
                    else: st.info("Heatmap requires 'LikelihoodScore'/'ImpactScore' columns and filtered risks.")
                else: st.info("No risks found in the register for the projects currently predicted as high-risk.")
            else: st.success("✅ No projects are currently predicted as high-risk based on the model and threshold.")


# --- Model Analysis & Explainability Tab ---
with tabs[3]: # Index adjusted for new tab order
    st.header("🔬 Model Analysis & Explainability")
    st.markdown("_Evaluate the trained model's performance, understand feature importance, compare models, and explain individual predictions._")

    if not st.session_state.predictions_made:
        st.info("ℹ️ Train a model using the sidebar configuration to view analysis results.")
    elif st.session_state.model_pipeline is None or st.session_state.y_test is None or st.session_state.y_pred_prob is None:
         st.warning("⚠️ Model analysis requires a trained model and test set results. Please retrain if necessary.")
    else:
        # --- Model Evaluation ---
        st.subheader("📊 Model Performance Evaluation (on Test Set)")
        with st.container():
            try:
                y_true = st.session_state.y_test
                y_pred_prob = st.session_state.y_pred_prob

                if len(y_true) != len(y_pred_prob):
                    st.error(f"Evaluation Error: Mismatch between true labels ({len(y_true)}) and predictions ({len(y_pred_prob)}).")
                else:
                    y_pred_class = (y_pred_prob >= st.session_state.current_prediction_threshold).astype(int)
                    accuracy = accuracy_score(y_true, y_pred_class)
                    report = classification_report(y_true, y_pred_class, output_dict=True, zero_division=0)
                    cm = confusion_matrix(y_true, y_pred_class)
                    roc_auc_val = roc_auc_score(y_true, y_pred_prob) if len(np.unique(y_true)) > 1 else None
                    pr_auc_val = None
                    if len(np.unique(y_true)) > 1:
                        precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
                        pr_auc_val = auc(recall, precision)
                    f1 = f1_score(y_true, y_pred_class, zero_division=0)
                    precision_score_val = precision_score(y_true, y_pred_class, zero_division=0)
                    recall_score_val = recall_score(y_true, y_pred_class, zero_division=0)

                    # Display Metrics
                    eval_col1, eval_col2 = st.columns(2)
                    with eval_col1:
                        st.metric("✅ Accuracy", f"{accuracy:.3f}")
                        st.metric("📈 ROC AUC", f"{roc_auc_val:.3f}" if roc_auc_val is not None else "N/A")
                        st.metric("📊 PR AUC", f"{pr_auc_val:.3f}" if pr_auc_val is not None else "N/A")
                    with eval_col2:
                        st.metric("⚖️ F1 Score", f"{f1:.3f}")
                        st.metric("🎯 Precision", f"{precision_score_val:.3f}")
                        st.metric("🔍 Recall (Sensitivity)", f"{recall_score_val:.3f}")
                    st.caption(f"Metrics calculated at threshold {st.session_state.current_prediction_threshold:.2f}")

                    # Display Plots
                    plot_col1, plot_col2 = st.columns(2)
                    with plot_col1:
                        fig_cm = plot_confusion_matrix_plotly(cm, labels=['No Derailment (0)', 'Derailment (1)'])
                        st.plotly_chart(fig_cm, use_container_width=True)
                    with plot_col2:
                         if report:
                            st.text("Classification Report:")
                            report_df = pd.DataFrame(report).transpose().round(3)
                            st.dataframe(report_df)
                         else: st.info("Classification report not available.")

                    plot_col3, plot_col4 = st.columns(2)
                    with plot_col3:
                        if roc_auc_val is not None:
                            fig_roc = plot_roc_curve_plotly(y_true, y_pred_prob, st.session_state.current_model_choice)
                            st.plotly_chart(fig_roc, use_container_width=True)
                        else: st.info("ROC curve not available.")
                    with plot_col4:
                        if pr_auc_val is not None:
                            fig_pr = plot_precision_recall_curve_plotly(y_true, y_pred_prob, st.session_state.current_model_choice)
                            st.plotly_chart(fig_pr, use_container_width=True)
                        else: st.info("PR curve not available.")

            except Exception as e: st.error(f"Error displaying model evaluation: {e}"); st.exception(e)

        # --- Display Feature Importance (Overall) ---
        st.markdown("---")
        st.subheader("💡 Overall Feature Importance")
        with st.container():
            if st.session_state.feature_importance is not None and not st.session_state.feature_importance.empty:
                st.markdown(f"_Which features have the biggest impact on the **{st.session_state.current_model_choice}** model's predictions overall?_")
                fig_importance = px.bar(st.session_state.feature_importance.head(20),
                                        x='Importance', y='Feature', orientation='h',
                                        title=f"Top 20 Feature Importances ({st.session_state.current_model_choice})",
                                        color='Importance', color_continuous_scale=px.colors.sequential.Oranges_r)
                fig_importance.update_layout(yaxis={'categoryorder':'total ascending'}, title_x=0.5, height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_importance, use_container_width=True)
            else: st.info("Overall feature importance not calculated or available.")

        # --- Model Comparison Dashboard ---
        st.markdown("---")
        st.subheader("🔄 Model Comparison Dashboard")
        with st.container():
            st.markdown("_Compare performance across different models using cross-validation on the **entire dataset**._")
            st.caption("ℹ️ This runs independently of the main model training. Results show average performance across 5 data folds.")
            compare_button_analysis = st.button("Run Model Comparison", key="compare_models_analysis")

            if compare_button_analysis:
                if st.session_state.project_data is None or not st.session_state.selected_numerical_features or not st.session_state.selected_categorical_features:
                     st.warning("Cannot run comparison without loaded data and selected features in the sidebar.")
                else:
                    with st.spinner("Running cross-validation... This may take a moment."):
                        # (CV Logic remains the same as previous version)
                        try:
                            models_to_compare = ["Random Forest", "XGBoost", "Logistic Regression"]
                            comparison_results = {}
                            metrics_to_calc = {
                                'Accuracy': make_scorer(accuracy_score), 'ROC_AUC': make_scorer(roc_auc_score, needs_proba=True, multi_class='ignore'),
                                'F1_Score': make_scorer(f1_score, zero_division=0), 'Precision': make_scorer(precision_score, zero_division=0),
                                'Recall': make_scorer(recall_score, zero_division=0)
                            }
                            features_cv = st.session_state.selected_numerical_features + st.session_state.selected_categorical_features
                            target_cv = st.session_state.target_variable
                            X_full = st.session_state.project_data[features_cv].copy()
                            y_full = st.session_state.project_data[target_cv].copy()
                            valid_indices_full = y_full.dropna().index
                            X_full = X_full.loc[valid_indices_full]; y_full = y_full.loc[valid_indices_full]
                            for col in st.session_state.selected_numerical_features:
                                if X_full[col].isnull().any(): X_full[col] = X_full[col].fillna(X_full[col].mean())
                            for col in st.session_state.selected_categorical_features:
                                if X_full[col].isnull().any(): X_full[col] = X_full[col].fillna(X_full[col].mode()[0] if not X_full[col].mode().empty else 'Unknown')

                            for model_name in models_to_compare:
                                pipeline_comp = get_model_pipeline(model_name, st.session_state.selected_numerical_features, st.session_state.selected_categorical_features, X_full.shape, y_full)
                                scores = {}
                                for metric_name, scorer in metrics_to_calc.items():
                                    try:
                                        cv_scores = cross_val_score(pipeline_comp, X_full, y_full, cv=5, scoring=scorer, n_jobs=-1)
                                        scores[metric_name] = np.mean(cv_scores)
                                    except ValueError as e_cv:
                                        st.warning(f"CV {metric_name} failed for {model_name}: {e_cv}")
                                        scores[metric_name] = np.nan
                                comparison_results[model_name] = scores
                            st.session_state.model_comparison_results = pd.DataFrame(comparison_results).T
                            st.success("✅ Model comparison complete.")
                        except Exception as e:
                            st.error(f"❌ Error during model comparison: {e}"); st.exception(e)
                            st.session_state.model_comparison_results = None

            # Display comparison results
            if st.session_state.model_comparison_results is not None:
                st.dataframe(st.session_state.model_comparison_results.style.format("{:.3f}", na_rep="N/A").highlight_max(axis=0, color=ARCADIS_ORANGE))
                st.caption("Higher values are generally better. Compares average performance across 5 folds.")
            else:
                st.info("Click 'Run Model Comparison' to see results.")

        # --- Project Deep Dive & Individual Explanations (LIME) ---
        st.markdown("---")
        st.subheader("🔎 Local Explainability (LIME)")
        with st.container():
            st.markdown("_Why did a specific project receive its risk prediction? Select a project **from the test set** to see its local drivers._")
            lime_ready = st.session_state.lime_explainer is not None

            if lime_ready and st.session_state.X_test_original is not None and not st.session_state.X_test_original.empty:
                try:
                    test_project_ids = st.session_state.X_test_original.index
                    valid_test_indices_for_display = test_project_ids.intersection(st.session_state.project_data.index)
                    if not valid_test_indices_for_display.empty:
                        test_project_info = st.session_state.project_data.loc[valid_test_indices_for_display, ['ProjectID', 'ProjectName']].reset_index()
                        selected_project_index = st.selectbox(
                            "Select Test Set Project for Explanation:",
                            options=test_project_info['index'].tolist(),
                            format_func=lambda x: f"{test_project_info.set_index('index').loc[x, 'ProjectID']} - {test_project_info.set_index('index').loc[x, 'ProjectName']}",
                            key="project_explain_select_analysis"
                        )

                        if selected_project_index is not None:
                            instance_original_data = st.session_state.X_test_original.loc[[selected_project_index]]
                            processed_data_row_index = st.session_state.X_test_original.index.get_loc(selected_project_index)
                            instance_processed_data = st.session_state.X_test_processed.iloc[processed_data_row_index]

                            if processed_data_row_index < len(st.session_state.y_pred_prob):
                                instance_pred_prob = st.session_state.y_pred_prob[processed_data_row_index]
                                instance_pred = 1 if instance_pred_prob >= st.session_state.current_prediction_threshold else 0
                                project_id_display = test_project_info.set_index('index').loc[selected_project_index, 'ProjectID']

                                explain_col1, explain_col2 = st.columns([2, 1])
                                with explain_col1:
                                    st.markdown(f"**Explanation for: {project_id_display}**")
                                    st.markdown(f"*Predicted Risk Probability:* `{instance_pred_prob:.1%}` | *Predicted Class:* `{'High Risk' if instance_pred == 1 else 'Low Risk'}`")
                                    st.markdown("---")
                                    with st.spinner("Generating LIME explanation..."):
                                        try:
                                            exp_lime = st.session_state.lime_explainer.explain_instance(
                                                data_row=np.array(instance_processed_data), predict_fn=st.session_state.model_pipeline.predict_proba,
                                                num_features=10, top_labels=1
                                            )
                                            lime_explanation = exp_lime.as_list(label=1) # Explain derailment class (1)
                                            lime_df = pd.DataFrame(lime_explanation, columns=['Feature Rule', 'Contribution (to High Risk)'])
                                            st.dataframe(lime_df, use_container_width=True, hide_index=True)
                                            st.caption("Positive contributions increase the High Risk probability, negative decrease it.")

                                            # Waterfall Chart
                                            feature_contribs = [tup[1] for tup in lime_explanation]
                                            feature_names = [tup[0] for tup in lime_explanation]
                                            base_value = exp_lime.intercept[1]
                                            predicted_value = instance_pred_prob
                                            fig_waterfall = plot_waterfall_plotly(feature_contribs, base_value, predicted_value, feature_names)
                                            st.plotly_chart(fig_waterfall, use_container_width=True)

                                        except IndexError as e_lime_index: st.error(f"LIME Error: Could not get explanation for class 1. Intercept: {exp_lime.intercept}. Error: {e_lime_index}")
                                        except Exception as e_lime: st.error(f"LIME Error: {e_lime}"); st.exception(e_lime)
                                with explain_col2:
                                    st.markdown("**Project Feature Values**")
                                    project_features = st.session_state.numerical_features + st.session_state.categorical_features
                                    project_details = st.session_state.project_data.loc[selected_project_index, project_features + ['ProjectID', 'ProjectName']]
                                    detail_df = pd.DataFrame(project_details).T
                                    st.dataframe(detail_df, use_container_width=True, hide_index=True)
                            else: st.error("Prediction probability not found for the selected project index.")
                    else: st.warning("No projects from the test set found in the current project data.")
                except KeyError as e_key: st.error(f"Error accessing project data (KeyError): {e_key}.")
                except Exception as e: st.error(f"Error in project deep dive: {e}"); st.exception(e)
            elif not lime_ready: st.info("LIME explainer not available. Train the model first.")
            else: st.info("Test set data is empty or unavailable.")


# --- Simulation & Scenarios Tab ---
with tabs[4]: # Index adjusted
    st.header("🎲 Simulation & Scenarios")
    st.markdown("_Explore prediction uncertainty and the impact of potential changes._")

    if not st.session_state.predictions_made:
        st.info("ℹ️ Train a model using the sidebar configuration to enable simulations.")
    elif st.session_state.project_data is None or st.session_state.model_pipeline is None:
        st.warning("⚠️ Simulations require loaded data and a trained model.")
    else:
        df_simulation_tab = st.session_state.project_data.copy()

        # --- Monte Carlo Simulation ---
        st.subheader("🔄 Monte Carlo Simulation: Understanding Uncertainty")
        with st.container():
            st.markdown("""
            **What is it?** Monte Carlo simulation helps quantify the uncertainty in the model's predictions. It works by:
            1. Taking the projects in the **test set**.
            2. Running many simulations (`{N_MONTE_CARLO_RUNS}` times).
            3. In each simulation, slightly changing the input feature values based on their typical variation (adding random 'noise' to numerical features, randomly sampling categories based on their frequency).
            4. Making a prediction for each simulated version of the project.

            **Why is it useful?** A single prediction gives a point estimate, but doesn't tell you how sensitive that prediction is to small changes in the input data.
            * If a project's predicted probability varies a lot across simulations (high standard deviation), the prediction is less certain.
            * If the probability stays consistent, the prediction is more robust.

            **How to interpret:**
            * **Mean Prob:** The average predicted probability across all simulations.
            * **Std Dev:** Measures the spread or variability of the predictions. Higher means less certainty.
            * **P10 / P90:** The 10th and 90th percentile probabilities. 80% of the simulated predictions fall between these two values. A narrow range indicates higher confidence.
            """)

            mc_button_sim = st.button("Run Monte Carlo Simulation", key="mc_simulate_tab")

            if mc_button_sim:
                 if st.session_state.X_test_original is None or st.session_state.X_test_original.empty:
                     st.error("❌ Test set data not found or is empty. Please retrain the model.")
                 else:
                    with st.spinner(f"Running {N_MONTE_CARLO_RUNS} Monte Carlo simulations..."):
                        # (MC Logic remains the same as previous version)
                        try:
                            X_test = st.session_state.X_test_original.copy()
                            feature_distributions = {}
                            for feat in st.session_state.numerical_features:
                                if feat in X_test.columns:
                                    feat_mean = X_test[feat].mean(); feat_std = X_test[feat].std()
                                    if pd.isna(feat_std) or feat_std == 0: feat_std = feat_mean * 0.1 if pd.notna(feat_mean) and feat_mean != 0 else 0.1
                                    feature_distributions[feat] = {'type': 'normal', 'mean': feat_mean, 'std': feat_std}
                            for feat in st.session_state.categorical_features:
                                if feat in X_test.columns:
                                    value_counts = X_test[feat].value_counts(normalize=True)
                                    if not value_counts.empty: feature_distributions[feat] = {'type': 'categorical', 'values': value_counts.index, 'probs': value_counts.values}

                            mc_predictions = []
                            test_project_ids_mc = X_test.index
                            test_project_map = st.session_state.project_data.loc[test_project_ids_mc, 'ProjectID'].to_dict()

                            for i in range(N_MONTE_CARLO_RUNS):
                                X_sim = X_test.copy()
                                for feat, dist in feature_distributions.items():
                                    if feat not in X_sim.columns: continue
                                    if dist['type'] == 'normal':
                                        noise = np.random.normal(0, dist['std'] * 0.1, size=len(X_sim))
                                        X_sim[feat] = X_sim[feat] + noise
                                        if feat in ['InitialCostEstimate', 'InitialScheduleDays', 'ScopeChanges']: X_sim[feat] = X_sim[feat].clip(lower=0)
                                    elif dist['type'] == 'categorical':
                                        if 'values' in dist and 'probs' in dist and len(dist['values']) > 0 and not np.isnan(dist['probs']).any():
                                             X_sim[feat] = np.random.choice(dist['values'], size=len(X_sim), p=dist['probs'])
                                        else: X_sim[feat] = X_test[feat]
                                sim_probs = st.session_state.model_pipeline.predict_proba(X_sim)[:, 1]
                                mc_predictions.append(sim_probs)

                            mc_predictions = np.array(mc_predictions)
                            mc_stats = pd.DataFrame({'Index': test_project_ids_mc, 'Mean_Prob': mc_predictions.mean(axis=0), 'Std_Prob': mc_predictions.std(axis=0), 'Min_Prob': mc_predictions.min(axis=0), 'Max_Prob': mc_predictions.max(axis=0), 'P10_Prob': np.percentile(mc_predictions, 10, axis=0), 'P90_Prob': np.percentile(mc_predictions, 90, axis=0)})
                            mc_stats['ProjectID'] = mc_stats['Index'].map(test_project_map)
                            mc_stats = mc_stats.dropna(subset=['ProjectID']).drop(columns=['Index'])
                            st.session_state.monte_carlo_results = mc_stats
                            st.success(f"✅ Monte Carlo simulation complete.")
                        except Exception as e: st.error(f"❌ Monte Carlo Simulation Error: {e}"); st.exception(e)

            # Display Monte Carlo Results
            if st.session_state.monte_carlo_results is not None:
                st.markdown("**Simulation Results (Test Set Projects)**")
                mc_display = st.session_state.monte_carlo_results.copy()
                project_names_df = df_simulation_tab[['ProjectID', 'ProjectName']].drop_duplicates(subset=['ProjectID'])
                mc_display = mc_display.merge(project_names_df, on='ProjectID', how='left')
                mc_display['Mean_Prob (%)'] = (mc_display['Mean_Prob'] * 100).round(1)
                mc_display['Std_Prob (%)'] = (mc_display['Std_Prob'] * 100).round(1)
                mc_display['P10_Prob (%)'] = (mc_display['P10_Prob'] * 100).round(1)
                mc_display['P90_Prob (%)'] = (mc_display['P90_Prob'] * 100).round(1)
                display_cols = ['ProjectID', 'ProjectName', 'Mean_Prob (%)', 'Std_Prob (%)', 'P10_Prob (%)', 'P90_Prob (%)']
                if 'ProjectName' not in mc_display.columns: display_cols.remove('ProjectName')
                st.dataframe(mc_display[display_cols], use_container_width=True,
                    column_config={ 'Mean_Prob (%)': st.column_config.ProgressColumn("Mean Risk Prob.", format="%.1f%%", min_value=0, max_value=100), 'Std_Prob (%)': st.column_config.NumberColumn("Std Dev (%)", format="%.1f"), 'P10_Prob (%)': st.column_config.NumberColumn("P10 (%)", format="%.1f"), 'P90_Prob (%)': st.column_config.NumberColumn("P90 (%)", format="%.1f") }
                )
            else:
                st.info("Click 'Run Monte Carlo Simulation' to see uncertainty results for test set projects.")


        st.markdown("---")
        # --- Batch Scenario Planning ---
        st.subheader("🔮 Batch Scenario Planning: 'What-If' Analysis")
        with st.container():
            st.markdown("""
            **What is it?** This feature allows you to explore the impact of specific changes on project risk predictions. You can define scenarios by:
            1. Selecting a project.
            2. Modifying one or more of its input features (e.g., increase 'ScopeChanges', change 'ResourceAvailability' to 'Low').
            3. Running the prediction with the modified data.

            You can define multiple scenarios for one or more projects, either interactively below or by uploading a CSV file.

            **Why is it useful?** It helps answer "What if...?" questions, such as:
            * What is the likely impact on risk if we experience vendor delays?
            * How much does risk decrease if we improve client engagement?
            * Can we simulate the effect of potential budget cuts or resource shortages?

            **How to use:**
            * Use the expander below to define scenarios one by one and add them to the batch.
            * Alternatively, upload a CSV file (use the template from 'Data Management') with columns for 'ProjectID', 'ScenarioName', and any features you want to change.
            * Click 'Run Batch Scenarios' to see the predicted risk probability for each defined scenario.
            * Compare the scenario predictions to the project's original prediction (shown in the 'Portfolio Deep Dive' tab).
            """)

            # Scenario Upload/Definition (Logic remains similar)
            scenario_file_sim = st.file_uploader("Upload Scenario CSV (Optional)", type=['csv'], key="scenario_upload_tab")
            # ... (rest of scenario upload/definition/run/display logic remains the same as previous version) ...
            if scenario_file_sim:
                try:
                    scenario_df_up = pd.read_csv(scenario_file_sim)
                    if all(col in scenario_df_up.columns for col in ['ProjectID', 'ScenarioName']):
                        st.session_state.batch_scenario_data = scenario_df_up
                        st.success("✅ Scenario CSV uploaded successfully.")
                        st.session_state.batch_scenario_results = None # Clear old results
                    else: st.error("Scenario CSV must include 'ProjectID' and 'ScenarioName' columns.")
                except Exception as e: st.error(f"Error reading scenario CSV: {e}")

            with st.expander("Define Scenario Interactively", expanded=False):
                if df_simulation_tab is not None and not df_simulation_tab.empty:
                    project_list = df_simulation_tab['ProjectID'].unique().tolist()
                    project_name_map = df_simulation_tab.drop_duplicates(subset=['ProjectID']).set_index('ProjectID')['ProjectName'].to_dict()
                    scenario_project_id = st.selectbox("Select Project for Scenario:", options=project_list,
                        format_func=lambda x: f"{x} - {project_name_map.get(x, '')}",
                        key="scenario_project_select_tab")
                else:
                    st.warning("Load project data to define scenarios.")
                    scenario_project_id = None

                scenario_name = st.text_input("Scenario Name:", value="Custom Scenario", key="scenario_name_tab")
                scenario_changes = {}

                if scenario_project_id:
                    current_project_data = df_simulation_tab[df_simulation_tab['ProjectID'] == scenario_project_id].iloc[0]
                    st.markdown("**Numerical Features**")
                    num_col1, num_col2 = st.columns(2)
                    for idx, feat in enumerate(SCENARIO_FEATURES_NUMERIC):
                         with num_col1 if idx % 2 == 0 else num_col2:
                             if feat in current_project_data:
                                 current_value = current_project_data[feat]
                                 change_type = st.selectbox(f"{feat} Change:", ['No Change', 'Set Value', '% Change'], key=f"{feat}_change_type_tab_{scenario_project_id}")
                                 if change_type == 'Set Value':
                                     new_value = st.number_input(f"New {feat}:", value=float(current_value), step=1000.0 if 'Cost' in feat else 1.0, key=f"{feat}_set_value_tab_{scenario_project_id}", format="%f")
                                     scenario_changes[feat] = new_value
                                 elif change_type == '% Change':
                                     perc_change = st.slider(f"{feat} % Change:", -50.0, 100.0, 0.0, 5.0, key=f"{feat}_perc_change_tab_{scenario_project_id}")
                                     scenario_changes[feat] = current_value * (1 + perc_change / 100)
                             else: st.text(f"{feat}: N/A")
                    st.markdown("**Categorical Features**")
                    cat_col1, cat_col2 = st.columns(2)
                    for idx, feat in enumerate(SCENARIO_FEATURES_CATEGORICAL):
                         with cat_col1 if idx % 2 == 0 else cat_col2:
                             if feat in current_project_data:
                                 current_value = current_project_data[feat]
                                 options = sorted(df_simulation_tab[feat].dropna().unique())
                                 try: current_index = options.index(current_value) if current_value in options else 0
                                 except ValueError: current_index = 0
                                 new_value = st.selectbox(f"{feat}:", options=options, index=current_index, key=f"{feat}_cat_value_tab_{scenario_project_id}")
                                 if new_value != current_value: scenario_changes[feat] = new_value
                             else: st.text(f"{feat}: N/A")

                    if st.button("Add Scenario to Batch", key="add_scenario_tab"):
                        scenario_row = {'ProjectID': scenario_project_id, 'ScenarioName': scenario_name}
                        scenario_row.update(scenario_changes)
                        scenario_df_new = pd.DataFrame([scenario_row])
                        if st.session_state.batch_scenario_data is not None:
                            st.session_state.batch_scenario_data = pd.concat([st.session_state.batch_scenario_data, scenario_df_new], ignore_index=True)
                        else: st.session_state.batch_scenario_data = scenario_df_new
                        st.success(f"✅ Scenario '{scenario_name}' added for {scenario_project_id}.")
                        st.session_state.batch_scenario_results = None
                else: st.info("Select a project to define a scenario.")

            # Display Current Scenarios & Run Button
            if st.session_state.batch_scenario_data is not None and not st.session_state.batch_scenario_data.empty:
                st.markdown("**Current Scenarios Defined**")
                st.dataframe(st.session_state.batch_scenario_data, use_container_width=True)
                if st.button("Run Batch Scenarios", key="run_scenarios_tab"):
                    with st.spinner("Running batch scenario predictions..."):
                        # (Scenario Running Logic remains same)
                        try:
                            scenario_results_list = []
                            base_features = st.session_state.numerical_features + st.session_state.categorical_features
                            if not base_features: st.error("Model features not defined. Retrain model.")
                            else:
                                for _, scenario in st.session_state.batch_scenario_data.iterrows():
                                    project_id = scenario['ProjectID']
                                    scenario_name = scenario['ScenarioName']
                                    if project_id in df_simulation_tab['ProjectID'].values:
                                        project_data_base = df_simulation_tab[df_simulation_tab['ProjectID'] == project_id][base_features].head(1).copy()
                                        if project_data_base.empty: st.warning(f"Skipping {scenario_name} for {project_id}: Base data not found."); continue
                                        project_data_scenario = project_data_base.copy()
                                        for feat in scenario.index:
                                            if feat in project_data_scenario.columns and pd.notna(scenario[feat]): project_data_scenario[feat] = scenario[feat]
                                        pred_prob = st.session_state.model_pipeline.predict_proba(project_data_scenario[base_features])[:, 1][0]
                                        scenario_results_list.append({'ProjectID': project_id, 'ProjectName': df_simulation_tab.loc[df_simulation_tab['ProjectID'] == project_id, 'ProjectName'].iloc[0], 'ScenarioName': scenario_name, 'PredictedRiskProb': pred_prob})
                                    else: st.warning(f"Skipping {scenario_name} for {project_id}: ProjectID not found.")
                                if scenario_results_list: st.session_state.batch_scenario_results = pd.DataFrame(scenario_results_list); st.success("✅ Batch scenario predictions complete.")
                                else: st.warning("No valid scenarios processed."); st.session_state.batch_scenario_results = None
                        except Exception as e: st.error(f"❌ Batch Scenario Error: {e}"); st.exception(e); st.session_state.batch_scenario_results = None

            # Display Scenario Results
            if st.session_state.batch_scenario_results is not None and not st.session_state.batch_scenario_results.empty:
                st.markdown("**Scenario Prediction Results**")
                results_display = st.session_state.batch_scenario_results.copy()
                results_display['PredictedRiskProb (%)'] = (results_display['PredictedRiskProb'] * 100).round(1)
                st.dataframe(results_display[['ProjectID', 'ProjectName', 'ScenarioName', 'PredictedRiskProb (%)']], use_container_width=True,
                    column_config={'PredictedRiskProb (%)': st.column_config.ProgressColumn("Risk Prob.", format="%.1f%%", min_value=0, max_value=100)}
                )
                try:
                    results_display_pivot = results_display.drop_duplicates(subset=['ProjectID', 'ScenarioName'], keep='last')
                    pivot_results = results_display_pivot.pivot_table(index=['ProjectID', 'ProjectName'], columns='ScenarioName', values='PredictedRiskProb').reset_index()
                    if not pivot_results.empty and len(pivot_results.columns) > 2:
                        fig_scenarios = px.bar(pivot_results.melt(id_vars=['ProjectID', 'ProjectName'], value_name='RiskProb', var_name='ScenarioName'), x='ProjectName', y='RiskProb', color='ScenarioName', barmode='group', title="Scenario Comparison", labels={'RiskProb': 'Predicted Risk Probability', 'ProjectName': 'Project'}, color_discrete_sequence=[ARCADIS_ORANGE] + ARCADIS_SECONDARY_PALETTE)
                        fig_scenarios.update_layout(title_x=0.5, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_scenarios, use_container_width=True)
                    elif not pivot_results.empty: st.info("Only one scenario type found, comparison chart requires multiple.")
                except Exception as e_pivot: st.warning(f"Could not generate scenario comparison chart: {e_pivot}")


# --- Data Management Tab ---
with tabs[5]: # Index adjusted
    st.header("💾 Data Management")
    st.markdown("_Upload and preview project data, risk register, or scenarios. Download templates._")

    # --- Data Upload ---
    st.subheader("📤 Upload Data")
    with st.container():
        data_type = st.selectbox("Data Type to Upload:", ["Project Data", "Risk Register", "Scenario Data"], key="data_type_upload_tab")
        uploaded_file = st.file_uploader(f"Upload {data_type} (CSV)", type=['csv'], key="data_uploader_tab")

        if uploaded_file:
            try:
                df_uploaded = pd.read_csv(uploaded_file)
                if data_type == "Project Data":
                    if 'ProjectID' in df_uploaded.columns and st.session_state.target_variable in df_uploaded.columns:
                        st.session_state.project_data = df_uploaded
                        st.success("✅ Project data uploaded successfully. Retrain model via sidebar.")
                        # Reset dependent states
                        st.session_state.model_pipeline = None; st.session_state.trained_models = {}
                        st.session_state.predictions_made = False; st.session_state.lime_explainer = None
                        st.session_state.monte_carlo_results = None; st.session_state.batch_scenario_results = None
                        st.session_state.auto_insights = None; st.session_state.risk_register = None
                        try: # Try generating mock register
                            st.session_state.risk_register = generate_mock_risk_register(st.session_state.project_data['ProjectID'].tolist())
                            st.info("Generated mock risk register.")
                        except Exception: pass
                        st.rerun()
                    else: st.error(f"Project data must include 'ProjectID' and '{st.session_state.target_variable}'.")
                elif data_type == "Risk Register":
                    if 'ProjectID' in df_uploaded.columns:
                        st.session_state.risk_register = df_uploaded
                        st.success("✅ Risk register uploaded successfully.")
                        st.rerun()
                    else: st.error("Risk register must include 'ProjectID'.")
                elif data_type == "Scenario Data":
                    if all(col in df_uploaded.columns for col in ['ProjectID', 'ScenarioName']):
                        st.session_state.batch_scenario_data = df_uploaded
                        st.success("✅ Scenario data uploaded successfully.")
                        st.session_state.batch_scenario_results = None
                        st.rerun()
                    else: st.error("Scenario data must include 'ProjectID' and 'ScenarioName'.")
            except Exception as e: st.error(f"Error uploading {data_type}: {e}")

    st.markdown("---")
    # --- Data Preview ---
    st.subheader("📊 Data Preview")
    with st.container():
        preview_type = st.selectbox("Preview Data:", ["Project Data", "Risk Register", "Scenario Data"], key="data_preview_select_tab")
        data_to_preview = None
        if preview_type == "Project Data": data_to_preview = st.session_state.project_data
        elif preview_type == "Risk Register": data_to_preview = st.session_state.risk_register
        elif preview_type == "Scenario Data": data_to_preview = st.session_state.batch_scenario_data

        if data_to_preview is not None:
            st.dataframe(data_to_preview.head(10), use_container_width=True)
            st.markdown(f"**Total Rows:** {len(data_to_preview)}")
        else: st.info(f"No {preview_type.lower()} available.")

    st.markdown("---")
    # --- Download Templates ---
    st.subheader("📥 Download Data Templates")
    with st.container():
        st.markdown("Download sample CSV templates for uploading data.")
        num_feat_template = st.session_state.selected_numerical_features or SCENARIO_FEATURES_NUMERIC
        cat_feat_template = st.session_state.selected_categorical_features or SCENARIO_FEATURES_CATEGORICAL

        template_cols = {
            'Project Data': ['ProjectID', 'ProjectName', 'Region', 'ProjectType'] + num_feat_template + cat_feat_template + ['PermittingDelays', 'RiskEventOccurred', st.session_state.target_variable, 'ActualCost', 'ActualScheduleDays', 'CostVariancePerc', 'ScheduleVariancePerc', 'CompletionDate'],
            'Risk Register': ['ProjectID', 'RiskID', 'RiskDescription', 'Likelihood', 'Impact', 'ImpactArea', 'MitigationStatus', 'Owner', 'DueDate', 'MitigationAction'],
            'Scenario Data': ['ProjectID', 'ScenarioName'] + SCENARIO_FEATURES_NUMERIC + SCENARIO_FEATURES_CATEGORICAL
        }
        for data_type, cols in template_cols.items():
            unique_cols = pd.Index(cols).unique().tolist()
            template_df = pd.DataFrame(columns=unique_cols)
            csv_bytes = df_to_csv(template_df)
            st.download_button(
                label=f"Download {data_type} Template", data=csv_bytes,
                file_name=f"{data_type.lower().replace(' ', '_')}_template.csv", mime='text/csv',
                key=f"download_template_{data_type.lower().replace(' ', '_')}_tab"
            )

# --- End of App ---
st.markdown("---")
st.caption("_RiskLens Pro - Narrative Driven Analysis for Arcadis_")
