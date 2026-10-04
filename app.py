import streamlit as st
import pandas as pd
import json
import numpy as np

from classes.train_model import TrainModel
from classes.data_source import Model
from classes.description import (
    IndividualDescription,
)
import copy
import types

from classes.visual import DistributionModelPlot
from PIL import Image, ImageOps, ImageDraw
# Set Streamlit page layout
st.set_page_config(page_title="Logistic Regression App", layout="wide")

# Demo files used when the user does not upload their own
DEMO_DATA_FILE = "data/medical/training_data_70000_Ind.csv"
DEMO_EXPLANATION_FILE = "data/medical/feature_explanation.csv"
DEMO_CATEGORICAL_FILE = "data/medical/categorical_features.json"
# Features left out of the demo model (self-reported smoking and alcohol came out as protective)
DEMO_EXCLUDED_FEATURES = ["smoke", "alco"]


def build_description(model, individual_id, scale, threshold_type, risk_age, baseline):
    """
    Compute contributions and thresholds for one configuration and describe one individual.
    Shared by the Wordalisation tab and the configuration export so they always agree.
    """
    target = st.session_state["target"]
    model_features = st.session_state["model_features"]
    odds_space = (scale == 'odds')

    model.weight_contributions(scale=scale, baseline=baseline)
    age_column = 'age_contribution' if threshold_type == "age" else None
    thresholds, _, _ = model.calulcate_threshold(odds_space=odds_space, column=age_column)
    bins = model.risk_thresholds(odds_space=odds_space)

    # Create the individual from the updated model so contributions reflect this configuration
    individual_data = copy.copy(model)
    individual_data.df = model.df[model.df["ID"] == individual_id].copy()
    individual = individual_data.to_data_point(columns=["ID", target])

    description = IndividualDescription(
        individual,
        metrics=model.parameters['Parameter'],
        parameter_explanation=model.parameter_explanation,
        categorical_interpretations=st.session_state["categorical_interpretations"],
        thresholds=thresholds,
        target=target,
        bins=bins,
        model_features=model_features,
        individuals=model.df,
        threshold_type=threshold_type,
        odds_space=odds_space,
        risk_age=risk_age,
    )
    return description, individual, thresholds, bins


def get_summary(description):
    # Only call the LLM when the synthesized text changes, not on every Streamlit rerun
    summaries = st.session_state.setdefault("summaries", {})
    if description.synthesized_text not in summaries:
        summaries[description.synthesized_text] = description.stream_gpt()
    return summaries[description.synthesized_text]


def export_configurations(model, individual_id, output_path='patient_export.xlsx'):
    from itertools import product
    from openpyxl.styles import Alignment

    axes = {
        'scale':     ['linear', 'odds'],
        'reference': ['independent', 'largest', 'population'],
        'risk_age':  [False, True],
        'baseline':  ['specific', 'population']
    }
    reference_to_threshold_type = {'independent': 'feature_specific', 'largest': 'highest', 'population': 'average'}

    # Work on a copy so the configurations don't change the model shown on the page
    model = copy.copy(model)

    rows = []
    for config_idx, config_values in enumerate(product(*axes.values())):
        config = dict(zip(axes.keys(), config_values))

        description, _, _, _ = build_description(
            model, individual_id,
            scale=config['scale'],
            threshold_type=reference_to_threshold_type[config['reference']],
            risk_age=config['risk_age'],
            baseline=config['baseline'],
        )

        rows.append({
            'config_id':        config_idx + 1,
            'scale':            config['scale'],
            'reference':        config['reference'],
            'risk_age':         config['risk_age'],
            'baseline':         config['baseline'],
            'synthesized_text': description.synthesized_text,
            'llm_summary':      get_summary(description),
        })
        print(f"Config {config_idx+1}/24 done")

    df_export = pd.DataFrame(rows)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_export.to_excel(writer, sheet_name='Configurations', index=False)
        ws = writer.sheets['Configurations']
        for col_idx, col_name in enumerate(df_export.columns, 1):
            col_letter = ws.cell(1, col_idx).column_letter
            ws.column_dimensions[col_letter].width = 80 if 'text' in col_name or 'summary' in col_name else 14
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical='top')

    print(f"Exported to {output_path}")
    return df_export
# Logo in the sidebar
def add_white_round_bg(image_path, size=120):
    img = Image.open(image_path).convert("RGBA")
    # Resize image to fit inside the circle
    img.thumbnail((size, size), Image.LANCZOS)
    # Create white circle background
    bg = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, size, size), fill=255)
    bg.paste((255, 255, 255, 255), (0, 0), mask)
    # Center the image on the circle
    img_pos = ((size - img.width) // 2, (size - img.height) // 2)
    bg.paste(img, img_pos, img)
    return bg

img_with_bg = add_white_round_bg("data/ressources/img/heart-sense.png", size=120)
# Center the image in the sidebar using markdown and HTML
import base64
from io import BytesIO

# Encode the image as PNG and then base64
buffer = BytesIO()
ImageOps.exif_transpose(img_with_bg).convert('RGBA').resize((120,120)).save(buffer, format="PNG")
img_base64 = base64.b64encode(buffer.getvalue()).decode()

st.sidebar.markdown(
    f"<div style='display: flex; justify-content: center; align-items: center;'>"
    f"<img src='data:image/png;base64,{img_base64}' "
    f"style='width:120px; height:120px; border-radius:60px; background:white;'/>"
    f"</div>",
    unsafe_allow_html=True
)
st.sidebar.divider()


@st.cache_resource(show_spinner="Training model...")
def train_model(data, target, features, feature_selection):
    # Cached so the model is only refit when the data, target, features or selection change,
    # not on every Streamlit rerun. The returned object is shared: don't mutate it.
    model = TrainModel(data, target, features)
    if feature_selection:
        model.selectFeatures()
    return model


def coefficient_table(model, feature_data):
    coef_df = model.coef_df.copy()
    coef_explanations = {row['Parameter']: row['Explanation'] for _, row in feature_data.iterrows()}
    coef_df['Explanation'] = coef_df['Parameter'].map(coef_explanations)
    coef_df['P-Value'] = coef_df['Parameter'].map(model.p_values)
    return coef_df


def setup_model(train=False):
    st.write("Upload a CSV file to use as the data source.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="data_file")
    if uploaded_file is None:
        st.caption("No file uploaded: using the demo cardiovascular dataset.")
        uploaded_file = DEMO_DATA_FILE

    data = pd.read_csv(uploaded_file)
    all_columns = data.columns.tolist()

    st.write("Upload a CSV file explaining the features (columns 'Parameter' and 'Explanation'), or edit the explanations below.")
    feature_file = st.file_uploader("Choose a CSV file", type="csv", key="feature_file")
    if feature_file is None:
        feature_file = DEMO_EXPLANATION_FILE

    default_features = [c for c in all_columns if not (uploaded_file == DEMO_DATA_FILE and c in DEMO_EXCLUDED_FEATURES)]
    selected_features = st.multiselect("Remove features you want to discard", all_columns, default=default_features)
    # Create editable DataFrame
    explanation_df = pd.DataFrame({
        "Parameter": selected_features,
        "Explanation": [_ for _ in selected_features]
    })

    # Let users edit explanations directly
    st.write("Provide short explanations for the selected features:")

    # Prefill explanations from the feature file
    feature_data = pd.read_csv(feature_file)
    explanation_map = dict(zip(feature_data['Parameter'].str.strip(), feature_data['Explanation'].str.strip()))
    explanation_df['Explanation'] = explanation_df['Parameter'].map(explanation_map).fillna("No explanation provided")

    edited_df = st.data_editor(
        explanation_df,
        num_rows="fixed",
        column_config={
        "Parameter": st.column_config.TextColumn(disabled=True),
        "Explanation": st.column_config.TextColumn("Explanation")
        },
        use_container_width=False,
        width=700
    )

    categorical_interpretations = None
    has_categorical = st.radio("Does your data have categorical features?", ("Yes", "No"))
    if has_categorical == "Yes":

        st.write("Upload a JSON file detailing the interpretations of the categorical data, or fill them in below.")
        json_file = st.file_uploader("Choose a JSON file", type="json", key="category_json_file")
        if json_file is None:
            json_file = open(DEMO_CATEGORICAL_FILE, "rb")
        preloaded_categorical_interpretations = json.load(json_file)

        categorical_feautres = st.multiselect(
            "Select categorical features", selected_features,
            default=[f for f in preloaded_categorical_interpretations if f in selected_features],
        )
        categorical_interpretations = {}
        st.write(f"Define meanings for values in your categorical features")
        for feature in categorical_feautres:
            st.markdown(f"**{feature}**")
            unique_values = sorted(data[feature].dropna().unique().tolist())
            value_expl_df = pd.DataFrame({
                "Value": unique_values,
                "Meaning": ["" for _ in unique_values]
            })
            # Use preloaded values if available
            prefilled = preloaded_categorical_interpretations.get(feature, {})
            value_expl_df['Meaning'] = [prefilled.get(str(val), "") for val in unique_values]

            edited_value_df = st.data_editor(
                value_expl_df,
                num_rows="fixed",
                use_container_width=False,
                hide_index=True,
                column_config={
                    "Value": st.column_config.TextColumn(disabled=True, width="auto"),
                    "Meaning": st.column_config.TextColumn("Meaning", width="auto")
                }, key=f"{feature}_editor"
            )
            # Convert to dictionary format expected
            value_map = dict(zip(edited_value_df["Value"].astype(str), edited_value_df["Meaning"]))
            categorical_interpretations[feature] = value_map

    feature_data = edited_df

    # is there is a pre-trained model, upload the weights
    if not train:
        st.write("Upload a CSV file of the weights (coefficients) found in your model")
        weights_file = st.file_uploader("Choose a CSV file", type="csv", key="weight_file")
        if weights_file is None:
            return None
        weights_data = pd.read_csv(weights_file)
        parameter_explanations = {row['Parameter']: row['Explanation'] for _, row in feature_data.iterrows()}
        weights_data['Explanation'] = weights_data['Parameter'].map(parameter_explanations)
        st.write("Feature Explanation:", weights_data)

        # The target is the column that is neither a model parameter nor an ID
        candidates = [c for c in selected_features if c not in set(weights_data['Parameter']) and c.upper() != 'ID']
        if not candidates:
            st.error("Could not find a target column: every selected column is a model parameter.")
            return None
        target = candidates[0] if len(candidates) == 1 else st.radio("Target column", candidates, key="existing_target_column")
        return (data, weights_data, target, categorical_interpretations)

    # if training a new model, ask for target column and features
    st.write("Choose target column")
    default_target = selected_features.index("cardio") if "cardio" in selected_features else len(selected_features) - 1
    target = st.radio("Target column", selected_features, index=default_target, key="target_column")
    features = [col for col in selected_features if col != target]

    feature_selection = st.radio("Do you want to perform feature selection using stepwise backward elimination?", ("No", "Yes"), key="feature_selection") == "Yes"
    try:
        model = train_model(data, target, features, feature_selection)
    except Exception as e:
        st.error(f"An error occurred while training the model: {e} Pick a target column that is binary.")
        return None

    if model.rows_removed:
        st.info(f"Removed {model.rows_removed} rows with statistical outliers (IQR method) before training. Per-column flags: {model.dropped_counts}")
    st.expander("Summary of the trained model", expanded=False).write(model.summary)
    st.write(f"AUC on the held-out 20% test set: {model.test_auc:.3f}")
    if feature_selection:
        if model.elimination_log:
            for feature, p_value in model.elimination_log:
                st.write(f"Removed feature {feature} with p-value {p_value:.3g}")
        else:
            st.write("All p-values are below the significance level: no features removed")
    else:
        st.write("No feature selection performed")

    coef_df = coefficient_table(model, feature_data)
    st.write("Training Output:", coef_df)

    # Keep only the target and the model parameters in data
    columns_to_keep = [target] + [param for param in coef_df['Parameter'].tolist() if param in data.columns]
    data = data[columns_to_keep]
    if categorical_interpretations is not None:
        categorical_interpretations = {k: v for k, v in categorical_interpretations.items() if k in columns_to_keep}
    return (data, coef_df, target, categorical_interpretations)


def setup_data(data, model_features, categorical_interpretations=None, target=None):
    model=Model()
    model.set_data(data.copy(), model_features, categorical_interpretations=categorical_interpretations)
    model.process_data()
    if model.rows_removed:
        st.info(f"Removed {model.rows_removed} rows with statistical outliers (IQR method) before analysis. Per-column flags: {model.dropped_counts}")

    st.session_state["model"] = model
    st.session_state["metrics"] = model.parameters["Parameter"]
    st.session_state["target"] = target
    st.session_state["categorical_interpretations"] = categorical_interpretations
    st.session_state["model_features"] = model_features

    st.expander("Dataframe used", expanded=True).write(model.df)
    st.success("Model data is processed! Go to the 'Wordalisation' tab to select an individual.")

def setup_chat():
    model = st.session_state["model"]
    target = st.session_state["target"]
    model_features = st.session_state["model_features"]

    individual_id = st.selectbox("Select Individual", model.df['ID'].unique(), index=0)

    metrics = model.parameters['Parameter']

    # Define axes explicitly
    AXES = {
        "scale": {"log_odds": "Log-odds", "odds_ratio": "Odds Ratio / % Change"},
        "reference": {"independent": "Independent", "largest": "Relative to Largest", "average": "Relative to Average", "age": "Relative to Age"},
        "risk_age": {"yes": "Include Risk-Age", "no": "Exclude Risk-Age"},
        "baseline": {"specific": "Specific Group", "population": "Population Average"}
    }

    has_age = 'age' in set(metrics)
    references = list(AXES["reference"].values())
    if not has_age:
        references.remove("Relative to Age")

    st.sidebar.header("Configuration")
    scale     = st.sidebar.radio("Scale", list(AXES["scale"].values()))
    reference = st.sidebar.radio("Risk Reference", references)
    risk_age  = st.sidebar.toggle("Include Risk-Age Mapping", disabled=not has_age,
                                  help=None if has_age else "Needs an 'age' feature in the model")
    baseline  = st.sidebar.radio("Categorical Baseline", list(AXES["baseline"].values()))

    odds_space = (scale == "Odds Ratio / % Change")

    threshold_type_map = {
        "Independent": "feature_specific",
        "Relative to Largest": "highest",
        "Relative to Average": "average",
        "Relative to Age": "age",
    }
    threshold_type = threshold_type_map[reference]

    description, individual, thresholds, bins = build_description(
        model, individual_id,
        scale='odds' if odds_space else 'linear',
        threshold_type=threshold_type,
        risk_age=risk_age,
        baseline='specific' if baseline == "Specific Group" else 'population',
    )
    st.session_state["individual"] = individual

    if st.button("Generate 24 configurations for this patient"):
        with st.spinner("Generating 24 configurations (this may take a minute)..."):
            export_configurations(model, individual_id=individual_id)
        with open('patient_export.xlsx', 'rb') as f:
            st.download_button(
                label="Download Excel",
                data=f,
                file_name=f'patient_{individual.id}_configurations.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

    st.markdown("###### **Synthesized Text about individual:**")
    st.write(description.synthesized_text)

    if threshold_type == "average":
        # Exclude total_risk_contribution — it's a sum of per-feature values and has a
        # different scale, which skews the average threshold positions.
        feature_bins = [v for k, v in bins.items() if k != 'total_risk_contribution']
        if odds_space:
            # Odds ratios are multiplicative: use geometric mean (average in log space)
            plot_thresholds = [round(np.exp(np.mean(np.log(col))), 4) for col in zip(*feature_bins)]
        else:
            plot_thresholds = [sum(col) / len(col) for col in zip(*feature_bins)]
    else:
        plot_thresholds = thresholds

    plot_df = model.df.sample(90, random_state=42)
    contribution_cols = [c for c in plot_df.columns if '_contribution' in c and c != 'total_risk_contribution']
    plot_min = plot_df[contribution_cols].min().min()
    plot_max = plot_df[contribution_cols].max().max()
    if threshold_type != "feature_specific" and plot_thresholds:
        plot_min_max = [min(plot_min, plot_thresholds[0]), max(plot_max, plot_thresholds[-1])]
    else:
        plot_min_max = [plot_min, plot_max]

    visual = DistributionModelPlot(plot_thresholds, plot_min_max, metrics, model_features=model_features, key="main", threshold_type=threshold_type)
    visual.add_title('Evaluation of individual', '')
    visual.add_individuals(types.SimpleNamespace(df=plot_df), metrics=metrics, target=target)
    visual.add_individual(individual, len(model.df), metrics=metrics, center=1 if odds_space else 0)
    visual.show()
    summary = get_summary(description)
    st.markdown("###### **Summary of the individual:**")
    st.write(summary)
    st.caption("This summary is generated by a language model from a statistical model. It is not medical advice.")


# Tabs for the main section
tab2, tab1 = st.tabs(["Wordalisation", "Setup Model"])

with tab1:
    st.header("Setup Model")
    model_option = st.radio("Do you want to train a new model or use an existing trained model?", ("Train a new model (data pre-loaded for demo)", "Use an existing trained model"))
    if model_option == "Train a new model (data pre-loaded for demo)":
        st.write("You chose to train a new model. Please upload the raw data, and the explanations of the parameters in the data in CSV format.")
        result = setup_model(train=True)
    else:
        result = setup_model(train=False)
    if result is not None:
        data, model_features, target, categorical_interpretations = result
        setup_data(data=data, model_features=model_features, categorical_interpretations=categorical_interpretations, target=target)

with tab2:
    st.header("Wordalisation")

    if("model" in st.session_state):
        setup_chat()
    else:
        st.info("You need to set up the model first. Please go to the 'Setup Model' tab to do that.")
