import streamlit as st
import pandas as pd
import json

from classes.train_model import TrainModel
from classes.data_source import Model
from classes.chat import ModelChat
from classes.description import (
    IndividualDescription,
)
import copy
from utils.page_components import (
    create_chat,
)



from classes.visual import DistributionModelPlot
from PIL import Image, ImageOps, ImageDraw
# Set Streamlit page layout
st.set_page_config(page_title="Logistic Regression App", layout="wide")

# Sidebar with a link using `st.page_link`
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
st.sidebar.page_link("app.py", label="Logistic Regression")


def setup_model(train=False):
    st.write("Upload a CSV file to use as the data source.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="data_file")
    # Default file for testing
    uploaded_file = open("data/medical/training_data_70000_Ind.csv", "rb")
    
    
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        all_columns = data.columns.tolist() 
        # st.write("Upload a CSV file to explaining the features.")
        # feature_file = st.file_uploader("Choose a CSV file", type="csv", key="feature_file")
        # Default file for testing
        feature_file= open("data/medical/feature_explanation.csv", "rb")
        selected_features = st.multiselect("Remove features you want to discard", all_columns, default=all_columns)
        # Create editable DataFrame
        explanation_df = pd.DataFrame({
            "Parameter": selected_features,
            "Explanation": [_ for _ in selected_features]
        })

        # Let users edit explanations directly
        st.write("Provide short explanations for the selected features:")
        
        # fill in explanation in edited_df from feature_file for testing and demo
        feature_data = pd.read_csv(feature_file)
        explanation_map = dict(zip(feature_data['Parameter'], feature_data['Explanation']))
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

        
            

        if feature_file is not None:
            # feature_data = pd.read_csv(feature_file)
            categorical_interpretations=None
            has_categorical = st.radio("Does your data have categorical features?", ("Yes", "No"))
            if has_categorical == "Yes":

                # st.write("Upload a JSON file detailing the interpretations of the categorical data.")
                # json_file = st.file_uploader("Choose a JSON file", type="json", key="category_json_file")
                # Default file for testing
                json_file = open("data/medical/categorical_features.json", "rb")
                if json_file is not None:
                    preloaded_categorical_interpretations = json.load(json_file)
                    

                # categorical_feautres=st.multiselect("Select categorical features", selected_features)
                # testing selecte features in json file
                categorical_feautres=st.multiselect("Select categorical features", selected_features, default=list(preloaded_categorical_interpretations.keys()))
                categorical_interpretations={}
                st.write(f"Define meanings for values in your categorical features")
                for feature in categorical_feautres:
                    st.markdown(f"**{feature}**")
                    unique_values= sorted(data[feature].dropna().unique().tolist())
                    value_expl_df = pd.DataFrame({
                        "Value": sorted(unique_values),
                        "Meaning": ["" for _ in unique_values]
                    })
                    # Use preloaded values if available
                    prefilled = preloaded_categorical_interpretations.get(feature, {})
                    value_expl_df['Meaning'] = [prefilled.get(str(val), "") for val in unique_values]
                    
                    edited_value_df=st.data_editor(
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


            feature_data= edited_df 
            
            # either they've said they don't have any categorical features or they've uploaded the interpretations 
            if has_categorical == "No" or categorical_interpretations is not None:
                
                if not train:
                    st.write("Upload a CSV file of the weights (coefficients) found in your model")
                    weights_file = st.file_uploader("Choose a CSV file", type="csv", key="weight_file")
                    if weights_file is not None:
                        weights_data = pd.read_csv(weights_file)
                        parameter_explanations = {row['Parameter']: row['Explanation'] for _, row in feature_data.iterrows()}
                        weights_data['Explanation'] = weights_data['Parameter'].map(parameter_explanations)
                        st.write("Feature Explanation:", weights_data)
                        target= list(set(data.columns) - set(weights_data['Parameter']))[0]
                        if categorical_interpretations is not None:
                            return (data, weights_data, target, categorical_interpretations)
                        else:
                            return (data, weights_data, target)
                # if training a new model, ask for target column and features 
                else:
                        st.write("Choose target column")
                        target = st.radio("Target column", selected_features, index=selected_features.index("cardio"), key="target_column")
                        features = [col for col in data.columns if col != target]
                        # model=TrainModel(data, target, features)
                        try:
                            model=TrainModel(data, target, features)
                        except Exception as e:
                            st.error(f"An error occurred while training the model: {e} Pick a traget column that is binary.")
                            return
                        
                        # merge explantions with coef_df on matching feature names
                        coef_explanations = {row['Parameter']: row['Explanation'] for _, row in feature_data.iterrows()}
                        model.coef_df['Explanation'] = model.coef_df['Parameter'].map(coef_explanations)
                        model.coef_df['P-Value'] = model.coef_df['Parameter'].map(model.p_values)
                        st.write("Training Output:", model.coef_df)

                        feature_selection=st.radio("Do you want to perform feature selection using stepwise backward elimination?", ("No", "Yes"), key="feature_selection")
                        if feature_selection == "Yes":
                            st.write("Performing stepwise backward elimination")
                            model.selectFeatures()
     
                            # merge explantions with coef_df on matching feature names
                            coef_explanations = {row['Parameter']: row['Explanation'] for _, row in feature_data.iterrows()}
                            model.coef_df['Explanation'] = model.coef_df['Parameter'].map(coef_explanations)
                            model.coef_df['P-Value'] = model.coef_df['Parameter'].map(model.p_values)
                            st.write("New Training Output:", model.coef_df)

                        else:
                            st.write("No feature selection performed")
                        
                        # Keep only the Intercept, target, and parameters in model.coef_df in data
                        columns_to_keep = [target] + [param for param in model.coef_df['Parameter'].tolist() if param in data.columns]
                        data = data[columns_to_keep]
                        return (data, model.coef_df, target,categorical_interpretations)
                        # if categorical_interpretations is not None:
                            
                        # else:
                        #     return (data, model.coef_df, target)
                        

def setup_data(data, model_features, categorical_interpretations=None, target=None):
    model=Model()
    model.set_data(data.head(90), model_features, categorical_interpretations=categorical_interpretations)
    model.process_data()
    model.weight_contributions()
    # bins=model.risk_thresholds()
    

    st.session_state["model"] = model
    # st.session_state["bins"] = model.risk_thresholds()
    st.session_state["metrics"] = model.parameters["Parameter"]
    st.session_state["target"] = target
    st.session_state["categorical_interpretations"] = categorical_interpretations
    st.session_state["model_features"] = model_features

    st.expander("Dataframe used", expanded=True).write(model.df)
    st.success("Model data is processed! Go to the 'Chat' tab to select an individual.")
    
def setup_chat():
    model = st.session_state["model"]

    # Select individual
    individuals_copy = copy.deepcopy(model)
    individuals_copy.select_and_filter(column_name="ID", label="Select Individual")
    individual = individuals_copy.to_data_point(columns=["ID", st.session_state["target"]])
    st.session_state["individual"] = individual

    # Calculate thresholds and bins
    thresholds, x_range, min_max_range = model.calulcate_threshold()
    bins = model.risk_thresholds()

    # Chat state hash
    to_hash = (individual.id,)
    chat = create_chat(to_hash, ModelChat, individual, model)

    metrics = model.parameters['Parameter']

    if chat.state == "empty":
        
        st.markdown("##### **Individual Description **")
        with st.expander("Description with your selected settings", expanded=False):

            # Create IndividualDescription based on dropdowns
            description = IndividualDescription(
                individual,
                metrics,
                parameter_explanation=model.parameter_explanation,
                categorical_interpretations=categorical_interpretations,
                thresholds=thresholds,
                target=st.session_state["target"],
                bins=bins,
                model_features=model_features,
                individuals=model.df,
                fixed=True,
                odds_space=True,
                # Dropdown-driven customization
                risk_scale=st.session_state.get("risk_scale", "Average"),
                log_linear_scale=st.session_state.get("log_linear_scale", "Linear Scale"),
                risk_age=st.session_state.get("risk_age", "Include"),
                categorical_treatment=st.session_state.get("categorical_treatment", "Average")
            )

            st.markdown("###### **Synthesized Text about individual:**")
            st.write(description.synthesized_text)

            # Visuals
            visual = DistributionModelPlot(thresholds, min_max_range, metrics, model_features=model_features)
            visual.add_title('Evaluation of individual','')
            visual.add_individuals(model, metrics=metrics, target=st.session_state["target"])
            visual.add_individual(individual, len(model.df), metrics=metrics)
            st.write(visual.show())

            summary = description.stream_gpt()
            st.markdown("###### **Summary of the individual:**")
            st.write(summary)

            chat.state = "empty"

    # Now get the user input, display messages, and save state
    chat.get_input()
    chat.display_messages()
    chat.save_state()


# Tabs
tab1, tab2 = st.tabs(["Setup Model", "Chat"])

with tab1:
    st.header("Setup Model")
    model_option = st.radio(
        "Do you want to train a new model or use an existing trained model?",
        ("Train a new model (data pre-loaded for demo)", "Use an existing trained model")
    )

    if model_option == "Train a new model (data pre-loaded for demo)":
        st.write("You chose to train a new model. Please upload the raw data and parameter explanations in CSV format.")
        result = setup_model(train=True)
        if result is not None:
            data, model_features, target, categorical_interpretations = result
            setup_data(data=data, model_features=model_features, categorical_interpretations=categorical_interpretations, target=target)
    else:
        result = setup_model(train=False)
        if result is not None:
            data, model_features, target, categorical_interpretations = result
            setup_data(data=data, target=target, model_features=model_features, categorical_interpretations=categorical_interpretations)

with tab2:
    st.header("Chat & Visualization")
    if "model" in st.session_state:
        st.markdown("### Select Settings for Individual Description")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.session_state["risk_scale"] = st.selectbox(
                "Risk Scale",
                ["Independent", "Highest feature", "Average"]
            )

        with col2:
            st.session_state["log_linear_scale"] = st.selectbox(
                "Log/Linear Scale",
                ["Log Scale", "Linear Scale"]
            )

        with col3:
            st.session_state["risk_age"] = st.selectbox(
                "Risk Age",
                ["Include", "Exclude"]
            )

        with col4:
            st.session_state["categorical_treatment"] = st.selectbox(
                "Categorical Variable Treatment",
                ["Reference Type", "Average"]
            )

        setup_chat()
    else:
        st.info("You need to setup the model first before you can chat. Please go to the 'Setup Model' tab.")
