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
# Set Streamlit page layout
st.set_page_config(page_title="Logistic Regression App", layout="wide")

# Sidebar with a link using `st.page_link`
st.sidebar.page_link("app.py", label="Logistic Regression")


def setup_model(train=False):
    st.write("Upload a CSV file to use as the data source.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="data_file")
    # Default file for testing
    # uploaded_file = open("C:/Users/beimn/Documents/workdir/Python for Data Science/anuerysm/bmi_train_data_70000_Ind.csv", "rb")
    
    
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Upload a CSV file to explaining the features.")
        feature_file = st.file_uploader("Choose a CSV file", type="csv", key="feature_file")
        # Default file for testing
        # feature_file= open("C:/Users/beimn/Documents/workdir/Python for Data Science/anuerysm/train_data_explanation.csv", "rb")
        if feature_file is not None:
            feature_data = pd.read_csv(feature_file)
            categorical_interpretations=None
            has_categorical = st.radio("Does your data have categorical features?", ("Yes", "No"))
            if has_categorical == "Yes":
                st.write("Upload a JSON file detailing the interpretations of the categorical data.")
                json_file = st.file_uploader("Choose a JSON file", type="json", key="category_json_file")
                # Default file for testing
                # json_file = open("C:/Users/beimn/Documents/workdir/Python for Data Science/anuerysm/train_data_categorical_features.json", "rb")
                if json_file is not None:
                    categorical_interpretations = json.load(json_file)
                    # st.write("Categorical Interpretations:", categorical_interpretations)
            # either they've said they don't have any categorical features or they've uploaded the interpretations 
            if has_categorical == "No" or categorical_interpretations is not None:
                # is there is a pre-trained model, upload the weights
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
                        target = st.radio("Target column", data.columns)
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
                        if categorical_interpretations is not None:
                            return (data, model.coef_df, target,categorical_interpretations)
                        else:
                            return (data, model.coef_df, target)
                        

def setup_data(data, model_features, categorical_interpretations=None, target=None):
    model=Model()
    model.set_data(data.head(5000), model_features)
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
    model=st.session_state["model"]
    # # Now select the focal player
    # columns = ["ID", target]
    # individual = select_individual(sidebar_container, model, columns=columns)

    # select individual
    # individual_ids= model.df['ID'].unique().tolist()
    # selected_individual_id = st.selectbox("Select individual", individual_ids)
    # process selected individual
    individuals_copy=copy.deepcopy(model) 
    individuals_copy.select_and_filter(column_name="ID",
            label="Select Individual",)
    individual=individuals_copy.to_data_point(columns=["ID", st.session_state["target"]])

    st.session_state["individual"] = individual

    # Gengerate Chat and Visualization
    thresholds = model.most_variable_data() 
    bins=model.risk_thresholds()           
    st.write("The most variable feature is",model.std_contributions.idxmax())
    st.write("The thresholds values for the fixed description are",str(thresholds), "based on the most variable feature", model.std_contributions.idxmax())
    # st.write("The bins value are",str(bins))

    # Chat state hash determines whether or not we should load a new chat or continue an old one
    # We can add or remove variables to this hash to change conditions for loading a new chat
    to_hash = (individual.id,)
    # Now create the chat as type PlayerChat
    chat = create_chat(to_hash, ModelChat, individual, model)

    metrics =  model.parameters['Parameter']

    # Now we want to add basic content to chat if it's empty
    if chat.state == "empty":

        # Make a plot of the distribution of the metrics for all players
        # We reverse the order of the elements in metrics for plotting (because they plot from bottom to top)
        visual = DistributionModelPlot(thresholds,metrics, model_features=model_features)
        visual.add_title('Evaluation of individual','')
        visual.add_individuals(model, metrics=metrics, target=target)
        visual.add_individual(individual, len(model.df), metrics=metrics)

        # visual= RidgelinePlot(model.df, metrics=metrics, target=target, individual_data=individual)
        # visual.plot_population()

        # Now call the description class to get the summary of the player
        
        description = IndividualDescription(individual,metrics,parameter_explanation=model.parameter_explanation, categorical_interpretations= categorical_interpretations , thresholds= thresholds, target=target, bins=bins, model_features=model_features)
        
        summary = description.stream_gpt()

        # Add the visual and summary to the chat
        chat.add_message(
            "Please can you summarise this individual for me?",
            role="user",
            user_only=False,
            visible=False,
        )
        chat.add_message(visual)
        chat.add_message(summary)

        chat.state = "default"

    # Now we want to get the user input, display the messages and save the state
    chat.get_input()
    chat.display_messages()
    chat.save_state()


# Tabs for the main section
tab1, tab2 = st.tabs(["Setup Model", "Chat"])

with tab1:
    st.header("Setup Model")
    model_option = st.radio("Do you want to train a new model or use an existing trained model?", ("Train a new model", "Use an existing trained model"))  
    if model_option == "Train a new model":
        st.write("You chose to train a new model. Please upload the raw data, and the explanations of the parameters in the data in CSV format.")
        # trainModel()
        result = setup_model(train=True)
        if result is not None:
            data, model_features, target, categorical_interpretations = result
            setup_data(data=data, model_features=model_features, categorical_interpretations=categorical_interpretations,target= target)
        
    else:
        result = setup_model(train=False)
        if result is not None:
            data, model_features, target, categorical_interpretations= result
            setup_data(data=data, target=target, model_features=model_features, categorical_interpretations=categorical_interpretations)
        # data, model_features= setup_model(train=False)

with tab2:
    st.header("Chat & Visualization")
    if("model" in st.session_state):
        setup_chat()
    else:
        st.info("You need to setup the model first before you can chat. Please go to the 'Setup Model' tab to do that.")
    # Placeholder for model results, visualizations, and chat logic
