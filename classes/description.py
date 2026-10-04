from abc import ABC, abstractmethod
from typing import List, Union, Dict

import pandas as pd
import openai
import numpy as np

import utils.sentences as sentences
from utils.gemini import convert_messages_format
from classes.data_point import Individual

from settings import (
    USE_GEMINI,
    GPT_BASE, GPT_VERSION, GPT_KEY, GPT_ENGINE,
    GEMINI_API_KEY, GEMINI_CHAT_MODEL,
)


openai.api_type = "azure"


class Description(ABC):
    gpt_examples_base = "data/gpt_examples"
    describe_base = "data/describe"

    @property
    @abstractmethod
    def gpt_examples_path(self) -> str:
        """
        Path to excel files containing examples of user and assistant messages for the GPT to learn from.
        """

    @property
    @abstractmethod
    def describe_paths(self) -> Union[str, List[str]]:
        """
        List of paths to excel files containing questions and answers for the GPT to learn from.
        """

    def __init__(self):
        self.synthesized_text = self.synthesize_text()
        self.messages = self.setup_messages()

    def synthesize_text(self) -> str:
        """
        Return a data description that will be used to prompt GPT.

        Returns:
        str
        """

    def get_prompt_messages(self) -> List[Dict[str, str]]:
        """
        Return the prompt that the GPT will see before self.synthesized_text.

        Returns:
        List of dicts with keys "role" and "content".
        """

    def get_intro_messages(self) -> List[Dict[str, str]]:
        """
        Constant introduction messages for the assistant.

        Returns:
        List of dicts with keys "role" and "content".
        """
        intro = [
            {
                "role": "system",
                "content": (
                    "You are a data analysis bot. "
                    "You provide succinct and to the point explanations about data using data. "
                    "You use the information given to you from the data and answers "
                    "to earlier user/assistant pairs to give summaries of individuals."
                ),
            },
        ]
        if len(self.describe_paths) > 0:
            intro += [
                {
                    "role": "user",
                    "content": "First, could you answer some questions about the data for me?",
                },
                {"role": "assistant", "content": "Sure!"},
            ]

        return intro

    def get_messages_from_excel(
        self,
        paths: Union[str, List[str]],
    ) -> List[Dict[str, str]]:
        """
        Turn an excel file containing user and assistant columns with str values into a list of dicts.

        Arguments:
        paths: str or list of str
            Path to the excel file containing the user and assistant columns.

        Returns:
        List of dicts with keys "role" and "content".

        """

        # Handle list and str paths arg
        if isinstance(paths, str):
            paths = [paths]
        elif len(paths) == 0:
            return []

        # Concatenate dfs read from paths
        df = pd.read_excel(paths[0])
        for path in paths[1:]:
            df = pd.concat([df, pd.read_excel(path)])

        if df.empty:
            return []

        # Convert to list of dicts
        messages = []
        for i, row in df.iterrows():
            if i == 0:
                messages.append({"role": "user", "content": row["user"]})
            else:
                messages.append({"role": "user", "content": row["user"]})
            messages.append({"role": "assistant", "content": row["assistant"]})

        return messages

    def setup_messages(self) -> List[Dict[str, str]]:
        messages = self.get_intro_messages()
        try:
            paths = self.describe_paths
            messages += self.get_messages_from_excel(paths)
        except (
            FileNotFoundError
        ) as e:  # FIXME: When merging with new_training, add the other exception
            print(e)
        messages += self.get_prompt_messages()

        messages = [
            message for message in messages if isinstance(message["content"], str)
        ]

        try:
            messages += self.get_messages_from_excel(
                paths=self.gpt_examples_path,
            )
        except (
            FileNotFoundError
        ) as e:  # FIXME: When merging with new_training, add the other exception
            print(e)

        messages += [
            {
                "role": "user",
                "content": f"Now do the same thing with the following description. : ```{self.synthesized_text}```",
            }
        ]
        return messages

    def stream_gpt(self, temperature=1):
        """
        Run the GPT model on the messages and stream the output.

        Arguments:
        temperature: optional float
            The temperature of the GPT model.

        Yields:
            str
        """

        # st.write("Chat transcript:", self.messages)

        if USE_GEMINI:
            import google.generativeai as genai

            converted_msgs = convert_messages_format(self.messages)

            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(
                model_name=GEMINI_CHAT_MODEL,
                system_instruction=converted_msgs["system_instruction"],
            )
            chat = model.start_chat(history=converted_msgs["history"])
            response = chat.send_message(content=converted_msgs["content"])

            answer = response.text
        else:
            # Use OpenAI API
            openai.api_base = GPT_BASE
            openai.api_version = GPT_VERSION
            openai.api_key = GPT_KEY

            response = openai.ChatCompletion.create(
                engine=GPT_ENGINE,
                messages=self.messages,
                temperature=temperature,
            )

            answer = response["choices"][0]["message"]["content"]

        return answer







class IndividualDescription(Description):
    output_token_limit = 150

    @property
    def gpt_examples_path(self):
        return f"{self.gpt_examples_base}/Cardio.xlsx"

    @property
    def describe_paths(self):
        return [f"{self.describe_base}/Anuerysm.xlsx"]

    def __init__(self, individual: Individual, metrics, parameter_explanation, categorical_interpretations, thresholds, target, bins, model_features, individuals, threshold_type, odds_space=False, risk_age=True):
        self.metrics = metrics
        self.individual = individual
        self.parameter_explanation = parameter_explanation
        self.categorical_interpretations = categorical_interpretations
        self.thresholds = thresholds
        self.target = target
        self.bins = bins
        self.model_features = model_features
        self.individuals = individuals
        self.threshold_type = threshold_type
        self.odds_space = odds_space
        # Risk age needs an 'age' coefficient in the model
        self.risk_age = risk_age and 'age' in set(model_features['Parameter'])
        super().__init__()


    def get_intro_messages(self) -> List[Dict[str, str]]:
        """
        Constant introduction messages for the assistant.

        Returns:
        List of dicts with keys "role" and "content".
        """
        intro = [
            {
                "role": "system",
                "content": (
                    "You are a doctor. "
                    f"You have been asked to provide a summary of a patient's risk of developing {self.target} issues. "
                ),
            },
        ]
        if len(self.describe_paths) > 0:
            intro += [
                {
                    "role": "user",
                    "content": "First, could you answer some questions about medical details for me?",
                },
                {"role": "assistant", "content": "Sure!"},
            ]

        return intro

    def synthesize_text(self):
        metrics = self.metrics
        calculated_age = self.calculate_risk_age() if self.risk_age else None
        beta_age = self.get_beta("age") if self.risk_age else None
        description = f"Here is a statistical description of the factors related to {self.target} issues for the patient. \n\n "
        for metric in metrics:
            description += (
                self.describe_metric_odds_space(metric, beta_age)
                if self.odds_space
                else self.describe_metric_linear(metric, beta_age)
            )

        description+= self.describe_overall_risk(calculated_age, odds_space=self.odds_space)

        return description
    def describe_metric_linear(self, metric, beta_age):
        individual = self.individual
        beta_feature = self.get_beta(metric)
        value = individual.ser_metrics[metric]
        
        if self.categorical_interpretations and metric in self.categorical_interpretations:
            # if categorical interpretation is available, look up the value in the interpretation dictionary
            interperation = self.categorical_interpretations[metric].get(str(int(value)), value)
            text = f" {interperation} "
        else:
            # if no interpretation is available, just use the value
            text = sentences.article(self.parameter_explanation[metric].lower()) + f" {self.parameter_explanation[metric].lower()} of {sentences.format_numbers(value)} "
        risk_increase = self.calculate_risk_age_increase(metric, value, beta_feature, beta_age) if self.risk_age else 0
        
        feature_bins = {k: v for k, v in self.bins.items() if k != 'total_risk_contribution'}
        avg = np.array(list(feature_bins.values())).mean(axis=0).tolist()
        
        thresholds = (
    self.thresholds
    if self.threshold_type in ("highest", "age")
    else self.bins.get(f"{metric}_contribution", self.thresholds)
    if self.threshold_type == "feature_specific"
    else avg
)
        words = [
            "implies a strongly reduced risk",
            "implies a moderately reduced risk",
            "implies no significant effect",
            "implies a moderately increased risk",
            "implies a strongly increased risk"
        ]
        text +=sentences.describe_contributions(individual.ser_metrics[metric + "_contribution"], thresholds=thresholds, words=words)
        text += f" of developing {self.target} issues."

        if metric != "age" and self.risk_age:
            effect = "decreases" if risk_increase < 0 else "increases"
            risk_increase = abs(risk_increase)
            if self.categorical_interpretations and metric in self.categorical_interpretations:
                interperation = self.categorical_interpretations[metric].get(str(int(value)), value)
                text += f" {interperation} "
            else:
                text+= sentences.article(self.parameter_explanation[metric].lower())+ f" {self.parameter_explanation[metric].lower()} of {sentences.format_numbers(value)} "
            text += f" {effect} your risk age by {sentences.format_numbers(risk_increase)} years. "
        return text
    
    def describe_metric_odds_space(self, metric, beta_age):
        individual = self.individual
        beta_feature = self.get_beta(metric)
        value = individual.ser_metrics[metric]
        
        if self.categorical_interpretations and metric in self.categorical_interpretations:
            # if categorical interpretation is available, look up the value in the interpretation dictionary
            interperation = self.categorical_interpretations[metric].get(str(int(value)), value)
            text = f" {interperation} "
            
        else:
            # if no interpretation is available, just use the value
            text = sentences.article(self.parameter_explanation[metric].lower()) + f" {self.parameter_explanation[metric].lower()} of {sentences.format_numbers(value)} "
        risk_increase = self.calculate_risk_age_increase(metric, value, beta_feature, beta_age) if self.risk_age else 0
        
        contribution = individual.ser_metrics[metric + "_contribution"]
        if contribution>1:
            percent_change= (contribution-1)*100
            direction = "increases"
        elif contribution<1:
            percent_change= (1-contribution)*100
            direction = "decreases"
        else:
            percent_change = 0
            direction = "does not significantly affect"

        #compose sentence
        if percent_change == 0:
            text += f"  does not significantly affect your odds of developing {self.target} issues."
        else:
            text += f"  {direction} your odds of developing {self.target} issues by {percent_change:.1f}% compared to the average patient {self.get_average_value(metric)}. "
        
        
        
        if metric != "age" and self.risk_age:
            effect = "decreases" if risk_increase < 0 else "increases"
            risk_increase = abs(risk_increase)
            text+= f"This corresponds to a {sentences.format_numbers(risk_increase)} years {effect} in risk age. "
        return text
    
    def describe_overall_risk(self, calculated_age, odds_space=False):
        individual = self.individual
        words= ["strongly reduced", "moderately reduced", "average", "moderately increased", "strongly increased"]
        risk_or_odds = "odds" if odds_space else "risk"
        text=(
            f" The patient's overall {risk_or_odds} of developing {self.target} issues is "
            f"{sentences.describe_contributions(individual.ser_metrics['total_risk_contribution'], thresholds=self.bins['total_risk_contribution'], words=words)} "
            f"compared to other patients who come into the clinic."
        )
        if individual.ser_metrics['total_risk_contribution'] > self.bins['total_risk_contribution'][1]:
            max_metric = max(
                {k: v for k, v in individual.ser_metrics.items() if k.endswith("_contribution") and k != "total_risk_contribution"},
                key=individual.ser_metrics.get 
            )
            max_metric = max_metric.replace("_contribution", "")
            text += f" The highest contribution factor for developing {self.target} issues is the patient's {self.parameter_explanation[max_metric].lower()}."
        if self.risk_age:
            text += f" The patient's risk of developing {self.target} issues is equivalent to that of a {calculated_age:.0f} year old."
        return text

    def get_prompt_messages(self):
        prompt = (
            f"Please use the statistical description enclosed with ``` to give a concise summary of the patients health metrics focusing on factors that negatively and postively affect their risk of developing {self.target} issues. Use second person language to address the patient. Write a concise, conversational paragraph (4-7 sentneces) that:"
            f"1. Begins with a greeting and a summary of the user’s overall risk, including their risk age if applicable. "
            "2. Highlights protective or positive factors first."
            "3. Explains the main risk factors, why they matter, and their impact on the patient's health."
            "4. Offers practical guidance or next steps the user can take to reduce risk or maintain good health. "
            "5. Uses an approachable, supportive, and professional tone — interpretive and evidence-based, but avoid speculation beyond the provided data."
            "6. Refers to the user’s data directly in context, balances positives and negatives, and flows naturally like a short personal summary rather than a bulleted list. "
        )
        return [{"role": "user", "content": prompt}]
    
    def get_average_value(self,param):
        # Get the average value for a given parameter from the individuals dataframe
        if self.categorical_interpretations and param in self.categorical_interpretations:
            # if categorical interpretation is available, look up the value in the interpretation dictionary
            mode_val= str(int(self.individuals[param].mode()[0]))
            interpretation = self.categorical_interpretations[param].get(str(int(mode_val)), mode_val)
            return self.naturalize_avg_description(interpretation)
        else:
            mean_val= self.individuals[param].mean()
            return f"typically has an average value of {mean_val:.1f}"
    def naturalize_avg_description(self,desc):
        if desc.startswith("Being "):
            return "who is " + desc.replace("Being ", "")
        elif desc.startswith("Having "):
            return "who has " + desc.replace("Having ", "")
        elif desc.startswith("Not "):
            return "who " + desc.lower()
        return "who typically has " + desc.lower()

    def calculate_risk_age(self):
        individual = self.individual
        beta_age = self.get_beta("age")
        
        total_contribution=0

        for param in self.model_features['Parameter']:
            if param == 'age':
                continue
            if param not in individual.ser_metrics:
                continue
            contribution_param= individual.ser_metrics[param] * self.get_beta(param)
            # center the contribution around 0
            contribution_param-= (self.individuals[param] * self.get_beta(param)).mean()

            total_contribution += contribution_param
        
        calucalted_risk_age= individual.ser_metrics['age'] + (total_contribution / beta_age)

        # Keep the risk age within the ages seen in the data
        return min(max(calucalted_risk_age, self.individuals['age'].min()), self.individuals['age'].max())
        # return calucalted_risk_age
        
    def calculate_risk_age_increase(self,metric, feature_value, beta_feature, beta_age):
        # Calculate the increase in risk age based on the feature value and baseline feature value
        feature_baseline = self.individuals[metric].mean()
        return (feature_value - feature_baseline) * beta_feature / beta_age
        
    
    def get_beta(self, param):
        """
        Get the beta value for a given metric from the model features.
        """
        return self.model_features.loc[self.model_features['Parameter'] == param, 'Value'].values[0]