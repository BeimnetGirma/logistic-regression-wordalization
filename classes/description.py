import math
from abc import ABC, abstractmethod
from typing import List, Union, Dict, Optional

import pandas as pd
import tiktoken
import openai
import numpy as np

import utils.sentences as sentences
from utils.gemini import convert_messages_format
from classes.data_point import Player, Country, Person, Individual
from classes.data_source import PersonStat

import json

from settings import USE_GEMINI

if USE_GEMINI:
    from settings import USE_GEMINI, GEMINI_API_KEY, GEMINI_CHAT_MODEL
else:
    from settings import GPT_BASE, GPT_VERSION, GPT_KEY, GPT_ENGINE

import streamlit as st
import random

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
                    "to earlier user/assistant pairs to give summaries of players."
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
                "content": f"Now do the same thing with the following 2 descriptions. : ```{self.synthesized_text}```",
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

        st.write("Chat transcript:", self.messages)

        if USE_GEMINI:
            import google.generativeai as genai

            converted_msgs = convert_messages_format(self.messages)

            # # save converted messages to json
            # with open("data/wvs/msgs_0.json", "w") as f:
            #     json.dump(converted_msgs, f)

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


class PlayerDescription(Description):
    output_token_limit = 150

    @property
    def gpt_examples_path(self):
        return f"{self.gpt_examples_base}/Forward.xlsx"

    @property
    def describe_paths(self):
        return [f"{self.describe_base}/Forward.xlsx"]

    def __init__(self, player: Player):
        self.player = player
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
                    "You are a UK-based football scout. "
                    "You provide succinct and to the point explanations about football players using data. "
                    "You use the information given to you from the data and answers "
                    "to earlier user/assistant pairs to give summaries of players."
                ),
            },
            {
                "role": "user",
                "content": "Do you refer to the game you are an expert in as soccer or football?",
            },
            {
                "role": "assistant",
                "content": (
                    "I refer to the game as football. "
                    "When I say football, I don't mean American football, I mean what Americans call soccer. "
                    "But I always talk about football, as people do in the United Kingdom."
                ),
            },
        ]
        if len(self.describe_paths) > 0:
            intro += [
                {
                    "role": "user",
                    "content": "First, could you answer some questions about football for me?",
                },
                {"role": "assistant", "content": "Sure!"},
            ]

        return intro

    def synthesize_text(self):

        player = self.player
        metrics = self.player.relevant_metrics
        description = f"Here is a statistical description of {player.name}, who played for {player.minutes_played} minutes as a {player.position}. \n\n "

        subject_p, object_p, possessive_p = sentences.pronouns(player.gender)

        for metric in metrics:

            description += f"{subject_p.capitalize()} was "
            description += sentences.describe_level(player.ser_metrics[metric + "_Z"])
            description += " in " + sentences.write_out_metric(metric)
            description += " compared to other players in the same playing position. "

        # st.write(description)

        return description

    def get_prompt_messages(self):
        prompt = (
            f"Please use the statistical description enclosed with ``` to give a concise, 4 sentence summary of the player's playing style, strengths and weaknesses. "
            f"The first sentence should use varied language to give an overview of the player. "
            "The second sentence should describe the player's specific strengths based on the metrics. "
            "The third sentence should describe aspects in which the player is average and/or weak based on the statistics. "
            "Finally, summarise exactly how the player compares to others in the same position. "
        )
        return [{"role": "user", "content": prompt}]


class CountryDescription(Description):
    output_token_limit = 150

    @property
    def gpt_examples_path(self):
        return f"{self.gpt_examples_base}/WVS_examples.xlsx"

    @property
    def describe_paths(self):
        return [f"{self.describe_base}/WVS_qualities.xlsx"]

    def __init__(self, country: Country, description_dict, thresholds_dict):
        self.country = country
        self.description_dict = description_dict
        self.thresholds_dict = thresholds_dict

        # read data/wvs/intermediate_data/relevant_questions.json
        with open("data/wvs/intermediate_data/relevant_questions.json", "r") as f:
            self.relevant_questions = json.load(f)

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
                    "You are a data analyst and a social scientist. "
                    "You provide succinct and to the point explanations about countries using metrics derived from data collected in the World Value Survey. "
                    "You use the information given to you from the data and answers to earlier questions to give summaries of how countries score in various metrics that attempt to measure the social values held by the population of that country."
                ),
            },
            # {
            #     "role": "user",
            #     "content": "Do you refer to the game you are an expert in as soccer or football?",
            # },
            # {
            #     "role": "assistant",
            #     "content": (
            #         "I refer to the game as football. "
            #         "When I say football, I don't mean American football, I mean what Americans call soccer. "
            #         "But I always talk about football, as people do in the United Kingdom."
            #     ),
            # },
        ]
        if len(self.describe_paths) > 0:
            intro += [
                {
                    "role": "user",
                    "content": "First, could you answer some questions about a the World Value Survey for me?",
                },
                {"role": "assistant", "content": "Sure!"},
            ]

        return intro

    def synthesize_text(self):

        description = f"Here is a statistical description of the core values of {self.country.name.capitalize()}. \n\n"

        # subject_p, object_p, possessive_p = sentences.pronouns(country.gender)

        for metric in self.country.relevant_metrics:

            description += (
                f"According to the WVS, {self.country.name.capitalize()} was found to "
            )
            description += sentences.describe_level(
                self.country.ser_metrics[metric + "_Z"],
                thresholds=self.thresholds_dict[metric],
                words=self.description_dict[metric],
            )
            description += " compared to other countries in the same wave. "

            if metric in self.country.drill_down_metrics:

                if self.country.ser_metrics[metric + "_Z"] > 0:
                    index = 1
                else:
                    index = 0

                question, value = self.country.drill_down_metrics[metric]
                question, value = question[index], value[index]
                description += "In response to the question '"
                description += self.relevant_questions[metric][question][0]
                description += "', on average participants "
                description += self.relevant_questions[metric][question][1]
                description += " '"
                description += self.relevant_questions[metric][question][2][str(value)]
                description += "' "
                description += self.relevant_questions[metric][question][3]
                description += ". "

            description += "\n\n"
        # st.write(description)

        return description

    def get_prompt_messages(self):
        prompt = (
            f"Please use the statistical description enclosed with ``` to give a concise, 2 short paragraph summary of the social values held by population of the country. "
            f"The first paragraph should focus on any factors or values for which the country is above or bellow average. If the country is neither above nor below average in any values, mention that. "
            f"The remaining paragraph should mention any specific values or factors that are neither high nor low compared to the average. "
        )
        return [{"role": "user", "content": prompt}]


class PersonDescription(Description):
    output_token_limit = 150

    @property
    def gpt_examples_path(self):
        return f"{self.gpt_examples_base}/Forward_bigfive.xlsx"

    @property
    def describe_paths(self):
        return [f"{self.describe_base}/Forward_bigfive.xlsx"]

    def __init__(self, person: Person):
        self.person = person
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
                    "You are a recruiter. "
                    "You provide succinct and to the point explanations about a candiate using data.  "
                    "You use the information given to you from the data and answers"
                    "to earlier user/assistant pairs to give summaries of candidates."
                ),
            },
            {
                "role": "user",
                "content": "Do you refer to the candidate as a candidate or a person?",
            },
            {
                "role": "assistant",
                "content": (
                    "I refer to the candidate as a person. "
                    "When I say candidate, I mean person. "
                    "But I always talk about the candidate, as a person."
                ),
            },
        ]
        if len(self.describe_paths) > 0:
            intro += [
                {
                    "role": "user",
                    "content": "First, could you answer some questions about a candidate for me?",
                },
                {"role": "assistant", "content": "Sure!"},
            ]

        return intro

    def categorie_description(self, value):
        if value <= -2:
            return "The candidate is extremely "
        elif -2 < value <= -1:
            return "The candidate is very "
        elif -1 < value <= -0.5:
            return "The candidate is quite "
        elif -0.5 < value <= 0.5:
            return "The candidate is relatively "
        elif 0.5 < value <= 1:
            return "The candidate is quite "
        elif 1 < value <= 2:
            return "The candidate is very "
        else:
            return "The candidate is extremely "

    def all_max_indices(self, row):
        max_value = row.max()
        return list(row[row == max_value].index)

    def all_min_indices(self, row):
        min_value = row.min()
        return list(row[row == min_value].index)

    def get_description(self, person):
        # here we need the dataset to check the min and max score of the person

        person_metrics = person.ser_metrics
        person_stat = PersonStat()
        questions = person_stat.get_questions()

        name = person.name
        extraversion = person_metrics["extraversion_Z"]
        neuroticism = person_metrics["neuroticism_Z"]
        agreeableness = person_metrics["agreeableness_Z"]
        conscientiousness = person_metrics["conscientiousness_Z"]
        openness = person_metrics["openness_Z"]

        text = []

        # extraversion
        cat_0 = "solitary and reserved. "
        cat_1 = "outgoing and energetic. "

        if extraversion > 0:
            text_t = self.categorie_description(extraversion) + cat_1
            if extraversion > 1:
                index_max = person_metrics[0:10].idxmax()
                text_2 = (
                    "In particular they said that " + questions[index_max][0] + ". "
                )
                text_t += text_2
        else:
            text_t = self.categorie_description(extraversion) + cat_0
            if extraversion < -1:
                index_min = person_metrics[0:10].idxmin()
                text_2 = (
                    "In particular they said that " + questions[index_min][0] + ". "
                )
                text_t += text_2
        text.append(text_t)

        # neuroticism
        cat_0 = "resilient and confident. "
        cat_1 = "sensitive and nervous. "

        if neuroticism > 0:
            text_t = (
                self.categorie_description(neuroticism)
                + cat_1
                + "The candidate tends to feel more negative emotions, anxiety. "
            )
            if neuroticism > 1:
                index_max = person_metrics[10:20].idxmax()
                text_2 = (
                    "In particular they said that " + questions[index_max][0] + ". "
                )
                text_t += text_2

        else:
            text_t = (
                self.categorie_description(neuroticism)
                + cat_0
                + "The candidate tends to feel less negative emotions, anxiety. "
            )
            if neuroticism < -1:
                index_min = person_metrics[10:20].idxmin()
                text_2 = (
                    "In particular they said that " + questions[index_min][0] + ". "
                )
                text_t += text_2
        text.append(text_t)

        # agreeableness
        cat_0 = "critical and rational. "
        cat_1 = "friendly and compassionate. "

        if agreeableness > 0:
            text_t = (
                self.categorie_description(agreeableness)
                + cat_1
                + "The candidate tends to be more cooperative, polite, kind and friendly. "
            )
            if agreeableness > 1:
                index_max = person_metrics[20:30].idxmax()
                text_2 = (
                    "In particular they said that " + questions[index_max][0] + ". "
                )
                text_t += text_2

        else:
            text_t = (
                self.categorie_description(agreeableness)
                + cat_0
                + "The candidate tends to be less cooperative, polite, kind and friendly. "
            )
            if agreeableness < -1:
                index_min = person_metrics[20:30].idxmin()
                text_2 = (
                    "In particular they said that " + questions[index_min][0] + ". "
                )
                text_t += text_2
        text.append(text_t)

        # conscientiousness
        cat_0 = "extravagant and careless. "
        cat_1 = "efficient and organized. "

        if conscientiousness > 0:
            text_t = (
                self.categorie_description(conscientiousness)
                + cat_1
                + "The candidate tends to be more careful or diligent. "
            )
            if conscientiousness > 1:
                index_max = person_metrics[30:40].idxmax()
                text_2 = (
                    "In particular they said that " + questions[index_max][0] + ". "
                )
                text_t += text_2
        else:
            text_t = (
                self.categorie_description(conscientiousness)
                + cat_0
                + "The candidate tends to be less careful or diligent. "
            )
            if conscientiousness < -1:
                index_min = person_metrics[30:40].idxmin()
                text_2 = (
                    "In particular they said that " + questions[index_min][0] + ". "
                )
                text_t += text_2
        text.append(text_t)

        # openness
        cat_0 = "consistent and cautious. "
        cat_1 = "inventive and curious. "

        if openness > 0:
            text_t = (
                self.categorie_description(openness)
                + cat_1
                + "The candidate tends to be more open. "
            )
            if openness > 1:
                index_max = person_metrics[40:50].idxmax()
                text_2 = (
                    "In particular they said that " + questions[index_max][0] + ". "
                )
                text_t += text_2
        else:
            text_t = (
                self.categorie_description(openness)
                + cat_0
                + "The candidate tends to be less open. "
            )
            if openness < -1:
                index_min = person_metrics[40:50].idxmin()
                text_2 = (
                    "In particular they said that " + questions[index_min][0] + ". "
                )
                text_t += text_2
        text.append(text_t)

        text = "".join(text)
        text = text.replace(",", "")
        return text

    def synthesize_text(self):
        person = self.person
        metrics = self.person.ser_metrics
        description = self.get_description(person)

        return description

    def get_prompt_messages(self):
        prompt = (
            f"Please use the statistical description enclosed with ``` to give a concise, 4 sentence summary of the person's personality, strengths and weaknesses. "
            f"The first sentence should use varied language to give an overview of the person. "
            "The second sentence should describe the person's specific strengths based on the metrics. "
            "The third sentence should describe aspects in which the person is average and/or weak based on the statistics. "
            "Finally, summarise exactly how the person compares to others in the same position. "
        )
        return [{"role": "user", "content": prompt}]

class IndividualDescription(Description):
    output_token_limit = 150

    @property
    def gpt_examples_path(self):
        return f"{self.gpt_examples_base}/Cardio.xlsx"

    @property
    def describe_paths(self):
        return [f"{self.describe_base}/Anuerysm.xlsx"]

    def __init__(self, individual: Individual,metrics,parameter_explanation, categorical_interpretations , thresholds, target, bins, model_features, individuals, fixed, odds_space=False):
        self.metrics = metrics
        self.individual = individual
        self.parameter_explanation = parameter_explanation
        self.categorical_interpretations = categorical_interpretations
        self.thresholds = thresholds
        self.target = target
        self.bins = bins
        self.model_features = model_features
        self.individuals=individuals
        self.fixed_description=fixed
        self.odds_space=odds_space
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
                    "You are doctor. "
                    "You have been asked to provide a summary of a patient's risk of developing cardio-vascular disease. "
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
        calculated_age= self.calculate_risk_age()
        beta_age= self.get_beta("age")
        description = f"Here is a statistical description of the factors related to {self.target} issues for the patient. \n\n "
        for metric in metrics:
            description += (
                self.describe_metric_odds_space(metric, beta_age)
                if self.odds_space
                else self.describe_metric_linear(metric, beta_age)
            )

        description+= self.describe_overall_risk(calculated_age)

        return description
    def describe_metric_linear(self, metric, beta_age):
        individual = self.individual
        beta_feature = self.get_beta(metric)
        value = individual.ser_metrics[metric]
        
        if self.categorical_interpretations and metric in self.categorical_interpretations:
            # if categorical interpretation is available, look up the value in the interpretation dictionary
            interperation = self.categorical_interpretations[metric].get(str(int(value)), value)
            risk_increase = self.calcaulte_catergorical_risk_age_increase(beta_feature, beta_age)
            text = f" {interperation} "
        else:
            # if no interpretation is available, just use the value
            text = sentences.article(self.parameter_explanation[metric].lower()) + f" {self.parameter_explanation[metric].lower()} of {sentences.format_numbers(value)} "
            risk_increase = self.calculate_risk_age_increase(metric, value, beta_feature, beta_age)
        
        thresholds = self.thresholds if self.fixed_description else self.bins.get(f"{metric}_contribution", self.thresholds)        
        words = [
            "implies a strongly reduced risk",
            "implies a moderately reduced risk",
            "implies no significant effect",
            "implies a moderately increased risk",
            "implies a strongly increased risk"
        ]
        text +=sentences.describe_contributions(individual.ser_metrics[metric + "_contribution"], thresholds=thresholds, words=words)
        text += f" of developing {self.target} issues."

        if metric != "age":
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
            risk_increase = self.calcaulte_catergorical_risk_age_increase(beta_feature, beta_age)
            text = f" {interperation} "
            
        else:
            # if no interpretation is available, just use the value
            text = sentences.article(self.parameter_explanation[metric].lower()) + f" {self.parameter_explanation[metric].lower()} of {sentences.format_numbers(value)} "
            risk_increase = self.calculate_risk_age_increase(metric, value, beta_feature, beta_age)
        
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
            text += f"  does not significantly affect your risk of developing {self.target} issues."
        else:
            text += f"  {direction} your risk of developing {self.target} issues by {percent_change:.1f}% compared to the average patient {self.get_average_value(metric)}. "
        
        
        
        if metric != "age":
            effect = "decreases" if risk_increase < 0 else "increases"
            risk_increase = abs(risk_increase)
            text+= f"This corresponds to a {sentences.format_numbers(risk_increase)} years {effect} in risk age. "
        return text
    
    def describe_overall_risk(self, calculated_age):
        individual = self.individual
        words= ["strongly reduced", "moderately reduced", "average", "moderately increased", "strongly increased"]
        text=(
                        f" The patient's overall risk of developing {self.target} issues is "
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
        text += f" The patient's risk of developing {self.target} issues is equivalent to that of a {calculated_age:.0f} year old."
        return text

    def get_prompt_messages(self):
        prompt = (
            f"Please use the statistical description enclosed with ``` to give a concise summary of the patients health metrics focusing on factors that negatively and postively affect their risk of developing {self.target} issues. Use second person language to address the patient. Write a concise, conversational paragraph (4-7 sentneces) that:"
            f"1. Begins with a greeting and a summary of the user’s overall cardiovascular risk, including heart age if applicable. "
            "2. Highlights protective or positive factors first (blood pressure, BMI, blood sugar, lifestyle habits, sex/age effects)."
            "3. Explains the main risk factors, why they matter, and their impact on heart health."
            "4. Offers practical guidance or next steps the user can take to reduce risk or maintain heart health"
            "5. Uses an approachable, supportive, and professional tone — interpretive and evidence-based, but avoid speculation beyond the provided data."
            "6. Refers to the user’s data directly in context, balances positives and negatives, and flows naturally like a short personal summary rather than a bulleted list."
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

        return min(max(calucalted_risk_age, 33), 90)
        # return calucalted_risk_age
        
    def calculate_risk_age_increase(self,metric, feature_value, beta_feature, beta_age):
        # Calculate the increase in risk age based on the feature value and baseline feature value
        feature_baseline = self.individuals[metric].mean()
        return (feature_value - feature_baseline) * beta_feature / beta_age
        
    def calcaulte_catergorical_risk_age_increase(self, beta_category, beta_age):
        # Calculate the increase in risk age based on the feature value and baseline feature value
        return beta_category / beta_age
        
    
    def get_beta(self, param):
        """
        Get the beta value for a given metric from the model features.
        """
        return self.model_features.loc[self.model_features['Parameter'] == param, 'Value'].values[0]