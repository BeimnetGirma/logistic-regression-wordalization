# Logistic Regression Wordalisation

A Streamlit app that turns a logistic regression model into a written explanation of one individual's risk. It fits (or loads) a logistic regression, works out how much each feature contributes to a chosen individual's predicted risk, describes those contributions in words, and asks a large language model (LLM) to rewrite that description as a short, readable summary.

The bundled demo uses a dataset of 70,000 patients with and without cardiovascular disease. The method works for any binary outcome with numeric or integer-coded features.

See the [model card](docs/model-card.md) for intended use, data, evaluation and limitations.

The write-up of the paper has been moved out of this repository into a separate directory. This repository contains only the app.

## Running the app

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

You also need access to an LLM. Create `.streamlit/secrets.toml` with either an Azure OpenAI deployment:

```toml
[services.gpt]
AZURE_OPENAI_ENDPOINT = "https://<your-resource>.openai.azure.com/"
AZURE_OPENAI_API_VERSION = "<api version>"
AZURE_OPENAI_API_KEY = "<your key>"
AZURE_OPENAI_DEPLOYMENT = "<deployment name>"
```

or Gemini:

```toml
[settings]
USE_GEMINI = true

[services.gemini]
GEMINI_API_KEY = "<your key>"
GEMINI_CHAT_MODEL = "<model name>"
```

`secrets.toml` is in `.gitignore`. Never commit it.

## Using the app

### Setup Model tab

1. **Data.** Upload a CSV with one row per individual. Without an upload, the demo file `data/medical/training_data_70000_Ind.csv` is used. You can deselect columns you don't want to use.
2. **Feature explanations.** Short plain-language names for each feature (for example `ap_hi` → "Systolic blood pressure"). These are used in the generated text. Upload a CSV with `Parameter` and `Explanation` columns or edit them in the table.
3. **Categorical features.** Give a meaning for each value of each categorical feature (for example `cholesterol` 3 → "Having well above normal cholesterol levels"). Upload a JSON file in the format of `data/medical/categorical_features.json` or fill in the table.
4. **Model.** Either:
   - **Train a new model.** Choose the binary target column. The app removes rows with outliers (any feature outside Q1 − 3·IQR to Q3 + 3·IQR), fits a logistic regression on 80% of the remaining data and reports the AUC on the other 20%. Answer "Yes" to stepwise backward elimination to drop features with p > 0.05; the demo model was chosen this way.
   - **Use an existing model.** Upload a CSV of coefficients with `Parameter` and `Value` columns, including a row named `Intercept`.

### Wordalisation tab

Pick an individual. The app shows:

- a **synthesized text**: a rule-based description of each feature's contribution to that individual's risk,
- a **plot** of the individual's contributions against a sample of 90 other individuals,
- an **LLM summary** written in the second person from the synthesized text.

The sidebar controls how risk is described:

| Setting | Options | Effect |
|---|---|---|
| Scale | Log-odds / Odds ratio | Contributions as `x·β − mean(x·β)` or as `exp` of that (a % change in odds) |
| Risk Reference | Independent / Relative to Largest / Relative to Average / Relative to Age | Which ±0.5σ and ±1σ thresholds decide words like "moderately increased risk" |
| Include Risk-Age Mapping | on / off | Converts each contribution into years of "risk age" (needs an `age` feature) |
| Categorical Baseline | Specific Group / Population Average | Whether a categorical feature is compared with its lowest category or with the population average |

On the odds-ratio scale the text states percentage changes directly, so the Risk Reference setting only changes the plot, not the text. [docs/axes_implementation_audit.md](docs/axes_implementation_audit.md) gives the exact formulas for each setting.

**Generate 24 configurations for this patient** writes the synthesized text and LLM summary for every combination of settings to an Excel file. This is useful for comparing wordings.

## How it works

| File | Role |
|---|---|
| [app.py](app.py) | Streamlit UI. `build_description` is shared by the page and the export so they always agree. |
| [classes/train_model.py](classes/train_model.py) | Fits the logistic regression with statsmodels, plus backward elimination and the held-out AUC. |
| [utils/utils.py](utils/utils.py) | `drop_outlier_rows`: the IQR outlier filter used before training and analysis. |
| [classes/data_source.py](classes/data_source.py) | `Model`: computes per-feature contributions (`weight_contributions`) and the wording thresholds (`calulcate_threshold`, `risk_thresholds`). |
| [classes/description.py](classes/description.py) | `IndividualDescription`: builds the synthesized text and the LLM prompt, and calls the LLM. |
| [classes/visual.py](classes/visual.py) | `DistributionModelPlot`: the Plotly contribution plot. |
| [utils/sentences.py](utils/sentences.py) | Maps a value and thresholds to a phrase. |

The LLM prompt is assembled in `Description.setup_messages` from:

1. a system message and the instruction in `IndividualDescription.get_prompt_messages`,
2. background question-and-answer pairs from `data/describe/Anuerysm.xlsx`, explaining clinical terms to a patient. These were written for an earlier aneurysm model, so they cover blood pressure and age but not cholesterol, glucose, BMI, smoking, alcohol or activity,
3. worked examples of synthesized text → summary from `data/gpt_examples/Cardio.xlsx`,
4. the synthesized text for the selected individual.

To adapt the app to another outcome, replace the files in steps 2 and 3 with examples for your domain.

LLM summaries are cached per synthesized text for the session, so the LLM is only called when the text changes.

## Credits and licence

This project grew out of [TwelveGPT Educational](https://github.com/soccermatics/twelve-gpt-educational) by Matthias Green, David Sumpter and Ágúst Pálmason Morthens. The logistic regression wordalisation was developed by Beimnet Girma and David Sumpter.

Released under the GNU GPL v3 (see [LICENSE](LICENSE)), as required for a derivative of TwelveGPT Educational.
