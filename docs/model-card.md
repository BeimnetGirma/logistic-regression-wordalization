# Model card: Logistic Regression Wordalisation

This app explains a logistic regression model's prediction for one individual in words. It computes how much each feature contributes to that individual's predicted risk, turns those contributions into a rule-based text (the *synthesized text*), and asks a large language model (LLM) to rewrite the text as a short summary addressed to the individual.

This card follows the structure of [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993). It covers the demo configuration: the cardiovascular dataset bundled in `data/medical/`, with outliers removed and a model chosen by stepwise backward elimination in the app.

Jump to section:

- [Intended use](#intended-use)
- [Data](#data)
- [Model](#model)
- [Evaluation](#evaluation)
- [Ethical considerations](#ethical-considerations)
- [Caveats and recommendations](#caveats-and-recommendations)

## Intended use

**Primary use: research and teaching.** The app is a testbed for *wordalisation*: studying how different ways of putting a statistical model into words (scale, reference point, risk age, categorical baseline) change what a reader takes away. The 24-configuration export exists to compare these wordings side by side.

**Out of scope:**

- **Clinical use.** The app must not be used to inform the diagnosis, treatment or risk assessment of a real patient. The demo model has not been clinically validated (see [Evaluation](#evaluation)), and the LLM summary gives lifestyle advice in the second person that no clinician has reviewed.
- **Data analysis.** The app explains a fitted model. It doesn't check whether that model is any good, and it doesn't test causal claims. A large contribution means the model weights that feature heavily, not that changing it would change the outcome.
- **Free-form questions.** There is no chat. The LLM only rewrites the synthesized text.

## Data

### Demo dataset

`data/medical/training_data_70000_Ind.csv` has 70,000 rows and 11 columns. It appears to be derived from the publicly available Kaggle *Cardiovascular Disease dataset*, with height and weight replaced by BMI and age in years. **The source and licence should be confirmed before the data is redistributed.**

| Column | Meaning | Values in the data |
|---|---|---|
| `age` | Age in years | 30–65, median 54 |
| `gender` | 1 = female, 2 = male | 65% female |
| `bmi` | Body mass index | 3.5–298.7, median 26.4 |
| `ap_hi` | Systolic blood pressure | −150 to 16,020, median 120 |
| `ap_lo` | Diastolic blood pressure | −70 to 11,000, median 80 |
| `cholesterol` | 1 normal, 2 above normal, 3 well above normal | |
| `gluc` | Glucose, coded like cholesterol | |
| `smoke`, `alco` | Self-reported smoking, alcohol intake (0/1). **Not used in the demo model** (see [Model](#model)) | 9%, 5% |
| `active` | Self-reported physical activity (0/1) | 80% |
| `cardio` | Target: cardiovascular disease present | 50% |

Things to know about this data:

- **The raw file contains impossible values.** 47 rows have systolic pressure ≤ 0 or > 250, 1,234 rows have diastolic higher than systolic, and BMI ranges from 3.5 to almost 300. See [Outlier removal](#outlier-removal) for how these are handled.
- **It is balanced (50% cases).** Predicted probabilities and the intercept therefore don't reflect the prevalence in any real population. Everything is relative to "other patients in this dataset", not to the general public.
- **Ages 30–65 only.** Risk ages are kept within this range.
- **Lifestyle variables are self-reported** and coarse (yes/no).

### Outlier removal

Before training, and again before computing contributions, the app removes every row where any model feature lies outside `[Q1 − 3·IQR, Q3 + 3·IQR]` (`drop_outlier_rows` in `utils/utils.py`). Features with an IQR of 0 (glucose, activity) are skipped.

On the demo data this removes **1,649 of 70,000 rows** (2.4%), leaving 68,351. Rows flagged per feature (a row can be flagged by more than one): diastolic 1,136, systolic 288, BMI 282. After cleaning:

- systolic pressure is 70–200 and diastolic 50–120,
- BMI is 7.0–49.2 (the low end is still implausible),
- 46 rows still have diastolic higher than systolic,
- prevalence is 49.4% and 65% are female, almost unchanged.

The rule is statistical, not clinical. It removes the impossible readings, but it would also remove genuine extreme values in other datasets.

### Text prompts

- `data/gpt_examples/Cardio.xlsx`: four worked examples (synthesized text → summary) that set the LLM's tone and format.
- `data/describe/Anuerysm.xlsx`: eleven Q&A pairs explaining clinical terms to a patient. These were written for an earlier aneurysm model and cover terms such as systolic blood pressure, LDL and aortic stenosis. They don't explain cholesterol codes, glucose, BMI, smoking, alcohol or physical activity.

## Model

### Logistic regression

The model is a logistic regression fit with statsmodels on a random 80% of the cleaned data (`random_state=42`).

**Smoking and alcohol are left out of the demo model by choice.** In earlier fits both came out as protective (β ≈ −0.14 and −0.18, p < 0.001), which contradicts clinical knowledge and is most likely due to self-reporting and how the dataset was collected. The app leaves them unselected by default for the demo data.

The remaining features are chosen by stepwise backward elimination: the feature with the highest p-value is removed while that p-value is above 0.05. On the demo data this removes only **gender** (p = 0.31), leaving seven features. Categorical features are entered as their integer codes, not one-hot encoded, so for example the effect of cholesterol is assumed to be the same size going from 1 → 2 as from 2 → 3.

Coefficients of the demo model after stepwise elimination (intercept −11.87; all p < 0.001):

| Feature | β | Direction |
|---|---|---|
| age | 0.051 per year | ↑ risk |
| bmi | 0.028 per unit | ↑ |
| ap_hi | 0.055 per mmHg | ↑ |
| ap_lo | 0.013 per mmHg | ↑ |
| cholesterol | 0.509 per level | ↑ |
| gluc | −0.118 per level | ↓ |
| active | −0.260 | ↓ |

**Higher glucose still comes out as protective.** This contradicts clinical knowledge and may be due to confounding with cholesterol, which is correlated with glucose in this data (r = 0.45). The app reports the coefficient faithfully, so it will tell a patient with high glucose that it *lowers* their risk. This is the main known failure of the demo model, and a good example of why a model should be checked before it is put into words.

### Contributions and wording

For feature *j* and individual *i* the contribution is `c_ij = x_ij·β_j − mean_i(x_ij·β_j)` on the log-odds scale, or `exp(c_ij)` on the odds-ratio scale. Contributions are put into words by comparing them with thresholds at ±0.5σ and ±1σ of a reference distribution. The **Risk Reference** setting picks that distribution: the feature's own, the most variable feature's, the average over features, or age's. The full definitions are in [axes_implementation_audit.md](axes_implementation_audit.md).

**Risk age** expresses the summed contributions of all non-age features as the number of years of ageing that would have the same effect on the log-odds: `age + Σ c_ij / β_age`.

### Language model

The summary is generated by whichever LLM is configured in `secrets.toml` (an Azure OpenAI deployment or Gemini), at temperature 1. The prompt asks for 4–7 sentences in the second person: overall risk, protective factors, risk factors, and next steps. The LLM sees only the synthesized text and the example files, not the raw data.

## Evaluation

- **Discrimination.** The app reports the AUC on the 20% held-out split. It is 0.788 for the demo model. Calibration is not evaluated, and on a balanced dataset it wouldn't carry over to a real population anyway.
- **Faithfulness of the synthesized text.** This text is deterministic and follows the formulas above.
- **Faithfulness of the LLM summary is not evaluated systematically.** The summary can add advice, reorder or merge factors, change hedging, or (as with any LLM) state things that aren't in the synthesized text. The 24-configuration export is a starting point for evaluating this by hand.
- **The wording settings are not all distinct.** On the odds-ratio scale the text gives percentage changes directly, so the Risk Reference setting has no effect on the text. Of the 24 exported configurations, 10 give distinct texts.

## Ethical considerations

- **It reads like medical advice.** The summary greets the reader, gives them a "risk age" and suggests what to do. Readers may take this as personal medical advice, especially when it contradicts what they've been told. The app shows a "not medical advice" caption, but that is weak protection on its own.
- **The demo model is wrong in places** (glucose), and leaving out smoking and alcohol means the text never mentions them, even for a smoker. Fluent, confident text makes a wrong model more persuasive, not less.
- **Gender is binary in the data.** It is dropped by stepwise elimination in the demo model, but it would be described as a risk factor in a model that keeps it.
- **Uploaded data is sent to the LLM.** The synthesized text, which includes the individual's feature values, goes to the configured LLM provider. Don't upload identifiable patient data unless your agreement with the provider allows it.

## Caveats and recommendations

- Check the coefficients for clinical plausibility before wordalising a model. Check what the outlier rule removes in your data: it is statistical, not clinical.
- Treat categorical features with care. Integer coding assumes equal steps between levels, and the "Specific Group" baseline assumes the lowest code is the natural reference.
- Replace `data/describe/Anuerysm.xlsx` with background Q&As that match your features.
- Have a domain expert review a sample of LLM summaries for each configuration before using the app in any study with participants.
