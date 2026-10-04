import numpy as np

import classes.data_point as data_point
from utils.utils import drop_outlier_rows


class Model:
    """
    The data for a fitted logistic regression model and the per-feature contributions
    of each individual to their predicted risk.
    """
    data_point_class = data_point.Individual

    def __init__(self):
        self.df = None
        self.parameters = None

    def set_data(self, df_raw, parameters, categorical_interpretations):
        self.df = df_raw
        self.parameters = parameters
        self.categorical_interpretations = categorical_interpretations

    def process_data(self):
        df_raw = self.df
        if not any(col.lower() == 'id' for col in df_raw.columns):
            df_raw['ID'] = df_raw.index
        else:
            df_raw.columns = [col if col.upper() != 'ID' else 'ID' for col in df_raw.columns]

        self.df = df_raw

        parameters = self.parameters

        self.intercept = parameters[parameters['Parameter'] == 'Intercept']['Value'].values[0]
        # Drop intercept
        self.parameters = parameters[parameters['Parameter'] != 'Intercept']

        # Remove rows with statistical outliers (IQR method) in the model's features before analysis
        feature_columns = [p for p in self.parameters['Parameter'] if p in df_raw.columns]
        n_before = len(df_raw)
        df_raw, self.dropped_counts = drop_outlier_rows(df_raw, columns=feature_columns)
        self.rows_removed = n_before - len(df_raw)

        self.df = df_raw
        return df_raw

    def weight_contributions(self, scale='linear', baseline='population'):
        # Start from the raw features every time: this is called on each rerun and
        # with different configurations, so stale contribution columns (in particular
        # a previous total_risk_contribution) must not leak into the new totals.
        df = self.df.drop(columns=[col for col in self.df.columns if col.endswith('_contribution')])
        categorical = self.categorical_interpretations or {}

        for i, row in self.parameters.iterrows():
            raw = df[row['Parameter']] * row['Value']
            is_categorical = row['Parameter'] in categorical
            
            if scale == 'linear':
                if baseline == 'population':
                    # Same formula for both — mean-centring works for continuous
                    # AND for categorical (mean is just the prevalence × β)
                    df[row['Parameter'] + '_contribution'] = raw - raw.mean()
                    
                elif baseline == 'specific':
                    if is_categorical:
                        # Reference category (minimum value) scores exactly 0
                        # e.g. cholesterol: 1→0, 2→β, 3→2β
                        min_raw = (df[row['Parameter']].min()) * row['Value']
                        df[row['Parameter'] + '_contribution'] = raw - min_raw
                    else:
                        # Continuous has no reference group — population average is
                        # the only meaningful anchor, so we still centre
                        df[row['Parameter'] + '_contribution'] = raw - raw.mean()
                    
            elif scale == 'odds':
                if baseline == 'population':
                    if is_categorical:
                        # Population average for categorical:
                        # e^(x·β - mean(x·β)) — mean is prevalence-weighted
                        df[row['Parameter'] + '_contribution'] = np.exp(raw - raw.mean())
                    else:
                        # method-4: same formula for continuous
                        df[row['Parameter'] + '_contribution'] = np.exp(raw - raw.mean())
                        
                elif baseline == 'specific':
                    if is_categorical:
                        # Reference category (minimum value) scores exp(0)=1 (neutral)
                        # e.g. cholesterol: 1→exp(0)=1, 2→exp(β), 3→exp(2β)
                        min_raw = (df[row['Parameter']].min()) * row['Value']
                        df[row['Parameter'] + '_contribution'] = np.exp(raw - min_raw)
                    else:
                        # Continuous with specific baseline is ambiguous —
                        # no reference group exists, so fall back to population average
                        df[row['Parameter'] + '_contribution'] = np.exp(raw - raw.mean()) 


        self.parameter_explanation = self.parameters.set_index('Parameter')['Explanation'].to_dict()

        self.df = df

        # total risk: product of odds-ratio contributions (multiplicative), sum in linear (log-odds) space
        contribution_cols = [col for col in df.columns if col.endswith('_contribution')]
        if scale == 'odds':
            df['total_risk_contribution'] = df[contribution_cols].prod(axis=1)
        else:
            df['total_risk_contribution'] = df[contribution_cols].sum(axis=1)


    def calulcate_threshold(self, odds_space=False, column=None):
        df = self.df
        contribution_columns = [col for col in df.columns if '_contribution' in col and col != 'total_risk_contribution']

        log_df = np.log(df[contribution_columns].clip(lower=1e-9)) if odds_space else df[contribution_columns]
        self.std_contributions = log_df.std()
        if column is not None and column in log_df.columns:
            most_variable_column = column
        else:
            most_variable_column = self.std_contributions.idxmax()
        series = log_df[most_variable_column]
        mean, sigma = series.mean(), series.std()
        log_thresholds = [mean + i * sigma for i in [-1, -0.5, 0.5, 1]]
        thresholds = [round(np.exp(t), 2) if odds_space else round(t, 2) for t in log_thresholds]

        plot_range = [thresholds[0], thresholds[-1]]
        min_value, max_value = df[most_variable_column].min(), df[most_variable_column].max()
        return thresholds, plot_range, [min_value, max_value]

    def risk_thresholds(self, odds_space=False):
        bins_dict = {}
        for col in self.df.columns:
            if 'contribution' in col:
                data = self.df[col]
                
                if odds_space:
                    # Work in log space, then exponentiate back
                    # Assumes contributions are already in odds space (positive, multiplicative)
                    log_data = np.log(data.clip(lower=1e-9))  # guard against log(0)
                    log_mean = log_data.mean()
                    log_std = log_data.std()
                    bins = [round(np.exp(log_mean + i * log_std), 4) 
                            for i in [-1, -0.5, 0.5, 1]]
                else:
                    mean = data.mean()
                    std = data.std()
                    bins = [round(mean + i * std, 4) for i in [-1, -0.5, 0.5, 1]]
                
                bins_dict[col] = bins
        return bins_dict
        


    def to_data_point(self, columns=None ) -> data_point.Individual:
        
        id = self.df['ID'].iloc[0]

        #Reindexing dataframe
        self.df.reset_index(drop=True, inplace=True)
        self.df = self.df.drop(columns=[col for col in columns if col in self.df.columns])
        
        # Convert to series
        ser_metrics = self.df.squeeze()
        return self.data_point_class(id=id,ser_metrics=ser_metrics)
