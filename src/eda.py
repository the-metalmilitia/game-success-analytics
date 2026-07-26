import pandas as pd
import numpy as np

class EDA:

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def analyse_numeric_overview(self):
        numeric_columns = self.df.select_dtypes(include=['number']).columns.to_list()
        print('\nNumeric Columns:')
        print(f'{'Col':<25} {'Min':>15} {'Max':>15} {'Mean':>15} {'Median':>15} {'std':>15} {'Lower Outliers':>25} {'Upper Outliers':>25}')
        print('-'*150)
        for col in numeric_columns:
            min = self.df[col].min()
            max = self.df[col].max()
            mean = self.df[col].mean()
            median = self.df[col].median()
            std = self.df[col].std()

            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_outlier = Q1 - 1.5 * IQR
            upper_outlier = Q3 + 1.5 * IQR

            print(f'{col:<25} {min:>15} {max:>15} {mean:>15.2f} {median:>15.2f} {std:>15.2f} {lower_outlier:>25.2f} {upper_outlier:>25.2f}')


    def analyse_missing_data(self):
        cols = self.df.isnull().columns.to_list()

        print('\nMissing Values in Each Column:')
        for col in cols:
            missing = self.df[col].isnull().sum()
            total = len(self.df[col])
            print(f'{col:<25}: {missing}/{total}')


    def analyse_duplicate_data(self):
        duplicates = self.df[self.df.duplicated()]
        print(duplicates)


    def analyse_categorical_data(self):
        cols = self.df.select_dtypes(include=['object']).columns.to_list()
        print('\nCategorical Columns:')
        for col in cols:
            # print(col)
            top_items = self.df[col].value_counts()
            most_common = top_items.head(1).index.to_list()
            most_common_count = top_items.iloc[0]
            rare = top_items.tail(1).index.to_list()
            rare_count = top_items.iloc[-1]
            print(f'\n{col}: Most Common: {most_common}({most_common_count}), Rarest: {rare}({rare_count})')


    def analyse_target_variables(self):
        # print(self.df['rating'].describe())
        # print(self.df['ratings_count'].describe())
        # print(self.df['metacritic'].describe())
        # print(self.df['reviews_count'].describe())

        cols = ['rating', 'ratings_count', 'metacritic', 'reviews_count']

        for col in cols:
            balance = self.df[col].value_counts(normalize=True)
            skew = self.df[col].skew()
            print(f'\n{col}:')
            print(f'\nBalance:')
            print(balance.head(10))
            print(f'\nSkew:{skew:.2f}')


    def correlation_matrix(self):
        print(self.df.corr(numeric_only=True))