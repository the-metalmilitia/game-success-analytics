import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

        cols = ['rating', 'ratings_count', 'metacritic', 'reviews_count']

        for col in cols:
            balance = self.df[col].value_counts(normalize=True)
            skew = self.df[col].skew()
            print(f'\n{col}:')
            print(f'\nBalance:')
            print(balance.head(10))
            print(f'\nSkew:{skew:.2f}')


    def correlation_matrix(self, path: str):
        selected_cols = ['rating', 'ratings_count', 'playtime', 'reviews_count', 'added_status_yet', 'added_status_beaten', 'added_status_toplay']
        corr = self.df[selected_cols].corr(numeric_only=True)
        print(corr)

        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.savefig(f'{path}/figures/corr_heatmap.png')
        plt.show()


    def business_observations_visualizations(self, path: str):
        ratings = self.df['rating'].value_counts()
        genres = self.df['genres'].value_counts().head(10)
        missing_values = self.df.isnull().sum()
        platforms = self.df['platforms'].value_counts()

        separator = '||'

        genres_dict = {}
        for all_genres in genres.index:
            if separator in all_genres:
                genre = all_genres.split(separator)
                for g in genre:
                    if g in genres_dict:
                        genres_dict[g] += genres[all_genres]
                    else:
                        genres_dict[g] = genres[all_genres]

        platforms_dict = {}
        for plats in platforms.index:
            if separator in plats:
                plat = plats.split(separator)
                for p in plat:
                    if p in platforms_dict:
                        platforms_dict[p] += platforms[plats]
                    else:
                        platforms_dict[p] = platforms[plats]


        print(platforms_dict)

        fig, axes = plt.subplots(2, 2, figsize=(25,25))
        axes[0, 0].bar(ratings.index, ratings)
        axes[0, 0].set_xlabel('Ratings')
        axes[0, 0].set_ylabel('Counts')
        axes[0, 0].set_title('Ratings Distribution')

        axes[0, 1].barh(list(genres_dict.keys()), list(genres_dict.values()))
        axes[0, 1].set_xlabel('Counts')
        axes[0, 1].set_ylabel('Genres')
        axes[0, 1].tick_params(axis='y', labelsize=8)
        axes[0, 1].set_title('Top 10 Genres')

        axes[1, 0].barh(missing_values.index, missing_values)
        axes[1, 0].set_xlabel('Count')
        axes[1, 0].set_ylabel('Columns')
        axes[1, 0].tick_params(axis='y', labelsize=8)
        axes[1, 0].set_title('Missing Values Per Column')


        axes[1, 1].barh(list(platforms_dict.keys()), list(platforms_dict.values()))
        axes[1, 1].set_xlabel('Platforms')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].tick_params(axis='y', labelsize=6)
        axes[1, 1].set_title('Platforms Distribution')

        plt.grid(True)
        plt.savefig(f'{path}/figures/eda_viz.png')
        plt.show()

        missing_values_sorted = missing_values.sort_values(ascending=False)
        with open(f'{path}/reports/business_observation.md', 'w') as file:
            file.write(f"""
A correlation matrix created for a few selected columns provides a great picture where reviews are very closely related to status_beaten, but also status_yet.
On the other hand, playtime seems to have very less to do with ratings or even anything else.

In other observations, the first figure in the eda_viz.png shows the ratings skewed to right with most ratings falling around {ratings.idxmax()} stars, while the lowest ratings around {ratings.idxmin()}.

In genres, {max(genres_dict, key=genres_dict.get)} turns out to be the most popular choice for developers and {min(genres_dict, key=genres_dict.get)} as the least.

{max(platforms_dict, key=platforms_dict.get)} has been the most favourite choice of the developers to launch a game and {min(platforms_dict, key=platforms_dict.get)} has not seen to much of an action.

Coming to the point of missing values, the 3 following have the most missing values, given with the missing values count:
{missing_values_sorted.head(3)}
            """)