import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast

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
        print('\n',duplicates)


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

        cols = ['positive', 'negative', 'price', 'achievements', 'peak_ccu', 'average_playtime_forever']

        for col in cols:
            balance = self.df[col].value_counts(normalize=True)
            skew = self.df[col].skew()
            print(f'\n{col}:')
            print(f'\nBalance:')
            print(balance.head(10))
            print(f'\nSkew:{skew:.2f}')


    def correlation_matrix(self, path: str):
        selected_cols = ['positive', 'negative', 'median_playtime_forever', 'peak_ccu', 'genres', 'tags', 'recommendations']
        corr = self.df[selected_cols].corr(numeric_only=True)
        print(corr)

        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.tight_layout()
        plt.savefig(f'{path}/figures/corr_heatmap.png')
        plt.show()


    def column_to_dict_converter(self, column_name: str) -> dict:
        self.df[f'{column_name}_literal'] = self.df[column_name].apply(lambda x: tuple(ast.literal_eval(x)))
        values = self.df[f'{column_name}_literal'].value_counts()

        values_dict = {}
        for all_values in values.index:
            if isinstance(all_values, tuple):
                for val in all_values:
                    if not isinstance(val, dict):
                        if val in values_dict:
                            values_dict[val] += 1
                        else:
                            values_dict[val] = 1

        print('\n', column_name)
        print(values_dict)
        return values_dict


    def business_observations_visualizations(self, path: str):

        ## Key business observations for columns Tag, Developers, Publishers and Categories
        tags = self.column_to_dict_converter('tags')
        tags = dict(sorted(tags.items(), key=lambda x: x[1], reverse=True))

        developers = self.column_to_dict_converter('developers')
        developers = dict(sorted(developers.items(), key=lambda x: x[1], reverse=True))

        publishers = self.column_to_dict_converter('publishers')
        publishers = dict(sorted(publishers.items(), key=lambda x: x[1], reverse=True))

        categories = self.column_to_dict_converter('categories')
        categories = dict(sorted(categories.items(), key=lambda x: x[1], reverse=True))

        ##Visualization of Top 20 Genres with highest ratings (+ive or -ive)
        ratings_by_genres = self.df[['genres', 'positive', 'negative']].copy()
        ratings_by_genres['sum_of_all_ratings'] = ratings_by_genres['positive'] + ratings_by_genres['negative']
        ratings_by_genres = ratings_by_genres.sort_values(by='sum_of_all_ratings', ascending=False)
        ratings_by_genres['avg_ratings'] = round(100 * ratings_by_genres['positive'] / (ratings_by_genres['positive'] + ratings_by_genres['negative']), 2)

        ratings_by_genres['genres'] = ratings_by_genres['genres'].apply(lambda x: tuple(ast.literal_eval(x)))
        ratings_by_genres = ratings_by_genres[ratings_by_genres['genres'].apply(lambda x: len(x) > 0 and not isinstance(x[0], dict))]
        ratings_per_genre = ratings_by_genres.explode('genres').groupby('genres')['avg_ratings'].median().sort_values(ascending=False)

        ax = ratings_per_genre.head(20).plot.bar()
        ax.set_xlabel('Genres')
        ax.set_ylabel('Average Ratings')
        ax.set_title('Top Genres')
        ax.tick_params(axis='x', labelsize=8)
        plt.tight_layout()
        plt.savefig(f'{path}/figures/EDA_Top_Genres.png')
        plt.show()

        ##Visualization of median playtime forever
        playtime = self.df[['median_playtime_forever', 'name']].copy()
        playtime = playtime.sort_values(by='median_playtime_forever', ascending=False).head(20)

        plt.barh(playtime['name'], playtime['median_playtime_forever'])
        plt.xlabel('Median Playtime')
        plt.ylabel('Game')
        plt.title('Top Played Games')
        plt.tight_layout()
        plt.savefig(f'{path}/figures/EDA_Top_PlayedGames.png')
        plt.show()

        #missing values
        missing_values = self.df.isnull().sum().sort_values(ascending=False).head(10)
        ax = missing_values.plot.barh()
        ax.set_xlabel('Count')
        ax.set_ylabel('Columns')
        ax.set_title('Top 10 Missing Values Per Column')
        plt.tight_layout()
        plt.savefig(f'{path}/figures/EDA_Columns_Missing_Values.png')
        plt.show()

        #top categories
        categories = dict(sorted(categories.items(), key=lambda x: x[1], reverse=True)[:20])
        plt.barh(list(categories.keys()), list(categories.values()))
        plt.xlabel('Categories')
        plt.ylabel('Count')
        plt.title('Top Categories')

        plt.tight_layout()
        plt.savefig(f'{path}/figures/EDA_Top_Categories.png')
        plt.show()

        with open(f'{path}/reports/business_observation.md', 'w', encoding='utf-8') as file:
            file.write(f"""

Some of the most common tags are as follows:
{list(tags.keys())[:5]}

The most common developers on the list are:
{list(developers.keys())[:5]}

While the most common publishers are:
{list(publishers.keys())[:5]}

A correlation matrix created for a few selected columns provides a great picture where ratings are closely associated with recommendations and peak CCU, but almost not at all with playtime.

In other observations, the first figure in the visualization shows the top 20 genres, with the top genre containing {list(ratings_per_genre.keys())[0]} as it's keyword.

In top played games, in the next visualization, {playtime['name'].iloc[0]} has the highest playtime at {playtime['median_playtime_forever'].iloc[0]}

Coming to the point of missing values {missing_values.iloc[0]} has the highest missing values

Lastly, the top categories are shows in the last visualization with, {next(iter(categories))} as the top one.
            """)