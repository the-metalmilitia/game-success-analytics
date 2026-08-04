import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
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


    def extract_literal_values(self, column_name: str) -> dict:
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

        return values_dict

    def parse_average_owners(self, owner_string: str):
        # owner_string = str(owner_numbers)
        owner_substrings = owner_string.split(' - ')
        if len(owner_substrings) > 1 and owner_substrings[0] and owner_substrings[1]:
            floor_count = int(owner_substrings[0])
            ceiling_count = int(owner_substrings[1])
            avg_owners = (floor_count + ceiling_count) / 2
            return np.log1p(avg_owners)


    def wilson_lower_bound(self, pos, neg, z=1.96):
        n = pos + neg
        if n == 0:
            return 0
        p = pos / n
        return (p + z**2/(2*n) - z * np.sqrt((p*(1-p) + z**2/(4*n)) / n)) / (1 + z**2/n)

    def save_figure(self, path: str):
        plt.tight_layout()
        plt.savefig(path)
        plt.show()


    def business_observations_visualizations(self, path: str):

        ## Key business observations for columns Tag, Developers, Publishers and Categories
        tags = self.extract_literal_values('tags')
        tags = dict(sorted(tags.items(), key=lambda x: x[1], reverse=True))
        print('\nMost Common Tags: ', list(tags.keys())[:20])

        developers = self.extract_literal_values('developers')
        developers = dict(sorted(developers.items(), key=lambda x: x[1], reverse=True))
        print('\nMost Common Developers: ', list(developers.keys())[:20])

        publishers = self.extract_literal_values('publishers')
        publishers = dict(sorted(publishers.items(), key=lambda x: x[1], reverse=True))
        print('\nMost Common Publishers: ', list(publishers.keys())[:20])

        self.plot_missing_values(path)
        estimated_owners_games = self.plot_actual_favourites(path)
        self.plot_top_fav_genres(path, estimated_owners_games)
        self.plot_most_common_categories(path)
        self.business_observations_report(path)



    def plot_top_fav_genres(self, path, estimated_owners_games):
        #top genres based on actual favourites
        estimated_owners_games['genres'] = estimated_owners_games['genres'].apply(lambda x: tuple(ast.literal_eval(x)))
        estimated_owners_games['categories'] = estimated_owners_games['categories'].apply(lambda x: tuple(ast.literal_eval(x)))

        estimated_owners_games = estimated_owners_games[estimated_owners_games['genres'].apply(lambda x: len(x) > 0 and not isinstance(x[0], dict))]
        estimated_owners_games = estimated_owners_games[estimated_owners_games['categories'].apply(lambda x: len(x) > 0 and not isinstance(x[0], dict))]

        fav_per_genre = estimated_owners_games.explode('genres').groupby('genres')['fav_score'].median().sort_values(ascending=False)
        fav_per_category = estimated_owners_games.explode('categories').groupby('categories')['fav_score'].median().sort_values(ascending=False)

        ax = fav_per_genre.head(20).plot.bar()
        ax.set_xlabel('Genres')
        ax.set_ylabel('Fav Score')
        ax.set_title('Top Favourite Genres')
        ax.tick_params(axis='x', labelsize=8)
        self.save_figure(f'{path}/figures/EDA_Top_Favourite_Genres.png')

        ax = fav_per_category.head(20).plot.bar()
        ax.set_xlabel('Categories')
        ax.set_ylabel('Fav Score')
        ax.set_title('Top Favourite Categories')
        ax.tick_params(axis='x', labelsize=8)
        self.save_figure(f'{path}/figures/EDA_Top_Favourite_Categories.png')

    def business_observations_report(self, path):
        with open(f'{path}/reports/business_observation.md', 'w', encoding='utf-8') as file:
            file.write(f"""
A correlation matrix created for a few selected columns provides a great picture where ratings are closely associated with recommendations and peak CCU, but almost not at all with playtime.

Among missing values, metacritic_url has the most missing values which means the metacritic score cannot unfortunately be used in figuring out the most played, most loved games in this dataset.

There are 2 separate columns for ratings, one for positive and the other for negative, which can be combined or used separately to calculate the best games and the worst.

There are separate columns for tags, genres and categories, which means we can do a more elaborate judgement on what all does one need for making a great game.

On applying some log calculations and wilson score to the average of positive and negative reviews along with average number of owners per game, Crab Game and Black Myth: Wukong turn out to be most favourite at the moment.
Top genres include keywords such as Web Publishing, Photo Editing, Action, Indie, RPG, etc. along with training, design and production tools, as well as, interestingly, nudity & sexual content.

The top most category by far is any game that has Valve's Anti-Cheat enabled. This is followed by Steam's trading cards and remote play on tablet. There are multiple remote playing, VR and Steam associated features included.
On the other hand, if you see Most Common Categories, they do match with the Top Categories somewhere, but not so much. This means that all the top games abide very much by the frameworks set by Steam.
            """)

    def plot_most_common_categories(self, path):
        #most common categories
        categories = self.extract_literal_values('categories')
        categories = dict(sorted(categories.items(), key=lambda x: x[1], reverse=True))
        print('\nMost Common Tags: ', list(categories.keys())[:20])
        categories = dict(sorted(categories.items(), key=lambda x: x[1], reverse=True)[:20])
        plt.barh(list(categories.keys()), list(categories.values()))
        plt.xlabel('Categories')
        plt.ylabel('Count')
        plt.title('Most Common Categories')
        self.save_figure(f'{path}/figures/EDA_Most_Common_Categories.png')

    def plot_missing_values(self, path):
        #missing values
        missing_values = self.df.isnull().sum().sort_values(ascending=False).head(10)
        ax = missing_values.plot.barh()
        ax.set_xlabel('Count')
        ax.set_ylabel('Columns')
        ax.set_title('Top 10 Missing Values Per Column')
        self.save_figure(f'{path}/figures/EDA_Columns_Missing_Values.png')

    def plot_actual_favourites(self, path):
        #Visualizing actual favourite games.
        estimated_owners_games = self.df[['name', 'estimated_owners', 'genres', 'categories', 'positive', 'negative', 'median_playtime_2weeks']].copy()
        estimated_owners_games['estimated_owners_log'] = estimated_owners_games['estimated_owners'].astype("string").apply(lambda x: self.parse_average_owners(x))
        estimated_owners_games['avg_ratings'] = estimated_owners_games.apply(lambda x: self.wilson_lower_bound(x['positive'], x['negative']), axis=1)
        # estimated_owners_games['avg_ratings'] = round(100 * estimated_owners_games['positive'] / (estimated_owners_games['positive'] + estimated_owners_games['negative']), 2)
        estimated_owners_games['avg_ratings_log'] = np.log1p(estimated_owners_games['avg_ratings'])
        estimated_owners_games['median_playtime_2weeks_log'] = np.log1p(estimated_owners_games['median_playtime_2weeks'])

        for col in ['estimated_owners_log', "avg_ratings_log", 'median_playtime_2weeks_log']:
            estimated_owners_games[col + '_norm'] = (estimated_owners_games[col] - estimated_owners_games[col].min()) / (estimated_owners_games[col].max() - estimated_owners_games[col].min())
        
        estimated_owners_games['fav_score'] = estimated_owners_games['avg_ratings_log_norm'] * 0.4 + estimated_owners_games['estimated_owners_log_norm'] * 0.4 + estimated_owners_games['median_playtime_2weeks_log_norm'] * 0.2
        estimated_owners_games = estimated_owners_games.sort_values(by='fav_score', ascending=False)
        plot = estimated_owners_games.head(20).plot.barh(x='name', y='fav_score')
        plot.set_xlabel('Name')
        plot.set_ylabel('Fav Score')
        plot.set_title('Most Favourite Games')
        self.save_figure(f'{path}/figures/EDA_Most_Favourite_Games.png')
        return estimated_owners_games