from pathlib import Path
from datetime import datetime
from sklearn.metrics import r2_score
from utils.reportwriter import ReportWriter
from utils.config import GAME_PRICE_BIN, GAME_PRICE_LABELS
from utils.util_functions import literal_values_per_row, wilson_lower_bound, parse_average_owners
from sklearn.preprocessing import MultiLabelBinarizer
from typing import cast
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.reportwriter import ReportWriter
from utils.model_saver import Model_Saver


class FeatureEngineering:

    model_saver = Model_Saver()

    def __init__(self, target_cleaned_df_path: Path, report_path: Path) -> None:
        self.reportWriter = ReportWriter(report_path)

        target_cleaned_df = pd.read_csv(target_cleaned_df_path)
        self.df = target_cleaned_df

        self.game_features = pd.DataFrame()

#Binning prices.
    def bin_prices(self):
        self.df['price_category'] = pd.cut(self.df['price'], bins=list(GAME_PRICE_BIN), labels=list(GAME_PRICE_LABELS))
        self.df['price_category'] = self.df['price_category'].apply(lambda x: x if isinstance(x, str)
                                                                    else "MicroIndie")
        print('\nAdded price_category')
        print(self.df[['price', 'price_category']].head())
        mlb = MultiLabelBinarizer(classes=GAME_PRICE_LABELS)
        binarized_array = cast(np.ndarray, mlb.fit_transform(self.df['price_category'].apply(lambda x: [x])))
        self.model_saver.save_artifact(mlb, 'bin_prices')
        new_df = pd.DataFrame(binarized_array, columns=mlb.classes_, index=self.df.index).add_prefix(f'{'price_category'}_')
        print(f'{'price_category'} column count: {len(new_df.columns)}')
        self.df = self.df.join(new_df).drop(columns=['price_category'])

        self.reportWriter.append_to_report("""
\n# Price Bins

Prices have been binned on the basis of standard price divisions on Steam. The bins are as given below:
Free to Play: $0
Micro Indie: $0.1 - $9.9
Standard Tier: $10.0 - $19.9
Upper Tier: $20.0 - $29.99
Premium Tier: $30 and above

The value will be stored in price_category column and will be fed to the correlation matrix and the model along with the prices.
""")

#Date parsing from string to year, month and game age.
    def parse_date(self):
        self.df = self.df.dropna(subset=['release_date'], axis=0)
        self.df['release_year'] = self.df['release_date'].apply(lambda x: datetime.strptime(str(x), "%Y-%m-%d").year)
        self.df['release_month'] = self.df['release_date'].apply(lambda x: datetime.strptime(str(x), "%Y-%m-%d").month)
        self.df['game_age'] = self.df['release_date'].apply(lambda x: (datetime.now() - datetime.strptime(str(x), "%Y-%m-%d")).days)
        print('\nAdded release_year and release_month')
        print(self.df[['release_date', 'release_month', 'release_year', 'game_age']].head())
        self.df = self.df.drop(columns=['release_date'])

        self.reportWriter.append_to_report("""
\n\n# Dates

The dates have been divided into game year, game month and game age.

While game year might not prove useful for a game still under development to know if it could be
successful, game month could be a useful parameter to see if releasing at the right month in a year could influence
commercial or critical success. For example, near Holiday season, black friday, etc.
Game age would also be a good parameter to check if a game gets more successful as it ages, or people tend to forget it and move on.

The dates are stored in release_month, release_year and game_age columns.
""")



#Counting literals in list cells (genres, tags, categories, supported languages)
    def count_literals_per_row(self, column_names: list):
        for col in column_names:
            literal_values_per_row(self.df, col)
            print(f'{col + '_count Max:'}{self.df[col + '_count'].max()}')

        print('\nLiterals Per Row')
        print(self.df.head(5))

        self.reportWriter.append_to_report("""
\n\n# Literals and Their Counts

Genres, tags, categories and supported languages are in lists in each row. 
A column has been added for each of these to keep the count of genres, categories, etc. and fed to the model,
so that a correlation can be seen between these lists and their effect on commercial and critical success.
In actual feature importance, these counts seem to take away the work of actual genre, tag and category and hence, will be ignored.
A count of Supported languages however, will be fed in, to see if support of larger number of languages can influence success.

The columns are genres_count, categories_count, tags_count and supported_languages_count.
""")



#Review Score through Wilson Lower Bound formula, applied with Log1p for a proper scaling.
    def calculate_review_score(self):
        self.df['review_score'] = self.df.apply(lambda x: wilson_lower_bound(x['positive'], x['negative']), axis=1)
        print('\nReview Score')
        print(self.df[['positive', 'negative', 'review_score']].head())

        self.reportWriter.append_to_report("""
\n\n# Review Score

Postive and Negative scores are fed into a wilson Lower Bound formula to get a review scoring,
and applied with log1p internally to scale the review score for all the rows.
Positive and negative review count columns will be removed when fed to the model and only the final
review score will be accounted.

The score is stored in review_score column.
""")



#Parsing average owners from estimated owners range strings.
    def parse_avg_owners(self):
        self.df['avg_owners'] = self.df['estimated_owners'].astype("string").apply(lambda x: parse_average_owners(x))
        self.df = self.df.dropna(subset=['avg_owners'], axis=0)
        print('\nAverage Owner Count')
        print(self.df[['estimated_owners', 'avg_owners']].head())
        self.df = self.df.drop(columns='estimated_owners')

        self.reportWriter.append_to_report("""
\n\n# Average Owners

Since estimated_owners is a string of range of customers (ex. '0-20000'), this is parsed into an integer value,
that is an average between the floor and the ceiling values in the string. Some rows have NaNs, which have been removed.

The average is stored in avg_owners column, which acts as a Y axis for the commercial success model.
It will also be fed to the correlation matrix to have a look at the factors immediately affecting it.

""")



#Dropping columns from the final dataset.
    def drop_columns(self, col_names: list):
        self.df = self.df.drop(columns=col_names)


#Meant for some investigation before multilabel binarizing. Not called in final code.
    def pre_binarizing_investigation(self, column_name: str):
        self.df[column_name] = self.df[column_name].apply(self._parse_dirty_categoricals)
        tags = self.df[column_name].explode().value_counts()
        usable_tags = tags[tags >= 12].index
        print(usable_tags)


#Column encoding
    def encode_column(self, column_names: list, threshold = 0):
        for col in column_names:
            self.df[col] = self.df[col].apply(self._parse_dirty_categoricals)

            val_counts = self.df[col].explode().value_counts()
            usable_classes = val_counts[val_counts >= threshold].index

            self.df = self.df.dropna(subset=[col], axis=0)
            mlb = MultiLabelBinarizer(classes=usable_classes)
            binarized_array = cast(np.ndarray, mlb.fit_transform(self.df[col]))
            self.model_saver.save_artifact(mlb, col)
            new_df = pd.DataFrame(binarized_array, columns=mlb.classes_, index=self.df.index).add_prefix(f'{col}_')
            print(f'{col} column count: {len(new_df.columns)}')
            self.df = self.df.join(new_df).drop(columns=[col])



    def _save_feature_csv(self, path: Path):
        self.df.to_csv(path, index=False)

#Feature Importance through Random Forest Regression models.
    def feature_importance(self, path: Path):
        #A = model A that will target predicting estimated_owners
        #B = model B that will target predicting review_score

        self.reportWriter.append_to_report("""
        \n\n# Encoding strategy
        
Certain columns which are labels, viz, genres, categories and tags have been encoded into multiple binary columns
through Multilabel Binarizer. This results in a pretty large number of final columns, but it's workable and in fact quite required.
        
Tags have been encoded with a threshold value, as there are a lot of tags which would blow up the column count.
So a tag is only encoded if it is found in x number of games or more, x being the threshold value.
        """)

        self.reportWriter.append_to_report("""
\n\n# Feature Importance

Two Random Forest Regression models are trained to calculate feature importance.
Both models take all the other values in the dataset except avg_owners and review_score as their X axis features.

Model A takes avg_owners as Y axis, which is what we have to predict for commercial success.
Model B takes review_score as Y axis, which is to be predicted for critical success.

An R2 accuracy of more than 65 or 70, should be a decent acceptance point.
""")

        rf_A = RandomForestRegressor(n_estimators=100, random_state=42, verbose=2)
        rf_B = RandomForestRegressor(n_estimators=100, random_state=42, verbose=2)
        target_col_A = 'avg_owners'
        target_col_B = 'review_score'

        X = self.df.drop(columns=[target_col_A, target_col_B])
        Y_A = self.df[target_col_A]
        Y_B = self.df[target_col_B]

        if 'Unnamed: 0' in X.columns:
            X = X.drop(columns='Unnamed: 0')

        if 'game_age' in X.columns:
            X = X.drop(columns='game_age')

        print('\nX Columns:')
        print(X.columns)

        self.model_saver.save_artifact(list(X.columns), 'feature_columns')

        if path:
            self._save_feature_csv(path)

        self._feature_importance_model_training('Commercial Success', X, Y_A, rf_A)
        self._feature_importance_model_training('Critical Success', X, Y_B, rf_B)
        


    def _feature_importance_model_training(self, model_name: str, x: pd.DataFrame, y: pd.Series, rf_regressor: RandomForestRegressor):
        X_Train, X_Test, Y_Train, Y_Test = train_test_split(x, y, test_size=0.2, random_state=42)
        rf_regressor.fit(X_Train, Y_Train)
        Y_Pred = rf_regressor.predict(X_Test)
        accuracy = r2_score(Y_Test, Y_Pred)
        features = rf_regressor.feature_names_in_
        importances = rf_regressor.feature_importances_
        print(f'Model {model_name} RandomForest Accuracy: {accuracy}')
        print(f'Model {model_name} Feature Importance:')
        self.reportWriter.append_to_report(f"""
        \n\n## Model {model_name} Feature Importance Results
        Top 10 Features:
        """)

        features_zip = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
        for i, (feat, imp) in enumerate(features_zip):
            feature_data = f'\n{feat}: {imp*100:.3f}%'
            print(feature_data)
            if i < 10:
                self.reportWriter.append_to_report(feature_data)



#Correlation matrix
    def correlation_matrix(self, column_names: list, path: Path):
        corr_df = self.df[column_names]
        corr_matrix = corr_df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.tight_layout()
        plt.savefig(path)
        plt.show()


#private method to parse dirty categorical lists.
    def _parse_dirty_categoricals(self, x):
        literal = ast.literal_eval(x)
        if not literal:
            return []
        elif isinstance(literal, dict):
            return []
        elif isinstance(literal, list) and len(literal) > 0:
            if isinstance(literal[0], dict):
                return []
            else:
                return literal
