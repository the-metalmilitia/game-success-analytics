import ast
from datetime import datetime
from utils.config import GAME_PRICE_BIN, GAME_PRICE_LABELS

import pandas as pd
from utils.model_saver import Model_Saver

class GameSuccessInference:
    model_saver = Model_Saver()

    def __init__(self):
        self.commercial_model = self.model_saver.load_commercial()
        self.critical_model = self.model_saver.load_critical()

        self.feature_columns_encoder = self.model_saver.load_artifact('feature_columns')
        self.bin_prices_encoder = self.model_saver.load_artifact('bin_prices')
        self.categories_encoder = self.model_saver.load_artifact('categories')
        self.genres_encoder = self.model_saver.load_artifact('genres')
        self.tags_encoder = self.model_saver.load_artifact('tags')

    @staticmethod
    def _parse_list(values):
        if isinstance(values, str):
            values = ast.literal_eval(values)

        if not isinstance(values, (list, tuple, set)):
            raise ValueError('Expected a list of values')

        return list(values)

    def prepare_features(self, game: dict) -> pd.DataFrame:
        release_date = datetime.strptime(game['release_date'], "%Y-%m-%d")

        genres = self._parse_list(game['genres'])
        categories = self._parse_list(game['categories'])
        tags = self._parse_list(game['tags'])
        languages = self._parse_list(game['supported_languages'])

        row = {
            'price': float(game['price']),
            'release_month': release_date.month,
            'supported_languages_count': len(languages),
        }

        features = pd.DataFrame([row])

        price_category = pd.cut([row['price']], bins=list(GAME_PRICE_BIN), labels=list(GAME_PRICE_LABELS))

        if pd.isna(price_category):
            price_category = 'Micro Indie'

        price_features = self.bin_prices_encoder.transform([[price_category]])
        price_df = pd.DataFrame(price_features, columns=[f'price_category_{label}'
                                                         for label in self.bin_prices_encoder.classes_],)

        features = pd.concat([features, price_df], axis=1)

        genres_features = self.genres_encoder.transform([[genres]])
        categories_features = self.categories_encoder.transform([[categories]])
        tags_features = self.tags_encoder.transform([[tags]])
        
        genres_df = pd.DataFrame(genres_features, columns=[f'genres_{label}'
                                                           for label in self.genres_encoder.classes_],)
        categories_df = pd.DataFrame(categories_features, columns=[f'genres_{label}'
                                                                   for label in self.categories_encoder.classes_],)
        tags_df = pd.DataFrame(tags_features, columns=[f'genres_{label}'
                                                                           for label in self.tags_encoder.classes_],)

        features = pd.concat([features, genres_df, categories_df, tags_df], axis=1)
        features = features.reindex(columns=self.feature_columns_encoder, fill_value=0)
        
        return features

    
    def predict(self, game: dict) -> dict:
        features = self.prepare_features(game)

        commercial_prediction = self.commercial_model.predict(features)[0]
        critical_prediction = self.critical_model.predict(features)[0]

        return {
            'commercial_prediction': commercial_prediction,
            'critical_prediction': critical_prediction
        }