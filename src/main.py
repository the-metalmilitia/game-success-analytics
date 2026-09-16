from data_loader import DataLoader
from data_loader import pd
from data_profiler import DataProfiler
from model_development_comparision import ModelComparision
from preprocessing import PreProcessor
from eda import EDA
from feature_engineering import FeatureEngineering
from config import RAW_DATA_DIR, OUTPUT_DIR, PROCESSED_DATA_DIR

def eda(df: pd.DataFrame):
    eda = EDA(df)
    eda.analyse_numeric_overview()
    eda.analyse_categorical_data()
    eda.analyse_missing_data()
    eda.analyse_duplicate_data()
    eda.analyse_target_variables()
    eda.correlation_matrix(str(OUTPUT_DIR))
    eda.business_observations_visualizations(str(OUTPUT_DIR))


def preprocessing(df: pd.DataFrame):
    preprocessing = PreProcessor(df, OUTPUT_DIR / 'reports/preprocessing_report.md')
    preprocessing.reporting_prediction_targets()
    preprocessing.feature_audit()
    preprocessing.report_invalid_data()
    preprocessing.save_final_dataset(PROCESSED_DATA_DIR / 'steam_games_cleaned.csv')


def feature_engineering():
    feature_eng = FeatureEngineering(PROCESSED_DATA_DIR / 'steam_games_cleaned.csv', OUTPUT_DIR / 'reports/feature_engineering_report.md')
    feature_eng.bin_prices()
    feature_eng.parse_date()
    feature_eng.parse_avg_owners()
    feature_eng.calculate_review_score()
    feature_eng.count_literals_per_row(['genres', 'tags', 'categories', 'supported_languages'])

    feature_eng.correlation_matrix(column_names=['release_month', 'price',
                                                 'avg_owners', 'genres_count', 
                                                 'categories_count', 'tags_count', 
                                                 'review_score', 'supported_languages_count']
                                                 , path=OUTPUT_DIR / 'figures/feature_eng_corr_matrix.png')

    feature_eng.encode_column(['genres', 'categories'])
    feature_eng.encode_column(['tags'], threshold=500)

    feature_eng.drop_columns(['positive', 'negative',
                              'tags_count', 'categories_count', 'genres_count',
                               'release_year', 'supported_languages'])

    feature_eng.feature_importance(path=PROCESSED_DATA_DIR / 'features.csv')


def xgb_model_investigation(model_compare: ModelComparision):
    # xgb_investigation = ModelComparision(PROCESSED_DATA_DIR / 'features.csv', OUTPUT_DIR / 'reports/xgb_model_investigation.md')
    model_compare.feature_importance_xgb()


def build_rf_xgb(model_compare: ModelComparision):
    model_compare.build_models()

def main():

    data_loader = DataLoader(RAW_DATA_DIR / 'steam_games.csv')
    df = data_loader.load()

    # data_profiler = DataProfiler()
    # data_profiler.profile(df)

# #EDA with visualization and EDA report.
    #eda(df)

#Preprocessing
    #preprocessing(df)

#Feature Engineering
    #feature_engineering()

#XGB Model Investigation
    model_compare = ModelComparision(PROCESSED_DATA_DIR / 'features.csv', report_path=OUTPUT_DIR / 'reports/xgb_model_investigation.md')
    xgb_model_investigation(model_compare)

#Buildin & Comparing XGB and RF 0Models
    build_rf_xgb(model_compare) 

# result is
# Model Commercial Success RF RF Accuracy: 0.7890185467921406
# Model Critical Success RF RF Accuracy: 0.7302207834757064
# Model Commercial Success XGB XGB Accuracy: 79.807%
# Model Critical Success XGB XGB Accuracy: 74.512%

    

if __name__ == '__main__':
    main()