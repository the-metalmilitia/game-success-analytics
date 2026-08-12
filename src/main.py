from data_loader import DataLoader
from data_profiler import DataProfiler
from preprocessing import PreProcessor
from eda import EDA
from config import RAW_DATA_DIR, OUTPUT_DIR, PROCESSED_DATA_DIR

def main():

    data_loader = DataLoader(RAW_DATA_DIR / 'steam_games.csv')
    df = data_loader.load()

    data_profiler = DataProfiler()
    data_profiler.profile(df)

# #EDA with visualization and EDA report.
#     eda = EDA(df)
#     eda.analyse_numeric_overview()
#     eda.analyse_categorical_data()
#     eda.analyse_missing_data()
#     eda.analyse_duplicate_data()
#     eda.analyse_target_variables()
#     eda.correlation_matrix(str(OUTPUT_DIR))
#     eda.business_observations_visualizations(str(OUTPUT_DIR))

#Preprocessing
    preprocessing = PreProcessor(df, OUTPUT_DIR / 'reports/preprocessing_report.md')
    preprocessing.reporting_prediction_targets()
    preprocessing.feature_audit()
    preprocessing.report_invalid_data()
    preprocessing.save_final_dataset(PROCESSED_DATA_DIR / 'steam_games_cleaned.csv')
    

if __name__ == '__main__':
    main()