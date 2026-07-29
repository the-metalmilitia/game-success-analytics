from data_loader import DataLoader
from data_profiler import DataProfiler
from eda import EDA
from config import RAW_DATA_DIR, OUTPUT_DIR

def main():

    data_loader = DataLoader(RAW_DATA_DIR / 'games_raw_for_cleanup.csv')
    df = data_loader.load()

    data_profiler = DataProfiler()
    data_profiler.profile(df)

    eda = EDA(df)
    eda.analyse_numeric_overview()
    eda.analyse_missing_data()
    eda.analyse_duplicate_data()
    eda.analyse_categorical_data()
    eda.analyse_target_variables()
    eda.correlation_matrix(str(OUTPUT_DIR))
    eda.business_observations_visualizations(str(OUTPUT_DIR))
    

if __name__ == '__main__':
    main()