import pandas as pd

class DataProfiler:
    @staticmethod
    def profile(df: pd.DataFrame):
        print('=' * 50)
        print('Data Profile')
        print('=' * 50)

        print('\nDescription:')
        print(df.describe(include='all'))

        print(f'\nRows: {len(df)}')
        print(f'\nColumns: {len(df.columns)}')

        print('\nColumn Names')
        print(df.columns.tolist())

        print('\nData Types')
        print(df.dtypes)

        print(f'\nMissing Values (out of {len(df)})')
        print(df.isnull().sum())

        print(f'\nDuplicate Values: {df.duplicated().sum()}')