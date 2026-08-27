import ast
import pandas as pd
import numpy as np

def _count_tuple_size(x):
    if "</" in str(x) or "<br>" in str(x):
        return 0

    langs = [lang.strip for lang in str(x).split(',')]
    return len(langs)

@staticmethod
def extract_literal_values(df: pd.DataFrame, column_name: str) -> dict:
        df[f'{column_name}_literal'] = df[column_name].apply(lambda x: tuple(ast.literal_eval(x)))
        values = df[f'{column_name}_literal'].value_counts()

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


@staticmethod
def literal_values_per_row(df: pd.DataFrame, column_name: str):
    df[f'{column_name}_count'] = df[column_name].apply(_count_tuple_size)


@staticmethod
def parse_average_owners(owner_string: str):
        owner_substrings = owner_string.split(' - ')
        if len(owner_substrings) > 1 and owner_substrings[0] and owner_substrings[1]:
            floor_count = int(owner_substrings[0])
            ceiling_count = int(owner_substrings[1])
            avg_owners = (floor_count + ceiling_count) / 2
            return np.log1p(avg_owners)


@staticmethod
def wilson_lower_bound(pos, neg, z=1.96):
    n = pos + neg
    if n == 0:
        return 0
    p = pos / n
    return np.log1p((p + z**2/(2*n) - z * np.sqrt((p*(1-p) + z**2/(4*n)) / n)) / (1 + z**2/n))