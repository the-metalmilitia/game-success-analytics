import pandas as pd
import numpy as np
from pathlib import Path
from config import USABLE_DATASET_COLUMNS
from utils.reportwriter import ReportWriter

class PreProcessor:
    def __init__(self, target_df: pd.DataFrame, report_path: Path):
        self.reportWriter = ReportWriter(report_path)
        self.df = target_df
        self.cleaned_df = target_df


#Reporting prediction target
    def reporting_prediction_targets(self):
        content = """
        \n\n
# Objective
The purpose of this project is to be able to predict how a game can be successful on Steam.

# Prediction Target
To make a game successful, a few factors matter. We will try to predict these very factors in this project, given as follows:
Model A predicts commercial success: estimated_owners
Model B predicts critical success: review_score
"""
        self.reportWriter.append_to_report(content)



#Feature auditing for what all columns are really needed
    def feature_audit(self):
        content = """
\n\n# Column-wise feature audit and removal

Column                                  Keep                                  Reason
--------------------------------------------------------------------------------------
app_id                                   ❌                             App identifier
name                                     ❌                             Game name
release_date                             ✅                             Game's release date. Useful to see trends every year/month
price                                    ✅                             To see Price's affect on owners and ratings
price_status                             ❌                             Very similar to price.
estimated_owners                         ✅                             Needd as a target to train the model on commercial success
developers                               ❌                             Just Developer names
publishers                               ❌                             Publisher names would influence prediction which would be unfair for indie games
genres                                   ✅                             Testing and predicting how genres affect game success
categories                               ✅                             Testing and predicting which categories affect game success
tags                                     ✅                             Testing and predicting which tags affect game success
positive                                 ✅                             Needed as a target for training the model on critical success.
negative                                 ✅                             Needed as a target for training the model on critical success.
recommendations                          ❌                             A lost of games missing data
peak_ccu                                 ❌                             Concurrent User count not required
metacritic_score                         ❌                             Most games missing data
user_score                               ❌                             User score is 0 for every game.
average_playtime_forever                 ❌                             Cannot be known for an unreleased game. Will result in data leakage
median_playtime_forever                  ❌                             Cannot be known for an unreleased game. Will result in data leakage
average_playtime_2weeks                  ❌                             Cannot be known for an unreleased game. Will result in data leakage
median_playtime_2weeks                   ❌                             Cannot be known for an unreleased game. Will result in data leakage
short_description                        ❌                             Game description not required
about_the_game                           ❌                             About section of a game not required
detailed_description                     ❌                             Game description not required
notes                                    ❌                             Notes not required
achievements                             ✅                             Achievement count needs to be evaluated on a game's performance
dlc_count                                ❌                             DLC counts not required
packages                                 ❌                             Package counts not required
supported_languages                      ✅                             Languages could be useful to see if more languages affect 
full_audio_languages                     ❌                             Not required
windows                                  ❌                             Not required
mac                                      ❌                             Not required
linux                                    ❌                             Not required
header_image                             ❌                             Not required
screenshots                              ❌                             Not required
movies                                   ❌                             Not required
website                                  ❌                             Not required
support_url                              ❌                             Not required
support_email                            ❌                             Not required
metacritic_url                           ❌                             Not required
steam_store_available                    ❌                             Not required. All values are true.
steam_spy_available                      ❌                             Not required. All values are true.
\n\n
"""

        self.reportWriter.append_to_report(content)


#Detailed invalid data reporting and cleaning.
    def report_invalid_data(self):
        #Checking for missing prices of paid games.
        missing_prices_df = self.cleaned_df[(self.cleaned_df['price'].isnull()) & (self.cleaned_df['price_status'] == 'paid')]
        print(f"Actual Missing Prices for Paid Games: {missing_prices_df.shape[0]}")

        content = f"""
\n# Missing Value Strategy

Given is a table of all the columns we are using along with their number of missing values. 
For Price especially, a further calculation shows much lesser values as only {missing_prices_df.shape[0]} paid games are lacking a price in the data.

Column                                      Missing Rows (out of {self.df.shape[0]})
achievements                                            17152
positive                                                    0
negative                                                    0
estimated_owners                                            0
genres                                                      0
categories                                                  0
tags                                                        0
supported_languages                                         0
price                                                       {missing_prices_df.shape[0]}

Given that most of the columns chosen for training actually have all the values, we can drop Achievements column and 
feed an average price in the rows for the {missing_prices_df.shape[0]} paid games missing price values.
"""

        self.reportWriter.append_to_report(content)

        #Dropping Achievements column since a lot of values are missing and it's not necessary for the modelling.
        print('Dropping Achievements column...')
        self.cleaned_df.drop(columns='achievements')

        #Imputing missing prices for paid games.
        median_price = self.df['price'].median()
        print(f'Imputing Missing 19 Prices with median price {median_price}...')
        self.cleaned_df['price'] = np.where((self.df['price_status'] == 'paid') & (self.df['price'].isnull()), median_price, self.df['price'])

        #Dropping all unusable columns as reasoned in the report.        
        print('Dropping unusable columns...')
        self.cleaned_df = self.cleaned_df[self.cleaned_df.columns.intersection(list(USABLE_DATASET_COLUMNS.keys()))]

        #Checking for invalid rows.
        for col, val in USABLE_DATASET_COLUMNS.items():
            bad_rows = self.cleaned_df[col].apply(lambda x: not isinstance(x, val))
            sum = bad_rows.sum()
            print(f'{col} has {sum} bad values')

        content = """
\n# Invalid & Duplicate Data
There are 0 invalid and duplicate entries in the selected columns.
"""

        self.reportWriter.append_to_report(content)



#Final shape of the cleaned dataset and saving to CSV
    def save_final_dataset(self, path: Path):
        content = f"""
        #Final Data Details

Final Dataset shape: {self.cleaned_df.shape}
New data saved at: {path}
"""
        self.reportWriter.append_to_report(content)
        self.cleaned_df.to_csv(path, index=False)