
        


# Objective
The purpose of this project is to be able to predict how a game can be successful on Steam.

# Prediction Target
To make a game successful, a few factors matter. We will try to predict these very factors in this project, given as follows:
- Positive
- Negative
- Recommendations
- Estimated Owners
- Median Playtime Forever

        


        # Column-wise feature audit and removal

Column                                  Keep                                  Reason
--------------------------------------------------------------------------------------
app_id                                   ❌                             App identifier
name                                     ❌                             Game name
release_date                             ❌                             Game's release date
price                                    ✅                             To see Price's affect on owners and ratings
price_status                             ❌                             Very similar to price.
estimated_owners                         ✅                             To test and to predict how many owners could a game have
developers                               ❌                             Just Developer names
publishers                               ✅                             Publisher names could matter
genres                                   ✅                             Testing and predicting how genres affect game success
categories                               ✅                             Testing and predicting which categories affect game success
tags                                     ✅                             Testing and predicting which tags affect game success
positive                                 ✅                             Testing and predicting a game's ratings
negative                                 ✅                             Testing and predicting a game's ratings
recommendations                          ❌                             A lost of games missing data
peak_ccu                                 ✅                             Concurrent User count not required
metacritic_score                         ❌                             Most games missing data
user_score                               ❌                             User score is 0 for every game.
average_playtime_forever                 ✅                             Playtime needs to be tested as well as predicted
median_playtime_forever                  ✅                             Playtime needs to be tested as well as predicted
average_playtime_2weeks                  ✅                             Playtime needs to be tested as well as predicted
median_playtime_2weeks                   ✅                             Playtime needs to be tested as well as predicted
short_description                        ❌                             Game description not required
about_the_game                           ❌                             About section of a game not required
detailed_description                     ❌                             Game description not required
notes                                    ❌                             Notes not required
achievements                             ✅                             Achievement count needs to be evaluated on a game's performance
dlc_count                                ❌                             DLC counts not required
packages                                 ❌                             Package counts not required
supported_languages                      ❌                             Not required
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
steam_store_available                    ✅                             Needs to be evaluated for a game's performance on Steam
steam_spy_available                      ✅                             Needs to be evaluated for a game's performance on Steam





# Missing Value Strategy

Given is a table of all the columns we are using along with their number of missing values. 
For Price especially, a further calculation shows much lesser values as only 19 paid games are lacking a price in the data.

Column                                      Missing Rows (out of 136491)
achievements                                            17152
steam_store_available                                       0
steam_spy_available                                         0
average_playtime_forever                                    0
median_playtime_forever                                     0
average_playtime_2weeks                                     0
median_playtime_2weeks                                      0
peak_ccu                                                    0
publishers                                                  0
genres                                                      0
categories                                                  0
tags                                                        0
positive                                                    0
negative                                                    0
estimated_owners                                            0
price                                                       19

Given that most of the columns chosen for training actually have all the values, we can drop Achievements column and 
feed an average price in the rows for the 19 paid games missing price values.


# Invalid & Duplicate Data
There are 0 invalid and duplicate entries in the selected columns.

        #Final Data Details

Final Dataset shape: (136491, 16)
New data saved at: D:\Projects\Python\game-success-analytics\data\processed\steam_games_cleaned.csv
