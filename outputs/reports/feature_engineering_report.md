

# Price Bins

Prices have been binned on the basis of standard price divisions on Steam. The bins are as given below:
Free to Play: $0
Micro Indie: $1.0 - $9.9
Standard Tier: $10.0 - $29.9
Upper Tier: $30.0 - $49.99

The value will be stored in price_category column and will be fed to the correlation matrix and the model along with the prices.



# Dates

The dates have been divided into game year, game month and game age.

While game year might not prove useful for a game still under development to know if it could be
successful, game month could be a useful parameter to see if releasing at the right month in a year could influence
commercial or critical success. For example, near Holiday season, black friday, etc.
Game age would also be a good parameter to check if a game gets more successful as it ages, or people tend to forget it and move on.

The dates are stored in release_month, release_year and game_age columns.



# Average Owners

Since estimated_owners is a string of range of customers (ex. '0-20000'), this is parsed into an integer value,
that is an average between the floor and the ceiling values in the string. Some rows have NaNs, which have been removed.

The average is stored in avg_owners column, which acts as a Y axis for the commercial success model.
It will also be fed to the correlation matrix to have a look at the factors immediately affecting it.




# Review Score

Postive and Negative scores are fed into a wilson Lower Bound formula to get a review scoring,
and applied with log1p internally to scale the review score for all the rows.
Positive and negative review count columns will be removed when fed to the model and only the final
review score will be accounted.

The score is stored in review_score column.



# Literals and Their Counts

Genres, tags, categories and supported languages are in lists in each row. 
A column has been added for each of these to keep the count of genres, categories, etc. and fed to the model,
so that a correlation can be seen between these lists and their effect on commercial and critical success.
In actual feature importance, these counts seem to take away the work of actual genre, tag and category and hence, will be ignored.
A count of Supported languages however, will be fed in, to see if support of larger number of languages can influence success.

The columns are genres_count, categories_count, tags_count and supported_languages_count.

        

# Encoding strategy
        
Certain columns which are labels, viz, genres, categories and tags have been encoded into multiple binary columns
through Multilabel Binarizer. This results in a pretty large number of final columns, but it's workable and in fact quite required.
        
Tags have been encoded with a threshold value, as there are a lot of tags which would blow up the column count.
So a tag is only encoded if it is found in x number of games or more, x being the threshold value.
        


# Feature Importance

Two Random Forest Regression models are trained to calculate feature importance.
Both models take all the other values in the dataset except avg_owners and review_score as their X axis features.

Model A takes avg_owners as Y axis, which is what we have to predict for commercial success.
Model B takes review_score as Y axis, which is to be predicted for critical success.

An R2 accuracy of more than 65 or 70, should be a decent acceptance point.

        

## Model Commercial Success Feature Importance Results
        Top 10 Features:
        
price: 35.711%
game_age: 31.270%
tags_Singleplayer: 7.215%
supported_languages_count: 5.165%
tags_Action: 2.521%
tags_Casual: 1.290%
release_month: 0.817%
categories_Single-player: 0.682%
categories_Steam Trading Cards: 0.596%
categories_Steam Achievements: 0.594%
        

## Model Critical Success Feature Importance Results
        Top 10 Features:
        
tags_Singleplayer: 25.168%
game_age: 19.116%
tags_Action: 8.093%
tags_Casual: 4.816%
tags_Adventure: 3.047%
categories_Steam Achievements: 2.844%
price: 2.561%
release_month: 1.477%
tags_Strategy: 1.372%
supported_languages_count: 1.367%