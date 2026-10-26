

# Price Bins

Prices have been binned on the basis of standard price divisions on Steam. The bins are as given below:
Free to Play: $0
Micro Indie: $0.1 - $9.9
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
