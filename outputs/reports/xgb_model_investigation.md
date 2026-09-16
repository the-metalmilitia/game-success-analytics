# Investigating XGBoost Model

## Top 10 Features
Critical Success XGB:

tags_Singleplayer: 20.523%
tags_Action: 8.415%
tags_Casual: 3.595%
tags_Adventure: 2.889%
tags_Utilities: 1.885%
tags_2D: 1.569%
tags_RPG: 1.558%
tags_Strategy: 1.497%
categories_Family Sharing: 1.272%
categories_Steam Achievements: 1.237%

## Top 10 Features
Commercial Success XGB:

tags_Singleplayer: 14.659%
tags_Action: 8.480%
price: 7.768%
game_age: 5.341%
tags_Casual: 3.109%
categories_Family Sharing: 3.042%
tags_Early Access: 2.477%
categories_Single-player: 1.839%
tags_Free to Play: 1.464%
tags_RPG: 1.411%

## Accuracies
XGBoost model, with a little bit of tweaks suggests that it's accuracy in both cases can go a little farther.

Accuracies for both models trained on XGBoost are:
Critical Success Model: 74.512%
Commercial Success Model: 79.807%
        
# Model Accuracy Comparision

## Accuracies

Commercial Success RF: 78.902%
Critical Success RF: 73.022%

Commercial Success XGBoost: 79.807%
Critical Success XGBoost: 74.512%

# Final Conclusion

Based on the higher level of accuracy of XGBoost on both the models, combined with the fact that it takes lesser time to train, 
we'll go with XGBoost as our final model to predict our commercial and critical game successes on Steam.
