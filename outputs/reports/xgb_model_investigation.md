# Investigating XGBoost Model

## Top 20 Features
Critical Success XGB:

tags_Singleplayer: 17.489%
tags_Indie: 9.853%
tags_Action: 7.025%
tags_Casual: 4.738%
tags_Adventure: 3.142%
tags_Utilities: 2.825%
tags_Strategy: 2.371%
tags_RPG: 1.538%
tags_Simulation: 1.431%
tags_Design & Illustration: 1.371%
            
## Top 10 Permutation Importances
            Critical Success XGB:

                       Feature  Importance Mean  Importance Std
             tags_Singleplayer            0.280           0.004
                    tags_Indie            0.274           0.002
                   tags_Casual            0.179           0.003
                   tags_Action            0.171           0.003
                tags_Adventure            0.163           0.005
                 tags_Strategy            0.060           0.002
               tags_Simulation            0.051           0.002
                      tags_RPG            0.040           0.001
                       tags_2D            0.032           0.001
                         price            0.019           0.001
 categories_Steam Achievements            0.014           0.001
                       tags_3D            0.012           0.000
     categories_Family Sharing            0.012           0.001
        categories_Steam Cloud            0.009           0.001
     supported_languages_count            0.008           0.000
             tags_Free to Play            0.008           0.000
categories_Steam Trading Cards            0.008           0.001
                       tags_VR            0.008           0.000
                tags_Utilities            0.007           0.000
               tags_Story Rich            0.007           0.000

## Top 20 Features
Commercial Success XGB:

tags_Free to Play: 37.215%
tags_Indie: 8.989%
tags_Action: 4.743%
price: 3.600%
tags_VR: 2.740%
categories_Playable without Timed Input: 2.691%
tags_Singleplayer: 2.621%
tags_Casual: 1.672%
categories_Family Sharing: 1.621%
tags_Utilities: 1.233%
            
## Top 10 Permutation Importances
            Commercial Success XGB:

                                Feature  Importance Mean  Importance Std
                                  price            0.383           0.005
                             tags_Indie            0.251           0.005
                            tags_Action            0.204           0.005
                      tags_Singleplayer            0.172           0.005
                            tags_Casual            0.139           0.004
              categories_Family Sharing            0.067           0.001
              supported_languages_count            0.060           0.002
                         tags_Adventure            0.059           0.004
                      tags_Free to Play            0.038           0.003
                               tags_RPG            0.033           0.002
                                tags_VR            0.023           0.001
                          release_month            0.020           0.001
                        tags_Simulation            0.013           0.000
                          tags_Strategy            0.011           0.001
               categories_Single-player            0.011           0.000
                                tags_2D            0.008           0.001
         categories_Steam Trading Cards            0.007           0.000
                                tags_3D            0.005           0.001
categories_Playable without Timed Input            0.005           0.001
                         tags_Utilities            0.004           0.001

We're also removing game_age in our final investigation and modelling because it is not useful at all in predicting a new game's success,
but at the same time puts a lot of influence in modelling the provided data, which will lead to unnecessary overfitting, while we'll never know
the game age of a new game we're trying to predict for.


## XGBoost Scores

XGBoost model, with a little bit of tweaks suggests that it's scores in both cases can go a little farther.

Scores for both models trained on XGBoost are:
Critical Success Model R2: 0.747
Commercial Success Model R2: 0.773

Critical Success Model MAE: 0.079
Commercial Success Model MAE: 1.014

Critical Success Model RMSE: 0.122
Commercial Success Model RMSE: 1.859
        
# Model Scores Comparision

## Scores

Commercial Success RF R2: 0.751
Critical Success RF R2: 0.726

Commercial Success XGBoost R2: 0.773
Critical Success XGBoost R2: 0.747
----------

Commercial Success RF MAE: 1.019
Critical Success RF MAE: 0.082

Commercial Success XGBoost MAE: 1.014
Critical Success XGBoost MAE: 0.079
----------

Commercial Success RF RMSE: 1.924
Critical Success RF RMSE: 0.127

Commercial Success XGBoost RMSE: 1.859
Critical Success XGBoost RMSE: 0.122

# Final Conclusion
Based on the higher level of scoring of XGBoost on both the models, combined with the fact that it takes lesser time to train, 
we'll go with XGBoost as our final model to predict our commercial and critical game successes on Steam.
