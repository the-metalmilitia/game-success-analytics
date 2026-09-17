from typing import Optional

import pandas as pd
from joblib import parallel_backend
from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from utils.reportwriter import ReportWriter
from sklearn.metrics import (r2_score, mean_absolute_error, root_mean_squared_error)
from sklearn.inspection import permutation_importance


class ModelComparision:
    def __init__(self, features_path: Path, report_path: Optional[Path] = None):
        self.df = pd.read_csv(features_path)
        if report_path is not None:
            self.reportWriter = ReportWriter(report_path)


    def feature_scoring_xgb(self):
        target_col_A = 'avg_owners'
        target_col_B = 'review_score'

        X = self.df.drop(columns=[target_col_A, target_col_B, 'game_age'])
        Y_critical = self.df[target_col_B]
        Y_commercial = self.df[target_col_A]
        xgb_model = XGBRegressor(n_estimators=2000, max_depth=9,
                                    learning_rate=0.02,
                                    subsample=0.8,
                                    random_state=42,
                                    eval_metric='rmse',
                                    early_stopping_rounds=10)

        self.reportWriter.append_to_report("# Investigating XGBoost Model")

        xgb_score_critical = self.__xgb_modelling__('Critical Success XGB', X, Y_critical, xgb_model, True)
        xgb_score_commercial = self.__xgb_modelling__('Commercial Success XGB', X, Y_commercial, xgb_model, True)

        self.reportWriter.append_to_report(
        f"""
\nWe're also removing game_age in our final investigation and modelling because it is not useful at all in predicting a new game's success,
but at the same time puts a lot of influence in modelling the provided data, which will lead to unnecessary overfitting, while we'll never know
the game age of a new game we're trying to predict for.

\n## XGBoost Scores

XGBoost model, with a little bit of tweaks suggests that it's scores in both cases can go a little farther.

Scores for both models trained on XGBoost are:
Critical Success Model R2: {xgb_score_critical['r2']:.3f}
Commercial Success Model R2: {xgb_score_commercial['r2']:.3f}

Critical Success Model MAE: {xgb_score_critical['mae']:.3f}
Commercial Success Model MAE: {xgb_score_commercial['mae']:.3f}

Critical Success Model RMSE: {xgb_score_critical['rmse']:.3f}
Commercial Success Model RMSE: {xgb_score_commercial['rmse']:.3f}
        """
        )


    def build_models(self):
        target_col_A = 'avg_owners'
        target_col_B = 'review_score'

        X = self.df.drop(columns=[target_col_A, target_col_B, 'game_age'])
        Y_critical = self.df[target_col_B]
        Y_commercial = self.df[target_col_A]

        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

        xgb_model = XGBRegressor(n_estimators=2000, max_depth=9,
                                    learning_rate=0.02,
                                    subsample=0.8,
                                    random_state=42,
                                    eval_metric='rmse',
                                    early_stopping_rounds=10)

        rf_score_commercial = self.__rf_modelling__('Commercial Success RF', X, Y_commercial, rf_model)
        rf_score_critical = self.__rf_modelling__('Critical Success RF', X, Y_critical, rf_model)

        xgb_score_commercial = self.__xgb_modelling__('Commercial Success XGB', X, Y_commercial, xgb_model)
        xgb_score_critical = self.__xgb_modelling__('Critical Success XGB', X, Y_critical, xgb_model)

        self.reportWriter.append_to_report(f"""
# Model Scores Comparision

## Scores

Commercial Success RF R2: {rf_score_commercial['r2']:.3f}
Critical Success RF R2: {rf_score_critical['r2']:.3f}

Commercial Success XGBoost R2: {xgb_score_commercial['r2']:.3f}
Critical Success XGBoost R2: {xgb_score_critical['r2']:.3f}
----------

Commercial Success RF MAE: {rf_score_commercial['mae']:.3f}
Critical Success RF MAE: {rf_score_critical['mae']:.3f}

Commercial Success XGBoost MAE: {xgb_score_commercial['mae']:.3f}
Critical Success XGBoost MAE: {xgb_score_critical['mae']:.3f}
----------

Commercial Success RF RMSE: {rf_score_commercial['rmse']:.3f}
Critical Success RF RMSE: {rf_score_critical['rmse']:.3f}

Commercial Success XGBoost RMSE: {xgb_score_commercial['rmse']:.3f}
Critical Success XGBoost RMSE: {xgb_score_critical['rmse']:.3f}

# Final Conclusion
Based on the higher level of scoring of XGBoost on both the models, combined with the fact that it takes lesser time to train, 
we'll go with XGBoost as our final model to predict our commercial and critical game successes on Steam.
""")


    def __rf_modelling__(self, model_name: str, x: pd.DataFrame, y: pd.Series, rf_regressor: RandomForestRegressor):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        rf_regressor.fit(x_train, y_train)

        with parallel_backend('threading'):
            perm_importance_result = permutation_importance(rf_regressor, x_test, y_test, n_repeats=10, n_jobs=-1, random_state=42, scoring=['r2'])

        importance_df = pd.DataFrame({
            'Feature': x.columns,
            'Importance Mean': perm_importance_result['r2'].importances_mean,
            'Importance Std': perm_importance_result['r2'].importances_std
        }).sort_values(by='Importance Mean', ascending=False).round(3).head(20)

        Y_Pred = rf_regressor.predict(x_test)

        score_r2 = r2_score(y_test, Y_Pred)
        score_mae = mean_absolute_error(y_test, Y_Pred)
        score_rmse = root_mean_squared_error(y_test, Y_Pred)

        print(f'\nModel {model_name} Permutation Importances:')
        print(importance_df.to_string(index=False))
        print(f'\nModel {model_name} R2: {score_r2:.3f}, MAE: {score_mae:.3f}, RMSE: {score_rmse:.3f}')

        return {'r2': score_r2, 'mae': score_mae, 'rmse': score_rmse}


    def __xgb_modelling__(self, model_name: str, x: pd.DataFrame, y: pd.Series, xgb_regressor: XGBRegressor, assess_importances: bool = False):
        x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42)
        x_test, x_eval, y_test, y_eval = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

        xgb_regressor.fit(x_train, y_train, eval_set=[(x_eval, y_eval)], verbose=False)

        Y_Pred = xgb_regressor.predict(x_test)

        score_r2 = r2_score(y_test, Y_Pred)
        score_mae = mean_absolute_error(y_test, Y_Pred)
        score_rmse = root_mean_squared_error(y_test, Y_Pred)

        print(f'\nModel {model_name} R2: {score_r2:.3f}, MAE: {score_mae:.3f}, RMSE: {score_rmse:.3f}')

        if assess_importances:
            features = xgb_regressor.feature_names_in_
            importances = xgb_regressor.feature_importances_

            with parallel_backend('threading'):
                perm_importance_result = permutation_importance(xgb_regressor, x_test, y_test, n_repeats=10, n_jobs=-1, random_state=42, scoring=['r2'])

            perm_importance_df = pd.DataFrame({
                'Feature': x.columns,
                'Importance Mean': perm_importance_result['r2'].importances_mean,
                'Importance Std': perm_importance_result['r2'].importances_std
            }).sort_values(by='Importance Mean', ascending=False).round(3).head(20)

            print(f'\nModel {model_name} Feature Importances:')

            self.reportWriter.append_to_report(f"""
\n## Top 20 Features
{model_name}:
""")

            features_zip = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
            for i, (feat, imp) in enumerate(features_zip):
                feature_data = f'{feat}: {imp*100:.3f}%'
                if i < 10:
                    print(feature_data)
                    self.reportWriter.append_to_report(f"\n{feature_data}")


            print(f'\nModel {model_name} Permutation Importances:')
            print(perm_importance_df.to_string(index=False))

            self.reportWriter.append_to_report(f"""
            \n## Top 10 Permutation Importances
            {model_name}:
""")
            self.reportWriter.append_to_report(f'\n{perm_importance_df.to_string(index=False)}')

        return {'r2': score_r2, 'mae': score_mae, 'rmse': score_rmse}