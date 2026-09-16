from typing import Optional

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from utils.reportwriter import ReportWriter
from sklearn.metrics import r2_score


class ModelComparision:
    def __init__(self, features_path: Path, report_path: Optional[Path] = None):
        self.df = pd.read_csv(features_path)
        if report_path is not None:
            self.reportWriter = ReportWriter(report_path)


    def feature_importance_xgb(self):
        target_col_A = 'avg_owners'
        target_col_B = 'review_score'

        X = self.df.drop(columns=[target_col_A, target_col_B])
        Y_critical = self.df[target_col_B]
        Y_commercial = self.df[target_col_A]
        xgb_model = XGBRegressor(n_estimators=2000, max_depth=9,
                                    learning_rate=0.02,
                                    subsample=0.8,
                                    random_state=42,
                                    eval_metric='rmse',
                                    early_stopping_rounds=10)

        self.reportWriter.append_to_report("# Investigating XGBoost Model")

        xgb_acc_critical = self.__xgb_modelling__('Critical Success XGB', X, Y_critical, xgb_model, True)
        xgb_acc_commercial = self.__xgb_modelling__('Commercial Success XGB', X, Y_commercial, xgb_model, True)

        self.reportWriter.append_to_report(
        f"""
\n## Accuracies
XGBoost model, with a little bit of tweaks suggests that it's accuracy in both cases can go a little farther.

Accuracies for both models trained on XGBoost are:
Critical Success Model: {xgb_acc_critical*100:.3f}%
Commercial Success Model: {xgb_acc_commercial*100:.3f}%
        """
        )


    def build_models(self):
        target_col_A = 'avg_owners'
        target_col_B = 'review_score'

        X = self.df.drop(columns=[target_col_A, target_col_B])
        Y_critical = self.df[target_col_B]
        Y_commercial = self.df[target_col_A]

        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

        xgb_model = XGBRegressor(n_estimators=2000, max_depth=9,
                                    learning_rate=0.02,
                                    subsample=0.8,
                                    random_state=42,
                                    eval_metric='rmse',
                                    early_stopping_rounds=10)

        rf_acc_commercial = self.__rf_modelling__('Commercial Success RF', X, Y_commercial, rf_model)
        rf_acc_critical = self.__rf_modelling__('Critical Success RF', X, Y_critical, rf_model)

        xgb_acc_commercial = self.__xgb_modelling__('Commercial Success XGB', X, Y_commercial, xgb_model)
        xgb_acc_critical = self.__xgb_modelling__('Critical Success XGB', X, Y_critical, xgb_model)

        self.reportWriter.append_to_report(f"""
# Model Accuracy Comparision

## Accuracies

Commercial Success RF: {rf_acc_commercial*100:.3f}%
Critical Success RF: {rf_acc_critical*100:.3f}%

Commercial Success XGBoost: {xgb_acc_commercial*100:.3f}%
Critical Success XGBoost: {xgb_acc_critical*100:.3f}%

# Final Conclusion

Based on the higher level of accuracy of XGBoost on both the models, combined with the fact that it takes lesser time to train, 
we'll go with XGBoost as our final model to predict our commercial and critical game successes on Steam.
""")


    def __rf_modelling__(self, model_name: str, x: pd.DataFrame, y: pd.Series, rf_regressor: RandomForestRegressor):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        rf_regressor.fit(x_train, y_train)
        Y_Pred = rf_regressor.predict(x_test)
        accuracy = r2_score(y_test, Y_Pred)
        print(f'Model {model_name} Accuracy: {accuracy*100:.3f}%')
        return accuracy


    def __xgb_modelling__(self, model_name: str, x: pd.DataFrame, y: pd.Series, xgb_regressor: XGBRegressor, assess_importances: bool = False):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        xgb_regressor.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False)
        Y_Pred = xgb_regressor.predict(x_test)
        accuracy = r2_score(y_test, Y_Pred)
        print(f'Model {model_name} Accuracy: {accuracy*100:.3f}%')

        if assess_importances:
            features = xgb_regressor.feature_names_in_
            importances = xgb_regressor.feature_importances_
            print(f'\nModel {model_name} Feature Importances:')

            self.reportWriter.append_to_report(f"""
\n## Top 10 Features
{model_name}:
""")

            features_zip = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
            for i, (feat, imp) in enumerate(features_zip):
                feature_data = f'{feat}: {imp*100:.3f}%'
                if i < 10:
                    print(feature_data)
                    self.reportWriter.append_to_report(f"\n{feature_data}")

        return accuracy