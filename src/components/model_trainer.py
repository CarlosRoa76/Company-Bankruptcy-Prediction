import sys, os
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from sklearn.metrics import f1_score, recall_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr, preprocessor_path=None):
        try:
            logging.info("Split training and test input data")
            
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            # Kept models that perform best on imbalanced tabular datasets
            models = {
                "Random Forest": RandomForestClassifier(class_weight='balanced'),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000),
                "XGBClassifier": XGBClassifier(eval_metric='logloss')
            }

            params = {
                "Random Forest": {
                    'n_estimators': [64, 128, 256],
                    'max_depth': [10, 20, None]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.05, 0.01],
                    'n_estimators': [64, 128, 256]
                },
                "Logistic Regression": {
                    'C': [0.1, 1.0, 10.0]
                },
                "XGBClassifier": {
                    'learning_rate': [0.1, 0.05, 0.01],
                    'n_estimators': [64, 128, 256],
                    'scale_pos_weight': [1, 5, 10] # Great for XGBoost imbalance
                }
            }

            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                models=models, param=params
            )
            
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found. Scores fell below 0.60 threshold.")
            
            logging.info(f"Best found model on both training and testing dataset: {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            classification_f1 = f1_score(y_test, predicted, average='macro')
            classification_recall = recall_score(y_test, predicted)
            
            logging.info(f"Best Model [{best_model_name}] Macro F1 Score: {classification_f1}")
            logging.info(f"Best Model [{best_model_name}] Bankruptcy Recall (Detection Rate): {classification_recall}")
            
            return classification_f1
        
        except Exception as e:
            raise CustomException(e, sys)