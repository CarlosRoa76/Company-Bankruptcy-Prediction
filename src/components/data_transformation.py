import sys, os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Structures the preprocessing pipeline using RobustScaler to handle 
        extreme outliers typical in corporate financial ratios.
        """
        try:
            # We treat all selected financial features as numerical
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", RobustScaler()) # Better than MinMax for financial outliers
                ]
            )

            logging.info("Numerical pipeline configured with RobustScaler.")

            preprocessor = ColumnTransformer(
                transformers=[
                    # Applying num_pipeline to all columns (excluding target, handled later)
                    ("num_pipeline", num_pipeline, slice(0, None)) 
                ],
                remainder="passthrough"
            )

            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            # Adjust this if your notebook's target column is named differently
            target_column_name = "Bankrupt?" 

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes.")

            # 1. Scale the features
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # 2. Handle Class Imbalance using SMOTE (ONLY on training data to prevent data leakage)
            logging.info("Applying SMOTE to handle bankruptcy class imbalance on training data.")
            smote = SMOTE(sampling_strategy='minority', random_state=42)
            input_feature_train_arr_resampled, target_feature_train_resampled = smote.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )

            # Recombine features matrix and target vectors
            train_arr = np.c_[input_feature_train_arr_resampled, np.array(target_feature_train_resampled)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            raise CustomException(e, sys)