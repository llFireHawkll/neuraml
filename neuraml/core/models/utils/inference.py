from typing import Union

import numpy as np
import pandas as pd
import xgboost as xgb
from neuraml.core.models.metrics.classification import ClsClassificationMetrics
from neuraml.core.models.metrics.regression import ClsRegressionMetrics
from neuraml.exceptions.exceptions import ModelNotFittedError


class ClsModelInference:
    def __init__(self, model) -> None:
        """_summary_

        Args:
            model (_type_): _description_
        """
        self.model = model

    def score(
        self,
        dataframe: pd.DataFrame,
        target_label: Union[pd.Series, np.array],
        scoring_method: Union[ClsRegressionMetrics, ClsClassificationMetrics],
        **kwargs
    ):
        """_summary_

        Args:
            dataframe (pd.DataFrame): _description_
            target_label (Union[pd.Series, np.array]): _description_
            scoring_method (Union[ClsRegressionMetrics, ClsClassificationMetrics]): _description_

        Returns:
            _type_: _description_
        """
        # Step-1 Predict values from dataframe
        y_prediction = self.predict(dataframe=dataframe)

        # Step-2 Return score based on the scoring method
        return scoring_method(y_true=target_label, y_pred=y_prediction, **kwargs)

    def predict(self, dataframe: pd.DataFrame):
        """_summary_

        Args:
            dataframe (pd.DataFrame): _description_

        Raises:
            ModelNotFittedError: _description_
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """
        # Step-1 Based on the model_type == xgboost
        if self.model.model_type == "xgboost":
            if self.model._fit_flag:
                return self.model.predict(xgb.DMatrix(dataframe))
            else:
                raise ModelNotFittedError()
        else:
            raise NotImplementedError
