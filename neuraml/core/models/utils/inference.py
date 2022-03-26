from typing import Union

import numpy as np
import pandas as pd
from neuraml.core.models.metrics.classification import ClsClassificationMetrics
from neuraml.core.models.metrics.regression import ClsRegressionMetrics


class ClsModelInference:
    def __init__(self, model, **kwargs) -> None:
        pass

    def score(
        self,
        scoring_method: Union[ClsRegressionMetrics, ClsClassificationMetrics],
        dataframe: pd.DataFrame,
    ):
        pass

    def predict(self, dataframe: pd.DataFrame):
        pass
