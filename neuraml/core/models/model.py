from abc import ABC, abstractmethod

import pandas as pd
import xgboost as xgb
from neuraml.exceptions.exceptions import ModelNotFittedError
from pydantic import BaseModel
from typing_extensions import Literal

__all__ = ["ClsXgboostModel"]


class ClsBaseModel(ABC):
    def __init__(self, **kwargs) -> None:
        """_summary_"""
        pass

    @abstractmethod
    def fit(self):
        """_summary_"""
        raise NotImplementedError

    @abstractmethod
    def predict(self, dataframe: pd.DataFrame):
        """_summary_

        Args:
            dataframe (pd.DataFrame): _description_
        """
        raise NotImplementedError

    @abstractmethod
    def save(self):
        """_summary_"""
        raise NotImplementedError


class XGBMetrics(BaseModel):
    pass

    class Config:
        extra = "allow"


class ClsXgboostModel(ClsBaseModel, XGBMetrics):
    def __init__(self, **kwargs) -> None:
        ClsBaseModel.__init__(self, **kwargs)
        XGBMetrics.__init__(self, **kwargs)

        # Setting internal attributes
        self._model_type: str = "xgboost"
        self._fit_flag: bool = False

    def fit(self):
        pass

    def predict(self, dataframe: pd.DataFrame):
        """_summary_

        Args:
            dataframe (pd.DataFrame): _description_
        """
        if self._fit_flag:
            return self.trained_model.predict(xgb.DMatrix(dataframe))
        else:
            raise ModelNotFittedError()
