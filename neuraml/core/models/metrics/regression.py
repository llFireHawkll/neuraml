import inspect

import numpy as np
from pydantic import BaseModel, validator
from sklearn import metrics as skmetrics
from typing_extensions import Literal

__all__ = ["ClsRegressionMetrics"]


class RegressionMetrics(BaseModel):
    metric: Literal["R2", "MAE", "MSE", "MSLE", "RMSE", "RMSLE", "CUSTOM"]

    @validator("metric", pre=True)
    def check_metric(cls, value):
        return value.upper()

    class Config:
        extra = "allow"

    def _r2_metric(self, y_true, y_pred, **kwargs):
        return skmetrics.r2_score(y_true=y_true, y_pred=y_pred, **kwargs)

    def _mae_metric(self, y_true, y_pred, **kwargs):
        return skmetrics.mean_absolute_error(y_true=y_true, y_pred=y_pred, **kwargs)

    def _mse_metric(self, y_true, y_pred, **kwargs):
        return skmetrics.mean_squared_error(y_true=y_true, y_pred=y_pred, **kwargs)

    def _msle_metric(self, y_true, y_pred, **kwargs):
        return skmetrics.mean_squared_log_error(y_true=y_true, y_pred=y_pred, **kwargs)

    def _rmse_metric(self, y_true, y_pred, **kwargs):
        return np.sqrt(
            skmetrics.mean_squared_error(y_true=y_true, y_pred=y_pred, **kwargs)
        )

    def _rmsle_metric(self, y_true, y_pred, **kwargs):
        return np.sqrt(
            skmetrics.mean_squared_log_error(y_true=y_true, y_pred=y_pred, **kwargs)
        )


class ClsRegressionMetrics(RegressionMetrics):
    def __init__(self, **kwargs) -> None:
        RegressionMetrics.__init__(self, **kwargs)

        # Setting internal attributes
        self._metrics_method_config = {
            "R2": self._r2_metric,
            "MAE": self._mae_metric,
            "MSE": self._mse_metric,
            "MSLE": self._msle_metric,
            "RMSE": self._rmse_metric,
            "RMSLE": self._rmsle_metric,
        }

        # Setting custom method support
        if self.metric == "CUSTOM":
            # Establishing argument checking using inspect module
            custom_method_validator = inspect.getfullargspec(kwargs["custom_method"])

            if custom_method_validator.varkw is None:
                raise ValueError(
                    "Need to provide 'kwargs'. Follow syntax: custom_method(y_true, y_pred, **kwargs)"
                )

            if not all(x in custom_method_validator.args for x in ["y_pred", "y_true"]):
                raise ValueError(
                    "Need to provide 'y_true', 'y_pred'. Follow syntax: custom_method(y_true, y_pred, **kwargs)"
                )

            self._metrics_method_config["CUSTOM"] = kwargs["custom_method"]

    def __call__(self, y_true, y_pred, **kwargs):
        return self._metrics_method_config[self.metric](y_true, y_pred, **kwargs)

    def __repr__(self):
        return f"Regression Metric Configured: {str(self.metric)}"

    def __str__(self):
        return self.__repr__()
