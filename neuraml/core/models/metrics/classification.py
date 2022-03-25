import inspect

from pydantic import BaseModel, validator
from sklearn import metrics as skmetrics
from typing_extensions import Literal

__all__ = ["ClsClassificationMetrics"]


class ClassificationMetric(BaseModel):
    metric: Literal[
        "ACCURACY",
        "F1",
        "PRECISION",
        "RECALL",
        "AUC",
        "LOGLOSS",
        "KAPPA",
        "CUSTOM",
    ]

    @validator("metric", pre=True)
    def check_metric(cls, value):
        return value.upper()

    class Config:
        extra = "allow"

    def _accuracy_metric(self, y_true, y_pred, **kwargs):
        return skmetrics.accuracy_score(y_true=y_true, y_pred=y_pred, **kwargs)

    def _f1_metric(self, y_true, y_pred, **kwargs):
        return skmetrics.f1_score(y_true=y_true, y_pred=y_pred, **kwargs)

    def _precision_metric(self, y_true, y_pred, **kwargs):
        return skmetrics.precision_score(y_true=y_true, y_pred=y_pred, **kwargs)

    def _recall_metric(self, y_true, y_pred, **kwargs):
        return skmetrics.recall_score(y_true=y_true, y_pred=y_pred, **kwargs)

    def _auc_metric(self, y_true, y_pred, **kwargs):
        return skmetrics.roc_auc_score(y_true=y_true, y_pred=y_pred, **kwargs)

    def _logloss_metric(self, y_true, y_pred, **kwargs):
        return skmetrics.log_loss(y_true=y_true, y_pred=y_pred, **kwargs)

    def _kappa_metric(self, y_true, y_pred, **kwargs):
        return skmetrics.cohen_kappa_score(y1=y_true, y2=y_pred, **kwargs)


class ClsClassificationMetrics(ClassificationMetric):
    def __init__(self, **kwargs) -> None:
        ClassificationMetric.__init__(self, **kwargs)

        # Setting internal attributes
        self._metrics_method_config = {
            "ACCURACY": self._accuracy_metric,
            "F1": self._f1_metric,
            "PRECISION": self._precision_metric,
            "RECALL": self._recall_metric,
            "AUC": self._auc_metric,
            "LOGLOSS": self._logloss_metric,
            "KAPPA": self._kappa_metric,
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
        return f"Classification Metric Configured: {str(self.metric)}"

    def __str__(self):
        return self.__repr__()
