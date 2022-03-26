from typing import Dict, List, Optional, Union

from neuraml.core.models.metrics.classification import ClsClassificationMetrics
from neuraml.core.models.metrics.regression import ClsRegressionMetrics
from pydantic import BaseModel, validator
from typing_extensions import Literal

import optuna

__all__ = ["ClsOptunaOptimizer"]


class OptunaSampler(BaseModel):
    sampler_type: Literal["TPE"] = "TPE"

    @validator("sampler_type", pre=True)
    def check_sampler_type(cls, value):
        return value.upper()


class ModelParameters(BaseModel):
    model: str
    direction: Literal["maximize", "minimize"]
    scoring_method: Union[ClsRegressionMetrics, ClsClassificationMetrics]

    @validator("direction", pre=True)
    def check_direction(cls, value):
        return value.lower()


class ModelHyperparameters(BaseModel):
    min_value: Union[float, int, str]
    max_value: Union[float, int, str]
    step_value: Optional[Union[float, int, str]]


class OptunaMetrics(BaseModel):
    sampler_config: OptunaSampler
    model_config: ModelParameters
    params: Dict[str, Union[float, int, str]]
    model_hyperparams: Dict[str, Union[ModelHyperparameters, List[str]]]
    seed: int = 42

    class Config:
        extra = "allow"


class ClsOptunaOptimizer(OptunaMetrics):
    def __init__(self, **kwargs) -> None:
        OptunaMetrics.__init__(self, **kwargs)

    def _setup_sampler(self):
        if self.sampler_config.sampler_type == "TPE":
            self.sampler = optuna.samplers.TPESampler(seed=self.seed)

    def _setup_objective_function(self, trial):
        # Step-1 Create empty trial parameter dictionary
        trial_params_dict = dict()

        # Step-2 Fill the first parameter nthread to be -1
        trial_params_dict["nthread"] = -1

        # Step-3 Based on the model hyperparameters
        # we will iterate and fill the trial_params_dict
        for hyparam, value_packet in self.model_hyperparams.items():
            if isinstance(value_packet, list):
                # If value_packet is of type List then
                # we will use suggest_categorical method
                trial_params_dict[hyparam] = trial.suggest_categorical(
                    name=hyparam, choices=value_packet
                )
            else:
                if value_packet.step_value is None:
                    if isinstance(value_packet.min_value, int):
                        # If value_packet.step_value is None and
                        # value_packet.min_value is of type int
                        # we will use suggest_int method
                        trial_params_dict[hyparam] = trial.suggest_int(
                            name=hyparam,
                            low=value_packet.min_value,
                            high=value_packet.max_value,
                        )
                    else:
                        # Else we will be use suggest_loguniform method
                        trial_params_dict[hyparam] = trial.suggest_loguniform(
                            name=hyparam,
                            low=value_packet.min_value,
                            high=value_packet.max_value,
                        )
                else:
                    # If value_packet has step_value present then
                    # we will use suggest_discrete_uniform
                    trial_params_dict[hyparam] = trial.suggest_discrete_uniform(
                        name=hyparam,
                        low=value_packet.min_value,
                        high=value_packet.max_value,
                        q=value_packet.step_value,
                    )

        # Step-4 Calling the model to optimize

        # Step-5 Training the model on set of hyperparameters and scoring
        # However we are only in validation_score
        _, validation_score = self.model_hyperparams.scor

        return valid_score

    def _setup_optimizer(self):
        self.study = optuna.create_study(
            direction=self.model_config.direction, sampler=self.sampler
        )

    def begin_optimization(self):
        self.study.optimize(
            func=self._setup_objective_function, n_trials=self.params["n_trials"]
        )
