from neuraml.exceptions.exceptions import (
    NoneError,
    EmptyDataFrameError,
    InstanceNotCalledError,
)
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from pydantic import BaseModel, validator
from typing_extensions import Literal
from typing import Optional, Dict, Union

import numpy as np
import pandas as pd

__all__ = ["ClsDataPreProcessing"]


class Imputation(BaseModel):
    global_numerical_impute_strategy: Literal[
        "mean", "median", "mode", "max", "min", "zero"
    ] = "mean"
    global_categorical_impute_strategy: Literal["unknown", "mode"] = "unknown"
    impute_strategy: Dict[str, Dict[str, Union[str, int, float]]] = {}

    class config:
        extra = "allow"


class Encoding(BaseModel):
    pass


class Outliers(BaseModel):
    pass


class Bucketing(BaseModel):
    global_bucket_size: int = 10
    bucket_strategy: Dict[str, Dict[str, int]] = {}

    class config:
        extra = "allow"


class ClsVariableImputation(Imputation):
    def __init__(self, **kwargs) -> None:
        Imputation.__init__(self, **kwargs)

        # Setting up internal attributes
        self.state_flag = False
        self.applied_impute_details = dict()

    def _set_imputation(
        self,
        dataframe: pd.DataFrame,
        variable: str,
        variable_dtype: str,
        imputation: str,
        imputation_value: Union[str, int, float] = None,
    ):
        # Step-1 Check first that variable has missing values or not
        # we are creating a boolean flag
        missing_flag = dataframe[variable].isnull().sum() > 0

        # Step-2 Compare based on the below logic
        imputation = str(imputation).lower()

        if variable_dtype == "numeric":
            if missing_flag:
                if imputation == "value":
                    value = imputation_value
                elif imputation == "mean":
                    value = np.round(dataframe[variable].mean(), 2)
                elif imputation == "median":
                    value = np.round(dataframe[variable].median(), 2)
                elif imputation == "mode":
                    value = np.round(dataframe[variable].mode()[0], 2)
                elif imputation == "max":
                    value = np.round(dataframe[variable].max(), 2)
                elif imputation == "min":
                    value = np.round(dataframe[variable].min(), 2)
                elif imputation == "zero":
                    value = 0
                else:
                    raise ValueError("Imputation Method: Not Present!")
            else:
                value = imputation_value

        elif variable_dtype == "categoric":
            if missing_flag:
                if imputation == "value":
                    value = str(imputation_value)
                elif imputation == "unknown":
                    value = "unknown"
                elif imputation == "mode":
                    value = str(dataframe[variable].mode()[0])
                else:
                    raise ValueError("Imputation Method: Not Present!")
            else:
                value = imputation_value

        else:
            raise ValueError("Value should be from [numeric, categoric]")

        # Step-3 Filling up applied impute details dictionary
        self.applied_impute_details[variable] = {
            "imputation_value": value,
            "imputation_method": imputation,
            "missing_flag": missing_flag,
        }

    def _fit(self, dataframe: pd.DataFrame):
        """ """
        # Step-1 Iterate over all the columns mentioned in keys
        for variable in self.impute_strategy.keys():
            # Check what operation is specified in the dictionary for each variable
            # returns a dictionary
            # Format {'imputation' : 'Mean',
            #         'variable_dtype' : 'numeric/categoric',
            #         'impute_value' : int/float}
            imputation = self.impute_strategy[variable]["imputation"]
            variable_dtype = self.impute_strategy[variable]["variable_dtype"]
            imputation_value = self.impute_strategy[variable].get(
                "imputation_value", None
            )

            # Step-2 Setting the imputation and filling the
            # imputation strategy
            self._set_imputation(
                dataframe=dataframe,
                variable=variable,
                variable_dtype=variable_dtype,
                imputation=imputation,
                imputation_value=imputation_value,
            )

    def _transform(self, dataframe: pd.DataFrame):
        """_summary_

        Args:
            dataframe (pd.DataFrame): _description_
        """
        # Step-1 Iterate all the columns from the imputation strategy dictionary
        for variable in self.impute_strategy.keys():
            # Step-1a Get the imputation value from applied imputation details dictionary
            imputation_value = self.applied_impute_details[variable]["imputation_value"]

            # Step-2 Fill the NaNs in the variable
            if imputation_value is None:
                pass
            else:
                dataframe[variable].fillna(imputation_value, inplace=True)

        return dataframe

    def __call__(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """_summary_

        Args:
            dataframe (pd.DataFrame): _description_

        Raises:
            EmptyDataFrameError: _description_
            NoneError: _description_

        Returns:
            pd.DataFrame: _description_
        """
        # Step-1 Check for empty dataframe
        if isinstance(dataframe, pd.DataFrame):
            if dataframe.shape[0] == 0:
                raise EmptyDataFrameError()
        elif dataframe is None:
            raise NoneError()

        # Step-2 Setting state_flag of instance to True
        self.state_flag = True

        # Step-3 Fit and transform the data
        # which in turn will impute the data as mentioned
        self._fit(dataframe=dataframe)
        output_dataframe = self._transform(dataframe=dataframe)

        return output_dataframe

    def _get_dict_as_string(self, input_dict: dict) -> str:
        """_summary_

        Args:
            input_dict (dict): _description_

        Returns:
            str: _description_
        """
        output_list = []

        for column_name, impute_dict in input_dict.items():
            temp_list = []

            for key, value in impute_dict.items():
                temp_list.append(str(key) + ": " + str(value))
            output_list.append(str(column_name) + " -> " + ", ".join(temp_list))

        return " \n".join(output_list)

    def __repr__(self):
        # Step-1 Check state_flag status
        if self.state_flag:
            return self._get_dict_as_string(input_dict=self.applied_impute_details)
        else:
            # Step-2 Else raise error
            raise InstanceNotCalledError()

    def __str__(self):
        return self.__repr__()


class ClsVariableBucketing(Bucketing):
    def __init__(self, **kwargs) -> None:
        Bucketing.__init__(self, **kwargs)

        # Setting up internal attributes
        self.state_flag = False
        self.applied_bucket_details = dict()

    def _do_bucketing(self, dataframe: pd.DataFrame, variable: str, bucket_size: int):
        """_summary_

        Args:
            dataframe (pd.DataFrame): _description_
            variable (str): _description_
            bucket_size (int): _description_

        Returns:
            _type_: _description_
        """
        dataframe[variable + "_BK"] = pd.cut(
            dataframe[variable], bins=bucket_size
        ).apply(lambda x: x.mid)

        return dataframe

    def _fit(self, dataframe: pd.DataFrame):
        """ """
        # Step-1 Iterate over all the columns mentioned in keys
        for variable in self.bucket_strategy.keys():
            # Check what operation is specified in the dictionary for each variable
            # returns a dictionary
            # Format {'bucket_size' : 'int'}
            # Step-1a Fetch the bucket size
            bucket_size = self.bucket_strategy[variable].get(
                "bucket_size", self.global_bucket_size
            )

            # Step-2 Fill the applied bucket details dictionary
            self.applied_bucket_details[variable] = {
                "bucket_size": bucket_size,
                "bucket_variable_name": variable + "_BK",
            }

    def _transform(self, dataframe: pd.DataFrame):
        """_summary_

        Args:
            dataframe (pd.DataFrame): _description_
        """
        # Step-1 Iterate all the columns from the bucket strategy dictionary
        for variable in self.bucket_strategy.keys():
            # Step-1a Get the bucket_size from applied bucket details dictionary
            bucket_size = self.applied_bucket_details[variable]["bucket_size"]

            # Step-2 Transform the data to create bucket columns
            dataframe = self._do_bucketing(
                dataframe=dataframe, variable=variable, bucket_size=bucket_size
            )

        return dataframe

    def __call__(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """_summary_

        Args:
            dataframe (pd.DataFrame): _description_

        Raises:
            EmptyDataFrameError: _description_
            NoneError: _description_

        Returns:
            pd.DataFrame: _description_
        """
        # Step-1 Check for empty dataframe
        if isinstance(dataframe, pd.DataFrame):
            if dataframe.shape[0] == 0:
                raise EmptyDataFrameError()
        elif dataframe is None:
            raise NoneError()

        # Step-2 Setting state_flag of instance to True
        self.state_flag = True

        # Step-3 Fit and transform the data
        # which in turn will impute the data as mentioned
        self._fit(dataframe=dataframe)
        output_dataframe = self._transform(dataframe=dataframe)

        return output_dataframe

    def _get_dict_as_string(self, input_dict: dict) -> str:
        """_summary_

        Args:
            input_dict (dict): _description_

        Returns:
            str: _description_
        """
        output_list = []

        for column_name, impute_dict in input_dict.items():
            temp_list = []

            for key, value in impute_dict.items():
                temp_list.append(str(key) + ": " + str(value))
            output_list.append(str(column_name) + " -> " + ", ".join(temp_list))

        return " \n".join(output_list)

    def __repr__(self):
        # Step-1 Check state_flag status
        if self.state_flag:
            return self._get_dict_as_string(input_dict=self.applied_bucket_details)
        else:
            # Step-2 Else raise error
            raise InstanceNotCalledError()

    def __str__(self):
        return self.__repr__()
