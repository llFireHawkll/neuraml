from typing import Dict, List, Union

import numpy as np
import pandas as pd
from neuraml.exceptions.exceptions import (
    EmptyDataFrameError,
    InstanceNotCalledError,
    NoneError,
)
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from typing_extensions import Literal

__all__ = ["ClsDataPreProcessing"]

# TODO: 1. Add the global mechanism to check all the features if they needs all the preprocessing
# TODO: 2. Add rigrous documentation for each class based on pep8 standards


def _get_dict_as_string(input_dict) -> str:
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


class Imputation(BaseModel):
    global_numerical_impute_strategy: Literal[
        "mean", "median", "mode", "max", "min", "zero"
    ] = "mean"
    global_categorical_impute_strategy: Literal["unknown", "mode"] = "unknown"
    impute_strategy: Dict[str, Dict[str, Union[str, int, float]]] = {}

    class config:
        extra = "allow"


class Encoding(BaseModel):
    global_encoding_strategy: Literal[
        "LabelEncoder", "OneHotEncoder", "OrdinalEncoder"
    ] = "LabelEncoder"
    global_replace_flag: bool = False
    encoding_strategy: Dict[str, Dict[str, Union[str, bool]]] = {}

    class config:
        extra = "allow"


class Capping(BaseModel):
    global_capping_strategy: Literal["IQR", "PERCENTILES"] = "IQR"
    capping_strategy: Dict[str, Dict[str, Union[str, int, float]]] = {}

    class config:
        extra = "allow"


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

    def __repr__(self):
        # Step-1 Check state_flag status
        if self.state_flag:
            return _get_dict_as_string(input_dict=self.applied_impute_details)
        else:
            # Step-2 Else raise error
            raise InstanceNotCalledError()

    def __str__(self):
        return self.__repr__()


class ClsVariableEncoding(Encoding):
    def __init__(self, **kwargs) -> None:
        Encoding.__init__(self, **kwargs)

        # Setting up internal attributes
        self.state_flag = False
        self.applied_encoding_details = dict()

    def _fit(self, dataframe: pd.DataFrame):
        """ """
        # Step-1 Iterate over all the columns mentioned in keys
        for variable in self.capping_strategy.keys():
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

    def __repr__(self):
        # Step-1 Check state_flag status
        if self.state_flag:
            return _get_dict_as_string(input_dict=self.applied_capping_details)
        else:
            # Step-2 Else raise error
            raise InstanceNotCalledError()

    def __str__(self):
        return self.__repr__()


class ClsVariableCapping(Capping):
    def __init__(self, **kwargs) -> None:
        Capping.__init__(self, **kwargs)

        # Setting up internal attributes
        self.state_flag = False
        self.applied_capping_details = dict()

    def _get_capping_bounds(
        self,
        dataframe: pd.DataFrame,
        variable: str,
        capping_type: str,
        iqr_range_bound: float = 1.5,
        percentile_range: List[float] = [0.0, 99.9],
    ):
        """_summary_

        Args:
            dataframe (pd.DataFrame): _description_
            variable (str): _description_
            capping_type (str): _description_
            iqr_range_bound (float, optional): _description_. Defaults to 1.5.
            percentile_range (List[float], optional): _description_. Defaults to [0, 99.9].
        """
        # Step-1 Sort the dataframe specified column/variable - Ascending sort
        dataframe_column = np.sort(dataframe[variable])

        # Step-2 Based on the capping_type
        if capping_type == "IQR":
            # Step-2a. Calculate the percentile 25 and 75
            Q1, Q3 = np.percentile(dataframe_column, [25, 75])

            # Step-2b. Compute the IQR Range
            IQR = Q3 - Q1

            # Step-2c. Compute the Lower and Upper Bound
            lower_bound = Q1 - (iqr_range_bound * IQR)
            upper_bound = Q3 + (iqr_range_bound * IQR)

        elif capping_type == "PERCENTILES":
            # Step-2a. Calculate the percentile  and 75
            Qmin, Qmax = np.percentile(dataframe_column, percentile_range)

            # Step-2b. Compute the Lower and Upper Bound
            lower_bound = Qmin
            upper_bound = Qmax

        else:
            raise ValueError(
                "Incorrect value passed for capping_type should be in IQR, PERCENTILES"
            )

        # Step-3 Filling up applied capping details dictionary
        self.applied_capping_details[variable] = {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        }

    def _fit(self, dataframe: pd.DataFrame):
        """_summary_

        Args:
            dataframe (pd.DataFrame): _description_
        """
        # Step-1 Iterate over all the columns mentioned in keys
        for variable in self.capping_strategy.keys():
            # Check what operation is specified in the dictionary for each variable
            # returns a dictionary
            # Format {'capping_method' : 'IQR',
            #         'capping_params' : int/float}
            #        {'capping_method' : 'PERCENTILES',
            #         'capping_params' : []}

            # Step-2 Get the configuration for variable it is necessary to supply
            # configuration or pass an empty dictionary(we use global strategy to do capping)
            variable_capping_config = self.capping_strategy.get(variable, {})

            # Step-3 Check the configuration if it is empty then apply
            # global capping strategy for that particular variable
            if len(variable_capping_config) == 0:
                self._get_capping_bounds(
                    dataframe=dataframe,
                    variable=variable,
                    capping_type=self.global_capping_strategy,
                )
            else:
                # Step-4a Take the configuration variables out from dictionary
                capping_method = variable_capping_config.get(
                    "capping_method", self.global_capping_strategy
                )
                capping_params = variable_capping_config.get("capping_params", None)

                # Step-4b If the capping method is IQR
                if capping_method == "IQR":
                    # Step-4ba Checking if any params are passed or not
                    if capping_params is None:
                        self._get_capping_bounds(
                            dataframe=dataframe,
                            variable=variable,
                            capping_type=capping_method,
                        )
                    else:
                        # Step-4ba Ensuring that passed params are of correct type
                        assert isinstance(
                            capping_params, (int, float)
                        ), "IQR Range bound is not an integer/float type"

                        # Step-4bb Calling the method to populate the dictionary for
                        # variable with correct upper and lower bounds
                        self._get_capping_bounds(
                            dataframe=dataframe,
                            variable=variable,
                            capping_type=capping_method,
                            iqr_range_bound=capping_params,
                        )

                # Step-4b If the capping method is PERCENTILES
                elif capping_method == "PERCENTILES":
                    # Step-4ba Checking if any params are passed or not
                    if capping_params is None:
                        self._get_capping_bounds(
                            dataframe=dataframe,
                            variable=variable,
                            capping_type=capping_method,
                        )
                    else:
                        # Step-4ba Ensuring that passed params are of
                        # correct type and correct length
                        assert (
                            len(capping_params) == 2
                        ), "Please Supply List of [LB, UB] in the config"

                        assert isinstance(
                            capping_params[0], (int, float)
                        ), "LB is not an integer/float instance"
                        assert isinstance(
                            capping_params[1], (int, float)
                        ), "UB is not an integer/float instance"

                        # Step-4bb Calling the method to populate the dictionary for
                        # variable with correct upper and lower bounds
                        self._get_capping_bounds(
                            dataframe=dataframe,
                            variable=variable,
                            capping_type=capping_method,
                            percentile_range=capping_params,
                        )
                else:
                    # Step-4b Raising type error for incorrect capping method
                    raise NotImplementedError(
                        str(capping_method)
                        + " -- Not Implemented Select From [IQR, PERCENTILES]"
                    )

    def _transform(self, dataframe: pd.DataFrame):
        """_summary_

        Args:
            dataframe (pd.DataFrame): _description_

        Returns:
            _type_: _description_
        """
        # Step-1 Iterate all the columns from the applied bucket strategy dictionary
        for variable in self.applied_capping_details.keys():
            lower_bound = self.applied_capping_details[variable]["lower_bound"]
            upper_bound = self.applied_capping_details[variable]["upper_bound"]

            # Step-2 Removing outliers greater than upper bound from the dataframe
            dataframe[variable] = np.where(
                dataframe[variable] > upper_bound, upper_bound, dataframe[variable]
            )

            # Step-3 Removing outliers lower than lower bound from the dataframe
            dataframe[variable] = np.where(
                dataframe[variable] < lower_bound, lower_bound, dataframe[variable]
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

    def __repr__(self):
        # Step-1 Check state_flag status
        if self.state_flag:
            return _get_dict_as_string(input_dict=self.applied_capping_details)
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

    def _fit(self):
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
        self._fit()
        output_dataframe = self._transform(dataframe=dataframe)

        return output_dataframe

    def __repr__(self):
        # Step-1 Check state_flag status
        if self.state_flag:
            return _get_dict_as_string(input_dict=self.applied_bucket_details)
        else:
            # Step-2 Else raise error
            raise InstanceNotCalledError()

    def __str__(self):
        return self.__repr__()


class ClsDataPreProcessing:
    def __init__(
        self,
        global_numerical_impute_strategy: str,
        global_categorical_impute_strategy: str,
        impute_strategy_config: Dict[str, Dict[str, Union[str, int, float]]],
        global_capping_strategy: str,
        capping_strategy_config: Dict[str, Dict[str, int]],
        global_encoding_strategy: str,
        global_replace_flag: bool,
        encoding_strategy_config: Dict[str, Dict[str, Union[str, bool]]],
        global_bucket_size: int,
        bucket_strategy_config: Dict[str, Dict[str, int]],
        enable_imputation: bool,
        enable_capping: bool,
        enable_encoding: bool,
        enable_bucketing: bool,
    ) -> None:
        # Step-1 Create instance of each pre-processing class
        self._imputation = ClsVariableImputation(
            **{
                "global_numerical_impute_strategy": global_numerical_impute_strategy,
                "global_categorical_impute_strategy": global_categorical_impute_strategy,
                "impute_strategy": impute_strategy_config,
            }
        )
        self._capping = ClsVariableCapping(
            **{
                "global_capping_strategy": global_capping_strategy,
                "capping_strategy": capping_strategy_config,
            }
        )
        self._encoding = ClsVariableEncoding(
            **{
                "global_encoding_strategy": global_encoding_strategy,
                "global_replace_flag": global_replace_flag,
                "encoding_strategy": encoding_strategy_config,
            }
        )
        self._bucketing = ClsVariableBucketing(
            **{
                "global_bucket_size": global_bucket_size,
                "bucket_strategy": bucket_strategy_config,
            }
        )

        # Step- Setting up internal attributes
        self._imputation_status: bool = enable_imputation
        self._capping_status: bool = enable_capping
        self._encoding_status: bool = enable_encoding
        self._bucketing_status: bool = enable_bucketing
        self.processing_status: bool = False

    def __call__(self, dataframe: pd.DataFrame):
        """_summary_

        Args:
            dataframe (pd.DataFrame): _description_

        Raises:
            EmptyDataFrameError: _description_
            NoneError: _description_

        Returns:
            _type_: _description_
        """
        # Step-1 Check for empty dataframe
        if isinstance(dataframe, pd.DataFrame):
            if dataframe.shape[0] == 0:
                raise EmptyDataFrameError()
        elif dataframe is None:
            raise NoneError()

        # Step-2 Set processing_status = True
        self.processing_status = True

        # Step-3 Run the pipeline for preprocessing
        ## Step-3a Execute Data Imputation
        if self._imputation_status:
            dataframe = self._imputation(dataframe=dataframe)

        ## Step-3b Execute Outlier Capping
        if self._capping_status:
            dataframe = self._capping(dataframe=dataframe)

        ## Step-3c Execute Variable Encoding
        if self._encoding_status:
            dataframe = self._encoding(dataframe=dataframe)

        ## Step-3d Execute Variable Bucketing
        if self._bucketing_status:
            dataframe = self._bucketing(dataframe=dataframe)

        return dataframe

    def __repr__(self):
        # Step-1 Check processing_status
        if self.processing_status:
            return (
                "Data Pre-Processing: "
                + "\nImputation Done?: "
                + str(self._imputation_status)
                + "\nCapping Done?: "
                + str(self._capping_status)
                + "\nEncoding Done?: "
                + str(self._encoding_status)
                + "\nBucketing Done?: "
                + str(self._bucketing_status)
            )
        else:
            # Step-2 Else raise error
            raise InstanceNotCalledError()

    def __str__(self):
        return self.__repr__()
