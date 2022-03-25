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
    impute_strategy: Dict[str, Dict[str, Union[float, int, str]]] = {}

    class Config:
        extra = "allow"


class Capping(BaseModel):
    global_capping_strategy: Literal["IQR", "PERCENTILES"] = "IQR"
    capping_strategy: Dict[str, Dict[str, Union[float, int, List[float], str]]] = {}

    class Config:
        extra = "allow"


class Encoding(BaseModel):
    global_encoding_strategy: Literal[
        "LabelEncoder", "OneHotEncoder", "OrdinalEncoder"
    ] = "LabelEncoder"
    global_replace_flag: bool = False
    encoding_strategy: Dict[str, Dict[str, Union[bool, str]]] = {}

    class Config:
        extra = "allow"


class Bucketing(BaseModel):
    global_bucket_size: int = 10
    bucket_strategy: Dict[str, Dict[int, str]] = {}

    class Config:
        extra = "allow"


class PreProcessing(BaseModel):
    imputation: Imputation
    capping: Capping
    encoding: Encoding
    bucketing: Bucketing
    enable_imputation: bool = False
    enable_capping: bool = False
    enable_encoding: bool = False
    enable_bucketing: bool = False

    class Config:
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
                    value = float(imputation_value)
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
                value = float(imputation_value)

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
                value = str(imputation_value)

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
            #         'imputation_value' : int/float}
            variable_dtype = self.impute_strategy[variable]["variable_dtype"]

            if variable_dtype == "numeric":
                imputation = self.impute_strategy[variable].get(
                    "imputation", self.global_numerical_impute_strategy
                )
            elif variable_dtype == "categoric":
                imputation = self.impute_strategy[variable].get(
                    "imputation", self.global_categorical_impute_strategy
                )
            else:
                raise ValueError("Value should be from [numeric, categoric]")

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
        # Step-0 Create a copy of original dataframe
        imputated_dataframe = dataframe.copy()

        # Step-1 Iterate all the columns from the imputation strategy dictionary
        for variable in self.applied_impute_details.keys():
            # Step-1a Get the imputation value from applied imputation details dictionary
            imputation_value = self.applied_impute_details[variable]["imputation_value"]

            # Step-2 Fill the NaNs in the variable
            if imputation_value is None:
                pass
            else:
                imputated_dataframe[variable].fillna(imputation_value, inplace=True)

        return imputated_dataframe

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
                            float(capping_params), (int, float)
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
                            capping_type=float(capping_method),
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
        # Step-0 Create a copy of original dataframe
        capped_dataframe = dataframe.copy()

        # Step-1 Iterate all the columns from the applied bucket strategy dictionary
        for variable in self.applied_capping_details.keys():
            lower_bound = self.applied_capping_details[variable]["lower_bound"]
            upper_bound = self.applied_capping_details[variable]["upper_bound"]

            # Step-2 Removing outliers greater than upper bound from the capped_dataframe
            capped_dataframe[variable] = np.where(
                capped_dataframe[variable] > upper_bound,
                upper_bound,
                capped_dataframe[variable],
            )

            # Step-3 Removing outliers lower than lower bound from the capped_dataframe
            capped_dataframe[variable] = np.where(
                capped_dataframe[variable] < lower_bound,
                lower_bound,
                capped_dataframe[variable],
            )

        return capped_dataframe

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


class ClsVariableEncoding(Encoding):
    def __init__(self, **kwargs) -> None:
        Encoding.__init__(self, **kwargs)

        # Setting up internal attributes
        self.state_flag = False
        self.applied_encoding_details = dict()

    def _get_encoder(
        self,
        dataframe: pd.DataFrame,
        variable: str,
        encoding_method: str,
        replace_value: bool,
    ):
        """_summary_

        Args:
            dataframe (pd.DataFrame): _description_
            variable (str): _description_
            encoding_method (str): _description_
        """
        # Step-1 Based on the specified encoder
        # Create an encoder instance and save it
        if encoding_method == "LabelEncoder":
            # Fitting LabelEncoder
            encoder = LabelEncoder().fit(dataframe[variable].to_numpy())

        elif encoding_method == "OneHotEncoder":
            # Fitting OneHotEncoder
            encoder = OneHotEncoder(sparse=False).fit(
                dataframe[variable].to_numpy().reshape(-1, 1)
            )

        elif encoding_method == "OrdinalEncoder":
            # Fitting OrdinalEncoder
            encoder = OrdinalEncoder().fit(
                dataframe[variable].to_numpy().reshape(-1, 1)
            )
        else:
            raise ValueError(
                "encoding_method: Value Not Present In List! Please select from given list [LabelEncoder, OneHotEncoder, OrdinalEncoder]"
            )

        # Step-2 Saving the above created encoder instance
        self.applied_encoding_details[variable] = {
            "encoder": encoder,
            "replace_value": replace_value,
            "encoding_method": encoding_method,
        }

    def _fit(self, dataframe: pd.DataFrame):
        """_summary_

        Args:
            dataframe (pd.DataFrame): _description_
        """
        # Step-1 Iterate over all the columns mentioned in keys
        for variable in self.encoding_strategy.keys():
            # Check what operation is specified in the dictionary for each variable
            # Format : variable name will be key and each variable will have a dict
            # {'encoding_method' : 'LabelEncoder',
            #  'replace_value' : False}

            # Step-2 Get the specified encoding method and replace value flag
            encoding_method = self.encoding_strategy[variable].get(
                "encoding_method", self.global_encoding_strategy
            )

            replace_value = self.encoding_strategy[variable].get(
                "replace_value", self.global_replace_flag
            )

            # Step-3 Fit the encoder on the specified column and store
            # the details inside the appled_encoding_details dictionary
            self._get_encoder(
                dataframe=dataframe,
                variable=variable,
                encoding_method=encoding_method,
                replace_value=replace_value,
            )

    def _do_transform(self, dataframe: pd.DataFrame, variable: str):
        """_summary_

        Args:
            dataframe (pd.DataFrame): _description_
            variable (str): _description_
            replace_value (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        # Step-0 Create a copy of original dataframe
        encoded_dataframe = dataframe.copy()

        # Step-1 Get the encoder, replace_value & encoding_method
        # from the applied_encoding_details dictionary
        encoder = self.applied_encoding_details[variable]["encoder"]
        replace_value = self.applied_encoding_details[variable]["replace_value"]
        encoding_method = self.applied_encoding_details[variable]["encoding_method"]

        # Step-2 We are going to transform the column
        # and return the encoded_dataframe
        if encoding_method == "LabelEncoder":
            # replace_value flag determine if we need to drop the source column or not
            # if replace_value is true then drop column else do not drop the column
            if replace_value:
                encoded_dataframe[variable] = encoder.transform(
                    encoded_dataframe[variable].to_numpy().ravel()
                )
            else:
                encoded_dataframe[variable + "_Enc"] = encoder.transform(
                    encoded_dataframe[variable].to_numpy().ravel()
                )

        elif encoding_method == "OneHotEncoder":
            # Transformation of the column
            transformed = encoder.transform(
                encoded_dataframe[variable].to_numpy().reshape(-1, 1)
            )

            # one hot encoded encoded_dataframe
            columns = [
                variable + "_" + "_".join(i.split("_")[1:])
                for i in encoder.get_feature_names()
            ]
            ohe_df = pd.DataFrame(transformed, columns=columns)

            if replace_value:
                encoded_dataframe = pd.concat([encoded_dataframe, ohe_df], axis=1).drop(
                    [variable], axis=1
                )
            else:
                encoded_dataframe = pd.concat([encoded_dataframe, ohe_df], axis=1)

        elif encoding_method == "OrdinalEncoder":
            if replace_value:
                encoded_dataframe[variable] = encoder.transform(
                    encoded_dataframe[variable].to_numpy().reshape(-1, 1)
                ).ravel()
            else:
                encoded_dataframe[variable + "_Enc"] = encoder.transform(
                    encoded_dataframe[variable].to_numpy().reshape(-1, 1)
                ).ravel()

        else:
            raise ValueError(
                "Incorrect value passed for encoding_method should be in LabelEncoder, OneHotEncoder, OrdinalEncoder"
            )

        return encoded_dataframe

    def _transform(self, dataframe: pd.DataFrame):
        """_summary_

        Args:
            dataframe (pd.DataFrame): _description_
        """
        # Step-0 Create a copy of original dataframe
        encoded_dataframe = dataframe.copy()

        # Step-1 Iterate all the columns from the applied_encoding_details dictionary
        for variable in self.applied_encoding_details.keys():
            # Step-2 Based on the applied_encoding_details dictionary
            # perform the transformation
            encoded_dataframe = self._do_transform(
                dataframe=encoded_dataframe, variable=variable
            )

        return encoded_dataframe

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
        # Step-0 Create a copy of original dataframe
        bucketed_dataframe = dataframe.copy()

        # Step-1 Iterate all the columns from the bucket strategy dictionary
        for variable in self.applied_bucket_details.keys():
            # Step-1a Get the bucket_size from applied bucket details dictionary
            bucket_size = self.applied_bucket_details[variable]["bucket_size"]

            # Step-2 Transform the data to create bucket columns
            bucketed_dataframe = self._do_bucketing(
                dataframe=bucketed_dataframe, variable=variable, bucket_size=bucket_size
            )

        return bucketed_dataframe

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


class ClsDataPreProcessing(PreProcessing):
    def __init__(self, **kwargs) -> None:
        """_summary_"""
        # Step-1
        PreProcessing.__init__(self, **kwargs)

        # Step-2 Create instance of each pre-processing class
        self._imputation_ins = ClsVariableImputation(**self.imputation.dict())
        self._capping_ins = ClsVariableCapping(**self.capping.dict())
        self._encoding_ins = ClsVariableEncoding(**self.encoding.dict())
        self._bucketing_ins = ClsVariableBucketing(**self.bucketing.dict())

        # Step- Setting up internal attributes
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
        if self.enable_imputation:
            dataframe = self._imputation_ins(dataframe=dataframe)

        ## Step-3b Execute Outlier Capping
        if self.enable_capping:
            dataframe = self._capping_ins(dataframe=dataframe)

        ## Step-3c Execute Variable Encoding
        if self.enable_encoding:
            dataframe = self._encoding_ins(dataframe=dataframe)

        ## Step-3d Execute Variable Bucketing
        if self.enable_bucketing:
            dataframe = self._bucketing_ins(dataframe=dataframe)

        return dataframe

    def __repr__(self):
        # Step-1 Check processing_status
        if self.processing_status:
            return (
                "Data Pre-Processing: "
                + "\nImputation Done?: "
                + str(self.enable_imputation)
                + "\nCapping Done?: "
                + str(self.enable_capping)
                + "\nEncoding Done?: "
                + str(self.enable_encoding)
                + "\nBucketing Done?: "
                + str(self.enable_bucketing)
            )
        else:
            # Step-2 Else raise error
            raise InstanceNotCalledError()

    def __str__(self):
        return self.__repr__()
