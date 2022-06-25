from typing import Dict, List, Optional, Union

from neuraml.exceptions.exceptions import EmptyListError, InstanceNotCalledError
from pandas.api.types import is_object_dtype
from pydantic import BaseModel, validator

__all__ = ["ClsDataFeatureSet"]


class FeatureSetColumns(BaseModel):
    _id: str
    time: Optional[str]
    target: str
    regression: List[str]
    segmentation: Optional[List[str]]
    dtypes: Dict[str, Union[object, int, float]]
    nuniques: Dict[str, int]

    class Config:
        extra = "allow"
        arbitrary_types_allowed = True

    @validator("regression")
    def validate_regression(cls, value):
        if len(value) == 0:
            raise EmptyListError()
        return value


class ClsDataFeatureSet(FeatureSetColumns):
    """ClsDataFeatureSet class is responsible for maintaining
    the underlying configuration for our data model. This class
    has method to idenify variables as numerical, categorical and object types

    Args:
        FeatureSetColumns (_type_): _description_
    """

    def __init__(self, **kwargs) -> None:
        """Initializing the FeatureSetColumns pydantic basemodel with required parameters"""
        FeatureSetColumns.__init__(self, **kwargs)

        # Check and set the id variable
        self._check_and_set_id_variable()

        # Setting up internal attributes
        self.state_flag = False  # For storing the state of instance
        self.objects_cols = list()  # For storing object columns
        self.numerical_cols = list()  # For storing numerical columns
        self.categorical_cols = list()  # For storing categorical columns

    def _check_and_set_id_variable(self):
        """This method checks and set the index of the featureset
        This "id_" will be used as primary key
        """
        if self.id_ is None:
            self.id_ = "index"

    def _get_features_segmentation(self):
        """This method checks and classifies variables into
        numerical, categorical and object dtypes.
        """
        # Step-1 Iterate each variable from dtypes dict
        for variable in self.dtypes.keys():
            if variable == self.id_:
                pass
            elif variable == self.time:
                pass
            elif variable == self.target:
                pass
            else:
                if is_object_dtype(self.dtypes[variable]):
                    self.objects_cols.append(variable)
                else:
                    self.numerical_cols.append(variable)

        # Step-2 Iterate over all variables in object_cols list
        # and dump the variables which have atleast 10 unique values
        # into categorical_cols list
        for variable in self.objects_cols:
            if self.nuniques[variable] <= 10:
                self.categorical_cols.append(variable)

    def __call__(self):
        """This method is where the instance internal methods are called and all the
        magic happens here.
        """
        # Step-1 Setting state_flag of instance to True
        self.state_flag = True

        # Step-2 Populate object, numerical and categorical columns
        self._get_features_segmentation()

    def __repr__(self):
        """This method is used to get a summary of the featureset instance"""
        # Step-1 Check state_flag status
        if self.state_flag:
            output_string = f"""Feature Set Configuration:
            Target Variable: "{str(self.target)}"
            Regression Variables: ["{'", '.join(self.regression)}"]
            Segmentation Variables: ["{'", '.join(self.segmentation)}"]
            Regression Variables Count: {str(len(self.regression))}
            Segmentation Variables Count: {str(len(self.segmentation))}
            Numerical Variables Count: {str(len(self.numerical_cols))}
            Categorical Variables Count: {str(len(self.categorical_cols))}
            Object Variables Count: {str(len(self.objects_cols))}"""

            return output_string
        else:
            # Step-2 Else raise error
            raise InstanceNotCalledError()

    def __str__(self):
        """This method is used to get a summary of the featureset instance when print()
        is called. Internally calls __repr__() special method.

        Returns:
            __repr__: This method is used to get a summary of the featureset instance
        """
        return self.__repr__()
