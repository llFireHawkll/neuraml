from typing import Dict, List, Optional, Union

from neuraml.exceptions.exceptions import EmptyListError, InstanceNotCalledError
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

    class config:
        extra = "allow"

    @validator("regression")
    def validate_regression(cls, value):
        if len(value) == 0:
            raise EmptyListError()
        return value


class ClsDataFeatureSet(FeatureSetColumns):
    def __init__(self, **kwargs) -> None:
        FeatureSetColumns.__init__(self, **kwargs)

        # Setting up internal attributes
        self.state_flag = False  # For storing the state of instance
        self.objects_cols = list()  # For storing object columns
        self.numerical_cols = list()  # For storing numerical columns
        self.categorical_cols = list()  # For storing categorical columns

    def _get_features_segmentation(self):
        """_summary_"""
        # Step-1 Iterate each variable from dtypes dict
        for variable in self.dtypes.keys():
            if variable == self._id:
                pass
            elif variable == self.time:
                pass
            else:
                if isinstance(self.dtypes[variable], object):
                    self.objects_cols.append(variable)
                else:
                    self.numerical_cols.append(variable)

        # Step-2 Iterate over all variables in object_cols list
        # and dump the variables which have atleast 10 unique values
        # into categorical_cols list
        for variable in self.objects_cols:
            if self.dtypes[variable] <= 10:
                self.categorical_cols.append(variable)

    def __call__(self):
        """ """
        # Step-1 Setting state_flag of instance to True
        self.state_flag = True

        # Step-2 Populate object, numerical and categorical columns
        self._get_features_segmentation()

    def __repr__(self):
        # Step-1 Check state_flag status
        if self.state_flag:
            return (
                "Feature Set:"
                + "\nTarget Variable: "
                + str(self.target)
                + "\nRegression Variables: "
                + ", ".join(self.regression)
                + "\nSegmentation Variables: "
                + ", ".join(self.segmentation)
                + "\nRegression Variables Count: "
                + str(len(self.regression))
                + "\nSegmentation Variables Count: "
                + str(len(self.segmentation))
                + "\nNumerical Variables Count: "
                + str(len(self.numerical_cols))
                + "\nCategorical Variables Count: "
                + str(len(self.categorical_cols))
                + "\nObject Variables Count: "
                + str(len(self.objects_cols))
            )
        else:
            # Step-2 Else raise error
            raise InstanceNotCalledError()

    def __str__(self):
        return self.__repr__()
