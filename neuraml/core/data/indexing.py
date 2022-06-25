from typing import Optional

import pandas as pd
from neuraml.exceptions.exceptions import (
    EmptyDataFrameError,
    InstanceNotCalledError,
    NoneError,
)
from pydantic import BaseModel, Field, validator
from sklearn.model_selection import train_test_split

__all__ = ["ClsDataIndexing"]


class Indexing(BaseModel):
    stratify: bool = False
    enable_full_data: bool = False
    stratify_variable: Optional[str]
    train_test_split_size: float = Field(0.15, gt=0.0, lt=1.0, example=0.15)
    random_state: int = 42

    class Config:
        extra = "allow"

    @validator("stratify", pre=True, always=True)
    def validate_stratify(cls, value, values):
        stratify_variable = values.get("stratify_variable")
        if stratify_variable is None:
            if value:
                raise ValueError(
                    "Error stratify_variable cannot be set None, when stratify is set True!"
                )
            else:
                return value

        return value


class ClsDataIndexing(Indexing):
    """ClsDataIndexing class is responsible for maintaining
    the underlying indexing configuration for our data model. This class
    has method to idenify correct splitting criteria and get correct
    train and test dataframes indexes

    Args:
        Indexing (_type_): _description_
    """

    def __init__(self, **kwargs) -> None:
        Indexing.__init__(self, **kwargs)

        # Setting up internal attributes
        self.state_flag = False  # For storing the state of instance

    def __call__(self, dataframe: pd.DataFrame):
        """ """
        # Step-1 Check for empty dataframe
        if isinstance(dataframe, pd.DataFrame):
            if dataframe.shape[0] == 0:
                raise EmptyDataFrameError()
        elif dataframe is None:
            raise NoneError()

        # Step-2 Setting state_flag of instance to True
        self.state_flag = True

        # Step-3 Check if enable_full_data is True
        if self.enable_full_data:
            # Step-3a Set index list for train/test
            self.train_indexes = list(dataframe.index)
            self.test_indexes = list(dataframe.index)

        else:
            # Step-4 If enable_full_data is False then we need to stratify the data
            if self.stratify:
                # Step-4a We will split the whole data
                # considering a stratification criteria
                if self.stratify_variable in dataframe.columns:
                    train_data, test_data = train_test_split(
                        dataframe,
                        test_size=self.train_test_split_size,
                        random_state=self.random_state,
                        stratify=dataframe[self.stratify_variable],
                    )
                else:
                    raise ValueError(
                        "stratify_variable: Please provide a valid column name!"
                    )
            else:
                # Step-4b We will split the whole data without
                # considering a stratification criteria
                train_data, test_data = train_test_split(
                    dataframe,
                    test_size=self.train_test_split_size,
                    random_state=self.random_state,
                )

            # Step-5 Set index list for train/test
            self.train_indexes = list(train_data.index)
            self.test_indexes = list(test_data.index)

    def __repr__(self):
        # Step-1 Check state_flag status
        if self.state_flag:
            return (
                "Index:"
                + "\nShape of Input Data: "
                + str(len(self.train_indexes) + len(self.test_indexes))
                + "\nShape of Training Data: "
                + str(len(self.train_indexes))
                + "\nShape of Testing Data: "
                + str(len(self.test_indexes))
            )
        else:
            # Step-2 Else raise error
            raise InstanceNotCalledError()

    def __str__(self):
        return self.__repr__()
