from abc import ABC, abstractmethod

import pandas as pd
from neuraml.core.data.featureset import ClsDataFeatureSet, FeatureSetColumns
from neuraml.core.data.indexing import ClsDataIndexing, Indexing
from neuraml.core.data.preprocessing import ClsDataPreProcessing, PreProcessing
from neuraml.exceptions.exceptions import (
    EmptyDataFrameError,
    InstanceNotCalledError,
    NoneError,
)

__all__ = [
    "ClsPandasDataLoader",
]


class ClsBaseDataLoader(ABC):
    def __init__(self):
        """_summary_"""
        pass

    @abstractmethod
    def _read_data_s3(self):
        """_summary_"""
        raise NotImplementedError

    @abstractmethod
    def _write_data_s3(self):
        """_summary_"""
        raise NotImplementedError

    @abstractmethod
    def _read_data_local(self, path, file_type, **kwargs):
        """_summary_

        Args:
            path (_type_): _description_
            file_type (_type_): _description_
        """
        raise NotImplementedError

    @abstractmethod
    def _write_data_local(self, path, file_type, **kwargs):
        """_summary_

        Args:
            path (_type_): _description_
            file_type (_type_): _description_
        """
        raise NotImplementedError
