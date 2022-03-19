import pandas as pd
import pytest
from neuraml.core.data.indexing import ClsDataIndexing


@pytest.fixture(scope="module")
def get_test_dataframe():
    data = {
        "product_name": ["laptop", "printer", "tablet", "desk", "chair"],
        "price": [1200, 150, 300, 450, 200],
    }
    return pd.DataFrame(data=data)


@pytest.fixture(scope="module")
def setup_data_indexing_instance():
    """_summary_"""
    # Step-1 Instantiate the DataIndexing class object
    indexing_configuration = {
        "stratify": False,
        "enable_full_data": False,
        "train_test_split_size": 0.2,
        "random_state": 42,
    }

    obj_data_indexing = ClsDataIndexing(**indexing_configuration)

    # Step-3 Return the object
    return obj_data_indexing


def test_state_variable(setup_data_indexing_instance):
    obj_data_indexing = setup_data_indexing_instance
    assert (
        obj_data_indexing.state_flag == False
    ), "state_flag is set to true which means instance was called!"


def test_train_and_test_size(setup_data_indexing_instance, get_test_dataframe):
    obj_data_indexing = setup_data_indexing_instance
    obj_data_indexing(dataframe=get_test_dataframe)

    assert len(obj_data_indexing.train_indexes) == 4
