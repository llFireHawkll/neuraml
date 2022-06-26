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
