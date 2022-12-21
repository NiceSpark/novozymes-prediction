"""This file contains helper functions for writing json documents, as well as creating timestamp"""

import json
import os
from datetime import datetime


def open_json(filename: str):
    """
    open a .json file from the file path (filename)
    """
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    else:
        print(f"file {filename} does not exists !")
        return {}


def write_json(filename: str, json_file: dict):
    """
    write a json_file to the file path (filename)
    """
    with open(filename, "w+") as f:
        json.dump(json_file, f, indent=4, default=str)


def create_timestamp():
    dateTimeObj = datetime.now()
    return dateTimeObj
