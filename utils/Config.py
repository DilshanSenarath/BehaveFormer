import os
import json
from pathlib import Path

class Config:
    _instance = None
    _data = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, path = "../../config.json"):
        self.path = (Path(__file__)/path).resolve()

    def get_config_dict(self):
        if not Config._data:
            with open(self.path) as file:
                Config._data = json.load(file)

        return Config._data
