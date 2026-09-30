import os
import hashlib
import time
import random
import json

class Utils:
    @staticmethod
    def encode_participant():
        raw = f"{time.time()}_{random.random()}"
        hash = hashlib.sha1(raw.encode()).hexdigest()[:4].upper()
        return f"P_{hash}"
    @staticmethod
    def create_dir(hash: str):
        path = os.path.join("data", hash)
        os.makedirs(path)
        return path
    @staticmethod
    def save_particpant_info(path: str, info: dict):
        path = os.path.join(path, "info.json")
        with open(path, "w") as f:
            json.dump(info, f, indent=4)
    @staticmethod
    def create_participant_data(info : dict):
        hash = Utils.encode_participant()
        path = Utils.create_dir(hash)
        Utils.save_particpant_info(path, info)
        return path, hash