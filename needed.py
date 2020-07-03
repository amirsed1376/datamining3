import sqlite3
import os


class SqlManager:
    def __init__(self, file):
        self.conn = sqlite3.connect(file)
        self.crs = self.conn.cursor()


def create_folder(address):
    if not os.path.exists(address):
        os.makedirs(address)
