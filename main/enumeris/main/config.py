import os

__author__ = 'ThinkPad'

DB_URL = "postgresql+pg8000://postgres:pwd@localhost:5432/DBNAME"

CHART_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "charts")
