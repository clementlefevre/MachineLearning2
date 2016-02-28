import os

__author__ = 'ThinkPad'

DB_URL = "postgresql+pg8000://postgres:clement2014@localhost:5432/DWE_MAUSFRERES_2012"

CHART_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "charts")
