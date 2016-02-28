from calendar_service import getMonth, getDay, getSeason
import numpy as np

__author__ = 'ThinkPad'


def convertData(row):
    # return a list of site_id(1)totalIn(0), min temp(1), max temp(2), avg temp(3), month(4), season(5), day(6),
    site_id = int(row[0])
    site_name = row[1]
    totalIn = int(row[2])
    min_temp = int(row[5])
    max_temp = int(row[4])
    precipitations_mm = row[6]
    mean_temp = (np.mean([min_temp, max_temp]))
    month = getMonth(row[3])
    day = getDay(row[3])
    season = getSeason(month)
    return [site_id, site_name, totalIn, min_temp, max_temp, mean_temp, precipitations_mm, month, season, day]
