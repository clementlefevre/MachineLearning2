from datetime import datetime

__author__ = 'ThinkPad'


def getMonth(date_posted):
    date = datetime.strptime(date_posted, '%Y-%m-%d')
    return date.month


def getSeason(month):
    if month < 4 or month > 11:
        return 1
    elif month < 7:
        return 2
    elif month < 9:
        return 3
    elif month < 11:
        return 4


def getDay(date_posted):
    date = datetime.strptime(date_posted, '%Y-%m-%d')
    return date.day
