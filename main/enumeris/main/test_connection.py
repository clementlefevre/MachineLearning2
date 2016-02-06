from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from config import DB_URL
from main.enumeris.main.utils.settings import SQL_QUERY_PER_SITE, SQL_QUERY_ALL_SITES

__author__ = 'ThinkPad'

from sqlalchemy import create_engine, text, Column, Integer, ForeignKey, DATETIME, DATE, String

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

Base = declarative_base()

SEASONS = {1: 'winter', 2: 'spring', 3: 'summer', 4: 'autumn'}


class Site(Base):
    __tablename__ = 'dwe_bld_site'
    idbldsite = Column(Integer, primary_key=True)
    sname = Column(String)
    address = relationship('Address', backref='dwe_bld_site')
    count = relationship('Count', backref='dwe_bld_site')


class Address(Base):
    __tablename__ = 'dwe_bld_address'
    id = Column(Integer, primary_key=True)

    idbldsite = Column(Integer, ForeignKey('dwe_bld_site.idbldsite'))
    weather = relationship('Weather')


class Weather(Base):
    __tablename__ = "dwe_ext_weather_premium"
    id = Column(Integer, primary_key=True)
    day = Column(DATE)
    idbldaddress = Column(Integer, ForeignKey('dwe_bld_address.id'))


class Count(Base):
    __tablename__ = "dwe_cnt_site"
    id = Column(Integer, primary_key=True)
    idbldsite = Column(Integer, ForeignKey('dwe_bld_site.idbldsite'))
    timestamp = Column(DATETIME)


# ----------------------------------------------------------------------

def loadSession():
    engine = create_engine(DB_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    return session


def main():
    session = loadSession()
    allSites = session.query(Site).all()
    site1 = session.query(Site).filter_by(idbldsite=1).all()[0]

    result = retrieveData(session)
    data = []

    for row in result:
        data.append(convertData(row))

    np_array = np.array(data)
    print data

    x, y = filterData(np_array, 1, 1)

    plotFigure(x, y)

    print "over"


def retrieveData(session):
    sql = text(SQL_QUERY_PER_SITE)
    result = session.execute(sql, {'start_date': '2012-01-01', 'end_date': '2016-02-01', 'site_id': 1})

    sql = text(SQL_QUERY_ALL_SITES)
    result = session.execute(sql, {'start_date': '2015-12-01', 'end_date': '2016-02-01'})
    return result


def convertData(row):
    # return a list of totalIn(0), min temp(1), max temp(2), avg temp(3), month(4), season(5), day(6)
    site_id = int(row[0])
    site_name = row[1]
    totalIn = int(row[2])
    min_temp = int(row[5])
    max_temp = int(row[4])
    mean_temp = int(np.mean([min_temp, max_temp]))
    month = getMonth(row[3])
    day = getDay(row[3])
    season = getSeason(month)
    return [totalIn, min_temp, max_temp, mean_temp, month, day, season]


def filterData(np_array, season, day=None):
    # this function filter on a given day and a given season, and return the totalIn and meanTemp

    if day is not None:
        filtered_array = np_array[(np_array[:, 6] == season) & (np_array[:, 5] == day)]
    else:
        filtered_array = np_array[(np_array[:, 6] == season)]
    return filtered_array[:, 3], filtered_array[:, 0]


def plotFigure(x, y):
    plt.figure(num=None, figsize=(8, 6))
    plt.clf()
    fig, ax = plt.subplots()
    plt.xlim(np.min(x), np.max(x))
    plt.ylim(np.min(y), np.max(y))
    plt.scatter(x, y, s=4, color='r', marker='s')
    fig.savefig('Figure' + '.png')


def getMonth(date_posted):
    date = datetime.strptime(date_posted, '%Y-%m-%d')
    return date.month


def getSeason(month):
    if month < 4:
        return 1
    elif month < 6:
        return 2
    elif month < 10:
        return 3
    elif month > 9:
        return 1


def getDay(date_posted):
    date = datetime.strptime(date_posted, '%Y-%m-%d')
    return date.day
