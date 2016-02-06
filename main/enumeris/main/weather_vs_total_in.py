import calendar
from datetime import datetime
import os

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from config import DB_URL
from main.enumeris.main.utils.settings import SQL_QUERY_PER_SITE

__author__ = 'ThinkPad'

from sqlalchemy import create_engine, text, Column, Integer, ForeignKey, DATETIME, DATE, String

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

Base = declarative_base()

SEASONS = {1: 'winter', 2: 'spring', 3: 'summer', 4: 'autumn'}

CHART_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "charts")


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



    # allSites = [site1]
    analyzeAllSite(session, allSites)


def analyzeAllSite(session, allSites):
    data = []

    for site in allSites:
        result = retrieveData(session, site.idbldsite)

        for row in result:
            data.append(convertData(row))

    np_array = np.array(data)

    for season in range(1, 5):
        for day in range(1, 8):
            filterOnSeasonDay = filterDataOnSeasonDay(np_array, season, day)
            x, y = normalize(filterOnSeasonDay)

            plotFigure(x, y, season, day, "All Sites")
            # plot3DFigure(x, y, z, season, day, "All Sites")


def retrieveData(session, site_id):
    sql = text(SQL_QUERY_PER_SITE)
    result = session.execute(sql, {'start_date': '2013-01-01', 'end_date': '2016-02-01', 'site_id': site_id})

    # sql = text(SQL_QUERY_ALL_SITES)
    # result = session.execute(sql, {'start_date': '2015-12-01', 'end_date': '2016-02-01'})
    return result


def convertData(row):
    # return a list of totalIn(0), min temp(1), max temp(2), avg temp(3), month(4), season(5), day(6), site_id(7)
    site_id = int(row[0])
    site_name = row[1]
    totalIn = int(row[2])
    min_temp = int(row[5])
    max_temp = int(row[4])
    mean_temp = (np.mean([min_temp, max_temp]))
    month = getMonth(row[3])
    day = getDay(row[3])
    season = getSeason(month)
    return [totalIn, min_temp, max_temp, mean_temp, month, season, day, site_id]


def filterDataOnSeasonDay(np_array, season, day=None):
    # this function filter on a given day and a given season, and return the totalIn and meanTemp

    if day is not None:
        filtered_array = np_array[(np_array[:, 5] == season) & (np_array[:, 6] == day)]
    else:
        filtered_array = np_array[(np_array[:, 5] == season)]

    return filtered_array


def normalize(np_array):
    normalized_array = np.empty([1, 8])
    sites = np.unique(np_array[:, 7]).tolist()

    for site in sites:
        site_array = np_array[np_array[:, 7] == site]
        mean_id = np.mean(site_array[:, 0])
        std_id = np.std(site_array[:, 0])
        if mean_id * std_id > 0:
            site_array[:, 0] = (site_array[:, 0] - mean_id) / std_id
            normalized_array = np.concatenate([normalized_array, site_array])
    normalized_array = np.delete(normalized_array, 0, 0)
    return normalized_array[:, 3], normalized_array[:, 0]


def plotFigure(x, y, season, day, siteName):
    plt.figure(num=None, figsize=(8, 6))
    plt.clf()
    fig, ax = plt.subplots()
    plt.xlim(np.min(x), np.max(x))
    plt.ylim(np.min(y), np.max(y))
    plt.scatter(x, y, s=4, color='r', marker='s')
    plt.title('{0} - {1} - {2}.png'.format(siteName, SEASONS.get(season), calendar.day_name[day - 1]))
    plt.xlabel("Average Temperation (Celsius)")
    plt.ylabel("Optimized In")
    # pdb.set_trace()
    fname = '{0} - {1} - {2}.png'.format(siteName, SEASONS.get(season), calendar.day_name[day - 1])
    fig.savefig(os.path.join(CHART_DIR, fname))


def plot3DFigure(x, y, z, season, day, siteName):
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter(x, y, z, c='r', marker='o')

    ax.set_xlabel('Temp Min C')
    ax.set_ylabel('Temp Max C')
    ax.set_zlabel('Optimized In')

    fig.savefig('{0} - {1} - {2}-3D.png'.format(siteName, SEASONS.get(season), calendar.day_name[day - 1]))


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
