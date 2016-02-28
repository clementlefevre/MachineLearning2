__author__ = 'ThinkPad'

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import numpy as np
from model.site import Site
from services.converters import convertData
from services.db_connector import retrieveData
from services.filters import filterDataOnSeasonDay
from services.normalizer import normalize
from services.plotter import plotFigure

from config import DB_URL


def main(siteId=None, start_date=None, end_date=None):
    session = loadSession()
    interval = (start_date, end_date)
    if siteId is None:
        siteList = session.query(Site).all()
    else:
        siteList = [session.query(Site).filter_by(idbldsite=siteId).first()]
    if start_date or end_date == None:
        interval = ('2013-01-01', '2016-01-01')
    analyzeAllSite(session, siteList, interval)


def loadSession():
    engine = create_engine(DB_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    return session


def analyzeAllSite(session, allSites, interval):
    data = []

    for site in allSites:
        result = retrieveData(session, site.idbldsite, interval)

        for row in result:
            data.append(convertData(row))

    np_array = np.array(data)

    for season in range(1, 5):
        for day in range(1, 8):
            filterOnSeasonDay = filterDataOnSeasonDay(np_array, season, day)
            x1, x2, y = normalize(filterOnSeasonDay)

            plotFigure(x1, y, season, day, "All Sites", 'Mean Temp Celsius')
            plotFigure(x2, y, season, day, "All Sites", 'Precipitations mm')

            # plot3DFigure(x, y, z, season, day, "All Sites")
