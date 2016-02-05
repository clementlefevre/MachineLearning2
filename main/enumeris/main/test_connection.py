from datetime import datetime

import numpy as np

from main.enumeris.main.config import DB_URL

__author__ = 'ThinkPad'

from sqlalchemy import create_engine, text, Column, Integer, ForeignKey, DATETIME, DATE, String

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

Base = declarative_base()

SQL_QUERY_PER_SITE = "SELECT S.idbldsite,S.sname, xyz.totalin, xyz.dato," \
                     " M.tempmaxcelsius, M.tempmincelsius FROM dwe_bld_site S " \
                     "LEFT JOIN dwe_bld_address A " \
                     "ON A.idbldsite=S.idbldsite " \
                     "LEFT JOIN dwe_ext_weather_premium M ON M.idbldaddress=A.id" \
                     " LEFT JOIN ( " \
                     "SELECT idbldsite, to_char(dwe_cnt_site.timestamp, 'YYYY-MM-DD') as dato, sum(optimizedin) as totalin" \
                     " FROM dwe_cnt_site WHERE dwe_cnt_site.timestamp>:start_date and dwe_cnt_site.timestamp<:end_date and idbldsite= :site_id" \
                     " GROUP by dato, idbldsite ) AS xyz" \
                     " ON xyz.idbldsite = S.idbldsite where xyz.idbldsite = :site_id and to_char(M.day, 'YYYY-MM-DD') = xyz.dato " \
                     "ORDER BY dato "

SQL_QUERY_ALL_SITES = "SELECT S.idbldsite,S.sname, xyz.totalin, xyz.dato," \
                      " M.tempmaxcelsius, M.tempmincelsius FROM dwe_bld_site S " \
                      "LEFT JOIN dwe_bld_address A " \
                      "ON A.idbldsite=S.idbldsite " \
                      "LEFT JOIN dwe_ext_weather_premium M ON M.idbldaddress=A.id" \
                      " LEFT JOIN ( " \
                      "SELECT idbldsite, to_char(dwe_cnt_site.timestamp, 'YYYY-MM-DD') as dato, sum(optimizedin) as totalin" \
                      " FROM dwe_cnt_site WHERE dwe_cnt_site.timestamp>:start_date and dwe_cnt_site.timestamp<:end_date " \
                      " GROUP by dato, idbldsite ) AS xyz" \
                      " ON xyz.idbldsite = S.idbldsite where   to_char(M.day, 'YYYY-MM-DD') = xyz.dato " \
                      "ORDER BY dato "


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
    """"""

    engine = create_engine(DB_URL)

    Session = sessionmaker(bind=engine)
    session = Session()
    return session


def main():
    session = loadSession()
    allSites = session.query(Site).all()
    site1 = session.query(Site).filter_by(idbldsite=1).all()[0]

    sql = text(SQL_QUERY_PER_SITE)

    result = session.execute(sql, {'start_date': '2012-01-01', 'end_date': '2016-02-01', 'site_id': 1})
    data = []

    for row in result:
        data.append([int(row[2]), (int(row[4]) + int(row[5]) / 2), getMonth(row[3]), getDay(row[3])])
    np_array = np.array(data)
    print data

    np_array = np_array[(np_array[:, 3] == 3)]
    import matplotlib.pyplot as plt
    y = np_array[:, 0]
    x = np_array[:, 1]

    plt.figure(num=None, figsize=(8, 6))
    plt.clf()
    fig, ax = plt.subplots()

    plt.xlim(np.min(x), np.max(x))
    plt.ylim(np.min(y), np.max(y))

    plt.scatter(x, y, s=4, color='r', marker='s')

    fig.savefig('Figure' + '.png')

    print "over"


def getMonth(date_posted):
    date = datetime.strptime(date_posted, '%Y-%m-%d')
    return date.month


def getDay(date_posted):
    date = datetime.strptime(date_posted, '%Y-%m-%d')
    return date.day
