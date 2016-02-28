__author__ = 'ThinkPad'

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, ForeignKey, DATETIME, DATE, String
from sqlalchemy.orm import relationship

Base = declarative_base()

SEASONS = {1: 'winter', 2: 'spring', 3: 'summer', 4: 'autumn'}
PARAMS = {'site_id': 0, 'site_name': 1, 'totalIn': 2, 'min_temp': 3, 'max_temp': 4, 'mean_temp': 5,
          'precipitations_mm': 6, 'month': 7, 'season': 8, 'day': 9}


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
