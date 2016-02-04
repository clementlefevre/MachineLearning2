__author__ = 'ThinkPad'

from . import db


class Site(db.Model):
    __tablename__ = 'dwe_bld_site'
    idbldsite = db.Column(db.Integer, primary_key=True)
    sname = db.Column(db.String)
    address = db.relationship('Address', backref='dwe_bld_site')
    count = db.relationship('Count', backref='dwe_bld_site')


class Address(db.Model):
    __tablename__ = 'dwe_bld_address'
    id = db.Column(db.Integer, primary_key=True)

    idbldsite = db.Column(db.Integer, db.ForeignKey('dwe_bld_site.idbldsite'))
    weather = db.relationship('Weather')


class Weather(db.Model):
    __tablename__ = "dwe_ext_weather_premium"
    id = db.Column(db.Integer, primary_key=True)
    day = db.Column(db.DATE)
    idbldaddress = db.Column(db.Integer, db.ForeignKey('dwe_bld_address.id'))


class Count(db.Model):
    __tablename__ = "dwe_cnt_site"
    id = db.Column(db.Integer, primary_key=True)
    idbldsite = db.Column(db.Integer, db.ForeignKey('dwe_bld_site.idbldsite'))
    timestamp = db.Column(db.DATETIME)


# ----------------------------------------------------------------------

def loadSession():
    """"""

    engine = db.create_engine('postgresql://postgres:clement2014@localhost:5432/DWE_MAUSFRERES_2012')

    Session = db.sessionmaker(bind=engine)
    session = Session()
    return session


def main():
    session = loadSession()
    allAddresses = session.query(Address).all()
    allSites = session.query(Site).all()
    # allWeather = session.query(Weather).all()
    site1 = session.query(Site).filter_by(idbldsite=1).all()[0]
    allCount = session.query(Count).filter_by(idbldsite=1).filter_by(timestamp='2015-02-01').all()

    print "over"
