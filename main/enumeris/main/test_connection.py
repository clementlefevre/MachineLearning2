from main.enumeris.main.config import DB_URL

__author__ = 'ThinkPad'

from . import db

SQL_QUERY = "SELECT S.idbldsite,S.sname, xyz.totalin, xyz.dato," \
            " M.tempmaxcelsius FROM dwe_bld_site S " \
            "LEFT JOIN dwe_bld_address A " \
            "ON A.idbldsite=S.idbldsite " \
            "LEFT JOIN dwe_ext_weather_premium M ON M.idbldaddress=A.id" \
            " LEFT JOIN ( " \
            "SELECT idbldsite, to_char(dwe_cnt_site.timestamp, 'YYYY-MM-DD') as dato, sum(optimizedin) as totalin" \
            " FROM dwe_cnt_site WHERE dwe_cnt_site.timestamp>:start_date and dwe_cnt_site.timestamp<:end_date and idbldsite=1" \
            " GROUP by dato, idbldsite ) AS xyz" \
            " ON xyz.idbldsite = S.idbldsite where xyz.idbldsite = :site_id and to_char(M.day, 'YYYY-MM-DD') = xyz.dato " \
            "ORDER BY dato "


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

    engine = db.create_engine(DB_URL)

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

    sql = db.text(SQL_QUERY)
    result = session.execute(sql, {'start_date': '2015-01-01', 'end_date': '2015-02-01', 'site_id': 1})
    names = []
    for row in result:
        names.append(row)

    print names

    print "over"
