from ..utils.settings import SQL_QUERY_PER_SITE
from sqlalchemy import text

__author__ = 'ThinkPad'


def retrieveData(session, site_id, interval):
    sql = text(SQL_QUERY_PER_SITE)
    start_date, end_date = interval
    result = session.execute(sql, {'start_date': start_date, 'end_date': end_date, 'site_id': site_id})

    return result
