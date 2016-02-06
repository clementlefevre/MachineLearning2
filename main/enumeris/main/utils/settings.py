__author__ = 'ThinkPad'

SQL_QUERY_PER_SITE = "SELECT S.idbldsite,S.sname, xyz.totalin, xyz.dato," \
                     " M.tempmaxcelsius, M.tempmincelsius,M.precipitationinmilimiter FROM dwe_bld_site S " \
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
                      " M.tempmaxcelsius, M.tempmincelsius,M.precipitationinmilimiter FROM dwe_bld_site S " \
                      "LEFT JOIN dwe_bld_address A " \
                      "ON A.idbldsite=S.idbldsite " \
                      "LEFT JOIN dwe_ext_weather_premium M ON M.idbldaddress=A.id" \
                      " LEFT JOIN ( " \
                      "SELECT idbldsite, to_char(dwe_cnt_site.timestamp, 'YYYY-MM-DD') as dato, sum(optimizedin) as totalin" \
                      " FROM dwe_cnt_site WHERE dwe_cnt_site.timestamp>:start_date and dwe_cnt_site.timestamp<:end_date " \
                      " GROUP by dato, idbldsite ) AS xyz" \
                      " ON xyz.idbldsite = S.idbldsite where   to_char(M.day, 'YYYY-MM-DD') = xyz.dato " \
                      "ORDER BY dato "
