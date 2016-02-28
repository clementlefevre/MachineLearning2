from ..model.site import PARAMS

__author__ = 'ThinkPad'


def filterDataOnSeasonDay(np_array, season, day=None):
    # this function filter on a given day and a given season, and return the totalIn and meanTemp

    if day is not None:
        filtered_array = np_array[
            (np_array[:, PARAMS.get('season')] == season) & (np_array[:, PARAMS.get('day')] == day)]
    else:
        filtered_array = np_array[(np_array[:, PARAMS.get('season')] == season)]

    return filtered_array
