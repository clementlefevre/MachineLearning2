import pandas as pd
from ..model.site import PARAMS
import numpy as np

__author__ = 'ThinkPad'


def normalize(np_array):
    normalized_array = np.empty([1, 10])
    normalized_array = normalize_pandas(np_array)
    site_idx = PARAMS.get('site_id')
    total_in_idx = PARAMS.get('totalIn')
    #
    # sites = np.unique(np_array[:, PARAMS.get('site_id')]).tolist()
    #
    # for site in sites:
    #     site_array = np_array[np_array[:, site_idx] == site]
    #     mean_id = np.mean(site_array[:, total_in_idx])
    #     std_id = np.std(site_array[:, total_in_idx])
    #     if mean_id * std_id > 0:
    #         site_array[:, total_in_idx] = (site_array[:, total_in_idx] - mean_id) / std_id
    #         normalized_array = np.concatenate([normalized_array, site_array])
    # normalized_array = np.delete(normalized_array, 0, 0)
    return normalized_array[:, PARAMS.get('mean_temp')], normalized_array[:,
                                                         PARAMS.get('precipitations_mm')], normalized_array[:,
                                                                                           total_in_idx]


def normalize_pandas(np_array):
    df = pd.DataFrame({'ID': np_array[:, PARAMS.get('site_id')].astype(np.int64),
                       'value': np_array[:, PARAMS.get('totalIn')].astype(np.int64)
                       })

    dfAlldata = pd.DataFrame(np_array)

    byid = df.groupby('ID')
    mean = byid.mean()
    std = byid.std()

    df['normalized'] = df.apply(lambda x: (x.value - mean.ix[x.ID]) / std.ix[x.ID], axis=1)
    df['mean_temp'] = np_array[:, PARAMS.get('mean_temp')]
    print(df)
    dfValues = df.values
    return df.values
