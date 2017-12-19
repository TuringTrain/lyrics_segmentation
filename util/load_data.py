import pandas as pd
from os import path

# Open dataframes stored in hdf container
#
# Important:
#   segment_borders ids are the data points we want to model
#   they are a proper subset of ids in ssm stores, but we only model
#   for the 107k ids in segment_borders, as they are all english with 5+ segments


def load_segment_borders(data_path: str) -> pd.DataFrame:
    with pd.HDFStore(path.join(data_path, 'mldb_seg5p_segment_borders.hdf')) as store:
        borders = store['mldb_seg5p']
    return borders


def load_ssm_string(data_path: str) -> pd.DataFrame:
    with pd.HDFStore(path.join(data_path, 'ssm_store.hdf')) as store:
        sssm = store['mdb_127_en_string_1'].append(store['mdb_127_en_string_2']).append(store['mdb_127_en_string_3'])
        sssm.set_index(['id'], inplace=True)
    return sssm


def load_ssm_postag(data_path: str) -> pd.DataFrame:
    with pd.HDFStore(path.join(data_path, 'ssm_store.hdf')) as store:
        sppm = store['mdb_127_en_postag_1'].append(store['mdb_127_en_postag_2']).append(store['mdb_127_en_postag_3'])
        sppm.set_index(['id'], inplace=True)
    return sppm
