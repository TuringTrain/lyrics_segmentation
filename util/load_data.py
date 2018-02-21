import pandas as pd
from os import path

# Open dataframes stored in hdf container
#
# Important:
#   segment_borders ids are the data points we want to model
#   they are a proper subset of ids in ssm stores, but we only model
#   for the 107k ids in segment_borders, as they are all english with 5+ segments


def load_segment_borders(data_path: str) -> pd.DataFrame:
    with pd.HDFStore(path.join(data_path, 'borders_pub1.hdf')) as store:
        borders = store['mdb_127_en_seg5p']
    return borders


def load_ssm_string(data_path: str) -> pd.DataFrame:
    with pd.HDFStore(path.join(data_path, 'ssm_store_pub1.hdf')) as store:
        sssm = store['mdb_127_en_seg5p_string_1'].append(store['mdb_127_en_seg5p_string_2'])
    return sssm


def load_ssm_phonetics(data_path: str) -> pd.DataFrame:
    with pd.HDFStore(path.join(data_path, 'ssm_store_pub1.hdf')) as store:
        sppm = store['mdb_127_en_phonetics_1'].append(store['mdb_127_en_phonetics_2'])
    return sppm

def load_linewise_length_chars(data_path: str) -> pd.DataFrame:
    with pd.HDFStore(path.join(data_path, 'ssm_store_pub1.hdf')) as store:
        ssm_linewise = store['mdb_127_en_line_length_in_chars']
    return ssm_linewise

def load_linewise_length_tokens(data_path: str) -> pd.DataFrame:
    with pd.HDFStore(path.join(data_path, 'ssm_store_pub1.hdf')) as store:
        ssm_linewise = store['mdb_127_en_line_length_in_tokens']
    return ssm_linewise

def load_linewise_length_syllables(data_path: str) -> pd.DataFrame:
    with pd.HDFStore(path.join(data_path, 'ssm_store_pub1.hdf')) as store:
        ssm_linewise = store['mdb_127_en_line_length_in_syllables']
    return ssm_linewise


# train and test on all genres
def load_segment_borders_watanabe(data_path: str) -> pd.DataFrame:
    with pd.HDFStore(path.join(data_path, 'borders_pub2.hdf')) as store:
        train_borders = store['watanabe_train']
        dev_borders = store['watanabe_dev']
        test_borders = store['watanabe_test']
    return train_borders, dev_borders, test_borders

# train on all genres, test on single genre
def load_segment_borders_for_genre(data_path: str, genre_name: str) -> pd.DataFrame:
    with pd.HDFStore(path.join(data_path, 'borders_pub2.hdf')) as store:
        train_borders = store['watanabe_train']
        dev_borders = store[genre_name + '_watanabe_dev']
        test_borders = store[genre_name + '_watanabe_test']
    return train_borders, dev_borders, test_borders

################ current genres and song counts ###########
# /AlternativeRock_watanabe_dev  ->  855
# /AlternativeRock_watanabe_test  ->  875
# /HardRock_watanabe_dev  ->  433
# /HardRock_watanabe_test  ->  462
# /HeavyMetal_watanabe_dev  ->  254
# /HeavyMetal_watanabe_test  ->  269
# /HipHop_watanabe_dev  ->  1066
# /HipHop_watanabe_test  ->  1097
# /RnB_watanabe_dev  ->  930
# /RnB_watanabe_test  ->  905
# /Soul_watanabe_dev  ->  129
# /Soul_watanabe_test  ->  118
# /SouthernHipHop_watanabe_dev  ->  188
# /SouthernHipHop_watanabe_test  ->  183
# /Synthpop_watanabe_dev  ->  108
# /Synthpop_watanabe_test  ->  126
# /watanabe_dev  ->  20531
# /watanabe_test  ->  20583
# /watanabe_train  ->  61687
