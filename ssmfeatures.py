###########################################################
######RPF Feature Vectors from Watanabe et al. (2016)######
###########################################################
#############These functions extract features##############
##############from a self-similarity matrix################
###########################################################

from functools import reduce
import numpy as np


def ssm_feats_thresholds_watanabe(ssm_lines):
    feat0_for_threshold = {}
    feat1_for_threshold = {}
    feat2_for_threshold = {}
    feat3_for_threshold = {}
    for lam in [0.1 * factor for factor in range(1, 10)]:
        feat0_for_threshold[lam] = feat_rpf1_counts(ssm_lines, lam=lam)
        feat1_for_threshold[lam] = feat_rpf2_counts(ssm_lines, lam=lam)
        feat2_for_threshold[lam] = feat_rpf1_value_differences(ssm_lines, lam=lam)
        feat3_for_threshold[lam] = feat_rpf2_value_differences(ssm_lines, lam=lam)

    frpf3 = feat_rpf3(ssm_lines)
    frpf4b = feat_rpf4b(ssm_lines)
    frpf4e = feat_rpf4e(ssm_lines)
    return feat0_for_threshold, feat1_for_threshold,\
           feat2_for_threshold, feat3_for_threshold,\
           frpf3, frpf4b, frpf4e


#return a dictionary after updating it
def update_and_return(d, new_key, new_value):
    d.update({new_key: new_value})
    return d

def capped_row_count(matrix):
    return matrix.shape[0] - 1

def capped_col_count(matrix):
    return matrix.shape[1] - 1


def state_property(ssm, row):
    return ssm[row, row + 1]

def border_start_property(ssm, row):
    return ssm[row, 0]

def border_end_property(ssm, row):
    return ssm[row, capped_col_count(ssm)]

def sequence_indicators(ssm, row, lam, diagonal_property):
    seq_indicators = set()
    for col in range(capped_col_count(ssm)):
        if row == col:
            continue
        if diagonal_property(ssm, row, col, lam):
            seq_indicators.add(col)
    return seq_indicators

def sequence_indicator_value_differences(ssm, row, lam, diagonal_property):
    seq_indicators_diff = set()
    for col in range(capped_col_count(ssm)):
        if row == col:
            continue
        if diagonal_property(ssm, row, col, lam):
            value_difference = abs(ssm[row, col] - ssm[row + 1, col + 1])
            if value_difference != 0:
                seq_indicators_diff.add(value_difference)
    return seq_indicators_diff


def rpf(ssm, lam, indicator):
   #return {row : indicator(ssm, row, lam) for row in range(ssm.shape[0] - 1)}
    indicators = dict()
    for row in range(capped_row_count(ssm)):
        indicators_in_row = indicator(ssm, row, lam)
        if not indicators_in_row:
            continue
        indicators[row] = indicators_in_row
    return indicators

#f_lambda^RPF#
def feat_rpf_counts(ssm, lam, rpf_function):
    rpf_entries = rpf_function(ssm, lam)
    return reduce(lambda x,key: update_and_return(x, key, len(rpf_entries.get(key))), rpf_entries, {})

#f_lambda^RPFv
def feat_rpf_value_differences(ssm, lam, rpf_function):
    rpf_entries = rpf_function(ssm, lam)
    return reduce(lambda x,key: update_and_return(x, key, np.average(list(rpf_entries.get(key)))), rpf_entries, {})

#################################
#################################


#######RPF1#######
#sequence edge indicators, called g_lambda in the paper
rpf1_property = lambda ssm,row,col,lam: (ssm[row, col] - lam) * (ssm[row + 1, col + 1] - lam) < 0

def sequence_edge_indicators(ssm, row, lam):
    return sequence_indicators(ssm, row, lam, rpf1_property)

def sequence_edge_value_differences(ssm, row, lam):
    return sequence_indicator_value_differences(ssm, row, lam, rpf1_property)

def rpf1_counts(ssm, lam):
    return rpf(ssm, lam, sequence_edge_indicators)

def rpf1_value_differences(ssm, lam):
    return rpf(ssm, lam, sequence_edge_value_differences)

#f_lambda^RPF1#
def feat_rpf1_counts(ssm, lam):
    return feat_rpf_counts(ssm, lam, rpf1_counts)

#f_lambda^RPF1v
def feat_rpf1_value_differences(ssm, lam):
    return feat_rpf_value_differences(ssm, lam, rpf1_value_differences)



#######RPF2#######
#sequence body indicators, called c_lambda in the paper
rpf2_property = lambda ssm,row,col,lam: ssm[row, col] - lam >= 0 and ssm[row + 1, col + 1] - lam >= 0

def sequence_body_indicators(ssm, row, lam):
    return sequence_indicators(ssm, row, lam, rpf2_property)

def sequence_body_value_differences(ssm, row, lam):
    return sequence_indicator_value_differences(ssm, row, lam, rpf2_property)

def rpf2_counts(ssm, lam):
    return rpf(ssm, lam, sequence_body_indicators)

def rpf2_value_differences(ssm, lam):
    return rpf(ssm, lam, sequence_body_value_differences)

#f_lambda^RPF2#
def feat_rpf2_counts(ssm, lam):
    return feat_rpf_counts(ssm, lam, rpf2_counts)

#f_lambda^RPF2v
def feat_rpf2_value_differences(ssm, lam):
    return feat_rpf_value_differences(ssm, lam, rpf2_value_differences)


#######RPF3#######
####Similarity with subsequent line####
def feat_rpf3(ssm):
    block_sim = dict()
    for row in range(capped_row_count(ssm)):
        block_sim[row] = state_property(ssm, row)
    return block_sim

####RPF4#####
###Similarities with beginning line (4b) or ending line (4e)###
def feat_rpf4b(ssm):
    f4b = dict()
    for row in range(capped_row_count(ssm)):
        f4b[row] = border_start_property(ssm, row)
    return f4b

def feat_rpf4e(ssm):
    f4e = dict()
    for row in range(capped_row_count(ssm)):
        f4e[row] = border_end_property(ssm, row)
    return f4e
