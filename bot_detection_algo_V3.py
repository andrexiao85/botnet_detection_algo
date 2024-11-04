#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 12:52:30 2019
@author: emugambi
"""
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
#from functools import reduce 
#import random
#from scipy import stats

# bring in the data
def get_Data(day):
    df = pd.read_csv('/Users/emugambi/botnet_traffic/Data/lanl_nflow_a%s' %day, sep = ",", header = None, 
                           names=('time','duration','s_computer','s_port','d_computer','d_port',
                                  'protocol','packets','bytes'))
    df = df[['time','s_computer','d_computer','packets']]
    return df

def get_Data_2(day):
    df = pd.read_csv('/Users/emugambi/botnet_traffic/Data/lanl_nflow_a%s' %day, sep = ",", header = None, 
                           names=('time','duration','s_computer','s_port','d_computer','d_port',
                                  'protocol','packets','bytes'))
    #df = df[['time','s_computer','d_computer','packets']]
    return df

def shannon(p):
    """
    shannon entropy for discrete distributions
    Parameters
    ----------
    p : array-like, dtype=float, shape=n
        counts of hits seen in a network
    """
    eps = 0.00000000001   # v small number to get rid of zeros
    p = (np.asarray(p, dtype=np.float)/np.sum(p)) + eps
    return -1.0*(np.sum(np.where(p != 0, p * np.log2(p), 0)))

def count_diffs(d_hist,d_new):
    """
    counting rare users and their appearances in hosts from time to time
    """
    hist_set, new_set = set(d_hist), set(d_new)
    new_items = new_set.difference(hist_set)
    #all_items = set(d_hist + d_new)
    return len(new_items)/len(new_set)

def grp_quantity(dat,value):
    out = dat.groupby([value]).size().reset_index()
    return out[0]

# Computes g*
def g_star(Pxx):
    m = len(Pxx)
    max_ = max(Pxx)
    var = (1.0/(2.0*m))*sum(Pxx)
    return max_/var

# Computes z_alpha
def z_alpha(alpha, m):
    c = 1.0-alpha
    z = -2.0*np.log(1.0-c**(1.0/m))
    return z

# computes max_pxx, uses z_alpha function
def max_pxx(df_agg):
    alpha = 0.001
    f, pxx = signal.periodogram(df_agg)
    m, max_ = len(pxx), np.max(pxx)
    var = (1.0/(2.0*m))*sum(pxx)
    z = z_alpha(alpha, m)
    return max_, len(f), z, max_/var

# get traffic from a specific edge
def generate_hourly_multiedge_behavior(dat, this_src, this_dst):
    """
    # generates data segmented based on hosts of interest
    """
    #host = get_Data()                                                      # get your data in
    dat_src = dat[dat['s_computer']==this_src]   
    dat_src = dat_src[dat_src['d_computer']==this_dst]
    dat_dst = dat[dat['s_computer']==this_dst]   
    dat_dst = dat_dst[dat_dst['d_computer']==this_src]                       # select host
    out = pd.concat([dat_src,dat_dst])                         
    return out

# compare effect of intervals on pxx
def get_pxx(dat,src_computer,dst_computer,method):
    intervals = ['1s','2s','3s','4s','5s','6s','7s','8s','9s','10s','11s','12s','13s','14s','15s']
    edge_pxx = {}
    #s,d = 'C17693','C5074'
    s,d = src_computer,dst_computer
    df = dat
    edge_dat = generate_hourly_multiedge_behavior(df, s, d)
    #edge_dat['packets'] = ((edge_dat['packets'])-np.min(edge_dat['packets']))/(np.max(edge_dat['packets'])-np.min(edge_dat['packets']))
    edge_dat['new_time'] = pd.to_datetime(edge_dat['time'], unit='s')
    edge_dat.index = edge_dat['new_time']
    del edge_dat['time']
    del edge_dat['new_time']
    for interval in intervals:
        if method == 'packets':
            new_data = list(edge_dat.resample(interval).sum()['packets'])
            new_data_2 = [1 if x > 0 else 0 for x in new_data]     # tweak to reduce reliance on quantity of packets
        elif method == 'bytes':
            new_data = list(edge_dat.resample(interval).sum()['bytes'])
        h1, h2, h3, h4 = max_pxx(new_data_2)  # h1 = max periodogram, h2 = sample size , h3 = z-statistic , h4 = g* statistic 
        print (interval)
        edge_pxx[interval] = [h1,h2,h3,h4]
    out_df = pd.DataFrame.from_dict(edge_pxx)
    out_df = out_df.T
    out_df.columns = ['max_pxx','samples','z_const','g_x']
    out_df = out_df[out_df['g_x'] >= out_df['z_const']]
    if len(out_df) > 0:
        out_df = out_df.sort_values(by = 'max_pxx', ascending = False)
        out_df = out_df.iloc[0]
    #out_df.to_csv('/Users/emugambi/botnet_traffic/edge_17693_5074_pckts_v2$.csv')
    #print(out_df)
    return out_df

# run individual edges thru periodogram computation
def compute_pxx(data,lst,qty):
    all_pxx = {}
    for i in range(len(lst)):
        all_pxx[lst['s_computer'].iloc[i],lst['d_computer'].iloc[i]] = get_pxx(data,lst['s_computer'].iloc[i],lst['d_computer'].iloc[i],qty)
        print('edge no:',i)
    return all_pxx
        
# identify unidirectional edges     
def unidirectionality_test(data):
    d_x = data[['s_computer','d_computer']].drop_duplicates()
    d_y = d_x.copy()          # duplicate
    d_z = pd.merge(d_x, d_y, how = 'outer', left_on=['s_computer','d_computer'], right_on=['d_computer','s_computer']) # outer join
    d_z = d_z.fillna(0)
    d_z_0 = d_z[d_z['s_computer_y'] == 0]
    d_z_0 = d_z_0[['s_computer_x','d_computer_x']]
    d_z_0.columns = ['s_computer','d_computer']
    return d_z_0

# identify edges with repeated "similar byte" sequences
def byte_behavior(which_dat):
    df = get_Data_2(which_dat)
    dat = df[['s_computer','d_computer','packets','bytes']]
    dat['pckts_per_byte'] = np.floor(dat['bytes']/dat['packets'])
    out = dat.groupby(['s_computer','d_computer','pckts_per_byte']).size().reset_index()
    out.columns = ['s_computer','d_computer','pckts_per_byte','frequency']
    out = out.sort_values('frequency',ascending = False)
    out = out[out['frequency']>1000.0*np.median(out['frequency'])]
    # out.to_csv('/Users/emugambi/botnet_traffic/day_%s_byte_frq.csv' %which_dat)
    out = out[['s_computer','d_computer']].drop_duplicates()
    return out
    
# run botnet algo for each days traffic 
def run_detection_methods(which_day):
    day = which_day
    dat = get_Data(day)
    bb = byte_behavior(which_day)
    uni_data = unidirectionality_test(dat)
    uni_data_final = pd.merge(uni_data, bb, how = 'inner', on = ['s_computer','d_computer'])
    high_pxx_segment = compute_pxx(dat,uni_data_final,'packets')
    pxx_result = pd.DataFrame.from_dict(high_pxx_segment)
    pxx_result = pxx_result.T
    #pxx_result.to_csv('/Users/emugambi/botnet_traffic/results/day_%s_pxx_byte_output_001.csv' %day)
    return pxx_result
    
# plot distribution of edges with high pxx for validation
def edge_traffic_dist(what_day,src,dst):
    dat = get_Data(what_day)
    edge_dat = generate_hourly_multiedge_behavior(dat, src, dst)
    edge_diff = edge_dat['time'].diff()
    edge_diff = edge_diff.reset_index()
    #edge_diff.columns = ['t_diff']
    out = edge_diff.groupby(['time']).size().reset_index()
    out.columns = ['time','counts']
    out = out.sort_values('counts',ascending=False)
    print(edge_diff['time'].describe())
    print(out)
    out.plot.bar(x='time',y='counts',rot=0)
    #out.to_csv('/Users/emugambi/botnet_traffic/results/charts/day_%s_C1015_C15487_time_dist.csv' %what_day)
    return out
    
    
    
