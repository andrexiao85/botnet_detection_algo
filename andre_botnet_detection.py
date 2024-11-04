#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:02:33 2019

@author: xiaoa
"""
import os
import sys
import pandas as pd
import numpy as np
import random
from decimal import Decimal as D
from scipy import signal
import re
import math


def gen_nodes(num_nodes, num_bots):
    
    nodes = np.array([f"host_{i}" for i in range(int(num_nodes))])
    bots = np.random.choice(nodes, num_bots, replace = False)

    return nodes, bots

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def gen_params(num_nodes, prop_bots, time_int, bot_int, pack_count, prob_hostcomm):
    
    param_combos = np.array(np.meshgrid(num_nodes, 
                                        prop_bots, 
                                        time_int, 
                                        bot_int, 
                                        pack_count, 
                                        prob_hostcomm)).T.reshape(-1, 6)
    
    return param_combos
    
    


def readParams(filename):
    
    with open(filename, 'r') as file:
        params = [list(map(float, l.split(" "))) for l in file.readlines()]
        num_nodes, prop_bots, time_int, bot_int, pack_count, prob_hostcomm = params
    return num_nodes, prop_bots, time_int, bot_int, pack_count, prob_hostcomm


def sim_data(nodes, bots, time_int, bot_int, pack_count, prob_hostcomm):

    time = np.array([])
    host = np.array([])
    dest = np.array([])
    packets = np.array([])
    is_resp = np.array([])
    is_bot = np.array([])

    
    t = D('0')

    while t < time_int:
        for j in nodes:
            #print('Time: ', t)
            #print('Node: ', j)
            #print('Bot: ', j in bots)
            #print('Send Packets: ', t % bot_int == 0)
            if j in bots and t % int(bot_int) == 0:
                temp = 0
                d = np.random.choice([b for b in bots if b != j], len(bots)-1, replace = False)   
                for k in d:
                    
                    resp_time = random.randint(1, 3)

                    time = np.append(time, [t+temp, t+temp+D(f'{resp_time}')])
                    host = np.append(host, [j, k])
                    dest = np.append(dest, [k, j])
                    packets = np.append(packets, [1, 1])
                    is_resp = np.append(is_resp, [0, 1])
                    is_bot = np.append(is_bot, [1, 1])
                    temp += D('1')
            elif np.random.choice([True, False], p = [prob_hostcomm, 1-prob_hostcomm]):
                resp_time = random.randint(1, 3)
                d = np.random.choice([n for n in nodes if n != j])
                time = np.append(time, [t, t+D(f'{resp_time}')])
                host = np.append(host, [j, d])
                dest = np.append(dest, [d, j])
                p_send = np.random.randint(1, pack_count+1)
                #p_resp = np.random.randint(1, pack_count+1, 1)
                packets = np.append(packets, [p_send, 1])
                is_resp = np.append(is_resp, [0, 1])
                is_bot = np.append(is_bot, [0, 0])
        t += D('1')
        #print(t)
            
    
    df = pd.DataFrame({'Time': time,
                       'Host': host,
                       'Dest': dest,
                       'Packets': packets,
                       'Is_Resp': is_resp,
                       'Is_Bot': is_bot})
    
    
    bot_bool = list(map(int, np.isin(nodes, bots)))
    
    
    df2 = pd.DataFrame({'Host': nodes,
                        'Is_Bot': bot_bool})
    
    
    
    #df = df.sort_values(by=['Time', 'Host'])
    df = df[df.Time < time_int]
    #df['Intervals'] = np.array(list(map(int, df.Time)))
    #df = df.set_index('Time')
    return df, df2

def gen_data(out_full, out_agg, trials_path, params, repeats):
    
    num_nodes_arr, prop_bots_arr, time_int_arr, bot_int_arr, pack_count_arr, prob_hostcomm_arr = readParams(rf'/Users/xiaoa/Documents/botnet_data/{params}')
    param_combos = gen_params(num_nodes_arr, prop_bots_arr, time_int_arr, bot_int_arr, pack_count_arr, prob_hostcomm_arr)
    
    i = 0
    
    param_combos = np.repeat(param_combos, repeats, axis=0)
    

    for params in param_combos:
        
        
        
        num_nodes, prop_bots, time_int, bot_int, pack_count, prob_hostcomm = params
        num_bots = int(num_nodes*prop_bots)
        if num_bots < 2:
            num_bots = 2
        nodes, bots = gen_nodes(num_nodes, num_bots)
        
        df, df2 = sim_data(nodes, bots, time_int, bot_int, pack_count, prob_hostcomm)
    
        print(i+1, "/", len(param_combos))
        
        df.to_csv(rf'{out_full}/full_data_{i}.csv', index=False)
        df2.to_csv(rf'{out_agg}/agg_{i}.csv', index=False)

        #df_agg_packets = agg_packets(df, decimals)
        #df_agg_packets.to_csv(rf'{directory}/{file_base}_{i}.csv')
        
        i += 1
            
            
        
    trials_df = pd.DataFrame(data=param_combos, columns=['Nodes', 
                                                          'Prop_Bots', 
                                                          'Time_Int', 
                                                          'Bot_Int', 
                                                          'Pack_Count', 
                                                          'Prob_HostComm'])
    trials_df.to_csv(rf'{trials_path}', index=False)
        
    return trials_df

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


def snr(df, df2):
    idx = list(range(len(df2)))
    df_agg_addresses = df.groupby('Intervals').Host.nunique()
    df_agg_packets = df.groupby('Intervals').sum()[['Packets']]
    df_data = df_agg_packets.copy()
    df_data["Addresses"] = df_agg_addresses
    df_data = df_data.groupby('Intervals').sum().reindex(idx).fillna(0)
    df_data_agg = df_data.groupby('Intervals').sum()
       
    
    df_mean = df_data_agg.mean()
    df2_mean = df2.mean()
    df_diff = (df_data_agg-df_mean)**2
    df2_diff = (df2-df2_mean)**2
    df_vars = df_diff.mean()
    df2_vars = df2_diff.mean()
    df_log = df2_vars/df_vars
    snr_packets = 10*math.log(df_log[0])
    snr_addresses = 10*math.log(df_log[1])
    
    return snr_packets, snr_addresses

def max_pxx(df_agg):
    df = df_agg.groupby('Intervals').sum()
    f1, pxx1 = signal.periodogram(df.Packets)
    f2, pxx2 = signal.periodogram(df.Addresses)
    return max(pxx1), max(pxx2)

# Aggregates raw data into intervals and produces statistical metrics of the periodogramsmax  peak for analysis
def agg_data(full_dir, agg_dir, t_path):
    
    i = 0
    
    #directory = os.fsencode(directory)
    
    full_directory = os.listdir(full_dir)
    full_directory.sort(key=natural_keys)
    agg_directory = os.listdir(agg_dir)
    agg_directory.sort(key=natural_keys)
    full_directory = [f for f in full_directory if f.endswith(".csv")]
    agg_directory = [a for a in agg_directory if a.endswith(".csv")]
    
    trials = pd.read_csv(t_path)
    
    for full_file, agg_file in zip(full_directory, agg_directory):
        #print(os.listdir(directory))
        full_filename = os.fsdecode(full_file)
        agg_filename = os.fsdecode(agg_file)
        #print(filename)
        
            
        full_filename = rf'{full_dir}/' +  full_filename
        agg_filename = rf'{agg_dir}/' +  agg_filename 
        df_read = pd.read_csv(full_filename)
        df_agg = pd.read_csv(agg_filename)
        
        df_read['Intervals'] = [int(t) for t in df_read.Time]
        df_length = trials.Time_Int[i]
        print(f'{i+1}/{len(trials)}')
        agg_length = len(df_agg)
        
        snr_p = np.zeros(agg_length)
        snr_a = np.zeros(agg_length)
        max_pxx_pack = np.zeros(agg_length)
        max_pxx_addy = np.zeros(agg_length)
        g_pack = np.zeros(agg_length)
        g_addy = np.zeros(agg_length)
        z = np.zeros(agg_length)
        
        hosts = df_agg.Host
        
        j = 0
        
        for host in hosts:
            
            df = df_read[(df_read.Host == host) | (df_read.Dest == host)]
            idx = list(range(int(df_length))) 
            df_agg_addresses = df.groupby('Intervals').Host.nunique()
            df_agg_packets = df.groupby('Intervals').sum()[['Packets']]
            df_data = df_agg_packets.copy()
            df_data["Addresses"] = df_agg_addresses
            df_data = df_data.groupby('Intervals').sum().reindex(idx).fillna(0)
            
            df_data_agg = df_data.groupby('Intervals').sum()
            
            f_p, pxx_p = signal.periodogram(df_data_agg.Packets, fs=10)
            f_a, pxx_a = signal.periodogram(df_data_agg.Addresses, fs=10)
            
            #i1 = np.argmax(pxx_p)
            #i2 = np.argmax(pxx_a)
            
            #print(f_p[i1], f_a[i2])
            
            
            snr_packets, snr_addresses = snr(df_read, df_data_agg)
            
            snr_p[j] = snr_packets
            snr_a[j] = snr_addresses
            
            
            
            max_pxx_pack[j] = pxx_p.max()
            max_pxx_addy[j] = pxx_a.max()
            g_pack[j] = g_star(pxx_p)
            g_addy[j] = g_star(pxx_a)
            z[j] = z_alpha(.001, len(pxx_p))
            j += 1
            
        df_agg['SNR_Pack'] = snr_p
        df_agg['SNR_Addy'] = snr_a
        df_agg['Max_Peak_Pack'] = max_pxx_pack
        df_agg['Max_Peak_Addy'] = max_pxx_addy
        df_agg['g*_Pack'] = g_pack
        df_agg['g*_Addy'] = g_addy
        df_agg['z_alpha'] = z
        
        tp_p = list(map(int, (df_agg.Is_Bot == 1) & (df_agg['g*_Pack'] >= df_agg.z_alpha)))
        fp_p = list(map(int, (df_agg.Is_Bot == 0) & (df_agg['g*_Pack'] >= df_agg.z_alpha)))
        tn_p = list(map(int, (df_agg.Is_Bot == 0) & (df_agg['g*_Pack'] < df_agg.z_alpha)))
        fn_p = list(map(int, (df_agg.Is_Bot == 1) & (df_agg['g*_Pack'] < df_agg.z_alpha)))
        
        tp_a = list(map(int, (df_agg.Is_Bot == 1) & (df_agg['g*_Addy'] >= df_agg.z_alpha)))
        fp_a = list(map(int, (df_agg.Is_Bot == 0) & (df_agg['g*_Addy'] >= df_agg.z_alpha)))
        tn_a = list(map(int, (df_agg.Is_Bot == 0) & (df_agg['g*_Addy'] < df_agg.z_alpha)))
        fn_a = list(map(int, (df_agg.Is_Bot == 1) & (df_agg['g*_Addy'] < df_agg.z_alpha)))
        
        df_agg['TP_Pack'] = tp_p
        df_agg['FP_Pack'] = fp_p
        df_agg['TN_Pack'] = tn_p
        df_agg['FN_Pack'] = fn_p
        
        df_agg['TP_Addy'] = tp_a
        df_agg['FP_Addy'] = fp_a
        df_agg['TN_Addy'] = tn_a
        df_agg['FN_Addy'] = fn_a
        
            
            
            
            
            
        df_agg.to_csv(rf'{agg_dir}/agg_{i}.csv', index=False)

        i += 1
            
            
    
    
    #print(df_agg_packets)
    return r'{output}'

# Summarizes aggregated data
def summarize_data(agg_dir, t_path):
    
    i = 0
    
    #directory = os.fsencode(directory)
    
    agg_directory = os.listdir(agg_dir)
    agg_directory.sort(key=natural_keys)
    
    trials = pd.read_csv(t_path)
    df_length = len(trials)
    
    tp_p = np.zeros(df_length)
    fp_p = np.zeros(df_length)
    tn_p = np.zeros(df_length)
    fn_p = np.zeros(df_length)
    
    tp_a = np.zeros(df_length)
    fp_a = np.zeros(df_length)
    tn_a = np.zeros(df_length)
    fn_a = np.zeros(df_length)
    
    #f1_p = np.zeros(df_length)
    #f1_a = np.zeros(df_length)
    
    for agg_file in agg_directory:

        agg_filename = os.fsdecode(agg_file)

        if agg_filename.endswith(".csv"):
            
            
            print(f'{i+1}/{df_length}')
            agg_filename = rf'{agg_dir}/' +  agg_filename 
            
            #print(agg_filename)
            
            df_agg = pd.read_csv(agg_filename)

            df_sum = df_agg.sum()
            

            tp_p[i] = df_sum.TP_Pack
            fp_p[i] = df_sum.FP_Pack
            tn_p[i] = df_sum.TN_Pack
            fn_p[i] = df_sum.FN_Pack
            
            tp_a[i] = df_sum.TP_Addy
            fp_a[i] = df_sum.FP_Addy
            tn_a[i] = df_sum.TN_Addy
            fn_a[i] = df_sum.FN_Addy
            
           
            
            
            #f1_p[i] = 2*df_sum.TP_Pack/(2*df_sum.TP_Pack+df_sum.FP_Pack+df_sum.FN_Pack)
            #f1_a[i] = 2*df_sum.TP_Addy/(2*df_sum.TP_Addy+df_sum.FP_Addy+df_sum.FN_Addy)
            i += 1        
    
    #acc_p = (tp_p + tn_p) / (tp_p + fp_p + tn_p + fn_p)
    #prec_p = tp_p / (tp_p + fp_p)
    recall_p = tp_p / (tp_p + fn_p)
    fpr_p = fp_p / (fp_p + tn_p)
    f1_p = 2*(tp_p) / (2*tp_p + fp_p + fn_p)
    
    #acc_a = (tp_a + tn_a) / (tp_a + fp_a + tn_a + fn_a)
    #prec_a = tp_a / (tp_a + fp_a)
    recall_a = tp_a / (tp_a + fn_a)
    fpr_a = fp_a / (fp_a + tn_a)
    f1_a = 2*(tp_a) / (2*tp_a + fp_a + fn_a)
    
    #trials['Acc_P'] = acc_p
    #trials['Prec_P'] = prec_p
    trials['Recall_P'] = recall_p
    trials['FPR_P'] = fpr_p
    trials['F1_P'] = f1_p
    
    #trials['Acc_A'] = acc_a
    #trials['Prec_A'] = prec_a
    trials['Recall_A'] = recall_a
    trials['FPR_A'] = fpr_a
    trials['F1_A'] = f1_a
    
    
    
    #trials['TPR_Pack'] = tp_p
    
    #trials['TNR_Pack'] = tn_p
    #trials['FNR_Pack'] = fn_p
    
    #trials['TPR_Addy'] = tp_a
    
    #trials['TNR_Addy'] = tn_a
    #trials['FNR_Addy'] = fn_a

    
    trials.to_csv(t_path, index=False)
    
def analyze_data(t_path, recall=.2, fpr=.01, f1=0):
    trials = pd.read_csv(t_path)
    #bool_p = (trials.Prec_P > prec) & (trials.Recall_P > recall)
    #bool_a = (trials.Prec_A > prec) & (trials.Recall_A > recall)
    unique_hostcomm = trials.Prob_HostComm.unique()
    #unique_packcount = trials.Pack_Count.unique()
    packet = np.zeros(len(unique_hostcomm))
    address = np.zeros(len(unique_hostcomm))
    count = 0
    #for i in unique_packcount:
    for j in unique_hostcomm:
        temp_p = trials[(trials.Recall_P >= recall) & (trials.FPR_P <= fpr) & (trials.Prob_HostComm == j)]
        temp_a = trials[(trials.Recall_A >= recall) & (trials.FPR_A <= fpr) & (trials.Prob_HostComm == j)]
        packet[count] = len(temp_p)
        address[count] = len(temp_a)
        count += 1
            
    #param_combos = np.array(np.meshgrid(unique_packcount, unique_hostcomm)).T.reshape(-1, 2)
            
    df = pd.DataFrame(data=unique_hostcomm, columns=['Prob_HostComm'])
    df['Packets'] = packet
    df['Addresses'] = address
    
    return df
    
    
    

            