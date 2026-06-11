#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use akshare to test the cross-sectional IC and long-short performance of
the 20-day momentum factor of A-shares.
@author: shenyixin
"""
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import akshare as ak

SYMBOL = "000300"          # CSI 300 index code
MOMENTUM_WINDOW = 20       # Momentum look-back window
Q = 0.2                    # Long/short quantile (top & bottom 20%)
COST_RATE = 0.001          # transaction cost of portfolio adjustment
RETRIES = 2                # Number of retries
SLEEP = 0.5                # Delay between stocks
COST_GRID = [0, 0.0005, 0.001, 0.0015, 0.002]   # Cost sensitivity sweep

def get_one(code, momentum_window=MOMENTUM_WINDOW, retries=RETRIES):
    for attempt in range(retries):          
        try:
            if code.startswith('6'):
                name = 'sh' + code
            else:
                name = 'sz' + code
            raw = ak.stock_zh_a_hist_tx(symbol=name)
            result = raw[['date','close']].copy()
            result['code'] = code
            result['factor'] = result['close'].pct_change(momentum_window)
            result['next_ret'] = result['close'].pct_change().shift(-1) 
            return result                    
        except Exception as e:
            print(f"Attempt{attempt+1}failed: {type(e).__name__},retrying...")
            time.sleep(3)                   
    return None

def get_more(codes, momentum_window=MOMENTUM_WINDOW, f=get_one):
    frame = []
    total = len(codes)
    success = 0
    for i, code in enumerate(codes, 1):
        one = f(code,momentum_window)
        if one is not None:
            frame.append(one)
            success += 1
        print(f"progress {i}/{total} | success {success} | current {code}")
        time.sleep(SLEEP)
    panel = pd.concat(frame, ignore_index=True)
    panel = panel.dropna(subset=['factor','next_ret'])
    return panel

def cross_ic(x, min_stocks=5):
    x = x.dropna(subset=["factor", "next_ret"])
    if len(x) < min_stocks:                    
        return np.nan
    return x["factor"].corr(x["next_ret"], method="spearman")


def compute_daily_ic(panel, f=cross_ic):
    daily_ic = panel.groupby("date").apply(f).dropna()
    return daily_ic

def summarize_ic(daily_ic):
    ic_mean = daily_ic.mean()
    icir    = ic_mean / daily_ic.std()
    ic_win  = (daily_ic > 0).mean()
    return {'ic_mean': ic_mean, 'icir': icir, 'ic_win': ic_win}

def plot_cumulative_ic(daily_ic, momentum_window=MOMENTUM_WINDOW):
    daily_ic.cumsum().plot(figsize=(8, 5), title=f'Cumulative IC ({momentum_window}-Day Momentum)')
    plt.show()

def strategy(x, q=Q, min_stocks=10):
    if len(x) < min_stocks:
        return np.nan
    is_long  = x['factor'] < x['factor'].quantile(q)
    is_short = x['factor'] > x['factor'].quantile(1-q)
    long_ret  = x.loc[is_long, 'next_ret'].mean()
    short_ret = x.loc[is_short, 'next_ret'].mean()
    return long_ret - short_ret

def calc_sharpe_max_dd(panel, cost_rate=COST_RATE, trading_days=252):
    daily_ret = panel.groupby('date').apply(strategy).dropna()
    daily_net_ret = daily_ret - cost_rate           
    sharpe = daily_net_ret.mean() / daily_net_ret.std() * np.sqrt(trading_days)
    acc = (1 + daily_net_ret).cumprod()
    max_dd = (acc / acc.cummax() - 1).min()
    return {'sharpe': sharpe, 'max_dd': max_dd}

if __name__ == "__main__":
    cons = ak.index_stock_cons(symbol=SYMBOL)
    codes = cons['品种代码'].astype(str).str.zfill(6).tolist()    
    panel = get_more(codes, momentum_window=MOMENTUM_WINDOW)
    print(panel.shape, panel['code'].nunique())
    daily_ic = compute_daily_ic(panel, f=cross_ic)
    stats = summarize_ic(daily_ic)
    print(f"IC mean: {stats['ic_mean']:.4f}")
    print(f"ICIR:    {stats['icir']:.4f}")
    print(f"IC win:  {stats['ic_win']:.2%}")
    plot_cumulative_ic(daily_ic, momentum_window=MOMENTUM_WINDOW)   
    for c in COST_GRID:
        print(f'cost = {c}:',calc_sharpe_max_dd(panel, cost_rate=c))