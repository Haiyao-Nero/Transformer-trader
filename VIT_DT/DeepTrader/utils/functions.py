import math
import random

import numpy as np
import torch

switch2days = {'D': 1, 'W': 5, 'M': 252, 'Y': 252}
# For any number of trading length Ny = 252 (because we are getting daily data.. So I changed only variable 'M' in trading mode a
# and setting trade length I want.. after changing in Hyper.json, functions.py, we need to change in parse_config.py also for trade length)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def calculate_metrics(agent_wealth, trade_mode, MAR=0.):
    """
    Based on metric descriptions at AlphaStock
    """
    #print(type(agent_wealth))
    #print(agent_wealth.shape)
    #print(agent_wealth)

    trade_ror = agent_wealth[:, 1:] / agent_wealth[:, :-1] - 1
    # agent_wealth[:, 1:] means selecting all rows and selecting values from 2nd column till end
    # 
    #print("Trade ROR")
    #print(trade_ror.shape)
    #print(trade_ror)

    if agent_wealth.shape[0] == trade_ror.shape[0] == 1:
        agent_wealth = agent_wealth.flatten()
    trade_periods = trade_ror.shape[-1]
    if trade_mode == 'D':
        Ny = 252
    elif trade_mode == 'W':
        Ny = 50
    elif trade_mode == 'M':
        Ny = 252
    elif trade_mode == 'Y':
        Ny = 1
    else:
        assert ValueError, 'Please check the trading mode'
    
    Ny = 252 # Whatever trading length, average we are doing is of daily wealth. So we are taking 252 days as average for ARR.
    AT = np.mean(trade_ror, axis=-1, keepdims=True)
    VT = np.std(trade_ror, axis=-1, keepdims=True)

    """
    # monthly Rate of Return
    MRR = np.mean(trade_ror, axis=-1, keepdims=True)
    MRR = MRR*21
    
    # Cummulative Rate of Return
    CRR = []
    mul = 1
    for mrr in MRR:
        mul *= (1+mrr)
        CRR.append(mul-1)
    """

    APR = AT * Ny
    AVOL = VT * math.sqrt(Ny)
    ASR = APR / AVOL
    drawdown = (np.maximum.accumulate(agent_wealth, axis=-1) - agent_wealth) /\
                     np.maximum.accumulate(agent_wealth, axis=-1)
    MDD = np.max(drawdown, axis=-1)
    CR = APR / MDD

    tmp1 = np.sum(((np.clip(MAR-trade_ror, 0., math.inf))**2), axis=-1) / \
           np.sum(np.clip(MAR-trade_ror, 0., math.inf)>0)
    downside_deviation = np.sqrt(tmp1)
    DDR = APR / downside_deviation

    # sortino ratio
    tmp2 = np.sum(((np.clip(MAR-trade_ror, 0., math.inf))**2), axis=-1) / \
              np.sum(np.clip(MAR-trade_ror, 0., math.inf)>0)
    downside_deviation = np.sqrt(tmp2)
    SoR = APR / downside_deviation

    metrics = {
        'APR': APR,
        'AVOL': AVOL,
        'ASR': ASR,
        'MDD': MDD,
        'CR': CR,
        'DDR': DDR,
        'SoR': SoR
    }

    return metrics