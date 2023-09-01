
# Readme


## file function 


### 01_featuring

Create features based on alpha Trade

```Python
['00_price_rising_rate0001',
 '00_price_rising_rate0003',
 '00_price_rising_rate0006',
 '00_price_rising_rate0009',
 '00_price_rising_rate0030',
 '00_price_rising_rate0060',
 '00_price_rising_rate0090',
 '00_price_rising_rate0120',
 '03_market_cap',
 '04_PE',
 '05_book2market_ratio',
 '06_div',
 '00_target',
 '01_vol_7',
 '01_vol_30',
 '01_vol_60',
 '01_vol_90',
 '01_vol_180',
 '01_vol_360',
 '02_trade_vol_1',
 '02_trade_vol_3',
 '02_trade_vol_7',
 '02_trade_vol_30',
 '02_trade_vol_60',
 '02_trade_vol_90']
```

### 01_getting DJIA

DJIA index, been normalized with passed 14 days mean and std


```Python
def procoss(df):
    df_ = df.copy()
    df_.index = df_.index.date
    
    df_ = df_[["Close", "Volume"]]
    
    df_std = df_.rolling(14).std()
    df_mean = df_.rolling(14).mean()
    
    return (df_ - df_mean) / df_std
```


### 02_norm

Z-score, except std (vol features) and target




## 99 convert to DeepTrade format

- keep 22 stocks with 2899 days, remove those have fewer trading years
- added indutry classification
