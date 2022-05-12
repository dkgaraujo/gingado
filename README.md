# Welcome to gingado!
> A machine learning library for economics and finance


The purpose of `gingado` is to support usage of machine learning models in economics and finance use cases, promoting good modelling practices while being easy to use. `gingado` aims to be suitable for beginners and advanced users alike.

## Install

To install `gingado`, simply run the following code on the terminal:

`$ pip install gingado`

## Overview

`gingado` is built around three main functionalities:
* **data augmentation**, to add more data from official sources, improving the machine models being trained by the user;
* **automatic benchmark model**, to enable the user to assess their models against a reasonably well-performant model; and
* **support for model documentation**, to embed documentation and ethical considerations in the model development phase.

Each of these functionalities is illustrated below for a user trying to forecast GDP growth. Each step builds on top of the previous one, and they can be done with one line of code! But to highlight how `gingado` can benefit users in multiple ways, the brief walk-through below goes over them separtely. 

Before stepping in, let's import the necessary packages and data.

### Setup

This walk-through will use the [Jordà-Schularick-Taylor Macrohistory Database](https://www.macrohistory.net) on macroeconomics and finance as an example. As a preliminary step, let's import `gingado` and other necessary libraries, and proceed to download the data:

```python
import pandas as pd

JST_url = "http://data.macrohistory.net/JST/JSTdatasetR5.dta"
jst = pd.read_stata(JST_url, iterator=False)

jst.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>country</th>
      <th>iso</th>
      <th>ifs</th>
      <th>pop</th>
      <th>rgdpmad</th>
      <th>rgdppc</th>
      <th>rconpc</th>
      <th>gdp</th>
      <th>iy</th>
      <th>...</th>
      <th>eq_capgain</th>
      <th>eq_dp</th>
      <th>eq_capgain_interp</th>
      <th>eq_tr_interp</th>
      <th>eq_dp_interp</th>
      <th>bond_rate</th>
      <th>eq_div_rtn</th>
      <th>capital_tr</th>
      <th>risky_tr</th>
      <th>safe_tr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2659</th>
      <td>2013.0</td>
      <td>USA</td>
      <td>USA</td>
      <td>111</td>
      <td>315820.328999</td>
      <td>31571.993947</td>
      <td>103.425299</td>
      <td>101.892671</td>
      <td>16784.851</td>
      <td>0.192086</td>
      <td>...</td>
      <td>0.271035</td>
      <td>0.019355</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.023508</td>
      <td>0.024601</td>
      <td>0.139843</td>
      <td>0.212405</td>
      <td>-0.065168</td>
    </tr>
    <tr>
      <th>2660</th>
      <td>2014.0</td>
      <td>USA</td>
      <td>USA</td>
      <td>111</td>
      <td>318106.646578</td>
      <td>32113.618881</td>
      <td>105.186253</td>
      <td>104.113597</td>
      <td>17527.258</td>
      <td>0.196377</td>
      <td>...</td>
      <td>0.136350</td>
      <td>0.019199</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.025408</td>
      <td>0.021817</td>
      <td>0.130063</td>
      <td>0.132729</td>
      <td>0.122398</td>
    </tr>
    <tr>
      <th>2661</th>
      <td>2015.0</td>
      <td>USA</td>
      <td>USA</td>
      <td>111</td>
      <td>320413.930388</td>
      <td>32800.923063</td>
      <td>107.421590</td>
      <td>107.192931</td>
      <td>18224.780</td>
      <td>0.198301</td>
      <td>...</td>
      <td>-0.000092</td>
      <td>0.021124</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.021358</td>
      <td>0.021122</td>
      <td>0.046193</td>
      <td>0.065433</td>
      <td>-0.008779</td>
    </tr>
    <tr>
      <th>2662</th>
      <td>2016.0</td>
      <td>USA</td>
      <td>USA</td>
      <td>111</td>
      <td>322705.239927</td>
      <td>33078.508719</td>
      <td>108.318698</td>
      <td>109.333457</td>
      <td>18715.040</td>
      <td>0.195831</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2663</th>
      <td>2017.0</td>
      <td>USA</td>
      <td>USA</td>
      <td>111</td>
      <td>324802.861426</td>
      <td>33593.446309</td>
      <td>110.013284</td>
      <td>111.389150</td>
      <td>19519.424</td>
      <td>0.204547</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 57 columns</p>
</div>



For practical purposes, let's use for this database only the information from 1991 onwards:

```python
jst.drop(labels=['iso', 'ifs', 'crisisJST_old'], axis=1, inplace=True)
jst['year'] = pd.to_datetime(jst['year'], format='%Y')

start_year = '1991-01-01'
end_year = '2016-01-01'
jst = jst[jst['year'] >= start_year]
```

```python
jst.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>country</th>
      <th>pop</th>
      <th>rgdpmad</th>
      <th>rgdppc</th>
      <th>rconpc</th>
      <th>gdp</th>
      <th>iy</th>
      <th>cpi</th>
      <th>ca</th>
      <th>...</th>
      <th>eq_capgain</th>
      <th>eq_dp</th>
      <th>eq_capgain_interp</th>
      <th>eq_tr_interp</th>
      <th>eq_dp_interp</th>
      <th>bond_rate</th>
      <th>eq_div_rtn</th>
      <th>capital_tr</th>
      <th>risky_tr</th>
      <th>safe_tr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2659</th>
      <td>2013-01-01</td>
      <td>USA</td>
      <td>315820.328999</td>
      <td>31571.993947</td>
      <td>103.425299</td>
      <td>101.892671</td>
      <td>16784.851</td>
      <td>0.192086</td>
      <td>173.067206</td>
      <td>-426.197</td>
      <td>...</td>
      <td>0.271035</td>
      <td>0.019355</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.023508</td>
      <td>0.024601</td>
      <td>0.139843</td>
      <td>0.212405</td>
      <td>-0.065168</td>
    </tr>
    <tr>
      <th>2660</th>
      <td>2014-01-01</td>
      <td>USA</td>
      <td>318106.646578</td>
      <td>32113.618881</td>
      <td>105.186253</td>
      <td>104.113597</td>
      <td>17527.258</td>
      <td>0.196377</td>
      <td>175.997979</td>
      <td>-365.193</td>
      <td>...</td>
      <td>0.136350</td>
      <td>0.019199</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.025408</td>
      <td>0.021817</td>
      <td>0.130063</td>
      <td>0.132729</td>
      <td>0.122398</td>
    </tr>
    <tr>
      <th>2661</th>
      <td>2015-01-01</td>
      <td>USA</td>
      <td>320413.930388</td>
      <td>32800.923063</td>
      <td>107.421590</td>
      <td>107.192931</td>
      <td>18224.780</td>
      <td>0.198301</td>
      <td>176.301162</td>
      <td>-407.769</td>
      <td>...</td>
      <td>-0.000092</td>
      <td>0.021124</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.021358</td>
      <td>0.021122</td>
      <td>0.046193</td>
      <td>0.065433</td>
      <td>-0.008779</td>
    </tr>
    <tr>
      <th>2662</th>
      <td>2016-01-01</td>
      <td>USA</td>
      <td>322705.239927</td>
      <td>33078.508719</td>
      <td>108.318698</td>
      <td>109.333457</td>
      <td>18715.040</td>
      <td>0.195831</td>
      <td>178.575038</td>
      <td>-428.350</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2663</th>
      <td>2017-01-01</td>
      <td>USA</td>
      <td>324802.861426</td>
      <td>33593.446309</td>
      <td>110.013284</td>
      <td>111.389150</td>
      <td>19519.424</td>
      <td>0.204547</td>
      <td>182.415361</td>
      <td>-439.642</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 54 columns</p>
</div>



### Data augmentation

As we can see, the dataset contains mainly information on the GDP of other countries. There are many ways in which this dataset can be complement by a richer set of variables. For example, for the same time periods we can add detailed information on trade, inflation, labour markets, foreign currency exchange rates, and many other series that would help machine learning models better form predictions. 

`gingado` helps this step by facilitating *data augmentation*, ie complementing the user dataset with compatible data. `gingado` only sources data from official data sources, namely statistical agencies, central banks, and other relevant authorities at the domestic or international level.

In particular, `gingado`'s function `augm_with_sdmx` looks for all data available from the desired sources for the defined period and frequency and downloads them all, but only adds to the original dataset the variables that bring some level of variance in the data that can be explored by the machine learning models later on.

```python
from gingado.augmentation import augm_with_sdmx

jst_augm_gingado = augm_with_sdmx(jst, start_year, end_year, time_col='year', freq='A', sources='BIS')
```

      6%|▌         | 1/18 [00:00<00:07,  2.23it/s]

    Source: BIS, dataflow: WS_CBPOL_D ok!


     11%|█         | 2/18 [00:00<00:05,  2.89it/s]

    Source: BIS, dataflow: WS_CBPOL_M ok!


     17%|█▋        | 3/18 [00:01<00:04,  3.06it/s]

    Source: BIS, dataflow: WS_CBS_PUB ok!


     22%|██▏       | 4/18 [00:01<00:04,  3.47it/s]

    Source: BIS, dataflow: WS_CREDIT_GAP ok!


     28%|██▊       | 5/18 [00:01<00:03,  3.57it/s]

    Source: BIS, dataflow: WS_DEBT_SEC2_PUB ok!


     39%|███▉      | 7/18 [00:01<00:02,  4.11it/s]

    Source: BIS, dataflow: WS_DER_OTC_TOV ok!
    Source: BIS, dataflow: WS_DSR ok!


     50%|█████     | 9/18 [00:02<00:02,  3.89it/s]

    Source: BIS, dataflow: WS_EER_D ok!
    Source: BIS, dataflow: WS_EER_M ok!


     61%|██████    | 11/18 [00:02<00:01,  4.27it/s]

    Source: BIS, dataflow: WS_GLI ok!
    Source: BIS, dataflow: WS_LBS_D_PUB ok!


     67%|██████▋   | 12/18 [00:03<00:01,  4.42it/s]

    Source: BIS, dataflow: WS_LONG_CPI ok!


     78%|███████▊  | 14/18 [00:03<00:00,  4.51it/s]

    Source: BIS, dataflow: WS_OTC_DERIV2 ok!
    Source: BIS, dataflow: WS_SPP ok!


     89%|████████▉ | 16/18 [00:03<00:00,  5.14it/s]

    Source: BIS, dataflow: WS_TC ok!
    Source: BIS, dataflow: WS_XRU ok!


     94%|█████████▍| 17/18 [00:04<00:00,  5.33it/s]

    Source: BIS, dataflow: WS_XRU_D ok!


    100%|██████████| 18/18 [00:04<00:00,  3.86it/s]


    Source: BIS, dataflow: WS_XTD_DERIV ok!


     25%|██▌       | 1/4 [00:02<00:06,  2.05s/it]

    Source: WB, dataflow: DF_WITS_Tariff_TRAINS ok!


     50%|█████     | 2/4 [00:02<00:02,  1.09s/it]

    Source: WB, dataflow: DF_WITS_TradeStats_Development ok!


     75%|███████▌  | 3/4 [00:02<00:00,  1.28it/s]

    Source: WB, dataflow: DF_WITS_TradeStats_Tariff ok!


    100%|██████████| 4/4 [00:03<00:00,  1.21it/s]


    Source: WB, dataflow: DF_WITS_TradeStats_Trade ok!


      6%|▌         | 1/18 [00:00<00:03,  4.54it/s]

    Trying to download WS_CBPOL_D from BIS... not possible.


     11%|█         | 2/18 [00:00<00:03,  4.65it/s]

    Trying to download WS_CBPOL_M from BIS... not possible.


     17%|█▋        | 3/18 [00:00<00:03,  3.95it/s]

    Trying to download WS_CBS_PUB from BIS... not possible.


     22%|██▏       | 4/18 [00:00<00:03,  4.03it/s]

    Trying to download WS_CREDIT_GAP from BIS... not possible.


     28%|██▊       | 5/18 [00:01<00:03,  3.95it/s]

    Trying to download WS_DEBT_SEC2_PUB from BIS... not possible.


     33%|███▎      | 6/18 [02:41<10:57, 54.82s/it]

    Trying to download WS_DER_OTC_TOV from BIS... ok!


     39%|███▉      | 7/18 [02:42<06:47, 37.02s/it]

    Trying to download WS_DSR from BIS... not possible.


     44%|████▍     | 8/18 [02:42<04:12, 25.30s/it]

    Trying to download WS_EER_D from BIS... not possible.


     50%|█████     | 9/18 [02:42<02:37, 17.45s/it]

    Trying to download WS_EER_M from BIS... not possible.


     61%|██████    | 11/18 [02:43<00:59,  8.48s/it]

    Trying to download WS_GLI from BIS... not possible.
    Trying to download WS_LBS_D_PUB from BIS... not possible.


     67%|██████▋   | 12/18 [02:43<00:36,  6.10s/it]

    Trying to download WS_LONG_CPI from BIS... ok!


     78%|███████▊  | 14/18 [02:44<00:12,  3.06s/it]

    Trying to download WS_OTC_DERIV2 from BIS... not possible.
    Trying to download WS_SPP from BIS... not possible.


     83%|████████▎ | 15/18 [02:44<00:06,  2.20s/it]

    Trying to download WS_TC from BIS... not possible.


     94%|█████████▍| 17/18 [02:46<00:01,  1.44s/it]

    Trying to download WS_XRU from BIS... ok!
    Trying to download WS_XRU_D from BIS... not possible.


    100%|██████████| 18/18 [06:24<00:00, 21.37s/it]


    Trying to download WS_XTD_DERIV from BIS... ok!


      0%|          | 0/4 [00:00<?, ?it/s]/Users/douglasaraujo/Coding/.venv_gingado/lib/python3.10/site-packages/pandasdmx/api.py:260: UserWarning: 'provider' argument is redundant for <Resource.data: 'data'>
      warn(f"'provider' argument is redundant for {resource_type!r}")
     25%|██▌       | 1/4 [00:01<00:03,  1.24s/it]

    Trying to download DF_WITS_Tariff_TRAINS from WB... not possible.


     50%|█████     | 2/4 [00:01<00:01,  1.50it/s]

    Trying to download DF_WITS_TradeStats_Development from WB... not possible.


     75%|███████▌  | 3/4 [00:01<00:00,  2.08it/s]

    Trying to download DF_WITS_TradeStats_Tariff from WB... not possible.


    100%|██████████| 4/4 [00:02<00:00,  1.94it/s]


    Trying to download DF_WITS_TradeStats_Trade from WB... not possible.


      0%|          | 0/4 [00:00<?, ?it/s]

    Getting data from BIS's WS_DER_OTC_TOV


     25%|██▌       | 1/4 [00:08<00:24,  8.31s/it]

    Getting data from BIS's WS_LONG_CPI


     50%|█████     | 2/4 [00:08<00:07,  3.59s/it]

    Successful
    Getting data from BIS's WS_XRU


     75%|███████▌  | 3/4 [00:09<00:02,  2.36s/it]

    Successful
    Getting data from BIS's WS_XTD_DERIV


    100%|██████████| 4/4 [00:10<00:00,  2.61s/it]


    Successful


    0it [00:00, ?it/s]



    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    /Users/douglasaraujo/Coding/gingado/index.ipynb Cell 11' in <module>
          <a href='vscode-notebook-cell:/Users/douglasaraujo/Coding/gingado/index.ipynb#ch0000007?line=0'>1</a> from gingado.augmentation import augm_with_sdmx
    ----> <a href='vscode-notebook-cell:/Users/douglasaraujo/Coding/gingado/index.ipynb#ch0000007?line=2'>3</a> jst_augm_gingado = augm_with_sdmx(jst, start_year, end_year, time_col='year', freq='A', sources=['BIS', 'WB'])


    File ~/Coding/gingado/gingado/augmentation.py:33, in augm_with_sdmx(df, start_date, end_date, time_col, freq, sources)
         <a href='file:///Users/douglasaraujo/Coding/gingado/gingado/augmentation.py?line=24'>25</a>     end_date=df[time_col].max()
         <a href='file:///Users/douglasaraujo/Coding/gingado/gingado/augmentation.py?line=26'>27</a> sdmx_data = get_sdmx_data(
         <a href='file:///Users/douglasaraujo/Coding/gingado/gingado/augmentation.py?line=27'>28</a>     start_date=start_date,
         <a href='file:///Users/douglasaraujo/Coding/gingado/gingado/augmentation.py?line=28'>29</a>     end_date=end_date,
         <a href='file:///Users/douglasaraujo/Coding/gingado/gingado/augmentation.py?line=29'>30</a>     freq=freq,
         <a href='file:///Users/douglasaraujo/Coding/gingado/gingado/augmentation.py?line=30'>31</a>     sources=sources
         <a href='file:///Users/douglasaraujo/Coding/gingado/gingado/augmentation.py?line=31'>32</a>     )
    ---> <a href='file:///Users/douglasaraujo/Coding/gingado/gingado/augmentation.py?line=32'>33</a> sdmx_data = sdmx_data.dropna(axis=1).sort_index()
         <a href='file:///Users/douglasaraujo/Coding/gingado/gingado/augmentation.py?line=33'>34</a> sdmx_data.reset_index(inplace=True)
         <a href='file:///Users/douglasaraujo/Coding/gingado/gingado/augmentation.py?line=34'>35</a> sdmx_data['TIME_PERIOD'] = pd.to_datetime(sdmx_data['TIME_PERIOD'])


    AttributeError: 'NoneType' object has no attribute 'dropna'


### Automatic benchmark

Now

## Design principles

The choices made during development of `gingado` derive from the following principles, in no particular order:
* *lowering the barrier to use machine learning* can help more economists familiarise themselves with these techniques and use them when appopriate
* *promoting good practices* such as documenting ethical considerations and benchmarking models as part of machine learning development will help embed these habits in economists
* *offering compatibility with other existing software that is consolidated by wide practice* benefits users and should be promoted as much as possible
