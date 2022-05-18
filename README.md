# Welcome to gingado!
> A machine learning library for economics and finance


The purpose of `gingado` is to support usage of machine learning models in economics and finance use cases, promoting good modelling practices while being easy to use. `gingado` aims to be suitable for beginners and advanced users alike.

Most functionalities are likely to also be useful to a broader set of users. In addition to more general benefits, `gingado` is designed to align well with the workflow needs of economists due to its support for panel datasets and the functionality to quickly and easily add official statistical data on macroeconomics and finance to the user dataset.

## Install

To install `gingado`, simply run the following code on the terminal:

`$ pip install gingado`

## Overview

`gingado` is built around three main functionalities:
* **data augmentation**, to add more data from official sources, improving the machine models being trained by the user;
* **automatic benchmark model**, to enable the user to assess their models against a reasonably well-performant model; and
* **support for model documentation**, to embed documentation and ethical considerations in the model development phase.

Each of these functionalities is illustrated below for a user trying to forecast GDP growth. Each step builds on top of the previous one, and they can be used stand-alone, together, or even as part of a larger pipeline from data input to model training to documentation! But to highlight how `gingado` can benefit users in multiple ways, the brief walk-through below goes over them separtely and then highlights how they can work jointly. 

Before stepping in, let's import the necessary packages and data.

### Setup

This walk-through will use the [Jordà-Schularick-Taylor Macrohistory Database](https://www.macrohistory.net) on macroeconomics and finance as an example. As a preliminary step, let's import `gingado` and other necessary libraries, and proceed to download the data:

```python
import pandas as pd

JST_url = "http://data.macrohistory.net/JST/JSTdatasetR5.dta"
jst = pd.read_stata(JST_url, iterator=False)

jst.tail()
```

For practical purposes, let's use for this database only the information from 2011 onwards:

```python
jst.drop(labels=['iso', 'ifs', 'crisisJST_old'], axis=1, inplace=True)
jst['year'] = pd.to_datetime(jst['year'], format='%Y')

start_year = '2011-01-01'
end_year = '2016-01-01'
jst = jst[jst['year'] >= start_year]

jst.set_index(['year', 'country'], inplace=True)
```

```python
jst.tail()
```

### Data augmentation

As we can see, the dataset contains mainly information on the GDP of other countries. There are many ways in which this dataset can be complement by a richer set of variables. For example, for the same time periods we can add detailed information on trade, inflation, labour markets, foreign currency exchange rates, and many other series that would help machine learning models better form predictions. 

`gingado` helps this step by facilitating *data augmentation*, ie complementing the user dataset with compatible data. `gingado` only sources data from official data sources, namely statistical agencies, central banks, and other relevant authorities at the domestic or international level.

In particular, `gingado`'s function `augm_with_sdmx` looks for all data available from the desired sources for the defined period and frequency and downloads them all, but only adds to the original dataset the variables that bring some level of variance in the data that can be explored by the machine learning models later on.

```python
from gingado.augmentation import augm_with_sdmx

jst_augm_gingado = augm_with_sdmx(jst, start_year, end_year, time_col='year', freq='A', sources='BIS')
```

### Automatic benchmark

Creating a model card can be facilitated by using the template. Upon creation, two informations are already filled by gingado...

## Design principles

The choices made during development of `gingado` derive from the following principles, in no particular order:
* *lowering the barrier to use machine learning* can help more economists familiarise themselves with these techniques and use them when appopriate
* *promoting good practices* such as documenting ethical considerations and benchmarking models as part of machine learning development will help embed these habits in economists
* *offering compatibility with other existing software that is consolidated by wide practice* benefits users and should be promoted as much as possible
