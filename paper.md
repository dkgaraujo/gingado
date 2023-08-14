---
title: 'gingado: a machine learning library focused on economics and finance'
tags:
  - Python
  - finance
  - economics
  - machine learning
authors:
  - name: Douglas K. G. Araujo
    orcid: 0000-0001-8070-6828
    corresponding: true
    #equal-contrib: true
    affiliation: 1
affiliations:
 - name: Bank for International Settlements, Economist, Basel, Switzerland
   index: 1
date: 14 August 2023
bibliography: gingado.bib
---

# Summary

gingado is an open source Python library under active development that offers a variety of convenience functions and objects to support usage of machine learning in economics research. It is designed to be compatible with widely used machine learning libraries. gingado facilitates augmenting user datasets to include relevant data sourced directly from official sources by leveraging the SDMX data and metadata sharing protocol. The library also offers a model benchmarking object that creates a model with a reasonably good performance out-of-the-box and, if provided with candidate models, retains the model with the best performance. gingado also includes methods to help with machine learning model documentation and functions that flexibly create simulated datasets with a variety of non-linearities and treatment effects, to support model prototyping and benchmarking. The library is under active undervelopment and new functionalities are periodically added or improved.

# Statement of need

`gingado` is a package designed to faciliate the usage of machine learning techniques by economists. While `scikit-learn` and many other popular machine learning libraries count with extensive functionalities, they are not specifically attuned to needs that often occur with economists. More specifically, for some economics or finance use cases that involve prediction (such as forecasting), data discovery and the process of testing whether more data actually contributes to model quality is cumbersome to set up.

Also importantly, with the increase in importance of machine learning models and their use both in research and production, `gingado` facilitates the creation of easy to use and transmit documentation of the model. The baseline template with built-in questions can also serve as a guide for economists that need to tailor documentation templates for their own use cases.

# Introduction

Conceptually, every machine learning (ML) application entails the combination of a specific dataset, algorithm, cost function and optimisation procedure - and each of these components can be replaced more or less independently from the others (@DeepLearning). The set of possibilities is particularly wide in many economics and finance use cases (@doi:10.1146/annurev-economics-080217-053433, @10.1257/jep.31.2.87, @varian2014big, @doerr2021big @chakraborty2017machine). And there is often no definite answer as to which alternative is best (@10.1257/jep.31.2.87). Hence, in practice, creating a ML application can amount to a process that requires multiple iterations and attempted improvements until achieving a result considered to be satisfactory. In fact, as of writing this paper, different steps of the process of creating ML models in economics and finance could be more streamlined, ranging from selection and use of the dataset or simulation of a dataset with known generating process, comparison of different ML models, and crucially, also the model's documentation. 

`gingado` is a free, open source library that seeks to facilitate these and other steps in the use of ML in economics and finance in academic and practitioner settings, while promoting good modelling and documentation practices.[^code] It offers a number of main contributions. First, `gingado` facilitates data augmentation of the original user dataset with statistical series from official sources, in a way that is relevant for each case and empirically testable on its ability to improve the model's performance. Second, `gingado` provides automatic benchmark models that perform well on a broad set of situations and that can train quickly to achieve a reasonable if not the best performance for the dataset at hand; users can also make use of a generic benchmark object to create their own automatic benchmarks. Third, for when the user needs to test a causal inference models, or benchmark existing models, `gingado` offers functions that flexibly simulate panel data with customised data generating processes that include linear and non-linear interactions and diverse treatment assignment mechanisms and (homogenous or heterogenous) treatment effects. Fourth, `gingado` promotes documentation of the ML model as part of the development workflow, automating some documentation steps to leave users room for concentrating on more value-added documentation items such as ethical considerations. Also here `gingado` offers users the possibility for users to create their own model documentation templates that are easy to embed in the modelling practice. And fifth, `gingado` offers a number of other utilities for helping data science work in economics, in particular in time series of panel data.

[^code]: `gingado`'s instructions, documentation, practical examples and source code are available at https://dkgaraujo.github.io/gingado/. The library is named after the Brazilian concept that is difficult to translate but broadly represents a dance-like non-stop swing of the body that is often associated with flexibility to adjust oneself to adverse situations, eg in life or in a football match. The name is also as an homage to the Afro-Brazilian martial art of Capoeira, where having "gingado" is key. I chose "gingado" as the name for this library focused on using ML in economics and finance because it combines the idea of a constant swing, akin to how economic and financial series also "swing" around a trend through the course of business and financial cycles, with flexibility, similar to how ML models are considerably flexible when fitting functional forms.

`gingado` is in active development. The library follows three design principles:

1. **Flexibility**: `gingado` works well out of the box but users can customise its objects in ways that are more suitable for their use cases;
2. **Compatibility**: `gingado` works seamlessly with other widely used libraries in data science and ML; and
3. **Responsibility**: `gingado` promotes model documentation and ethical considerations as key steps in the modelling process.

In addition, `gingado` seeks to be a parsimonious library that complements, rather than redo, existing functionalities of other widely used libraries. `gingado`'s application programming interface (API) is compatible in particular with `scikit-learn` (@scikit-learn), which itself is the basis for a variety of other ML libraries; in addition, the `gingado` API can generally be made compatible with other Python ML libraries with minimal adjustment, and even with R or other languages (eg, via the `reticulate` package, @reticulate). The library can also be used in a modular way: users might prefer to use `gingado` only to augment their datasets, create automatic benchmark models, or document their models, etc.

These characteristics make `gingado` a useful tool across many domains in economics and finance. ML algorithms are already amply used in economics for prediction problems (in the sense of @PredictionProblems), causal inference[^causal] (eg, @chernozhukov2018generic and @athey2019generalized), model estimation (@maliar2021deep, @fernandez2019financial and @duarte2018machine), estimation of models with non-traditional data (@ferreira2021forecasting) and even being the subject of study themselves (@predunequal, @giannone2021illusion). `gingado`'s functionalities can be used as appropriate in each instance. Central banks have also been using ML in a variety of applications (@araujo2023machine), and the practitioner use cases in the industry are numerous and diverse.

[^causal]: A recent compilation of causal inference techniques from various domains is https://neurips.cc/virtual/2021/workshop/21871.

To showcase practical applications, the online documentation includes two end-to-end examples[^Examples] (with more to come over time): the automatic benchmark and model documentation functionalities are illustrated with the cross-country dataset with over 60 variables used to analyse drivers of GDP growth per capita [@BARRO19941]. The dataset was also recently used by @giannone2021illusion to study the different predictive abilities of sparse vs dense models. Another example focuses on attempts to forecast exchange rates (@rossi2013exchange), illustrating how `gingado`'s utilities can be used to compare different lags of the model, download specific data from SDMX sources, augment the original dataset with other relevant data, quickly create a benchmark model and use it compare different alternatives, and finally how to document the model to promote responsible model maintenance and usage. The examples illustrate ways in which `gingado` can help economists' workflow. 

[^Examples]: Available at: https://dkgaraujo.github.io/gingado/.

The next section describes how `gingado` facilitates data augmentation. Section @sec-bench outlines the automatic benchmark process, followed by a a discussion of real and simulated data functions in @sec-data. Section @sec-doc describes `gingado`'s model documentation framework. The last section concludes. The online documentation describes how to install the most up-to-date version of `gingado` and also more advanced topics like how to create customised automatic benchmark models and model documentation templates.

# Data augmentation {#sec-data_augm}
Publicly available data have a long tradition in economics and finance research and practice. For example, the Federal Reserve Bank of St. Louis' [Federal Reserve Economic Data (FRED)](https://fred.stlouisfed.org). system has developed in tandem with wider adoption of the internet itself (see @RePEc:fip:fedlrv:00023 for an interesting narrative of FRED's history). Similarly, numerous other central banks and statistical agencies, as well as the international financial institutions, put datasets in the public domain in one form or the other. A number of statistics organisations created in the early 2000's an initiative to promote the collection, compilation and dissemination of statistical data, the [Statistical Data and Metadata Exchange (SDMX)](https://sdmx.org), which is now in its version 3.0.[^SDMX3] In addition, other data aggregators such as [Base dos Dados](https://basedosdados.org/, @dahis2022data) and [DBnomics](https://db.nomics.world) host an incredible amount of economic and financial series. 

[^SDMX3]: A technical description of version 3.0 is found here: https://sdmx.org/wp-content/uploads/SDMX_3-0-0_SECTION_1_FINAL-1_0.pdf.

Many of these services allow users to access data in a programmatic way, ie setting up a computer programme to download the data instead of the user manually accessing the website, selecting the data, downloading it in a file and incorporating the data in the analyses. Thanks to SDMX and to the broader availability of user-friendly data APIs like the one offered by FRED, querying data from trusted sources to augment the user's original dataset has become much easier than in the past. Accessing data programmatically also allows any numeric transformations, consistency checks and data imputation routines that are applied on the dataset to be done in a reproducible and transparent way.

In addition, programmatic access to data can also ensure that any data that are added to the original dataset are done so in a consistent way. For example, SDMX includes the concept of "codelists", which are standardised definitions that apply across dataset domains. One specific codelist contains all possible realisations of the frequency of a dataset (ie, daily, weekly, monthly, etc.), and another codelist encompasses all possible codes for specific currencies. This technology ensures that the user has greater control over the datasets to be incorporated.

The considerations above are more important when models are in the production stage. In economics and finance, ML models are occasionally designed to be run in production settings instead of a one-off execution; and this is often performed by users that were not involved during development and therefore are oblivious to the model's internal functioning. For example, a model forecasting stock prices will be used extensively over time - and by stock traders or portfolio managers, not the developers. These situations tend to take place in situations where the ML model will require future observations of the datasets used for training. Therefore, automating the process of data augmentation in a way that can ensure consistency between datasets helps to insulate users from worrying about these processes related to the use of additional data.

`gingado` aims to facilitate this process of finding and adding new datasets, leveraging the public availability of reliable data from official sources and automatically ensuring that the augmented dataset is consistent with the original, having the same frequency and time period. With this, when the model is run in the future, potentially by other people or automatically by systems, the same publicly available dataset(s) used during model training are downloaded and used each time with the appropriate time period. The current version of `gingado` achieves this by means of its data augmentation object `AugmentSDMX`; the idea is to gradually add to the public codebase other "data augmenters" that can scan and download publicly available datasets in a way that is consistent with the original dataset.

## Basic data augmentation process

Similar to other parts of `gingado`, the API for data augmentation is designed for compatibility with `scikit-learn`. Specifically, the user decides which specific sources and dataflows will be scanned for data upon instantiation of the data augmenter object. So far, the object does not perform any activity other than store this and other parameters. It is only when one of the methods `fit`, `transform`, or `fit_transform` are called that the data augmenter will (a) read characteristics of the data and (b) look in the named sources and dataflows what would be adequate datasets corresponding to the same frequency and time period. This process is shown in listing @lst-AugmentSDMXAPI, which assumes the user already has a data frame called `X_train` with training data *indexed by time*: `gingado` uses this time period information behind the curtains to obtain data of the relevant frequency and time periods. In the listing, the user first imports the class `AugmentSDMX`, and then define a dictionary `src` to list the SDMX sources and their relevant dataflows - more on how to find the sources below. The data augmenter is set to look for the Composite indicator of systemic stress (CISS) by the European Central Bank (ECB) and the central bank policy rates dataflow from the Bank for International Settlement (BIS). In the next line, an instance of the class `AugmentSDMX` is created using those sources listed in `src`, and the method `fit_transform` learns the frequency and time periods from `X_train` and use that to fetch all data series from these two sources that comply with these time requirements.

```{#lst-AugmentSDMXAPI .python lst-cap="Basic example of the AugmentSDMX API"}
from gingado.augmentation import AugmentSDMX

src = {'ECB': ['CISS'], 'BIS': ['WS_CBPOL_D']}
augmented_X_train = AugmentSDMX(sources=src).fit_transform(X_train)
```

The process described above will result in `augmented_X_train`, a dataset that contains the original data, now augmented by potentially numerous other series. The data are indexed by time, and for this reason every combination of permissible codelists in the augmented dataset will create a different variable (ie, a different column). For example, even if the original dataset represents only data from Brazil, the augmentation process shown in @lst-AugmentSDMXAPI will append the CISS for every country in the list as a different column, as well as the central bank policy rates. `gingado` makes an explicit choice to allow for this, instead of filtering by individual (in this example, retrieving only the dataset about Brazil), because the newly added data can have some level of information on the target variable, and if that is the case the model would probably uncover it. Of course, users that desire to add only data relating to a country, currency, etc might add those as relevant filters, as described in more detail in the documentation.

One important note is that the user remains responsible for ensuring that the data being automatically added is not a covariate that will interfere with the desired statistical properties of the task at hand. For example, ensuring that the covariate is not endogenous or otherwise a bad control to include (@HünermundLouwCaspi2023 discuss the sensitiveness of some ML-based inference to inclusion of bad controls.) In some settings, one strategy to ensure only adequate covariates are included is to consider well which data sets will be included and tested in the model, ie that broad groups of data that might negatively interfere with the model are not selected in the data augmentation functions.

## Data augmentation with SDMX {#sec-AugmSDMX}

`gingado` explores and fetches available datasets from official sources to augment user data using SDMX, an initiative by international organisations (BIS, ECB, Organisation for Economic Cooperation and Development - OECD, International Monetary Fund - IMF, World Bank - WB, Eurostat, and United Nations- UN) that develop and publish statistics from these sources. Since its inception in 2001, SDMX has grown to be used by other sources as well - primarily central banks and statistics agencies - as a standard to disseminate their data.

Downloading data from SDMX sources can be advantageous to users because of the variety of sources and the consistency of the concepts describing the datasets. Instead of dealing with the specific data descriptions and download processes of each of the SDMX participant institutions, the user can rely on an API based on standardised concepts to fetch the data. For example, using SDMX to look across multiple sources for data at quarterly frequency related to the countries of Argentina, Brazil and South Africa for the period spanning the first quarter of 2015 to the fourth quarter of 2021 would use the codelists (introduced above) for frequency and for reference area. These are the same across the SDMX sources and dataflows, which facilitates the process of finding relevant datasets.

`AugmentSDMX` is based on the `pandasdmx` python package. The backend code searches all listed SDMX sources in this package, and retrieves the dataflows for those it is able to get (some sources may time out). Given that downloading all the relevant data from all sources could be an expensive operation, users define the SDMX source(s) from which to get the data, as well as the specific dataflows if they wish.

As of the time of writing, the data providers available for automatic download of data using `AugmentSDMX` are:[^Sources]

-  Australian Bureau of Statistics
-  BIS
-  Countdown 2030
-  Deutsche Bundesbank (Germany)
-  ECB
-  Eurostat
-  International Labour Organization - ILO
-  Internationa Monetary Fund - IMF
-  National Institute of Statistics and Geography (Mexico)
-  National Institute of Statistics and Economic Studies (France)
-  National Institute of Statistics (Italy)
-  National Institute of Statistics (Lithuania)
-  Norges Bank (Norway)
-  National Bank of Belgium
-  Organisation for Economic Cooperation and Development  -OECD
-  Pacific Data Hub
-  Statistics Estonia
-  United Nations Statistics Division
-  UN International Children’s Emergency Fund (UNICEF)
-  World Bank Group’s “World Integrated Trade Solution”
-  World Bank Group’s “World Development Indicators”

[^Sources]: Users can get the up-to-date list with the `gingado` method `list_SDMX_sources` in `gingado.utils`.

Each of those sources offers a number of dataflows, which are closely related datasets. For example, one dataflow related to foreign exchange rates could include the time series of multiple individual exchange rate pairs, and each pair can be downloaded in their nominal or real exchange rate. The dataflows from all of these sources could be available to train the ML model at hand. In total, the sources listed above result in 9,110 dataflows.

One potential drawback of the process is that the computation time might increase as the number of SDMX series are added. This is an important consideration, as production version models will also need to incorporate these data - depending on how expensive these transactions are, it could significantly affect the speed of computation at run time. However, for cases where immediate response is not required, the quantity of data could be considered reasonable.

## Data augmentation with other datasets

There are many other official agencies that offer freely (and easily) accessible datasets that would certainly be useful to the proposed data augmentation. For example, the Federal Reserve Bank of St Louis maintains the [Federal Reserve Economic Data (FRED)](https://fred.stlouisfed.org) and related APIs, and the Brazilian Institute for Geography and Statistics (IBGE) has a [range of APIs](https://servicodados.ibge.gov.br/api/docs/) that offer datasets that could be of use. It would be useful to also be able to use these tools programatically as seamlessly as for the SDMX-based data augmentation.

While `gingado` doesn't include these sources above off-the-shelf as automatic augmenters, future versions might concentrate on more flexibly adding data augmenters and crucially, offering users the possibility to do the same. The code already allows for that for the users willing to manually adjust the class `AugmentSDMX`, but it may not be currently straightforward for most users.

## Is it worth adding more and more data?
While an increasing amount of data can lead to better results by ML models, there are situations where users might want to consider limiting the amount of data being fed to a model. Therefore, the answer to this subsection's title will depend on each use case, and `gingado` can help the user answer it.

`AugmentSDMX`'s API is compatible with `scikit-learn` in a specific way that allows it to be included in a `pipeline`. Pipelines are objects that apply on data a specific sequence of transformations, and possibly a final step consisting of an estimator (such as a predictor). These pipelines exist to allow the cross-validation of the sequence of steps as a whole, and to allow for a consistent way to estimate different versions of the same model without losing control of the data pipeline.

What this means in practice is that the user can apply standard parameter search algorithms such as grid search to test whether or not the inclusion of a particular dataset will impact the model, eg by improving its performance, changing the importance of regressors, etc. This process is exemplified in listing @lst-AugmentSDMXSearch: the user created with a few lines a model that, when fitted, will estimate the model without augmentation (ie, will "pass through" `AugmentSDMX`) and with augmentation. The first four lines import the necessary objects from `gingado` and `scikit-learn`. Then, an instance of the data augmenter is created in the variable `sdmx`, in this case using all dataflows made available by the BIS. Note that unlike @lst-AugmentSDMXAPI, neither the method `fit` nor `fit_transform` was called yet and therefore no calculation or data download is yet done. In the next step, an instance of a `Pipeline` object is created - but still not fitted. This pipeline chains together two steps: the data augmentation and a random forest. Finally, a dictionary with a grid of parameters to be tested empirically and `grid`, a variable that performs an ML calibration using grid search with cross-validation, are created. `param_grid` is the key part of this listing, since it tells `grid` to test two versions of the ML model: one that bypasses the augmentation step and uses only the user-provided dataset, and the other one that uses the `sdmx` variable to augment automatically the user dataset with all relevant and compatible (ie, with the same frequency and time periods) data from the BIS. When fitted using training data on covariates and dependent variables, `grid` will select the parameters in `param_grid` that result in models with the best performance, and thus the question of whether or not to use more data can be answered empirically and in a completely data-driven way. Beyond that, the user can even jointly search richer combinations of different parameters governing data augmentation, data transformation and model estimation by combining `AugmentSDMX` with `scikit-learn`'s `Pipeline`.

```{#lst-AugmentSDMXSearch .python lst-cap="Use of AugmentSDMX in a scikit-learn pipeline"}
from gingado.augmentation import AugmentSDMX
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

sdmx = AugmentSDMX(sources={'BIS': 'all'})

pipeline = Pipeline([
    ('augmentation', sdmx),
    ('forest', RandomForestRegressor())
])

param_grid = {'augmentation': ['passthrough', sdmx]}

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid
    )
```

# Automatic benchmark {#sec-bench}

`gingado` offers users the possibility of automatically developing a benchmark ML model fine-tuned for their particular case, in a short time and with no input from the user other than the data (although users can also fine tune all aspects of the benchmark). This is achieved by means of an embedded parameter grid search mechanism that evaluates different versions of the underlying algorithm on the user's data, and selects that one that performs better as the benchmark.

The objective is to help the user during the exploratory phase of the development of a ML model. The practice of establishing a baseline model is common in ML practice. Without a goalpost, a baseline model that is relatively simple to understand and benchmark against, it can be difficult in practice to realise if one's model is performing well or just improving from a low base. So `gingado` allows users to quickly create a fully functioning model that can serve as benchmark, as shown in listing @lst-NewBenchmark. Similar to @lst-AugmentSDMXAPI, this listing assumes the user already has available two data frames: `X_train`, containing a panel of covariates, and `y_train`, which is the dependent variable. The first line imports the object `ClassificationBenchmark`. In the second line, an instance called `benchmark` is created and already fitted in the same line. Behind the scenes, this instance of `ClassificationBenchmark` will perform a grid search using a random forest with default parameters that tend to work well in my experience in a variety of datasets that tend to be in line with the size of empirical papers in economics. The fitted object then represents the model with the best performance, and can be used by the user for prediction, etc. Naturally, the user can pass other estimators and also other sets of parameters for the grid search, illustrating `gingado`'s combination of convenience with flexibility.

```{#lst-NewBenchmark .python lst-cap="Creation of an automatic benchmark model"}
from gingado.benchmark import ClassificationBenchmark
benchmark = ClassificationBenchmark().fit(X_train, y_train)
```

As mentioned, the off-the-shelf implementations of this automatic benchmark model are based on random forests (@breiman1996bagging, @breiman2001random), one type for regression tasks and another one for classification. Random forests, one of the most widely used ML methods according to industry practitioners (@howard2020deep) present several advantages that make them good candidates for benchmark models. They tend to have a very good out-of-sample fit, and require little data preparation work compared to other ML models. Random forests also provides an intuitive way to measure the individual importance of regressors, although they don't lend themselves to interpreting the channels by which the regressors contribute to the prediction (@varian2014big). These regressor importance measures are the mean reduction in impurity occurring in the trees that use that particular variable. The values are then typically scaled so that the sum of all feature importance measures is always one. Practitioners use this measurement as one possibility for variable selection (@géron2017hands-on, @kohlscheen2021does).

Whenever a benchmark model is fitted to the data, the model automatically creates a documentation. In the off-the-shelf implementations, this documentation is done via the `ModelCard` object, described in more detail in section @sec-doc. From then on, the model documentation can be accessed - and most importantly, filled out from that object.

Benchmark models also have a specific method to compare themselves with other candidate models. This allows users to directly compare their own candidate models with the existing benchmark. When this is done, via the module `compare`, `gingado` also includes amongst the candidates to be tested an ensemble combination of all the candidates and the current benchmark. The inclusion of this candidate ensembles serves to leverage the advantage that ensemble models have over simpler models (@giannone2021illusion).

## Other use cases for a benchmark model

In addition to serving as a baseline model during the development of ML, users can simply retain the automatic benchmark as their production model. Beyond benchmark-specific functionalities, these objects behave also as normal `scikit-learn` models and therefore can be applied as any normal model would. 

Benchmarks can be used as a test to see what if any regressors differentiate the most between two or more groups, for example to see if one group of samples has out-of-domain data (as proposed by @howard2020deep). The test proceed as follows: a random forest classifier is trained on the dataset, with group identifiers serving as the target variable. The forest's calculated regressor importance can then uncover what are the variables that differentiate between groups. This test can be generalised to check what regressors, if any, vary substantially between two or more groups. 

## Custom automatic benchmarks
In spite of the empirical qualities of random forests, no single algorithm could plausibly cover all potential use cases. For example, random forests could potentially not perform as well as other established algorithms if the economic or financial data at hand contains multimodal data, ie images, videos, texts and other such data. These data types are increasingly relevant in economics research. Some examples of this growing literature on non-traditional data types include @gentzkow2019text and @corpculture for text and @SatelliteAfrica and @bajari2023hedonic for images. In addition, random forests cannot extrapolate beyond the range of the training data that was fed to them. And there are some empirical settings in which gradient boosting trees are more used than random forests and other ML methods (@jrfm15040165). Similarly, @TabularDeepLearning show that neural networks can achieve similar, and in some cases better, performance than tree-based methods in some cases. And @taylor2018forecasting describe Facebook's own tool (now open sourced) for automatic forecasting, without relying on random forests.

So random forests might not be the first choice for all cases, even though they are expected to work well in a wide array of situations. In addition to the off-the-shelf implementations described above, `gingado` offers two possibilities for users to set up their own benchmark models. The most straightforward one is to simply pass as argument a model object to the benchmark's method `set_benchmark`. This will cause the benchmark object to put this new model in place of the previous benchmark. 

The second object involves the creation of a new benchmark object class altogether. `gingado` ships with a base class, `ggdBenchmark`, that contains all the necessary features for a benchmarker object. Users that wish to create their own benchmark models can sub-class from `ggdBenchmark` and implement the desired funcionalities. The user's benchmark will then work in the same way as the original `gingado` benchmark models. This user-created benchmark can include any algorithm or combination of algorithms, as long as the API is maintained.

# Real and synthetic datasets {#sec-data}

`gingado` aims to provide users with an easy way to load real datasets that can be used as benchmarks in research, along with functions that allow users to create synthetic datasets from a wide range of data generating processes. 

## Real datasets

Economics and finance research often relies on benchmark datasets used in canonical papers to explore new insights derived from the original work or to evaluation and question the original findings. Prominent (but not exhaustive) examples include the [Angrist Data Archive](https://economics.mit.edu/people/faculty/josh-angrist/angrist-data-archive); the data used by @giannone2021illusion to test ML models; the macro-history datasets of @jorda2017macrofinancial, @jorda2019rate and @jorda2021bank; the technology adoption (CHAT) dataset of @NBERw15319; and the datasets used in synthetic control studies by @abadie2003economic, @abadie2010synthetic and @abadie2015comparative; and others. Some of these benchmark data may be useful for building and testing ML algorithms.

In many cases, these datasets are shared in a way that makes them convenient to use with almost no manipulation. But even in these cases, these datasets are formatted using different platforms (Stata's DTA file, or in CSV/Excel files, etc). What `gingado` aims is to make available selected benchmark datasets in a standardised format that can be readily used by economists. At this point, the only dataset loaded this way in `gingado` is the one from @BARRO19941; @lst-BarroLee illustrates its use. After importing the `loan_BarroLee_1994` function in the first line, the user then attributes the result of calling this function to two datasets, `X` and `y`, which can then be used as normal for the training of ML models. The goal is for other datasets to have a similar structure, where their usage can be as simple as importing the appropriate function and calling it once. Naturally, the original source of the all data used should be cited and acknowledged accordingly by the end user.

Importantly, `gingado` does not aim to restrict this section only to datasets that support well-cited papers. Users that want to propose other datasets, including from their own work, are welcome to do so. In any case, the data must be used in a published academic paper. 


```{#lst-BarroLee .python lst-cap="Loading data from Barro and Lee (1994)"}
from gingado.datasets import load_BarroLee_1994()
X, y = load_BarroLee_1994()
```

## Simulated datasets

In many cases, researchers testing new econometric estimators use simulated data, created under "lab-like" conditions with a known data generating process. Such datasets enable the user to test whether proposed estimators really work as intended for a dataset with given characteristics, and can be especially helpful for causal estimands (@imbens2015causal). The possibility of simulating a datasets of varying lengths - on the time and the cross-sectional dimensions - also facilitate the testing of asymptotics. `gingado`'s `make_causal_effect` function offers users the possibility to simulate non-trivial data, including with non-linear interactions and a rich set of treatment-related variables.

More specifically, beyond the number of samples $i = (1, \dots, n)$ of the dataset, which can be interpreted as peers units or as time periods in a panel data, and $k = (1, \dots, M)$ number of features, `make_causal_effect` allows users to creates a dataset $\{y_i, X_i, D_i\}$ of outcome variable, covariates and treatment vectors respectively, by choosing the functional form of the following characteristics:

- $y_i | X_i$, the pre-treatment outcome. For the untreated units, this corresponds to the observations of $y_i$; for the treated units, this is the portion of $y_i$ before adding the treatment effect. The pre-treatment outcome depends on the covariates $X_i$ and on a constant. It might also have a random component[^reproduc]

[^reproduc]: All random components in `gingado` code can be reproduced with the use of the same random seed.

- $W_i = p(D_i \neq 0 | X_i)$, $i$'s propensity of being treated, ie having a treatment that is different than zero.[^treatval] The propensity can be a function that depends on $X_i$, either deterministically or with a random component. Alternatively, it can also be set with a scalar, in which case it is uniform across all samples, or with a random assignment to each sample according to a parameterised or empirical random distribution; in both cases the propensity simplifies to $p(W_i = 1 | X_i) = p(W_i = 1)$. 

[^treatval]: The treatment value itself can be set by the user, and therefore is not necessarily a 1, 0 dummy.

- $D_i \neq 0 | W_i$, the actual treatment assignment. The default value is purely a function of the treatment propensity through a binomial distribution: $p(D_i \neq 0 | W_i) = B(1, W_i)$. This is probably the most relevant use case. However, there are instances where researchers might want to explore a more complex treatment assignment relationship. For example, treatment can be rationed and only applied to the $\psi$ samples with the highest $W_i$, or applied with probability $B(1, W_i)$ but also subject to this rationing. In these cases, the treatment assignment would also be a function of $\psi$, in addition to $W_i$. For example, the case where only the $\psi$ most propense units would be treated can be defined as: $1(D_i \neq 0) = B(1, W_i) \times 1(B(1, W_i) \in \mathnormal{top}(W_i, \psi))$, where $\mathnormal{top}(\cdot, n)$ is the function that orders the set and returns the highest $n$ observations.

- $D_i$, the treatment value. The treatment can be set to 1 in the most simple of cases, but the treatment magnitude (and sign) can also vary as a function of covariates $X_i$, for $D_i | X_i$. A random variation in the sample-specific treatment can also be introduced. 

- $\tau_i | D_i$, the treatment effects on $y_i$ for each treated sample. This represents the difference between the actual observation $y_i$ and the pre-treatment outcome $y_i$ for treated samples. In the most simple cases, it may be a one-to-one mapping to the treatment value $D_i$. But, it may also vary by sample according to covariates $X_i$. 

All dependencies on covariates above can include non-linearities such as minimum or maximum comparisons, various types of interactions, exponentials, etc. In short, anything that can be coded using a NumPy array and respects the necessary contraints of each argument (for example, the treatment propensity must always be a number between 0 and 1). It is also important to note that a more complex treatment chain (propensity to assignment to value to effect) can depend on covariates in a different way at each step. Hopefully this flexibility in creating datasets with causal effects can provide researchers with ways to compare different ways to correct for interferences in the estimation of different potential outcomes.

# Model documentation {#sec-doc}

Stakeholders often require some level of documentation of the ML models. From simple descriptions of the model to standardised model reports to fully-fledged evaluation of models, `gingado` provides a way for models to be more easily documented. The basic idea is that a \textit{documentation template} outlines the specific items to be documented, and various methods allow the user to interact with this template. This template can be seen at any time by the user with the method `show_template`. A `gingado` documenter object also specifies which of those items are to be filled in automatically (eg, model description can be parsed from the ML algorithm itself), and how exactly this should be done.

After a documenter is instantiated, it can read information from an existing model as well. @lst-ModelCard demonstrates how straightforward it is to automatically read information from a model, and then see what are the questions from the documentation template that were not answered automatically by the `ModelCard` object, even when the ML model itself is not a `gingado` object.[^doc] The first two rows import the necessary objects: a `gingado.ModelCard` class and the `keras` ML library. The following chunk establishes the structure of a neural network comprising of two dense layers (each with 16 nodes) followed by an end layer that predicts the probability in a classification model. This neural network is then fit using training data frames `X_train` and `y_train` assumed in @lst-ModelCard to already exist, as before. Now, an instance of `ModelCard` object, called `model_doc_keras` is created, and it then reads the information from the keras classification ML model that was created and trained right before. Finally, in the last line the code presents to the user what are the questions, or fields, in the model documentation that are still open because they were not read automatically. This nudges the user to consider answering these questions (and thus documenting their model in an easy and recordable way).

[^doc]: Currently the base documenter `ggdModelDocumentation`, from which all documenters in this library derive, can read models created using `gingado`, `scikit-learn` and `keras`. Support for automatically reading information from models built with other libraries such as `PyTorch` is under development.

```{#lst-ModelCard .python lst-cap="Example of ModelCard reading information from an existing neural network model"}
from gingado.model_documentation import ModelCard
import keras_core as keras

keras_clf = keras.Sequential()
keras_clf.add(keras.layers.Dense(
    16, 
    activation='relu', 
    input_shape=(20,)
    ))
keras_clf.add(keras.layers.Dense(16, activation='relu'))
keras_clf.add(keras.layers.Dense(1, activation='sigmoid'))
keras_clf.compile(optimizer='sgd', loss='binary_crossentropy')

keras_clf.fit(X_train, y_train, batch_size=10, epochs=10)

model_doc_keras = ModelCard()
model_doc_keras.read_model(keras_clf)
model_doc_keras.open_questions()
```

At any time, the user can view the current state of the document with the method `show_json`, and save or read it to file with `save_json` and `read_json` respectively. Additional information items that were not included automatically are filled with `fill_info` (the user can also override automatic entries). And the remaining items from the template that are not yet filled are shown to the user with the method `open_questions`. These methods (and others, not shown for brevity) aim to provide the user with a more direct, hands-on approach to documenting the model, compared to a more traditional setting where a separate document (ie, an MS Word file) need to be written and maintained. Allowing users to document their models from inside their ML development environment will help embed the documentation process as another step of the model development. Another advantage of `gingado`'s approach is that it is much easier to keep the documentation aligned with the current version of the model. This is particularly important in the settings where the user expects to iterate over different specifications until a suitable model is achieved.

The model documentation is stored as a JSON file, a flexible format that is easy for machines to read, and that can quickly be transformed into objects humans can read more easily, too. `gingado` uses JSON files because they are a common language to serialise this type of structured information that works across platforms (ie, a JSON file works in Windows machines the same way it works in MacOS, Linux or any other system). Users that want to automatically produce documents in other formats (eg, PDF files) can do so by using these JSON files with other libraries of their choice. And in settings where multiple models are developed and in production, JSON files are a more streamlined format to offer information on each model to comparison scripts.

`gingado` includes two ready-to-use documenter objects, `ModelCard` and `ForecastCard`. The `ModelCard` documentation template is based on the work of @ModelCards, which I believe strikes a good balance between being a general documentation template while also prompting the user to answer questions about the model that are relevant from a broader stakeholder perspective.`ForecastCard` is a version of `ModelCard` with questions that are targeted for forecasting tasks. Similar to how users can create their own class of benchmark models, `gingado` enables users to create their own custom documenters, from the base class `ggdModelDocumentation`. This allows specific documentation templates to be used in a more automatic way, and can be of particular importance in the context of organisations using ML, since they might have their own documentation preferences. Users can also benefit from the machinery underlying the `ggdModelDocumentation` base class to create dataset documentation (eg, à la @gebru2021datasheets).

The template for the `ForecastCard` can provide an example of the type of information a documenter could either acquire automatically from reading the model object and from asking the economist; note that its language is only slightly adapted from @ModelCards with some fields directly reproducing a model card after those authors:

- Model details (basic information about the model)
  - Variable(s) being forecasted or nowcasted
  - Jurisdiction(s) of the variable being forecasted or nowcasted
  - Person or organisation developing the model
  - Model date
  - Model version
  - Model type
  - Description of the pipeline steps being used
  - Information about training algorithms, parameters, fairness constraints or other applied approaches, and features
  - Information about the econometric model or technique
  - Paper or other resource for more information
  - Citation details
  - License
  - Where to send questions or comments about the model
- Intended use (use cases that were envisioned during development)
  - Primary intended uses
  - Primary intended users
  - Out-of-scope use cases
- Metrics (metrics should be chosen to reflect potential real world impacts of the model)
  - Model performance measures,
  - How are the evaluation metrics calculated? Include information on the cross-validation approach, if used
- Data (details on the dataset(s) used for the training and evaluation of the model
  - Datasets
  - Preprocessing steps
  - Cut-off date that separates training from evaluation data
- Ethical considerations (considerations that went into model development, surfacing ethical challenges and solutions to stakeholders. Ethical analysis does not always lead to precise solutions, but the process of ethical contemplation is worthwhile to inform on responsible practices and next steps in future work)
  - Does the model use any sensitive data?
  - What risks may be present in model usage? Try to identify the potential recipients, likelihood, and magnitude of harms. If these cannot be determined, it can be noted that they were considered but remain unknown
  - Are there any known model use cases that are especially fraught?
  - If possible, this section should also include any additional ethical considerations that went into model development, for example, review by an external board, or testing with a specific community.
- Any other caveats or recommendations (additional concerns that were not covered in the previous sections)
  - For example, did the results suggest any further testing? Were there any relevant groups that were not represented in the evaluation dataset?
  - Are there additional recommendations for model use? What are the ideal characteristics of an evaluation dataset for this model?

While most economists trained in traditional econometric techniques using time series quantitative data for forecasting might find some of these fields "over-kill", the objective of this documenter is to be a reasonably simple template for models that might, who knows, forecast economically variables by, say, also including textual data ou some other form of non-traditional data.

## Ethical issues in economics and finance for ML models
One of the reasons why `gingado` facilitates model documentation is to promote a greater role for ethical considerations as part of the development of ML models in economics and finance: if the model documentation becomes part of development workflow and certain parts of the documentation are automated, then users are presumably more likely to consider these issues at development time, not *ex post*. The importance of ethical considerations in finance applications of ML is underscored by the ability of ML to drive results in economics and finance: for example, @frost2019bigtech show evidence that ML models can outperform traditional credit bureau data in predicting loan default rate in Argentina, which illustrates the strong incentives for investment in ML models by lenders. Other evidence of the real-life impact of ML applications might materially impact stakeholder outcomes is studied by @predunequal, in the context of US mortgage lending. These authors, who also document evidence of ML outperforming traditional models, find that greater use of ML would lead to relatively higher estimated propensities of default for Black and Hispanic borrowers, even when race is not used as a feature in the models. 

More broadly, as discussed by @rambachan2020economic, differences in predictions between groups that might be attributable to algorithmic bias can be seen as a combination of basal differences observed in society (and itself possibly the result of a societal bias), and of measurement error and estimation error differences. Of course, only the last two can be addressed by better developed ML models. Still, simply bringing the existence of this bias to light as done during the documentation process might be helpful in driving economists to make models that address this issue at least partially (@cowgill2020biased). Further to that, models that are likely to result in algorithmic bias even from predominantly underlying societal causes should ideally make it clear for users that this could be the case, so that users can exercise due discretion in whether and how to deploy these models. Another strategy that could also be documented with `gingado` is to choose different samples of the data based on subpopulations in which the outcome is perceived or known to be less biased, as suggested by @ludwig2021fragile. In any case, it is plausible to expect that real-life models where these issues loom large should include documentation describing the strategy taken by model developers.

There seems to be a wide awareness of the implications from biased datasets feeding into complex, black-box ML models in economics and finance.[^XAI] They are illustrated for example by @jrfm15040165, who present a practitioner view on using ML while seeking to originate loans without bias, by the law enforcement examples of ML gone awry discussed in @ludwig2021fragile and by @doerr2021big, who describe central banks' concerns with fairness implications from greater use of ML. In the broader language domain, @Parrots highlight ethical and societal issues stemming from the incredibly large size of datasets used to train large scale language models. 

[^XAI]: A related discussion where ethics and fairness in ML has made strides but needs more progress is related to explainable artificial intelligence (XAI) (for example, @BARREDOARRIETA202082).

But the ethical implications of ML do not stem just from the datasets used for training. It is likely that similar issues are important in a variety of uses, in varying degrees. For example, @Parrots also discuss model-related aspects: eg, the environmental impact of the large-size architecture of these models, and the risks originating from the higher (apparent) fluency of these models. @automateMH and @thomas2022reliance discuss pitfalls and risks from the metrics chosen during ML development. And importantly in the case of economics, @ludwig2021fragile pinpoint the poor experience with ML in the judicial system to a case of faulty development of models, including due statistical consideration to the fact that modeled outcomed are in many cases only a subset of the necessary information, and that decision is taken ultimately by a human decisionmaker. In addition, properly informing the user on algorithmic design choices and recommended and unrecommended use cases might prevent situations where the model is designed to be unbiased, but its deployment is  botched due to not considering that the model would work best if delivered equally to all subpopulations of interest, as shown by @lambrecht2019algorithmic in the case of an ML-delivered ad for careers in sciences, technology, engineering and maths (STEM) designed to be gender neutral but that was more widely seen by men due to the higher ad costs for women. Therefore, promoting opportunities for economists developing and deploying ML models to think about these implications from the complete model pipeline, from data to application to usage, seems more than warranted, in line with @cowgill2019economics.

Consideration of these and other ethical issues might be made easier by facilitating model documentation, which nudges users to explicitly document (and thus consider) features of the data and model that go beyond purely model architecture features, which are important but can be read automatically by software. In fact, I hope that the mere availability of model documentation functionalities might act as a reminder that this is an important step (@cowgill2020biased). At the same time, automatically covering model information in the documentation opens up space for the user to reflect on the open human-answerable questions, many of which include issues around ethics. In other words, can model documentation address all issues in ML ethics in economics? Definitely no. But the hope is that enabling and facilitating model documentation might help users take steps in that direction, both in academic as well as in practitioner settings.

# What to expect next?

The vision for `gingado` is for it to become a collection of tools that can help economists deploy ML in various use cases in research or practitioner work. In this sense, development of new tools to be added in `gingado` is guided by two considerations. First, what are the pain points of the ML workflow for economists that can be tackled ergonomically? Second, what are the areas of ML that can benefit economics use cases? Using these considerations as backdrop, four areas of active development as of writing of this paper are:

- new canonical datasets to help new users get up to speed with initial models, fomenting the build-up of AI skillsets for beginner-level users or simply being used as benchmarks;

- clustering functionalities that automatically divide a population of entities into clusters, and retain only those that are in the same cluster as a specified entity of interest. This can be used to find (and retain only) related entities to unit(s) of interest in multidimensional settings, providing a useful selection of control units for example, for matching applications (@imbens2009matching) or in the estimation of causal effects using synthetic controls (@abadie2003economic, @abadie2021using) in a more data-driven way, as proposed by @araujo2023machine.

- causal ML, implementing Python versions of algorithms that have been designed statistically to allow for causal inference. Examples that might be implemented include the generalised random forest of @athey2019generalized, the ML-based version of synthetic controls of @araujo2023machine, the double/debiased ML of @chernozhukov2018double and other algorithms.

- functionalities to fine-tune and use large language models (eg, @openai2023gpt4, @touvron2023llama) in settings of economic interest.

# Conclusion

`gingado` is a free, open source ML library focused on use cases in economics and finance - in both research and practice. Its compatibility with widely used ML frameworks, most notably `scikit-learn`, allows it to leverage on the wide familiarity with these tools and complement existing user codebases. And its flexible approach simultaneously affords users the ability to advance ML model development with few steps, while enabling users to tweak all tools to meet their goals, including adjusting models and their outcomes to better suit econometric use cases when necessary (@doi:10.1146/annurev-economics-080217-053433). The toolset provided by `gingado` was first created for my own use as a practitioner of ML in economics, and I will continue developing it over time to incorporate funcionalities that can be of most value-added to researchers and practitioners in economics and finance. Because the code is open, also the broader public can propose improvements and even new funcitonalities, either by directly suggesting it or by proposing code that implements these ideas. At the same time, I hope that `gingado` can contribute to facilitate greater use of ML in economics and finance, while promoting good modelling practices including a greater role for model documentation and ethical considerations. 

Of course, both research and empirical applications are already benefiting from more intense usage of ML, including in policy organisations (@board2017artificial, @araujo2023machine, @frost2019bigtech, @doerr2021central). And recent breakthroughs in the field (such as GPT-4, @openai2023gpt4) are likely to further promote this, as exemplified by @NBERw30957. In this context, `gingado` can help new and more experienced economists familiarise themselves with ML practice in an ergonomic way, by removing some of the complexity in getting such models set up, trained, and documented. The particular areas focused by `gingado` so far are data augmentation, automatic benchmark models, real and simulated datasets and model documentation.

Adding more data to one's dataset can often be cumbersome from an operational perspective, and especially so when multiple sources are involved. This is of course much harder when the model is used in a production setting, instead of a one-off analysis. `gingado` addresses this by augmenting the user dataset through an object that fits nicely in standard ML pipelines. This also allows the user to test whether or not adding more data actually improves the model (according to any criteria defined by the user). In addition, `gingado`'s data augmentation method focuses on ensuring that the data provided to users is from trusted sources. When more data augmentation objects are added to the `gingado` codebase, the reliability of data sources will be a key criterion.

Automatic benchmark models are not new. @fastai and @taylor2018forecasting for example enable users to quickly set up models with a reasonably good performance for the vast majority of use cases in their respective domains. `gingado` builds on that insight and provides users with additional functionalities that to my knowledge are novel. Namely, it offers a way to conveniently compare candidate models (and their ensemble) and pick the best one as the new benchmark. It also offers the automatic documentation of the benchmark model, and the ability to create one's own benchmark from a base class that that ensures users' customised benchmark models would be compatible with the other functionalities.

These models, of course, need to be trained on data. `gingado` provides the user with real and simulated datasets. The former currently contains one dataset, used by @BARRO19941, but others will be included over time with the goal of forming a portfolio of academic benchmark data representing various areas of economics and finance studies. And the functionality to simulate data can be used to generate a wide range of datasets, with rich treatment chains and non-linear dependences of the outcome variable on the covariates. These can be especially useful in the context of causal ML using the potential outcomes framework (@imbens2015causal).

And finally, there is model documentation. @10.1257/jep.31.2.87 mention the risk that ML models in economics and finance are applied naïvely or have their outputs misinterpreted. This risk increases with greater deployment of ML in important aspects of daily life, such as banking. But the risk probably also grows as the technical preconditions for ML, such as greater availability of higher-dimensional datasets and of compute power, facilitate model development by more people. As the popularity of ML models amongst economists grows, my goal with `gingado` is to contribute a small part to embed model documentation in the process of model development, automating some of the questions to afford the humans in control the opportunity to properly document their models and pay due consideration to ethical issues arising in each particular situation, hopefully facilitating research in the field and leading to better AI models developed and deployed in practice.


# Acknowledgements

This work represents my opinion and not necessarily that of the BIS. I thank Ben Cohen, Sebastian Doerr, Jon Frost, Frederik Hering and participants at the Irving Fisher Committee on Banking Statistics (IFC) 2nd Conference 'Data Science in Central Banking' and at the Informal Machine Learning Community seminar for helpful comments and feedback. All errors are my own. A more detailed version of this paper is available at https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4482553.

# References