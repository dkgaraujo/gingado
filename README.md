# Welcome to gingado!
> A machine learning library for economics and finance


The purpose of `gingado` is to support usage of machine learning models in economics and finance use cases, promoting good modelling practices while being easy to use. `gingado` aims to be suitable for beginners and advanced users alike.

Most functionalities are likely to also be useful to a broader set of users. In addition to more general benefits, `gingado` is designed to align well with the workflow needs of economists due to its support for panel datasets and the functionality to quickly and easily add official statistical data on macroeconomics and finance to the user dataset.

## Overview

`gingado` is built around three main functionalities:
* **data augmentation**, to add more data from official sources, improving the machine models being trained by the user;
* **automatic benchmark model**, to enable the user to assess their models against a reasonably well-performant model; and
* **support for model documentation**, to embed documentation and ethical considerations in the model development phase.

Each of these functionalities builds on top of the previous one, and they can be used stand-alone, together, or even as part of a larger pipeline from data input to model training to documentation!

## Install

To install `gingado`, simply run the following code on the terminal:

`$ pip install gingado`

## Design principles

The choices made during development of `gingado` derive from the following principles, in no particular order:
* *lowering the barrier to use machine learning* can help more economists familiarise themselves with these techniques and use them when appopriate
* *promoting good practices* such as documenting ethical considerations and benchmarking models as part of machine learning development will help embed these habits in economists
* *offering compatibility with other existing software that is consolidated by wide practice* benefits users and should be promoted as much as possible
