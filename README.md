# Welcome to gingado!
> A machine learning library for economics and finance


`gingado` seeks to facilitate the use of machine learning in economic and finance use cases, while promoting good practices. `gingado` aims to be suitable for beginners and advanced users alike.

## Overview

`gingado` is a free, open source library built around three main functionalities:
* **data augmentation**, to add more data from official sources, improving the machine models being trained by the user;
* **automatic benchmark model**, to enable the user to assess their models against a reasonably well-performant model; and
* **support for model documentation**, to embed documentation and ethical considerations in the model development phase.

Each of these functionalities builds on top of the previous one. They can be used on a stand-alone basis, together, or even as part of a larger pipeline from data input to model training to documentation!

## Design principles

The choices made during development of `gingado` derive from the following principles, in no particular order:
* **flexibility**: users can use `gingado` out of the box or build custom processes on top of it
* **compatibility**: `gingado` works well with other widely used libraries in machine learning, such as `scikit-learn` and `pandas`
* **responsibility**: `gingado` facilitates and promotes model documentation, including ethical considerations, as part of the machine learning development workflow

## Acknowledgements

`gingado`'s API is inspired on the following libraries:
* `scikit-learn` ([API description](https://arxiv.org/abs/1309.0238))
* `keras` (website [here](https://keras.io/about/) and also, [this essay](https://medium.com/s/story/notes-to-myself-on-software-engineering-c890f16f4e4d))
* `fastai` ([description here](https://www.mdpi.com/2078-2489/11/2/108))

In addition, `gingado` is developed and maintained using [`nbdev`](https://nbdev.fast.ai).

## Presentations, talks, papers

The material supporting public communication about `gingado` (ie, slide decks, papers) is kept in [this dedicated repository](https://github.com/dkgaraujo/gingado_comms). Interested users are welcome to visit the repository and comment on the drafts or slide decks, preferably by opening an [issue](https://github.com/dkgaraujo/gingado_comms/issues). I also store in this repository suggestions I receive as issues, so users can see what others commented (anonymously unless requested) and comment along as well!

## Install

To install `gingado`, simply run the following code on the terminal:

`$ pip install gingado`
