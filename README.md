# climate-prediction
Supervised classification techniques for predicting climate impacts

## Overview

This repository is home to data processing scripts used by Ecotrust to
perform supervised classification using modeled climate data in order to predict
likely impacts from climate change.

While the process is generalized to almost any phenomenon that is climate-driven,
it has been applied here to forestry and agriculture.

* Bioclimatic envelope modeling of tree species in the US West (see `bioclim` directory)
* Prediction of climate-driven shifts in agricultural production zones (see the `agzones` directory)

The basic process is this:
1. Gather explanatory raster datasets representing current climate
2. Draw a relationship between explanatory data and some spatially-explicit
   "response" (e.g. the presence of a tree species)
3. Use that relationship to model the response, predicting the spatial patterns
   of the response to future climatic changes.

The process is implemented in python and relies on many existing packages:
* pyimpute
* scikit-learn
* pandas
* numpy
* rasterio and GDAL

## Installation

All of these dependencies are pip-installable and specified in the `requirements.txt`

    pip install -r requirements.txt

## Quickstart

Change to the `agzones` directory and run the python scripts within.

## More reading

Check out the detailed docs and examples in [pyimpute](https://github.com/perrygeo/pyimpute)

## Videos
For an overview of the process applied to predicting agricultural zone shifts,
see my presentation at FOSS4G 2014:

http://vimeo.com/106235287


