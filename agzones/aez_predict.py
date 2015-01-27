import sys
import os
import json
import numpy as np
from pyimpute import load_training_rasters, load_targets, impute, stratified_sample_raster
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from pprint import pprint
import time

INPUTDIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "inputs")
if not os.path.exists(".cache"):
    os.mkdir(".cache")

###############################################################################
# Load the known data points or "training" data
# x ~ the explanatory variables which are full coverage/inexpensive
# y ~ the response variables which are sparse/expensive/impossible to collect

tdir = os.path.join(INPUTDIR, "training/")

explanatory_fields = [
    "tmin12c",
    "tmax8c",
    "p_ph_c",
    "pmean_wntrc",
    "pmean_sumrc",
    "irr_lands",
    "gt_demc",
    "grwsnc",
    "d2u2c",
    ]

explanatory_rasters = [os.path.join(tdir, r + ".tif")
                       for r in explanatory_fields]

response_raster = os.path.join(tdir, 'iso_zns3-27.tif')

# cache this so we can work with a consistent training set
sfile = ".cache/selected.json"
try:
    print "Loading training data"
    selected = np.array(json.load(open(sfile)))
    print "\tcached"
except IOError:
    print "\trandom stratified sampling"
    selected = stratified_sample_raster(response_raster,
                                    target_sample_size=20,
                                    min_sample_proportion=0.5)
    with open(sfile, 'w') as fh:
        fh.write(json.dumps(list(selected)))

print len(selected), "samples"

###############################################################################
# Set up classifier

from sklearn.externals import joblib
pfile = ".cache/cache_classifier.pkl"
try:
    print "Loading classifier;"
    rf = joblib.load(pfile)
    print "\tUsing cached @ %s" % pfile
except:
    print "\ttraining classifier..."
    train_xs, train_y = load_training_rasters(
        response_raster, explanatory_rasters, selected)

    import time
    start = time.time()

    # Instansiate the classifier
    rf = RandomForestClassifier(n_estimators=10, n_jobs=1)
    # fit the classifier to the training data
    rf.fit(train_xs, train_y)

    print "training time:", time.time() - start
    joblib.dump(rf, pfile)

###############################################################################
# Assess predictive accuracy
print "Cross validation"
from sklearn import cross_validation
cvfile = ".cache/cross_validation.txt"
try:
    acc = open(cvfile).read()
except IOError:
    cv = 5
    scores = cross_validation.cross_val_score(rf, train_xs, train_y, cv=cv)
    acc = "%d-fold Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (
        cv, scores.mean() * 100, scores.std() * 200)
    with open(cvfile, 'w') as fh:
        fh.write(acc)
print acc

###############################################################################
# Assess feature importance
featimp = sorted(zip([str(x) for x in explanatory_fields],
           [round(x, 3)*100 for x in rf.feature_importances_]),
       key=lambda tup: tup[1],
       reverse=True)

print "Feature importances:"
for fi in featimp:
    print "\t", fi[1], "% ", fi[0]

###############################################################################
# Load the target/explanatory raster data
# will be used to predict resposes

# rcps = ["RCP45", "RCP85"]
# years = ["2030s", "2050s", "2070s", "2080s"]
rcps = ["RCP85"]
years = ["2070s"]

print "Imputing response rasters FOR CURRENT DATA"
target_xs, raster_info = load_targets(explanatory_rasters)

start = time.time()
impute(target_xs, rf, raster_info, outdir="out_aezs_CURRENT",
       linechunk=250, class_prob=True, certainty=True)
run_time = time.time() - start
print run_time, "seconds"

for rcp in rcps:
    for year in years:
        print "Loading target explanatory raster data, swapping out for %s %s climate data" % (rcp, year)
        fdir = os.path.join(INPUTDIR, "%s/%s/" % (rcp, year))
        climate_rasters = [
            "grwsnc",
            "pmean_sumrc",
            "pmean_wntrc",
            "tmax8c",
            "tmin12c",]

        explanatory_rasters = []
        for ef in explanatory_fields:
            if ef in climate_rasters:
                # swap out for future
                explanatory_rasters.append(
                    os.path.join(fdir, ef + '.tif'))
            else:
                # use current/training
                explanatory_rasters.append(
                    os.path.join(tdir, ef + '.tif'))

        target_xs, raster_info = load_targets(explanatory_rasters)

        ########################################################################
        # Impute response rasters
        # default to standard naming convention for outputs
        # data gets dumped to an output directory
        print "Imputing response rasters"
        start = time.time()
        impute(target_xs, rf, raster_info, outdir="out_aezs_%s_%s" % (rcp, year),
               linechunk=250, class_prob=True, certainty=True)
        run_time = time.time() - start
        print run_time, "seconds"
