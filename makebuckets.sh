#!/bin/bash
for i in more_than_15_samples more_than_25_samples more_than_35_samples leq_than_60_samples
do
    mkdir ./Data/$i
    cp ./Data/onco-cellinedata/clin-$i-revised.csv ./Data/$i/clin.csv
    cp ./Data/onco-cellinedata/gex-new.csv ./Data/$i/gex.csv
done
