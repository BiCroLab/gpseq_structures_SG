#!/usr/bin/env bash

chr=$1 #chrs.csv
comms=$2 # gpseq_rank20_samples1000.csv
comm=$3

cut -d',' -f${comm} ${comms} > ${comm}.txt
paste $chr ${comm}.txt | awk '{print "chr"$1"\t"$2*1000000"\t"$2*1000000+1000000"\t"$3}'|
    awk '$4>0.001' |sed 's/chr23/chrX/' > ${comms}_comm-${comm}.bed
rm ${comm}.txt


