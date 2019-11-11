#!/usr/bin/env bash

commfile=$1
xyz=$2
comm=$3

cut -d',' -f-3 $xyz | head
tail -n+2 $commfile | cut -d',' -f3- | cut -d',' -f ${comm} | head

