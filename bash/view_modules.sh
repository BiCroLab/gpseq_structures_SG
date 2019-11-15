#!/usr/bin/env bash

cmm=$1
membership=$2
comm=$3

cut -f${comm} ${membership} | awk '{printf("%.6f\n", $1*1e1+1e-6)}' > ${cmm}.membership.simple   

head -1 ${cmm} > ${cmm}.header
tail -1 ${cmm} > ${cmm}.tails


cat ${cmm} |grep ^"<marker id" > ${cmm}.marker
cut -d'=' -f-8 ${cmm}.marker > ${cmm}.marker.left
cut -d'=' -f9- ${cmm}.marker > ${cmm}.marker.right

sed -i 's/^/"/' ${cmm}.membership.simple
sed -i 's/$/"/' ${cmm}.membership.simple
paste ${cmm}.marker.left ${cmm}.membership.simple|tr '\t' '='|awk '{print $0" />"}' > ${cmm}.marker

cat ${cmm} |grep ^"<link id" > ${cmm}.linker
cut -d'=' -f-6 ${cmm}.linker > ${cmm}.linker.left
cut -d'=' -f7- ${cmm}.linker > ${cmm}.linker.right
cat ${cmm}.linker.left | awk '{print $0"=\"0.00100\"/>"}' > ${cmm}.linker

cat ${cmm}.header ${cmm}.marker ${cmm}.linker ${cmm}.tails > ${cmm}.${comm}.cmm
rm ${cmm}.header ${cmm}.marker* ${cmm}.linker* ${cmm}.tails 
