#!/bin/bash
dump_folder=../dumps
today=`date +%Y-%m-%d.%H:%M`
echo $dump_folder
mkdir -p $dump_folder
zip -dgr ${dump_folder}/graphmatching_${today}.zip graphmatching\