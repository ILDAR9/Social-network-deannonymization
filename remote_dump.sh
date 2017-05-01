#!/bin/bash
# 
# Backup script
# 

# Format: YEAR MONTH DAY - HOUR MINUTE SECOND
DATE=$(date +%Y%m%d-%H%M%S)
TEMPFOLDER="dumps/temp_inst"
ZIPFILE="dumps/insta_$DATE.zip"

mkdir $TEMPFOLDER
scp -r ildar@dainfos_remote:/home/ildar/diplom/insta/\{*.ipynb,*.py\} $TEMPFOLDER
zip -r $ZIPFILE $TEMPFOLDER
rm -rf $TEMPFOLDER