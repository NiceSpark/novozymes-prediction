#!/bin/bash


ZIP_NAME=$(ls *.zip)
unzip $ZIP_NAME
DIR_NAME=${ZIP_NAME%.zip}
rm $ZIP_NAME