#!/usr/bin/env bash

## get script file base dir
if [[ $(echo $0 | awk '/^\//') == $0 ]]; then
    BASEDIR=$(dirname $0)
else
    BASEDIR=$PWD/$(dirname $0)
fi
echo "Base dir: $BASEDIR"

cd $BASEDIR/../

zip -9rv $BASEDIR/../packages.zip dependencies -x dependencies/__pycache__/\*
  
