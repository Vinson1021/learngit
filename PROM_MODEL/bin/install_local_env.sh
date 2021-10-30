#!/usr/bin/env bash

## get script file base dir
if [[ $(echo $0 | awk '/^\//') == $0 ]]; then
    BASEDIR=$(dirname $0)
else
    BASEDIR=$PWD/$(dirname $0)
fi
echo "Base dir: $BASEDIR"


## pipenv install function
## install pipenv development environment and production environment 
install_pipenv(){
  echo 'INFO -- Start to install dev environment --> running `pipenv install --dev `'
    pipenv install --dev

    if [[ $? -eq 0 ]]
    then 
        echo 'INFO -- Start to install prod environment --> running `pipenv install `'
        pipenv install
        
        if [[ $? -eq 0 ]]
        then
            echo 'INFO -- init environment SUCCESS!'
            exit 0
        else
            echo 'ERROR -- init environment FAILED!'
            exit 1
        fi
    else 
        echo 'ERROR -- init environment FAILED!'
        exit 1
    fi

}


## main script
if [ -x "$(which pipenv)" ]; then
    install_pipenv
else
    echo 'INFO -- pipenv is not installed --> running `pip3 install pipenv`'
    pip3 install pipenv

    if [[ $?  -eq 0 ]]; then 
        install_pipenv
    else 
        echo 'ERROR -- init environment FAILED!'
        exit 1
    fi
fi

