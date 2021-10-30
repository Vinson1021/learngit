#!/usr/bin/env bash

## get script file base dir
if [[ $(echo $0 | awk '/^\//') == $0 ]]; then
    BASEDIR=$(dirname $0)
else
    BASEDIR=$PWD/$(dirname $0)
fi
echo "Base dir: $BASEDIR"

## output error message and exit with code.
error() {
  local msg=$1
  local exit_code=$2

  echo "Error: $msg" >&2

  if [ -n "$exit_code" ] ; then
    exit $exit_code
  fi
}


## main function of spark submit. 
main() {
args=""
CONFIG=""
PY_FILES=packages.zip
JARS=""

  while ([ -n "$*" ] && [ "$1" != "--job-args" ] ) ; do
    arg=$1
    shift
    case "$arg" in
      --py-files|-p)
        [ -n "$1" ] || error "Option --py-files requires an argument" 1
        PY_FILES=$1
        shift
        ;;
      --main-py-file|-m)
        [ -n "$1" ] || error "Option --main-py-file requires an argument" 1
        MAIN_PY_FILE=$1
        shift
        ;;
       --jars|-j)
        [ -n "$1" ] || error "Option --jars requires an argument" 1
        JARS=$1
        shift
        ;;
    esac
  done

  CONFIG=$*

  echo $PY_FILES
  echo $CONFIG_FILE
  echo $CONFIG

  if [[ -z $MAIN_PY_FILE ]]; then
    error "params :'--main-py-file' are all required ！" 1
  else
    if [[ $JARS == "" ]];then
      JARS_CONF=""
    else
      JARS_CONF="--jars $BASEDIR/../libs/$JARS"
    fi

    spark-submit \
    --master local[*] \
    --py-files $BASEDIR/../$PY_FILES \
    $PARAMS_FILES \
    $JARS_CONF \
    $BASEDIR/../$MAIN_PY_FILE \
    $CONFIG
  fi
}


main $*

