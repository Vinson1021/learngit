#!/usr/bash

## get script file base dir
if [[ $(echo $0 | awk '/^\//') == $0 ]]; then
    BASEDIR=$(dirname $0)
else
    BASEDIR=$PWD/$(dirname $0)
fi
echo "Base dir: $BASEDIR"

SPARK_APP_NAME=data-forecast-spark-application-demo
SPARK_APP_OWNER='YOUR_MIS_NAME'
QUEUE=root.zw02.hadoop-caterb2b.etl
CONFIG_FILE=prod_job_demo_config.json
MAIN_PY_SCRIPT=job_demo.py

spark-submit \
--queue $QUEUE \
--executor-cores 1 \
--executor-memory 6G \
--deploy-mode cluster \
--master yarn \
--driver-memory 10G \
--conf spark.yarn.executor.memoryOverhead=4G \
--archives viewfs://hadoop-meituan/user/hadoop-caterb2b-forecast/libs/pyspark/scipy-notebook-348aef3.zip#ARCHIVE \
--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./ARCHIVE/notebook/bin/python \
--conf spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON=./ARCHIVE/notebook/bin/python \
--conf spark.task.cpus=1 \
--conf spark.yarn.maxAppAttempts=1 \
--conf spark.task.maxFailures=10 \
--conf spark.network.timeout=240s \
--conf spark.dynamicAllocation.enabled=true \
--conf spark.shuffle.service.enabled=true \
--conf spark.dynamicAllocation.maxExecutors=80 \
--conf spark.yarn.am.extraJavaOptions="-DappIdentify=mtmsp_None -Dport=AppMaster " \
--conf spark.driver.extraJavaOptions="-DappIdentify=mtmsp_None -Dport=Driver -XX:PermSize=128M -XX:MaxPermSize=256M " \
--conf spark.executor.extraJavaOptions="-DappIdentify=mtmsp_None -Dport=Executor " \
--name  $SPARK_APP_NAME \
--conf spark.job.owner=$SPARK_APP_OWNER \
--conf spark.yarn.job.priority=1 \
--conf spark.default.parallelism=1000 \
--conf spark.shuffle.memoryFraction=0.1 \
--conf spark.shuffle.spill.compress=true \
--conf spark.shuffle.compress=true \
--py-files $BASEDIR/../packages.zip \
--files $BASEDIR/hive-site.xml,$BASEDIR/../configs/$CONFIG_FILE \
$BASEDIR/../jobs/$MAIN_PY_SCRIPT \
--job-args config_json=$CONFIG_FILE $1
