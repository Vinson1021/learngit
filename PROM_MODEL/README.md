## 一.项目结构

项目的整个结构如下:

```
root/
 |-- configs/
 |   |-- dev_job_demo_config.json
 |   |-- prod_job_demo_config.json
 |-- dependencies/
 |   |-- common
 |   |-- algorithm
 |   |-- platform
 |   |   |-- logging.py
 |   |   |-- spark.py
 |-- jobs/
 |   |-- job_demo.py
 |-- tests/
 |   |-- test_data/
 |   |-- | -- employees/
 |   |-- | -- employees_expect/
 |-- bin/
 |   |-- build_dependencies.sh
 |   |-- install_local_env.sh
 |-- dev/
 |   |-- run_local.sh
 |-- packages.zip
 |-- Pipfile
 |-- Pipfile.lock
 |-- README.md
```
各个文件夹的功能和使用说明如下：
* **configs**：存放json配置文件，配置文件在任务执行的时候用```--job-args config_json=<?>```指定，用于存放与环境相关的配置参数，方便测试或者上线调度的时候进行环境隔离；文件命令方式以```dev```、```test```或者```prod
```等开头，以```config
.json```结尾；
* **dependencies**: 计算任务执行的依赖python文件，需要执行bin文件夹中```build_dependencies.sh```进行打包，生成```packages.zip```文件；
* **jobs**: 计算任务各业务的主方法；
* **bin**: 可执行的脚本，用于配置环境或者打包packages.zip包；
* **dev**: 本地调试用脚本文件，用于创建测试数据或者执行local模式的spark任务；
* **tests**: 用于存放测试数据和单元测试的脚本；
* **packages.zip**: dependencies的打包文件，用提交spark任务的时候用```--py-files```;

## 二.任务提交方式
### 1.本机调试
#### 本机环境安装
###### (1)pipfile方式：
执行bin目录下的```install_local_env.sh```进行安装。
安装成功调用```pipenv shell```进入虚拟环境。```exit```退出。
###### (2)requirement方式：
```
pip3 install -r requirements.txt
```
#### 本机spark代码调试
###### (1)打印日志：
通过调用```start_spark()```返回log对象：
```python
    spark, log, config = start_spark(
        app_name='my_job_demo')

    # log that main spark job is starting
    log.warn('job_demo is up-and-running')

```
###### (2)demo本机执行：
```shell script
 sh dev/run_local.sh --main-py-file jobs/job_demo.py --job-args config_json=configs/prod_job_demo_config.json 
```

```shell script
sh dev/run_local.sh --main-py-file jobs/job_demo_arima.py --job-args config_json=configs/prod_job_demo_arima_config.json fc_dt=20200729
```
### 2.mtjupyter调试

替换dev目录下的run_cluster.sh脚本中的变量并执行，可在mtjupyter上提交spark任务。
```shell script
cd dev
sh run_cluster.sh
```
### 3.任务托管平台调度
托管平台配置参数：
```shell script
--archives viewfs://hadoop-meituan/user/hadoop-phx-jupyter-rd/projects/jupyter/archives/scipy-notebook-348aef3.zip#ARCHIVE
--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./ARCHIVE/notebook/bin/python
--conf spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON=./ARCHIVE/notebook/bin/python
--conf spark.yarn.maxAppAttempts=0 
--conf spark.task.maxFailures=3
--conf spark.network.timeout=240s
--py-files packages.zip
--files configs/prod_job_demo_arima_config.json
## 自定义参数
--job-args config_json=prod_job_demo_arima_config.json fc_dt=20200731
```
## 三.参数配置化
为了实现线上线下环境的物理隔离，输入输出表或者其他参数可以在configs文件中以字典的形式存储，启动任务的时候利用```--files```和```--job-args```中的key:```config_json```指定配置文件；
### 如何指定配置文件
Spark配置参数：```--files configs/prod_job_demo_arima_config.json``` 将配置文件上传至HDFS;
Job参数传参：```--job-args config_json=prod_job_demo_arima_config.json ```指定自定义key=value的参数，其中config_json用来指定配置文件名;
> 注意：集群环境需要指定```--files```参数，目的是为了在程序提交的时候将配置文件上传到各节点可以访问的集群，单机调试可以不用设置该参数，但是需要指定配置文件的相对路径，而不仅仅是文件名；

## 四.算法说明
### 1.劳力预测
### 2.运力预测
### 3.指标预测
### 4.销量预测

#### 4.1 数据增强模型特征工程
###### (1)job script
```job_da_model_feature.py```

###### (2)配置文件
开发环境：```dev_da_model_feature_config.json```

线上环境：```prod_da_model_feature_config.json```

###### (3)spark提交参数
```shell script
--archives viewfs://hadoop-meituan/user/hadoop-phx-jupyter-rd/projects/jupyter/archives/scipy-notebook-348aef3.zip#ARCHIVE
--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./ARCHIVE/notebook/bin/python
--conf spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON=./ARCHIVE/notebook/bin/python
--conf spark.yarn.executor.memoryOverhead=2G
--conf spark.yarn.maxAppAttempts=0 
--conf spark.task.maxFailures=3
--conf spark.network.timeout=240s
--py-files packages.zip
--files configs/prod_da_model_feature_config.json
# 自定义参数
--job-args 
config_json=prod_da_model_feature_config.json
```
###### (4)Hadoop Shell Command
xt托管平台需要添加如下两条操作：
```shell script
hive -e "use mart_caterb2b_forecast; alter table app_caterb2b_forecast_sale_data_augmentation_raw_feature_v2 add partition(dt=`date +%Y%m%d`)"
```

#### 4.2 用户感知价格特征工程
###### (1)job script
```job_sgp_feature.py```

###### (2)配置文件
开发环境：```dev_sgp_feature_config.json```

线上环境：```prod_sgp_feature_config.json```

###### (3)spark提交参数
```shell script
--archives viewfs://hadoop-meituan/user/hadoop-phx-jupyter-rd/projects/jupyter/archives/scipy-notebook-348aef3.zip#ARCHIVE
--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./ARCHIVE/notebook/bin/python
--conf spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON=./ARCHIVE/notebook/bin/python
--conf spark.yarn.executor.memoryOverhead=2G
--conf spark.yarn.maxAppAttempts=0 
--conf spark.task.maxFailures=3
--conf spark.network.timeout=240s
--py-files packages.zip
--files configs/prod_sgp_feature_config.json
# 自定义参数
--job-args 
config_json=prod_sgp_feature_config.json
```
###### (4)Hadoop Shell Command
xt托管平台需要添加如下两条操作：
```shell script
hive -e "use mart_caterb2b_forecast; alter table topic_sale_price_sgp_day add partition(dt=$now.delta(0).datekey)"
```

#### 4.3 RDC蔬菜品类预测

###### (1)job script
```job_rdc_vegetables.py```

###### (2)配置文件
开发环境：```dev_rdc_vegetables_config.json```

线上环境：```prod_rdc_vegetables_config.json```

###### (3)spark提交参数
```shell script
--archives viewfs://hadoop-meituan/user/hadoop-phx-jupyter-rd/projects/jupyter/archives/scipy-notebook-348aef3.zip#ARCHIVE
--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./ARCHIVE/notebook/bin/python
--conf spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON=./ARCHIVE/notebook/bin/python
--conf spark.yarn.executor.memoryOverhead=2G
--conf spark.yarn.maxAppAttempts=0 
--conf spark.task.maxFailures=3
--conf spark.network.timeout=240s
--py-files packages.zip
--files configs/prod_rdc_vegetables_config.json
# 自定义参数
--job-args 
config_json=prod_rdc_vegetables_config.json
```
###### (4)Hadoop Shell Command
xt托管平台需要添加如下两条操作：
```shell script
hive -e "use mart_caterb2b_forecast; alter table topic_sale_price_sgp_day add partition(dt=$now.delta(0).datekey)"
```


### 5.量价预测
###### (1)job script
```job_pricing_strategy_feature.py```

###### (2)配置文件
开发环境：```dev_pricing_stragegy_feature_config.json```
线上环境：```prod_pricing_stragegy_feature_config.json```

###### (3)spark提交参数
```shell script
--archives viewfs://hadoop-meituan/user/hadoop-phx-jupyter-rd/projects/jupyter/archives/scipy-notebook-348aef3.zip#ARCHIVE
--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./ARCHIVE/notebook/bin/python
--conf spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON=./ARCHIVE/notebook/bin/python
--conf spark.yarn.executor.memoryOverhead=2G
--conf spark.yarn.maxAppAttempts=0 
--conf spark.task.maxFailures=3
--conf spark.network.timeout=240s
--py-files packages.zip
--files configs/prod_pricing_stragegy_feature_config.json
# 自定义参数
--job-args 
config_json=prod_pricing_stragegy_feature_config.json
```
###### (4)Hadoop Shell Command
xt托管平台需要添加如下两条操作：
```shell script
hive -e "use mart_caterb2b_forecast; alter table app_forecast_sale_pricing_strategy_feature add partition(dt=$now.delta(-1).datekey);"
hive -e "use mart_caterb2b_forecast; alter table app_forecast_sale_pricing_strategy_feature add partition(dt=$now.delta(0).datekey);"
```

### 6.POP 预测
#### 6.1 pop_longtail_baseline模型

###### (1)job script
```job_pop_longtail_baseline.py```

###### (2)配置文件
开发环境：```dev_pop_longtail_baseline_config.json```

线上环境：```prod_pop_longtail_baseline_config.json```

###### (3)spark提交参数
```shell script
--archives viewfs://hadoop-meituan/user/hadoop-phx-jupyter-rd/projects/jupyter/archives/scipy-notebook-348aef3.zip#ARCHIVE
--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./ARCHIVE/notebook/bin/python
--conf spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON=./ARCHIVE/notebook/bin/python
--conf spark.yarn.executor.memoryOverhead=2G
--conf spark.yarn.maxAppAttempts=0 
--conf spark.task.maxFailures=3
--conf spark.network.timeout=240s
--py-files packages.zip
--files configs/prod_pop_longtail_baseline_config.json.json
# 自定义参数
--job-args 
config_json=prod_pop_longtail_baseline_config.json.json
```
###### (4)Hadoop Shell Command
xt托管平台需要添加如下两条操作：
```shell script
hive -e "use mart_caterb2b_forecast; alter table app_sale_3p_fc_basic_day_output add partition(fc_dt=$now.delta(0).datekey,model_name='pop_longtail_baseline')"
```