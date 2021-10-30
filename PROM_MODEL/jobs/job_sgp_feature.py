"""
job_sgp_feature.py
~~~~~~~~~~

销量预测-用户感知价格

"""
from datetime import datetime

from pyspark.rdd import RDD
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType, DoubleType

from dependencies.algorithm.feature.feature_sgp_generator import generate_sgp_feats
from dependencies.common import forecast_utils as fu
from dependencies.platform.spark import start_spark


#############################################
# main function
#############################################
def main():
    """Main script definition.

    :return: None
    """
    # start Spark application and get Spark session, logger and config
    spark, log, config = start_spark(
        app_name='sgp_feature_engineering')

    # log that main spark job is starting
    log.warn('sgp_feature_engineering is up-and-running')

    # execute spark pipeline
    raw_data_rdd, config = extract_data(spark, config)
    feats_rdd, config = transform_data(raw_data_rdd, config)
    write_data(spark, feats_rdd, config)

    # log the success and terminate Spark application
    log.warn('sgp_feature_engineering is finished')
    spark.stop()
    return None


#############################################
# extract data
#############################################
def extract_data(spark: SparkSession, config: dict) -> (RDD, dict):
    """
    extract data from hive

    Args:
        spark: sparkSession
        config: config dict

    Returns:
        raw_data_rdd: data rdd
        config: updated config dict
    """

    # 日期参数设定
    current_date = datetime.now().strftime("%Y%m%d")
    if 'forecast_date' in config:
        current_date = config['forecast_date']
    label_start_date = fu.get_date(current_date, -730, '%Y%m%d', '%Y%m%d')

    actual_sql = """
            select sku.cat1_id,
                   sku.cat1_name,
                   sku.cat2_id,
                   sku.cat2_name,
                   sku.cat3_id,
                   sku.cat3_name,
                   sku.brand_id,
                   sku.brand_name,
                   sale.sku_id,
                   sale.wh_id,
                   sale.bu_id,
                   sale.dt cur_date,
                   sale.able_sell,
                   sale.arranged_cnt,
                   sale.arranged_amt
              from (
                    select sku_id,
                           wh_id,
                           bu_id,
                           dt,
                           able_sell,
                           sum(arranged_cnt) arranged_cnt,
                           sum(arranged_amt) arranged_amt
                      from mart_caterb2b_forecast.app_caterb2b_forecast_input_sales_dt_wh_sku
                      where dt >=  {label_start_date} and dt < {forecast_date}
                      group by sku_id, wh_id, bu_id, dt, able_sell
                   ) sale
              inner join mart_caterb2b.dim_caterb2b_sku sku
                on sale.sku_id = sku.sku_id
            where sku.cat1_id in ({cat1_id_list})
        """.format(forecast_date=current_date, cat1_id_list=config['cat1_id_list'], label_start_date=label_start_date)
    # 得到原始数据集
    schema_cols = spark.sql(actual_sql).schema.names
    feats_rdd = spark.sql(actual_sql).rdd.map(lambda x: ((x[schema_cols.index('sku_id')], x[schema_cols.index('wh_id')]), x)).groupByKey().persist()

    new_configs = {
        'schema_cols': schema_cols,
        'current_date': current_date
    }
    config.update(new_configs)

    return feats_rdd, config


#############################################
# data processing
#############################################
def transform_data(raw_data_rdd: RDD, config: dict) -> (RDD, dict):
    """
       transform data

       Args:
           raw_data_rdd: raw data rdd
           config: config dict

       Returns:
           feats_rdd: features rdd
           config: updated config dict
       """
    sgp_col_name = 'sgp'
    price_col_name = 'sale_price'

    feats_rdd = raw_data_rdd.map(
        lambda x: (x[0], generate_sgp_feats(x[1], config['schema_cols'], 30, 3))
    ).filter(
        lambda x: x[1] is not None
    ).flatMapValues(lambda x: x).map(lambda x: list(x[1])).persist()
    config['rst_schema_cols'] = config['schema_cols'] + [price_col_name, sgp_col_name]
    return feats_rdd, config


#############################################
# output data
#############################################
def write_data(spark: SparkSession, feats_rdd: RDD, config: dict) -> None:
    """
    write features to hive

    Args:
        spark: sparkSession
        feats_rdd: features rdd
        config: config dict

    Returns:
        None
    """
    schema = StructType([
        StructField('cat1_id', LongType(), True),
        StructField('cat1_name', StringType(), True),
        StructField('cat2_id', LongType(), True),
        StructField('cat2_name', StringType(), True),
        StructField('cat3_id', LongType(), True),
        StructField('cat3_name', StringType(), True),
        StructField('brand_id', LongType(), True),
        StructField('brand_name', StringType(), True),
        StructField('sku_id', LongType(), True),
        StructField('wh_id', LongType(), True),
        StructField('bu_id', LongType(), True),
        StructField('cur_date', StringType(), True),
        StructField('able_sell', IntegerType(), True),
        StructField('arranged_cnt', LongType(), True),
        StructField('arranged_amt', DoubleType(), True),
        StructField('sale_price', DoubleType(), True),
        StructField('sgp', DoubleType(), True),
    ])
    res_df = spark.createDataFrame(feats_rdd, schema).persist()
    res_df.select(config['rst_schema_cols']).repartition(100).write.mode('overwrite').orc(config['hive_path'].format(partition_dt=config['current_date']))
    return None


#############################################
# entry point for PySpark application
#############################################
if __name__ == '__main__':
    main()
