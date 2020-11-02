package com.sy.init;

import org.apache.spark.sql.SparkSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @Author Shi Yan
 * @Date 2020/8/6 22:52
 */
public class InitSpark {

    private static final Logger LOGGER = LoggerFactory.getLogger(InitSpark.class);

    private static SparkSession sparkSession = null;

    public static SparkSession getSparkSession() {
        LOGGER.info("Init and get sparksession!");
        sparkSession = SparkSession.builder().appName("Demo")
                .master("local[*]")
                .config("spark.driver.memory", "8g")
                .config("spark.executor.memory", "8g")
                .config("spark.executor.cores", 10)
                .config("spark.network.timeout", "1000")
                .config("spark.sql.broadcastTimeout", "2000")
                .config("spark.executor.heartbeatInterval", "2000")
                .getOrCreate();
        return sparkSession;
    }

    public static void closeSparkSession() {
        LOGGER.info("close sparksession!");
        if(null != sparkSession) {
            sparkSession.close();
        }
    }

}
