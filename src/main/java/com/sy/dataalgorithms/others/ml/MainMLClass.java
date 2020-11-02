package com.sy.dataalgorithms.others.ml;

import com.sy.init.InitSpark;
import com.sy.util.PropertiesReader;
import org.apache.spark.sql.SparkSession;

/**
 * @Author Shi Yan
 * @Date 2020/8/28 21:37
 */
public class MainMLClass {
    public static void main(String[] args) {
        SparkSession session = InitSpark.getSparkSession();
        session.sparkContext().setLogLevel("ERROR");
        //classificationDemo(session);
        //regressionDemo(session);
        //clusteringDemo(session);
        //recoSystem(session);
        statisticsDemo(session);
        InitSpark.closeSparkSession();
    }

    /**
     * 分类demo
     * @param session
     */
    public static void classificationDemo(SparkSession session) {
        String classificationDataPath = PropertiesReader.get("others_ml_binary_classification_txt");
        ClassificationDemo.fmClassification(session, classificationDataPath);
    }

    /**
     * 回归demo
     * @param session
     */
    public static void regressionDemo(SparkSession session) {
        String regressionDataPath = PropertiesReader.get("others_ml_regression_txt");
        RegressionDemo.gbRegression(session, regressionDataPath);
    }

    /**
     * 聚类demo
     * @param session
     */
    public static void clusteringDemo(SparkSession session) {
        String clusterDataPath = PropertiesReader.get("others_ml_cluster_txt");
        ClusteringDemo.kmCluster(session, clusterDataPath);
    }

    /**
     * 推荐demo
     * @param session
     */
    public static void recoSystem(SparkSession session) {
        String resysDataPath = PropertiesReader.get("others_ml_resys_txt");
        RecoSysDemo.alsRecsys(session, resysDataPath);
    }

    /**
     * 统计demo
     * @param session
     */
    public static void statisticsDemo(SparkSession session) {
        StatisticsDemo.correlationDemo(session);
    }

}
