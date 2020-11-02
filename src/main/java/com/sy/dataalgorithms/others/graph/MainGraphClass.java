package com.sy.dataalgorithms.others.graph;

import com.sy.init.InitSpark;
import com.sy.util.PropertiesReader;
import org.apache.spark.sql.SparkSession;

/**
 * @Author Shi Yan
 * @Date 2020/8/28 20:42
 */
public class MainGraphClass {
    public static void main(String[] args) {
        SparkSession session = InitSpark.getSparkSession();
        session.sparkContext().setLogLevel("ERROR");
        prDemo(session);
        InitSpark.closeSparkSession();
    }

    /**
     * pagerank
     * @param session
     */
    public static void prDemo(SparkSession session) {
        String filePath = PropertiesReader.get("others_graph_pagerank_txt");
        PRDemo.pageRank(session, filePath);
    }

}
