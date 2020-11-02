package com.sy.dataalgorithms.others.ml;

import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 * @Author Shi Yan
 * @Date 2020/8/28 21:47
 */
public class ClusteringDemo {

    /**
     * 聚类
     * @param session
     * @param filePath
     */
    public static void kmCluster(SparkSession session, String filePath) {
        Dataset<Row> data = session.read()
                .format("libsvm")
                .load(filePath);
        //model
        KMeans kMeans = new KMeans()
                .setK(2) //簇数
                .setMaxIter(4);//迭代次数
        //训练
        KMeansModel model = kMeans.fit(data);
        //预测
        Dataset<Row> predictions  = model.transform(data);
        //评估，轮廓系数
        ClusteringEvaluator evaluator = new ClusteringEvaluator();
        double sc = evaluator.evaluate(predictions);
        System.out.println("evaluator sc : " + sc);
        //显示聚类中心坐标
        Vector[] centers = model.clusterCenters();
        for(Vector center : centers) {
            System.out.println(center);
        }
    }

}
