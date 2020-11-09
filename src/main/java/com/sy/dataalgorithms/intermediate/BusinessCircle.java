package com.sy.dataalgorithms.intermediate;

import com.sy.init.InitSpark;
import com.sy.util.PropertiesReader;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.ml.clustering.BisectingKMeans;
import org.apache.spark.ml.clustering.BisectingKMeansModel;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.ml.feature.MinMaxScaler;
import org.apache.spark.ml.feature.MinMaxScalerModel;
import org.apache.spark.ml.linalg.SQLDataTypes;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.catalyst.encoders.RowEncoder;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.util.Random;

/**
 * 基于基站定位数据的商圈分析
 *   移动通信网络会记录用户手机的相关信息，比如手机所处的基站区域编号，所处基站的时间等。根据这些数据可以进行商圈划分，目的是
 *   为了研究潜在的顾客的分布以制定适宜的商业对策。如：可划分商业区、住宅区以及工作区
 * @Author Shi Yan
 * @Date 2020/11/9 21:43
 */
public class BusinessCircle {
    public static void main(String[] args) {
        SparkSession sparkSession = InitSpark.getSparkSession();
        sparkSession.sparkContext().setLogLevel("ERROR");
        businessCircleStatistics(sparkSession);
        InitSpark.closeSparkSession();
    }

    public static void businessCircleStatistics(SparkSession session) {
        /**
         * 原始数据集没有找到，找到的数据是被统计，转换，过滤，过的数据：
         * 这四个特征的统计方法是：
         *      对于某个基站(观测窗口为L天，基站N个，用户M个;
         *      某个用户i在j天在某个基站的 =》 工作日上班时间停留时间为weekday，凌晨停留时间为night，周末停留时间weekend，是否停留为stay【1：停留；0：无停留】)：
         *
         *
         *
         *      工作日上班时间人均停留时间=
         *      凌晨人均停留时间=
         *      凌晨人均停留时间=
         *      凌晨人均停留时间=
         *
         *      基站编号,工作日上班时间人均停留时间,凌晨人均停留时间,凌晨人均停留时间,凌晨人均停留时间
         *      36902,  78,                     521,           602,           2863
         *      36903,  144,                    600,           521,           2245
         *      36904,  95,                     457,           468,           1283
         *      36905,  69,                     596,           695,           1054
         *
         */
        String path = PropertiesReader.get("business_circle_csv");
        /**
         * +--------+--------------------------+----------------+----------------+----------+
         * |基站编号 |工作日上班时间人均停留时间   |凌晨人均停留时间  |周末人均停留时间  |日均人流量|
         * +--------+--------------------------+----------------+----------------+----------+
         * |   36902|                        78|             521|             602|      2863|
         * |   36903|                       144|             600|             521|      2245|
         * |   36904|                        95|             457|             468|      1283|
         * |   36905|                        69|             596|             695|      1054|
         * |   36906|                       190|             527|             691|      2051|
         * +--------+--------------------------+----------------+----------------+----------+
         *  |-- 基站编号: integer (nullable = true)
         *  |-- 工作日上班时间人均停留时间: integer (nullable = true)
         *  |-- 凌晨人均停留时间: integer (nullable = true)
         *  |-- 周末人均停留时间: integer (nullable = true)
         *  |-- 日均人流量: integer (nullable = true)
         */
        Dataset<Row> dataset = session.read()
                .option("sep", ",")
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(path);

        /**
         * 转为特征向量
         * +---------+--------------------+
         * |stationID|            features|
         * +---------+--------------------+
         * |    36902|[78.0,521.0,602.0...|
         * |    36903|[144.0,600.0,521....|
         * |    36904|[95.0,457.0,468.0...|
         * |    36905|[69.0,596.0,695.0...|
         * |    36906|[190.0,527.0,691....|
         * +---------+--------------------+
         */
        dataset = dataset.map((MapFunction<Row,Row>) row -> {
            int stationID = row.getInt(0);
            double weekdayAvg = (double) row.getInt(1);
            double nightAvg = (double) row.getInt(2);
            double weekendAvg = (double) row.getInt(3);
            double stayAvg = (double) row.getInt(4);
            return RowFactory.create(stationID, Vectors.dense(new double[]{weekdayAvg, nightAvg, weekendAvg, stayAvg}));
        }, RowEncoder.apply(new StructType(new StructField[]{
                new StructField("stationID", DataTypes.IntegerType,false, Metadata.empty()),
                new StructField("features", SQLDataTypes.VectorType(), false, Metadata.empty())
        })));

        /**
         * 数据标准化
         */
        MinMaxScalerModel featureScaler = new MinMaxScaler()
                .setInputCol("features")
                .setOutputCol("scaledFeatures")
                .fit(dataset);
        /**
         *+---------+--------------------+--------------------+
         * |stationID|            features|      scaledFeatures|
         * +---------+--------------------+--------------------+
         * |    36902|[78.0,521.0,602.0...|[0.10386473429951...|
         * |    36903|[144.0,600.0,521....|[0.26328502415458...|
         * |    36904|[95.0,457.0,468.0...|[0.14492753623188...|
         * |    36905|[69.0,596.0,695.0...|[0.08212560386473...|
         * |    36906|[190.0,527.0,691....|[0.37439613526570...|
         * +---------+--------------------+--------------------+
         */
        Dataset<Row> scaledData = featureScaler.transform(dataset);

        /**
         * 轮廓系统确定簇数,可以看出分为3类最佳
         * k: 2 silhouette: 0.5063659448997802
         * k: 3 silhouette: 0.629019144457301
         * k: 4 silhouette: 0.32319167016337247
         * k: 5 silhouette: 0.30681655682008674
         * k: 6 silhouette: 0.39947777279975305
         * k: 7 silhouette: 0.31054738863541337
         * k: 8 silhouette: 0.3417574406084828
         * k: 9 silhouette: 0.30133745097199804
         * k: 10 silhouette: 0.12586962519806658
         */
        int k = selectOptimalK(scaledData, 10);

        //model
        BisectingKMeans bkm = new BisectingKMeans().setFeaturesCol("scaledFeatures")
                .setK(k) //簇数
                .setSeed(1);
        BisectingKMeansModel model = bkm.fit(scaledData);
        /**
         * 预测结果，后面根据聚类结果划分出不同的商圈，对3类数据中的4个特征进行分析，定义3类商圈的不定定位进行商业活动，具体可看《python数据分析与挖掘实战》一书中的第14章。
         * 根据3类数据的活动并异，可划分商业区、住宅区以及工作区
         * +---------+--------------------+--------------------+----------+
         * |stationID|            features|      scaledFeatures|prediction|
         * +---------+--------------------+--------------------+----------+
         * |    36902|[78.0,521.0,602.0...|[0.10386473429951...|         1|
         * |    36903|[144.0,600.0,521....|[0.26328502415458...|         1|
         * |    36904|[95.0,457.0,468.0...|[0.14492753623188...|         1|
         * |    36905|[69.0,596.0,695.0...|[0.08212560386473...|         1|
         * |    36906|[190.0,527.0,691....|[0.37439613526570...|         1|
         * +---------+--------------------+--------------------+----------+
         */

        Dataset<Row> predictions  = model.transform(scaledData);
        predictions.show(5);

        /**
         * 聚类中心
         * [0.13227119317053798,0.04483188044831879,0.19956941131772793,0.7100471677339725]
         * [0.1886016451233843,0.8021375921375923,0.7629929621455044,0.09096028267984407]
         * [0.8643640466871185,0.048015925680159235,0.12134333562021299,0.3287583779747489]
         */

        Vector[] centers = model.clusterCenters();
        for(Vector center : centers) {
            System.out.println(center);
        }

    }

    /**
     * 利用轮廓系数获取最佳聚类簇数
     *
     * @param scaledData
     * @param kClusters
     * @return
     */
    public static int selectOptimalK(Dataset<Row> scaledData, int kClusters) {
        Random random = new Random();
        int optimalK = 2;
        double maxSilhouette = -1;
        for(int k = 2; k <= kClusters; k++) {
            double silhouette = 0.0;
                //model
                BisectingKMeans bkm = new BisectingKMeans().setFeaturesCol("scaledFeatures")
                        .setK(k)
                        .setSeed(random.nextInt(100));
                //训练
                BisectingKMeansModel model = bkm.fit(scaledData);
                //预测
                Dataset<Row> predictions  = model.transform(scaledData);
                //评估，轮廓系数
                ClusteringEvaluator evaluator = new ClusteringEvaluator();
                double sc = evaluator.evaluate(predictions);
                silhouette = sc;

            if(silhouette > maxSilhouette) {
                maxSilhouette = silhouette;
                optimalK = k;
            }
            System.out.println("k: " + k + " silhouette: " + silhouette);
        }
        return optimalK;
    }

}
