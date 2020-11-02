package com.sy.dataalgorithms.others.ml;

import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorIndexer;
import org.apache.spark.ml.feature.VectorIndexerModel;
import org.apache.spark.ml.regression.GBTRegressor;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 * @Author Shi Yan
 * @Date 2020/8/28 22:48
 */
public class RegressionDemo {

    /**
     * 回归
     * @param session
     * @param filePath
     */
    public static void gbRegression(SparkSession session, String filePath) {
        Dataset<Row> data = session.read()
                .format("libsvm")
                .load(filePath);
        //识别类别型特征，并索引
        VectorIndexerModel featureIndexer = new VectorIndexer()
                .setInputCol("features")
                .setOutputCol("indexedFeatures")
                .setMaxCategories(5)
                .fit(data);
        //分为训练和测试集
        Dataset<Row>[] splits = data.randomSplit(new double[] {0.8, 0.2});
        Dataset<Row> trainData = splits[0];
        Dataset<Row> testData = splits[1];
        //定义model
        GBTRegressor gbt = new GBTRegressor()
                .setLabelCol("label")
                .setFeaturesCol(featureIndexer.getOutputCol())
                .setMaxIter(10)
                .setMaxDepth(6)
                .setStepSize(0.1);
        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{featureIndexer, gbt});
        //训练
        PipelineModel model = pipeline.fit(trainData);
        //测试
        Dataset<Row> predictions = model.transform(testData);
        predictions.select("prediction", "label", "features").show(8);
        //评估
        RegressionEvaluator evaluator = new RegressionEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("rmse");
        double rmse = evaluator.evaluate(predictions);
        System.out.println("rmse on test data : " + rmse);
    }
}
