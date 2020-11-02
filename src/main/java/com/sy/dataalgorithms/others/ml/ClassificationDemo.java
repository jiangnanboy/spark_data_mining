package com.sy.dataalgorithms.others.ml;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.FMClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 * @Author Shi Yan
 * @Date 2020/8/28 21:47
 */
public class ClassificationDemo {

    /**
     * 分类
     * @param session
     */
    public static void fmClassification(SparkSession session, String filePath) {
        Dataset<Row> data = session
                .read()
                .format("libsvm")
                .load(filePath);
        //将label转为index
        StringIndexerModel labelIndexer = new StringIndexer()
                .setInputCol("label")
                .setOutputCol("indexedLabel")
                .setHandleInvalid("skip") //在转换多于原始数据label个数时跳过
                .fit(data);
        //对特征归一化到min-max
        MinMaxScalerModel featureScalar = new MinMaxScaler()
                .setInputCol("features")
                .setOutputCol("scaledFeatures")
                .fit(data);
        //数据分割为train:70%，test:30%
        Dataset<Row>[] splits = data.randomSplit(new double[] {0.7, 0.3});
        Dataset<Row> trainData = splits[0];
        Dataset<Row> testData = splits[1];
        //定义model
        FMClassifier fm = new FMClassifier()
                .setLabelCol(labelIndexer.getOutputCol())
                .setFeaturesCol(featureScalar.getOutputCol())
                .setMaxIter(10)
                .setStepSize(0.001);
        //将label转为原始label，用于预测展示真实的label
        IndexToString labelConverter = new IndexToString()
                .setInputCol("prediction") //默认预测的label
                .setOutputCol("predictedLabel") //转换为真实的label
                .setLabels(labelIndexer.labelsArray()[0]);
        //利用pipeline级联处理
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[] {labelIndexer, featureScalar, fm, labelConverter});
        //训练trainData
        PipelineModel model = pipeline.fit(trainData);
        //预测testData
        Dataset<Row> predictions = model.transform(testData);
        //展示真实label与预测label
        predictions.select(labelConverter.getOutputCol(), labelIndexer.getInputCol(), featureScalar.getInputCol()).show(5);
        //评估模型accuracy
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("indexedLabel")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");
        double accuracy = evaluator.evaluate(predictions);
        System.out.println("test accuracy = " + accuracy);
    }

}

