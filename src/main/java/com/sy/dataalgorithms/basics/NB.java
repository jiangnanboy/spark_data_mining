package com.sy.dataalgorithms.basics;

import com.sy.init.InitSpark;
import com.sy.util.PropertiesReader;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

import javax.print.attribute.standard.PresentationDirection;
import java.util.*;

/**
 * @Author Shi Yan
 * @Date 2020/11/2 20:40
 */
public class NB {
    public static void main(String[] args) {
        SparkSession sparkSession = InitSpark.getSparkSession();
        sparkSession.sparkContext().setLogLevel("ERROR");
        buildNBClassifier(sparkSession);
        InitSpark.closeSparkSession();
    }

    /**
     * 建立nb分类模型
     * @param session
     */
    public static void buildNBClassifier(SparkSession session) {
        String path = PropertiesReader.get("basic_nb_txt");
        JavaRDD<String> trainJavaRDD = session.read().textFile(path).toJavaRDD().repartition(10);
        long trainingDataSize = trainJavaRDD.count();//训练集大小

        //转为((属性1,类别1),1),((属性2,类别1),1),...,(("class",类别1),1)
        JavaPairRDD<Tuple2<String, String>, Integer> pairs = trainJavaRDD.flatMapToPair(s -> {
            List<Tuple2<Tuple2<String, String>, Integer>> list = new ArrayList<>();
            String[] tokens = s.split(",");
            int calssificationIndex = tokens.length - 1;
            String classification = tokens[calssificationIndex]; //类别
            for(int i = 0; i < calssificationIndex; i ++) {
                Tuple2<String, String> k = new Tuple2<>(tokens[i], classification);
                list.add(new Tuple2<>(k, 1));
            }
            Tuple2<String, String> k = new Tuple2<>("CLASS", classification);
            list.add(new Tuple2<>(k, 1));
            return list.iterator();
        });

        //按key对value加和，
        JavaPairRDD<Tuple2<String, String>, Integer> counts = pairs.reduceByKey((i1,i2) -> i1 + i2);

        Map<Tuple2<String, String>, Integer> countsMap = counts.collectAsMap();

        // 概率表包含在训练数据中统计的类别概率以及属性概率（连续型数值属性可使用高斯分布计算概率，离散型直接用统计频率）,可直接保存概率表ptMap以及存储的类别名classificationList，作为训练后的模型，供后期加载预测使用
        Map<Tuple2<String, String>, Double> ptMap = new HashMap<>();
        List<String> classificationList = new ArrayList<>();//存储类别名

        for(Map.Entry<Tuple2<String, String>, Integer> entry:countsMap.entrySet()) {
            Tuple2<String, String> k = entry.getKey();
            String classification = k._2;
            if(k._1.equals("CLASS")) {
                ptMap.put(k, entry.getValue() / (double)trainingDataSize);// 类别的概率 = 类别的数量 / 训练集大小
                classificationList.add(k._2);
            } else {
                Tuple2<String, String> k2 = new Tuple2<>("CLASS", classification);
                Integer count =countsMap.get(k2); // 类别为classification的数量
                if(null == count) {
                    ptMap.put(k, 0.0);
                } else {
                    ptMap.put(k, entry.getValue() / (double)count.intValue()); // 属性的概率 = 类别的数量 / 此类别包含该属性的个数
                }
            }
        }

        //打印概率
        System.out.println(ptMap);

    }

    /**
     * 预测新数据
     * @param session
     */
    public static void predictClassification(SparkSession session) {
        //test data，格式 =》 属性1，属性2，...
        String testPath = PropertiesReader.get("test_txt");
        JavaRDD<String> testJavaRDD = session.read().textFile(testPath).toJavaRDD().repartition(10);

        //load model
        String modelPath = PropertiesReader.get("model_txt");
        JavaRDD<String> modelJavaRDD = session.read().textFile(modelPath).toJavaRDD().repartition(10);

        // model 格式 => class,No 概率值... /n 属性名,类名 概率值
        JavaPairRDD<Tuple2<String, String>, Double> classifierRDD = modelJavaRDD.mapToPair(s -> {
            String[] tokens1 = s.split("\\s+");
            String[] tokens2 = tokens1[0].split(",");
            Tuple2<String, String> t2 = new Tuple2<>(tokens2[0], tokens2[1]);
            Double value = Double.valueOf(tokens1[1]);
            return new Tuple2<>(t2, value);
        });

        Map<Tuple2<String, String>, Double> classifierMap = classifierRDD.collectAsMap();
        Broadcast<Map<Tuple2<String, String>, Double>> mapBroadcast = JavaSparkContext.fromSparkContext(session.sparkContext()).broadcast(classifierMap);

        //加载类别名
        String classPath = PropertiesReader.get("class_txt");
        JavaRDD<String> classJavaRDD = session.read().textFile(classPath).toJavaRDD();
        List<String> classList = classJavaRDD.collect();

        Broadcast<List<String>> listBroadcast = JavaSparkContext.fromSparkContext(session.sparkContext()).broadcast(classList);

        JavaPairRDD<String, String> classified = testJavaRDD.mapToPair(s -> {
            Map<Tuple2<String, String>, Double> CLASSFIER = mapBroadcast.getValue();
            List<String> CLASSES = listBroadcast.getValue();

            String[] attributes = s.split(",");
            String selectedClass = null;
            double maxPosterior = 0.0;
            for(String aCLass : CLASSES) {
                double posterior = CLASSFIER.get(new Tuple2<>("CLASS", aCLass));//类别的概率
                for(int i = 0; i < attributes.length; i ++) {
                    Double prob = CLASSFIER.get(new Tuple2<>(attributes[i], aCLass)); //属性的概率
                    if(null == prob) {
                        posterior = 0.0;
                        break;
                    } else {
                        posterior *= prob.doubleValue();
                    }
                }
                if(null == selectedClass) {
                    selectedClass = aCLass;
                    maxPosterior = posterior;
                } else {
                    if(posterior > maxPosterior) {
                        selectedClass = aCLass;
                        maxPosterior = posterior;
                    }
                }
            }
            return new Tuple2<>(s, selectedClass);
        });

        //打印
        classified.collectAsMap().forEach((key, value) -> System.out.println(key + ","+ value));
    }

}
