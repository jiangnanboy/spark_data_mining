package com.sy.dataalgorithms.intermediate;

import com.sy.init.InitSpark;
import com.sy.util.PropertiesReader;
import org.apache.commons.lang3.StringUtils;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.sql.SparkSession;

import com.google.common.base.Splitter;
import scala.Tuple2;

import java.util.*;

/**
 * @Author Shi Yan
 * @Date 2020/11/2 9:46
 */
public class KNN {
    public static void main(String[] args) {
        SparkSession sparkSession = InitSpark.getSparkSession();
        sparkSession.sparkContext().setLogLevel("ERROR");
        Broadcast<Integer> broadcastK = JavaSparkContext.fromSparkContext(sparkSession.sparkContext()).broadcast(3);
        Broadcast<Integer> broadcastD = JavaSparkContext.fromSparkContext(sparkSession.sparkContext()).broadcast(2);
        knnCal(sparkSession, broadcastK, broadcastD);
        InitSpark.closeSparkSession();
    }

    public static void knnCal(SparkSession session, Broadcast<Integer> broadcastK, Broadcast<Integer> broadcastD) {
        String sPath = PropertiesReader.get("intermediate_knndata_s_txt");
        JavaRDD<String> sJavaRDD = session.read().textFile(sPath).toJavaRDD().repartition(10);

        String rPath = PropertiesReader.get("intermediate_knndata_r_txt");
        JavaRDD<String> rJavaRDD = session.read().textFile(rPath).toJavaRDD().repartition(10);

        JavaPairRDD<String, String> cartJavaRDD = rJavaRDD.cartesian(sJavaRDD);//笛卡尔集

        //距离计算 (rID,(sim1,class1)),...
        JavaPairRDD<String, Tuple2<Double, String>> knnJavaPairRDD = cartJavaRDD.mapToPair(cartRecord -> {
            //测试数据r
            String rReocrd = cartRecord._1;
            String[] rTokens = rReocrd.split(";");
            String rRecordID = rTokens[0];
            String r = rTokens[1];

            //训练数据s
            String sReocrd = cartRecord._2;
            String[] sTokens = sReocrd.split(";");
            String sClassificationID = sTokens[1];
            String s = sTokens[2];

            Integer dim = broadcastD.getValue();
            //r与s的距离
            double sim = calDistance(r, s, dim);

            Tuple2<Double, String> tuple2 = new Tuple2<>(sim, sClassificationID);
            return new Tuple2<>(rRecordID, tuple2);
        });

        //按训练数据id分组 =》 {r1,{(sim1,class1),(sim2, class2),...}}
        JavaPairRDD<String, Iterable<Tuple2<Double, String>>> knnGrouped = knnJavaPairRDD.groupByKey();

        JavaPairRDD<String, String> knnOutput = knnGrouped.mapValues(neighbors -> {
            Integer k = broadcastK.getValue();
            SortedMap<Double, String> nearestK = findNearestK(neighbors, k);
            Map<String, Integer> majority = buildClassificationCount(nearestK);
            String classificaiton = classifyByMajority(majority);
            return classificaiton;
        });

        // 打印
        List<Tuple2<String, String>> list = knnOutput.collect();
        list.stream().forEach(System.out::println);

    }

    static List<Double> splitOnToListOfDouble(String str, String delimiter) {
        Splitter splitter = Splitter.on(delimiter).trimResults();
        Iterable<String> tokens = splitter.split(str);
        if(null == tokens) {
            return null;
        }
        List<Double> list = new ArrayList<>();
        for(String token:tokens) {
            list.add(Double.valueOf(token));
        }
        return list;
    }

    static double calDistance(String r, String s, int dim) {
        List<Double> rVec = splitOnToListOfDouble(r, ",");
        List<Double> sVec = splitOnToListOfDouble(s, ",");
        if((dim != rVec.size()) && (dim != sVec.size())) {
            return Double.NaN;
        }
        double sim = 0.0;
        for(int i = 0; i < dim; i++) {
            double diff = rVec.get(i) - sVec.get(i);
            sim += diff * diff;
        }
        return Math.sqrt(sim);
    }

    static SortedMap<Double, String> findNearestK(Iterable<Tuple2<Double, String>> neighbors, int k) {
        SortedMap<Double, String> nearestK = new TreeMap<>();
        for(Tuple2<Double, String> neighbor:neighbors) {
            Double dis = neighbor._1;
            String classificationID = neighbor._2;
            nearestK.put(dis, classificationID);
            if(nearestK.size() > k) {
                nearestK.remove(nearestK.lastKey());
            }
        }
        return nearestK;
    }

    static Map<String, Integer> buildClassificationCount(Map<Double, String> nearestK) {
        Map<String, Integer> majority = new HashMap<>();
        for(Map.Entry<Double, String> entry:nearestK.entrySet()) {
            String id = entry.getValue();
            Integer count = majority.get(id);
            if(null == count) {
                majority.put(id, 1);
            } else {
                majority.put(id, count + 1);
            }
        }
        return majority;
    }

    static String classifyByMajority(Map<String, Integer> majority) {
        int votes = 0;
        String selectedClassification = null;
        for(Map.Entry<String, Integer> entry:majority.entrySet()) {
            if(null == selectedClassification) {
                selectedClassification = entry.getKey();
                votes = entry.getValue();
            } else {
                int count = entry.getValue();
                if(count > votes) {
                    selectedClassification = entry.getKey();
                    votes = count;
                }
            }
        }
        return selectedClassification;
    }

}

