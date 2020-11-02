package com.sy.dataalgorithms.basics;

import com.sy.init.InitSpark;
import com.sy.util.PropertiesReader;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.sql.SparkSession;
import scala.Serializable;
import scala.Tuple2;

import java.util.*;

/**
 * topn排序
 * @Author Shi Yan
 * @Date 2020/10/27 9:31
 */
public class TopN {
    public static void main(String[] args) {
        SparkSession sparkSession = InitSpark.getSparkSession();
        sparkSession.sparkContext().setLogLevel("ERROR");
        Broadcast<Integer> broadcastN = JavaSparkContext.fromSparkContext(sparkSession.sparkContext()).broadcast(10);
        Broadcast<String> broadcastDirection = JavaSparkContext.fromSparkContext(sparkSession.sparkContext()).broadcast("top");
        nonUniqueKeyTopN(sparkSession, broadcastN);
        InitSpark.closeSparkSession();
    }

    /**
     * key唯一
     * @param session
     * @param broadcastN
     * @param broadcastDirection
     */
    public static void uniqueKeyTopN(SparkSession session, Broadcast<Integer> broadcastN, Broadcast<String> broadcastDirection) {
        String path = PropertiesReader.get("basic_topndata_txt");
        JavaRDD<String> javaRDD = session.read().textFile(path).toJavaRDD().repartition(10);
        JavaPairRDD<String,Integer> pairRDD = javaRDD.mapToPair(s -> {
            String[] tokens = s.split(",");
            return new Tuple2<>(tokens[0], Integer.valueOf(tokens[1]));
        });

        //每个分区创建topn
        JavaRDD<SortedMap<Integer, String>> partitions = pairRDD.mapPartitions(iter -> {
            SortedMap<Integer, String> topN = new TreeMap<>();
            while (iter.hasNext()) {
                Tuple2<String, Integer> tuple = iter.next();
                topN.put(tuple._2, tuple._1);
                if(topN.size() > broadcastN.value()) {
                    if("top".equals(broadcastDirection.getValue())) {
                        topN.remove(topN.firstKey());
                    } else {
                        topN.remove(topN.lastKey());
                    }
                }
            }
            return Collections.singletonList(topN).iterator();
        });

        //对所有分区的topn进行总体的topn
        SortedMap<Integer, String> totalTopN = partitions.reduce((m1,m2) -> {
            SortedMap<Integer, String> top = new TreeMap<>();
            for(Map.Entry<Integer, String> entry : m1.entrySet()) {
                top.put(entry.getKey(), entry.getValue());
                if(top.size() > broadcastN.value()) {
                    if("top".equals(broadcastDirection.getValue())) {
                        top.remove(top.firstKey());
                    } else {
                        top.remove(top.lastKey());
                    }
                }
            }
            for(Map.Entry<Integer, String> entry : m2.entrySet()) {
                top.put(entry.getKey(), entry.getValue());
                if(top.size() > broadcastN.value()) {
                    if("top".equals(broadcastDirection.getValue())) {
                        top.remove(top.firstKey());
                    } else {
                        top.remove(top.lastKey());
                    }
                }
            }
            return top;
        });

        totalTopN.forEach((key, value) -> System.out.println(key + " -> " + value));
    }

    /**
     * key不唯一
     * @param session
     * @param broadcastN
     * @param broadcastDirection
     */
    public static void nonUniqueKeyTopN(SparkSession session, Broadcast<Integer> broadcastN, Broadcast<String> broadcastDirection) {
        String path = PropertiesReader.get("topndata_txt");
        JavaRDD<String> dataset = session.read().textFile(path).toJavaRDD().coalesce(10);
        JavaPairRDD<String,Integer> pairRDD = dataset.mapToPair(s -> {
            String[] tokens = s.split(",");
            return new Tuple2<>(tokens[0], Integer.valueOf(tokens[1]));
        });
        JavaPairRDD<String, Integer> uniqueKeys = pairRDD.reduceByKey((i1,i2) ->i1 + i2);
        //每个分区创建topn
        JavaRDD<SortedMap<Integer, String>> partitions = uniqueKeys.mapPartitions(iter -> {
            SortedMap<Integer, String> topN = new TreeMap<>();
            while (iter.hasNext()) {
                Tuple2<String, Integer> tuple = iter.next();
                topN.put(tuple._2, tuple._1);
                if(topN.size() > broadcastN.value()) {
                    if("top".equals(broadcastDirection.getValue())) {
                        topN.remove(topN.firstKey());
                    } else {
                        topN.remove(topN.lastKey());
                    }
                }
            }
            return Collections.singletonList(topN).iterator();
        });

        //对所有分区的topn进行总体的topn
        SortedMap<Integer, String> totalTopN = partitions.reduce((m1,m2) -> {
            SortedMap<Integer, String> top = new TreeMap<>();
            for(Map.Entry<Integer, String> entry : m1.entrySet()) {
                top.put(entry.getKey(), entry.getValue());
                if(top.size() > broadcastN.value()) {
                    if("top".equals(broadcastDirection.getValue())) {
                        top.remove(top.firstKey());
                    } else {
                        top.remove(top.lastKey());
                    }
                }
            }
            for(Map.Entry<Integer, String> entry : m2.entrySet()) {
                top.put(entry.getKey(), entry.getValue());
                if(top.size() > broadcastN.value()) {
                    if("top".equals(broadcastDirection.getValue())) {
                        top.remove(top.firstKey());
                    } else {
                        top.remove(top.lastKey());
                    }
                }
            }
            return top;
        });
        totalTopN.forEach((key, value) -> System.out.println(key + " -> " + value));
    }

    /**
     * key不唯一，利用函数takeOrdered
     * @param session
     * @param broadcastN
     */
    public static void nonUniqueKeyTopN(SparkSession session, Broadcast<Integer> broadcastN) {
        String path = PropertiesReader.get("topndata_txt");
        JavaRDD<String> dataset = session.read().textFile(path).toJavaRDD().coalesce(10);
        JavaPairRDD<String,Integer> pairRDD = dataset.mapToPair(s -> {
            String[] tokens = s.split(",");
            return new Tuple2<>(tokens[0], Integer.valueOf(tokens[1]));
        });
        JavaPairRDD<String, Integer> uniqueKeys = pairRDD.reduceByKey((i1,i2) ->i1 + i2);
        List<Tuple2<String, Integer>> topNResult = uniqueKeys.takeOrdered(broadcastN.getValue(), MyTupleComparatorDescending.INSTANCE);//takeOrdered <=> top
        for(Tuple2<String, Integer> entry:topNResult) {
            System.out.println(entry._2 + "->" + entry._1);
        }
    }
}

//升
class MyTupleComparatorAscending implements Comparator<Tuple2<String, Integer>>, Serializable {
    static MyTupleComparatorAscending INSTANCE = new MyTupleComparatorAscending();
    @Override
    public int compare(Tuple2<String, Integer> o1, Tuple2<String, Integer> o2) {
        return o1._2.compareTo(o2._2);
    }
}

//降
class MyTupleComparatorDescending implements Comparator<Tuple2<String, Integer>>, Serializable {
    static MyTupleComparatorDescending INSTANCE = new MyTupleComparatorDescending();
    @Override
    public int compare(Tuple2<String, Integer> o1, Tuple2<String, Integer> o2) {
        return -o1._2.compareTo(o2._2);
    }
}
