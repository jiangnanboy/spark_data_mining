package com.sy.dataalgorithms.basics;

import com.sy.init.InitSpark;
import com.sy.util.PropertiesReader;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.Optional;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

import java.util.*;

/**
 * 左外连接
 * @Author Shi Yan
 * @Date 2020/10/27 13:43
 */
public class LeftOuterJoin {
    public static void main(String[] args) {
        SparkSession sparkSession = InitSpark.getSparkSession();
        sparkSession.sparkContext().setLogLevel("ERROR");
        leftOutJoin2(sparkSession);
        InitSpark.closeSparkSession();
    }

    public static void leftOutJoin(SparkSession session) {
        //user rdd
        String userPath = PropertiesReader.get("basic_users_txt");
        JavaRDD<String> userJavaRDD = session.read().textFile(userPath).toJavaRDD();
        JavaPairRDD<String, Tuple2<String, String>> usersRDD = userJavaRDD.mapToPair(s -> {
            String[] userLine = s.split("\\s+");
            Tuple2<String, String> location = new Tuple2<>("L", userLine[1]);
            return new Tuple2<>(userLine[0], location);
        });

        //transaction rdd
        String transactionPath = PropertiesReader.get("basic_transactions_txt");
        JavaRDD<String> transactionJavaRDD = session.read().textFile(transactionPath).toJavaRDD();
        JavaPairRDD<String, Tuple2<String, String>> transactionRDD = transactionJavaRDD.mapToPair(s -> {
            String[] transactionLine = s.split("\\s+");
            Tuple2<String, String> product = new Tuple2<>("P", transactionLine[1]);
            return new Tuple2<>(transactionLine[2], product);
        });

        // union
        JavaPairRDD<String, Tuple2<String, String>> unionRDD = transactionRDD.union(usersRDD);

        //groupbykey
        JavaPairRDD<String, Iterable<Tuple2<String, String>>> groupRDD = unionRDD.groupByKey();

        //flatmaptopair
        JavaPairRDD<String, String> productLocationRDD = groupRDD.flatMapToPair(s -> {
            Iterable<Tuple2<String, String>> pairs = s._2;
            String location = "UNKONWN";
            List<String> products = new ArrayList<>();
            for(Tuple2<String, String> t2 : pairs) {
                if(t2._1.equals("L")) {
                    location = t2._2;
                } else {
                    products.add(t2._2);
                }
            }
            List<Tuple2<String, String>> kvList = new ArrayList<>();
            for(String product: products) {
                kvList.add(new Tuple2<>(product, location));
            }
            return kvList.iterator();
        });

        //一个product所有的location
        JavaPairRDD<String, Iterable<String>> productByLocations = productLocationRDD.groupByKey();

        //去重
        JavaPairRDD<String, Tuple2<Set<String>, Integer>> productByUniqueLocations = productByLocations.mapValues(s -> {
            Set<String> uniqueLocations  = new HashSet<>();
            for(String location : s) {
                uniqueLocations.add(location);
            }
            return new Tuple2<>(uniqueLocations, uniqueLocations.size());
        });

        //print
        List<Tuple2<String, Tuple2<Set<String>, Integer>>> result = productByUniqueLocations.collect();
        for(Tuple2<String, Tuple2<Set<String>, Integer>> t2 : result) {
            System.out.println(t2._1);
            System.out.println(t2._2);
        }
    }

    /**
     * 利用内置函数leftOuterJoin
     * @param session
     */
    public static void leftOutJoin2(SparkSession session) {
        //user rdd
        String userPath = PropertiesReader.get("users_txt");
        JavaRDD<String> userJavaRDD = session.read().textFile(userPath).toJavaRDD();
        JavaPairRDD<String, String> usersRDD = userJavaRDD.mapToPair(s -> {
            String[] userLine = s.split("\\s+");
            return new Tuple2<>(userLine[0], userLine[1]);
        });

        //transaction rdd
        String transactionPath = PropertiesReader.get("transactions_txt");
        JavaRDD<String> transactionJavaRDD = session.read().textFile(transactionPath).toJavaRDD();
        JavaPairRDD<String,String> transactionRDD = transactionJavaRDD.mapToPair(s -> {
            String[] transactionLine = s.split("\\s+");
            return new Tuple2<>(transactionLine[2], transactionLine[1]);
        });

        //leftoutjoin
        JavaPairRDD<String, Tuple2<String, Optional<String>>> leftJoinRDD = transactionRDD.leftOuterJoin(usersRDD);

        JavaPairRDD<String, String> products = leftJoinRDD.mapToPair(t2 -> {
            Tuple2<String, Optional<String>> value = t2._2;
            return new Tuple2<>(value._1, value._2.get());
        });

        //combinebykey
        JavaPairRDD<String, Set<String>> productByUniqueLocations = products.combineByKey(s -> {
            Set<String> set = new HashSet<>();
            set.add(s);
            return set;
        }, (set, s) -> {
            set.add(s);
            return set;

        }, (seta, setb) -> {
            seta.addAll(setb);
            return seta;
        });

        //print
        Map<String, Set<String>> productMap = productByUniqueLocations.collectAsMap();
        for(Map.Entry<String, Set<String>> entry : productMap.entrySet()) {
            System.out.println(entry.getKey());
            System.out.println(entry.getValue());
        }
    }

}

