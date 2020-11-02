package com.sy.dataalgorithms.others.base;

import com.sy.init.InitSpark;
import com.sy.util.PropertiesReader;
import org.apache.spark.api.java.Optional;
import org.apache.spark.sql.SparkSession;

import scala.Tuple2;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaPairRDD;

import java.util.Set;
import java.util.HashSet;

/**
 * @Author Shi Yan
 * @Date 2020/8/31 19:19
 */
public class RddDemo {

    public static void main(String[] args) {
        SparkSession session = InitSpark.getSparkSession();
        session.sparkContext().setLogLevel("ERROR");
        rddJoin(session);
        session.cloneSession();
    }

    /**
     * rdd join
     * @param session
     */
    public static void rddJoin(SparkSession session) {
        String usersPath = PropertiesReader.get("others_base_users_txt");
        String transactionsPath = PropertiesReader.get("others_base_transactions_txt");
        //读取users数据
        JavaRDD<String> userRDD = session.read().textFile(usersPath).toJavaRDD();
        JavaPairRDD<String, String> userPairRDD = userRDD.mapToPair( s -> {
            String userId = s.split("\\s+")[0];
            String location = s.split("\\s+")[1];
            return new Tuple2<>(userId, location);
        });
        //读取transactions数据
        JavaRDD<String> transactionRDD = session.read().textFile(transactionsPath).toJavaRDD();
        JavaPairRDD<String, String> transactionPairRDD = transactionRDD.mapToPair(s -> {
            String productId = s.split("\\s+")[1];
            String userId = s.split("\\s+")[2];
            return new Tuple2<>(userId, productId);
        });
        //leftoutjoin
        JavaPairRDD<String, Tuple2<String, Optional<String>>> rddJoin = transactionPairRDD.leftOuterJoin(userPairRDD); // <userId,Tuple2<productId, location>>
        JavaPairRDD<String, String> productPairRDD = rddJoin.mapToPair(tuple2 ->  { // <productId, location>
            if(tuple2._2._2.isPresent()) {
                return new Tuple2<>(tuple2._2._1, tuple2._2._2.get());
            }
            else {
                return new Tuple2<>(tuple2._2._1, null);
            }
        });
        // 分组统计 <productId, Tuple2<List<location, size>>>
        /*JavaPairRDD<String, Iterable<String>> productByLocation = productPairRDD.groupByKey();
        JavaPairRDD<String, Tuple2<Set<String>, Integer>> productUniqueLocation = productByLocation.mapValues(s -> {
            Set<String> locations = new HashSet<>();
            for(String l : s) {
                locations.add(l);
            }
            return new Tuple2<>(locations, locations.size());
        });*/
        // 分组统计<productId, Set<Location>>
        JavaPairRDD<String, Set<String>> productUniqueLocation = productPairRDD.combineByKey(s -> {
                Set<String> set = new HashSet<>();
                set.add(s);
                return set;
        }, (set, s) ->{
            set.add(s);
            return set;
        }, (set1, set2) -> {
            set1.addAll(set2);
            return set1;
        });
        //打印
        System.out.println(productUniqueLocation.take(10));
    }

}
