package com.sy.dataalgorithms.basics;

import com.google.common.collect.Sets;
import com.sy.init.InitSpark;
import com.sy.util.PropertiesReader;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

import java.util.*;

/**
 * 共同好友
 * @Author Shi Yan
 * @Date 2020/10/28 20:32
 */
public class FindCommonFriends {

    public static void main(String[] args) {
        SparkSession sparkSession = InitSpark.getSparkSession();
        sparkSession.sparkContext().setLogLevel("ERROR");
        commonFriends(sparkSession);
        InitSpark.closeSparkSession();
    }

    public static void commonFriends(SparkSession session) {
        String path = PropertiesReader.get("basic_usersfriends_txt");
        JavaRDD<String> javaRDD = session.read().textFile(path).toJavaRDD();

        //100,200 300 400 => key=(100,200),value=[200,300,400];key=(100,300),value=[200,300,400];key=(100,400),value=[200,300,400];
        //200,100 300 400 => key=(100,200),value=[100,300,400];key=(200,300),value=[100,300,400];key=(200,400),value=[100,300,400];
        JavaPairRDD<Tuple2<Long,Long>, Iterable<Long>> pairs =  javaRDD.flatMapToPair(s -> {
            String[] tokens = s.split(",");
            long person = Long.parseLong(tokens[0]);
            String friendString = tokens[1];
            String[] friendsTokens = friendString.split("\\s+");
            if(1 == friendsTokens.length) {
                Tuple2<Long, Long> key = buildSortedTuple(person, Long.parseLong(friendsTokens[0]));
                return Arrays.asList(new Tuple2<Tuple2<Long, Long>, Iterable<Long>>(key, new ArrayList<>())).iterator();
            }
            List<Long> friends = new ArrayList<>();
            for(String f:friendsTokens) {
                friends.add(Long.parseLong(f));
            }
            List<Tuple2<Tuple2<Long, Long>, Iterable<Long>>> result = new ArrayList<>();
            for(Long f:friends) {
                Tuple2<Long, Long> key = buildSortedTuple(person, f);
                result.add(new Tuple2<>(key, friends));
            }
            return result.iterator();
        });

        //key=(100,200),value=[[200,300,400],[100,300,400]],(100,200)的共同好友是[300,400]
        JavaPairRDD<Tuple2<Long, Long>, Iterable<Iterable<Long>>> groupRDD = pairs.groupByKey();
        JavaPairRDD<Tuple2<Long, Long>, Iterable<Long>> commonFriends = groupRDD.mapValues(s -> {
            Map<Long, Integer> countCommon = new HashMap<>();
            int size = 0;
            for(Iterable<Long> iter : s) {
                size ++;
                List<Long> list = iterableToList(iter);
                if((null == list) || (list.isEmpty())) {
                    continue;
                }
                for(Long f: list) {
                    Integer count = countCommon.get(f);
                    if(null == count) {
                        countCommon.put(f,1);
                    } else {
                        countCommon.put(f, ++count);
                    }
                }
            }
            //如果key的
            List<Long> finalCommonFriends = new ArrayList<>();
            for(Map.Entry<Long, Integer> entry:countCommon.entrySet()) {
                if(entry.getValue() == size) {
                    finalCommonFriends.add(entry.getKey());
                }
            }
            return finalCommonFriends;
        });

        //合并groupByKey和mapValues => reduceByKey or combineByKey
        /*JavaPairRDD<Tuple2<Long, Long>, Iterable<Long>> commonFriends = pairs.reduceByKey(new Function2<Iterable<Long>, Iterable<Long>, Iterable<Long>>() {
            @Override
            public Iterable<Long> call(Iterable<Long> a, Iterable<Long> b) throws Exception {
                Set<Long> x = Sets.newHashSet(a);
                Set<Long> intersection = new HashSet<>();
                for(Long item:b) {
                    if(x.contains(item)) {
                        intersection.add(item);
                    }
                }
                return intersection;
            }
        });*/

        //print
        Map<Tuple2<Long, Long>, Iterable<Long>> commonFriendsMap = commonFriends.collectAsMap();
        for(Map.Entry<Tuple2<Long, Long>, Iterable<Long>> entry:commonFriendsMap.entrySet()) {
            System.out.println(entry.getKey() + " => " + entry.getValue());
        }
    }

    static Tuple2<Long, Long> buildSortedTuple(long a, long b) {
        if(a < b) {
            return new Tuple2<>(a,b);
        } else {
            return new Tuple2<>(b, a);
        }
    }

    static List<Long> iterableToList(Iterable<Long> iter) {
        List<Long> list = new ArrayList<>();
        Iterator<Long> itr = iter.iterator();
        while(itr.hasNext()) {
            list.add(itr.next());
        }
        return list;
    }

}
