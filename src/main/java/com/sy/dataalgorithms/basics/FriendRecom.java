package com.sy.dataalgorithms.basics;

import com.sy.init.InitSpark;
import com.sy.util.PropertiesReader;
import jdk.nashorn.internal.runtime.arrays.IteratorAction;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

import java.util.*;

/**
 * 推荐好友
 * @Author Shi Yan
 * @Date 2020/10/28 21:54
 */
public class FriendRecom {
    public static void main(String[] args) {
        SparkSession sparkSession = InitSpark.getSparkSession();
        sparkSession.sparkContext().setLogLevel("ERROR");
        recommendation(sparkSession);
        InitSpark.closeSparkSession();
    }

    public static void recommendation(SparkSession session) {
        String path = PropertiesReader.get("basic_recommender_friends_txt");
        JavaRDD<String> javaRDD = session.read().textFile(path).toJavaRDD();

        //直接好友和可能好友
        JavaPairRDD<Long, Tuple2<Long, Long>> pairRDD = javaRDD.flatMapToPair(s -> {
            String[] tokens = s.split("\\s+");
            long person = Long.parseLong(tokens[0]);
            String friendString = tokens[1];
            String[] friendTokens = friendString.split(",");

            List<Long> friends = new ArrayList<>(); //此person所有直接朋友
            //直接friends
            List<Tuple2<Long, Tuple2<Long, Long>>> mappOutput = new ArrayList<>();
            for(String friend:friendTokens) {
                long toUser = Long.parseLong(friend);
                friends.add(toUser);
                Tuple2<Long, Long> directFriend = T2(toUser, -1L);
                mappOutput.add(T2(person, directFriend));
            }

            for(int i = 0; i < friends.size(); i++) {
                for(int j = i + 1; j < friends.size(); j++) {
                    //可能好友1
                    Tuple2<Long, Long> possibleFriend1 = T2(friends.get(j), person);
                    mappOutput.add(T2(friends.get(i), possibleFriend1));
                    //可能好友2
                    Tuple2<Long, Long> possibleFriend2 = T2(friends.get(i), person);
                    mappOutput.add(T2(friends.get(j), possibleFriend2));
                }
            }
            return mappOutput.iterator();
        });

        //1. 归约相同key
        /*JavaPairRDD<Long, Iterable<Tuple2<Long, Long>>> groupRDD = pairRDD.groupByKey();

        //mapValues
        JavaPairRDD<Long, String> recommendatons = groupRDD.mapValues(values -> {
            Map<Long, List<Long>> commonFriends= new HashMap<>();
            for(Tuple2<Long, Long> t2 : values) {
                long toUser = t2._1;
                long commonFriend = t2._2;
                boolean alreadyFriend = (commonFriend == -1);//直接好友，已经是好友
                if(commonFriends.containsKey(toUser)) {
                    if(alreadyFriend) {
                        commonFriends.put(toUser, null);
                    } else if(null != commonFriends.get(toUser)) {
                        commonFriends.get(toUser).add(commonFriend);
                    }
                } else {
                    if(alreadyFriend) {
                        commonFriends.put(toUser, null);
                    } else {
                        List<Long> list1 = new ArrayList<>(Arrays.asList(commonFriend));
                        commonFriends.put(toUser, list1);
                    }
                }
            }
            return buildRecommendations(commonFriends);
        });*/

        //以上 1. <=> 2.groupByKey与mapValues <=> reduceByKey or combineByKey
        JavaPairRDD<Long, String> recommendatons = pairRDD.combineByKey(t2 -> buildCommonFriends(t2),
                (commonFriends, t2) -> buildCommonFriends(commonFriends, t2)
        , (commonFriends1, commonFriends2) -> {
            for(Map.Entry<Long, List<Long>> entry:commonFriends2.entrySet()) {
                if(commonFriends1.containsKey(entry.getKey())) {
                    commonFriends1.get(entry.getKey()).addAll(entry.getValue());
                } else {
                    commonFriends1.put(entry.getKey(), entry.getValue());
                }
            }
            return commonFriends1;
        }).mapValues(commonFriends -> buildRecommendations(commonFriends));


        //print
        List<Tuple2<Long, String>> result = recommendatons.collect();
        for(Tuple2<Long, String> t2:result) {
            System.out.println(t2._1 + " => " + t2._2);
        }
    }

    static Map<Long, List<Long>> buildCommonFriends(Tuple2<Long, Long> t2) {
        Map<Long, List<Long>> commonFriends = new HashMap<>();
        long toUser = t2._1;
        long commonFriend = t2._2;
        boolean alreadyFriend = (commonFriend == -1);//直接好友，已经是好友
        if(commonFriends.containsKey(toUser)) {
            if(alreadyFriend) {
                commonFriends.put(toUser, null);
            } else if(null != commonFriends.get(toUser)) {
                commonFriends.get(toUser).add(commonFriend);
            }
        } else {
            if(alreadyFriend) {
                commonFriends.put(toUser, null);
            } else {
                List<Long> list1 = new ArrayList<>(Arrays.asList(commonFriend));
                commonFriends.put(toUser, list1);
            }
        }
        return commonFriends;
    }

    static Map<Long, List<Long>> buildCommonFriends(Map<Long, List<Long>> commonFriends, Tuple2<Long, Long> t2) {
        long toUser = t2._1;
        long commonFriend = t2._2;
        boolean alreadyFriend = (commonFriend == -1);//直接好友，已经是好友
        if(commonFriends.containsKey(toUser)) {
            if(alreadyFriend) {
                commonFriends.put(toUser, null);
            } else if(null != commonFriends.get(toUser)) {
                commonFriends.get(toUser).add(commonFriend);
            }
        } else {
            if(alreadyFriend) {
                commonFriends.put(toUser, null);
            } else {
                List<Long> list1 = new ArrayList<>(Arrays.asList(commonFriend));
                commonFriends.put(toUser, list1);
            }
        }
        return commonFriends;
    }

    static String buildRecommendations(Map<Long, List<Long>> friends) {
        StringBuffer sb = new StringBuffer();
        for(Map.Entry<Long, List<Long>> entry:friends.entrySet()) {
            if(null == entry.getValue()) { //直接好友
                continue;
            }
            sb.append("和 "+entry.getKey()).append(" 的共同好友(").append(entry.getValue().size()).append("个:").append(entry.getValue()).append("),");
        }
        return sb.toString();
    }

    static Tuple2<Long, Long> T2(long a, long b) {
        return new Tuple2<>(a, b);
    }

    static Tuple2<Long, Tuple2<Long, Long>> T2(long a, Tuple2<Long, Long> b) {
        return new Tuple2<>(a, b);
    }

}
