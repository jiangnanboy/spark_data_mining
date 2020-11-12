package com.sy.dataalgorithms.basics;

import com.sy.init.InitSpark;
import com.sy.util.PropertiesReader;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;
import scala.Tuple3;

import java.util.*;

/**
 * 关联规则
 * @Author Shi Yan
 * @Date 2020/10/27 21:14
 */
public class AssociationRules {
    public static void main(String[] args) {
        SparkSession sparkSession = InitSpark.getSparkSession();
        sparkSession.sparkContext().setLogLevel("ERROR");
        findAR(sparkSession);
        InitSpark.closeSparkSession();
    }

    /**
     * 关联规则挖掘
     * @param session
     */
    public static void findAR(SparkSession session) {
        String path = PropertiesReader.get("basic_associationrelus_txt");
        JavaRDD<String> javaRDD = session.read().textFile(path).toJavaRDD();
        //生成频繁模式 [a,b,c]=> [a]:1,[b]:1,[c]:1,[a,b]:1,[a,c]:1,[b,c]:1,[a,b,c]:1
        JavaPairRDD<List<String>, Integer> patterns = javaRDD.flatMapToPair(s -> {
            List<String> list = toList(s);
            List<List<String>> combinations = getSubsets(list);
            List<Tuple2<List<String>, Integer>> result = new ArrayList<>();
            for(List<String> combList:combinations) {
                if(combList.size() > 0) {
                    result.add(new Tuple2<>(combList, 1));
                }
            }
            return result.iterator();
        });

        //reducebykey归约
        JavaPairRDD<List<String>, Integer> combined = patterns.reduceByKey((i1, i2) -> i1 + i2);

        //生成所有子模式，如[a,b,c]:2 => [a,b,c]:(null,2), [a,b]:([a,b,c],2), [a,c]:([a,b,c],2), [b,c]:([a,b,c],2)
        JavaPairRDD<List<String>, Tuple2<List<String>, Integer>> subPatterns = combined.flatMapToPair(pattern -> {
            List<Tuple2<List<String>, Tuple2<List<String>, Integer>>> result = new ArrayList<>();
            List<String> list = pattern._1;
            Integer frequency = pattern._2;
            result.add(new Tuple2<>(list, new Tuple2<>(null, frequency)));
            if(1 == list.size()) {
                return result.iterator();
            }
            for(int i = 0; i < list.size(); i++) {
                List<String> subList = removeOneItem(list, i);
                result.add(new Tuple2<>(subList, new Tuple2<>(list, frequency)));
            }
            return result.iterator();
        });

        //组合所有子模式 ([a,b],[(null, 2), ([a,b,d], 1), ([a,b,c], 1)])
        JavaPairRDD<List<String>, Iterable<Tuple2<List<String>, Integer>>> rules = subPatterns.groupByKey();

        //生成关联规则
        JavaRDD<List<Tuple3<List<String>, List<String>, Double>>> assocRules = rules.map(rule -> {
            List<Tuple3<List<String>, List<String>, Double>> result = new ArrayList<>();
            List<String> fromList = rule._1;
            Iterable<Tuple2<List<String>, Integer>> to = rule._2;
            List<Tuple2<List<String>, Integer>> toList = new ArrayList<>();
            Tuple2<List<String>, Integer> fromCount = null;

            for(Tuple2<List<String>, Integer> t2 : to) {
                if(null == t2._1) {
                    fromCount = t2;
                } else {
                    toList.add(t2);
                }
            }

            if(toList.isEmpty()) {
                return result;
            }

            //计算置信度 Y的支持度/X的支持度
            for(Tuple2<List<String>, Integer> t2:toList) {
                double confidence =t2._2 / (double)fromCount._2;
                List<String> t2List = new ArrayList<>(t2._1);
                t2List.removeAll(fromList);
                result.add(new Tuple3<>(fromList, t2List, confidence)); // X -> Y : confidence
            }

            return result;
        });

        //打印
        List<List<Tuple3<List<String>, List<String>, Double>>> result = assocRules.collect();
        for(List<Tuple3<List<String>, List<String>, Double>> r1 : result) {
            System.out.println(r1);
        }
    }

    static List<String> toList(String line) {
        String[] items = line.trim().split(",");
        List<String> list = new ArrayList<>();
        for(String item : items) {
            list.add(item);
        }
        return list;
    }

    static List<String> removeOneItem(List<String> list, int i) {
        if((null == list) || (list.isEmpty())) {
            return list;
        }
        if((i < 0) || (i > (list.size() -1))) {
            return list;
        }
        List<String> clonedList = new ArrayList<>(list);
        clonedList.remove(i);
        return clonedList;
    }

    /**
     * 获得集合所有子集
     * @param elements
     * @return
     */
    static List<List<String>> getSubsets(List<String> elements) {
        List<List<String>> allSubsets = new ArrayList<>();
        int max = 1 << elements.size();
        for(int i = 0; i < max; i++) {
            int index = 0;
            int k = i;
            List<String> list = new ArrayList<>();
            while(k > 0) {
                if((k&1) > 0) {
                    list.add(elements.get(index));
                }
                k >>= 1;
                index ++;
            }
            allSubsets.add(list);
        }
        return allSubsets;
    }

}

