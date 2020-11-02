package com.sy.dataalgorithms.others.graph;

import org.apache.spark.sql.SparkSession;
import java.util.ArrayList;
import java.util.List;

import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;
import com.google.common.collect.Iterables;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;

/**
 * @Author Shi Yan
 * @Date 2020/8/28 20:32
 */
public class PRDemo {

    /**
     * pagerank计算
     * @param session
     */
    public static void pageRank(SparkSession session, String filePath) {
        int iteration = 10;
        JavaRDD<String> javaRDD = session.read().textFile(filePath).javaRDD();
        JavaPairRDD<String, Iterable<String>> links = javaRDD.mapToPair(
                s -> {
                        String[] pairs = s.split("\\s+");
                        return new Tuple2<>(pairs[0], pairs[1]);
                }
        ).distinct().groupByKey();
        links.persist(StorageLevel.MEMORY_ONLY());
        //对url的rank值进行初始化
        JavaPairRDD<String, Double> ranks = links.mapValues( s -> 1.0);
        //迭代更新url rank值
        for(int i = 0; i < iteration; i++) {
            //url对其它url的贡献
            JavaPairRDD<String, Double> contribution = links.join(ranks).values() // <String, Tuple2<Iterable<String>, Double>> -> Tuple2<Iterable<String>, Double>
                    .flatMapToPair(tuple2 ->  { // Tuple2<Iterable<String>, Double> -> String, Double
                        int urlSum = Iterables.size(tuple2._1);
                        List<Tuple2<String, Double>> list = new ArrayList<>();
                        for (String s : tuple2._1) {
                            list.add(new Tuple2<>(s, tuple2._2 / urlSum));
                        }
                        return list.iterator();
                    });
            //根据邻居url贡献值重计算url rank值
            ranks = contribution.reduceByKey((a, b) -> a + b).mapValues(sum ->
                    //0.15为随机跳转概率，0.85为阻尼系数
                    0.15 + sum * 0.85
            );
        }
        //打印结果
        List<Tuple2<String, Double>> result = ranks.collect();
        for(Tuple2<String, Double> tuple2 : result) {
            System.out.println(tuple2._1 + " rank value: " + tuple2._2);
        }
    }

}
