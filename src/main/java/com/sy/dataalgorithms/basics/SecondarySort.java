package com.sy.dataalgorithms.basics;

import com.sy.init.InitSpark;
import com.sy.util.PropertiesReader;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.*;
import scala.Tuple2;

import java.util.*;

/**
 * 二次排序
 * @Author Shi Yan
 * @Date 2020/10/26 15:48
 */
public class SecondarySort {
    public static void main(String[] args) {
        SparkSession sparkSession = InitSpark.getSparkSession();
        sparkSession.sparkContext().setLogLevel("ERROR");
        secondarySort(sparkSession);
        InitSpark.closeSparkSession();
    }

    public static void secondarySort(SparkSession session) {
        String dataPath = PropertiesReader.get("basic_secondarysort_txt");
        JavaRDD<String> lines = session.read().textFile(dataPath).toJavaRDD();
        JavaPairRDD<String, Tuple2<Integer, Integer>> paris = lines.mapToPair( s -> {
            String[] tokens = s.split(",");
            Integer time = new Integer(tokens[1]);
            Integer value = new Integer(tokens[2]);
            return new Tuple2<>(tokens[0], new Tuple2<>(time, value));

        });

        JavaPairRDD<String, Iterable<Tuple2<Integer, Integer>>> groups = paris.groupByKey();

        JavaPairRDD<String, Iterable<Tuple2<Integer, Integer>>> sorted = groups.mapValues( s -> {
            List<Tuple2<Integer, Integer>> newList = new ArrayList<>((Collection<? extends Tuple2<Integer, Integer>>) s);
            Collections.sort(newList, new TupleComparator());
            return newList;

        });

        List<Tuple2<String, Iterable<Tuple2<Integer, Integer>>>> output =  sorted.collect();
        for(Tuple2<String, Iterable<Tuple2<Integer, Integer>>> t : output) {
            Iterable<Tuple2<Integer, Integer>> list = t._2;
            System.out.println(t._1);
            for(Tuple2<Integer, Integer> t2: list) {
                System.out.println(t2._1 + "," + t2._2);
            }
            System.out.println("---------------------");
        }
    }

    /*public static void secondarySort2(SparkSession session) {
        String dataPath = PropertiesReader.get("secondarysort_txt");
        JavaRDD<Row> lines = session.read().textFile(dataPath).toJavaRDD()
                .map(s -> {
                    String[] tokens = s.split(",");
                    return RowFactory.create(tokens[0], Integer.valueOf(tokens[1]), Integer.valueOf(tokens[2]));
                });
        Dataset<Row> dataset = session.createDataFrame(lines, InitSchema.initSchema());
        dataset.groupBy("name").count().orderBy("name").show();

    }*/

}

class TupleComparator implements Comparator<Tuple2<Integer, Integer>> {

    @Override
    public int compare(Tuple2<Integer, Integer> o1, Tuple2<Integer, Integer> o2) {
        if(o1._1 < o2._1) {
            return -1;
        } else if(o1._1 > o2._1) {
            return 1;
        }
        return 0;
    }
}
