package com.sy.dataalgorithms.intermediate;

import com.sy.init.InitSpark;
import com.sy.util.PropertiesReader;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;
import scala.Tuple3;
import scala.Tuple7;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * item-cf相关电影
 * @Author Shi Yan
 * @Date 2020/10/28 14:29
 */
public class ItermCFMovieRecom {

    public static void main(String[] args) {
        SparkSession sparkSession = InitSpark.getSparkSession();
        sparkSession.sparkContext().setLogLevel("ERROR");
        movieRecommd(sparkSession);
        InitSpark.closeSparkSession();
    }

    public static void movieRecommd(SparkSession session) {
        String path = PropertiesReader.get("intermediate_movie_txt");
        JavaRDD<String> javaRDD = session.read().textFile(path).toJavaRDD();

        // (movie,(user,rating))
        JavaPairRDD<String, Tuple2<String, Integer>> javaPairRDD = javaRDD.mapToPair(line -> {
            String[] tokens = line.split("\\s+");
            String user = tokens[0];
            String movie = tokens[1];
            Integer rating = Integer.valueOf(tokens[2]);
            Tuple2<String, Integer> userRatings = new Tuple2<>(user, rating);
            return new Tuple2<>(movie, userRatings);
        });

        // movie1,[(user1,rating1),(user2,rating2),...]
        JavaPairRDD<String, Iterable<Tuple2<String, Integer>>> movieGroupRDD = javaPairRDD.groupByKey();

        //user,(movie,rating,numberOfRatingToMovie)
        JavaPairRDD<String, Tuple3<String, Integer, Integer>> usersRDD = movieGroupRDD.flatMapToPair(s -> {
            List<Tuple2<String, Integer>> listOfUsersAndRatings = new ArrayList<>();
            String movie = s._1;
            Iterable<Tuple2<String, Integer>> pairsOfUserAndRating = s._2;
            int numberOfRaters = 0;
            for(Tuple2<String, Integer> t2:pairsOfUserAndRating) {
                numberOfRaters++;
                listOfUsersAndRatings.add(t2);
            }
            List<Tuple2<String, Tuple3<String, Integer, Integer>>> results = new ArrayList<>();
            for(Tuple2<String, Integer> t2:listOfUsersAndRatings) {
                String user = t2._1;
                Integer rating = t2._2;
                Tuple3<String, Integer, Integer> t3 = new Tuple3<>(movie, rating, numberOfRaters);
                results.add(new Tuple2<>(user, t3));
            }
            return results.iterator();
        });

        //self-join (user1,[(movie1,1,2), (movie2,2,2)]),(user1,[(movie1,1,2), (movie3,2,2)])...
        JavaPairRDD<String, Tuple2<Tuple3<String, Integer, Integer>, Tuple3<String, Integer, Integer>>> joinedRDD = usersRDD.join(usersRDD);

        //filter重复movie对
        JavaPairRDD<String, Tuple2<Tuple3<String, Integer, Integer>, Tuple3<String, Integer, Integer>>> filteredRDD = joinedRDD.filter(
                s -> {
                    Tuple3<String, Integer, Integer> movie1 = s._2._1;
                    Tuple3<String, Integer, Integer> movie2 = s._2._2;
                    String movieName1 = movie1._1();
                    String movieName2 = movie2._1();
                    if(movieName1.compareTo(movieName2) < 0) {
                        return true;
                    } else {
                        return false;
                    }
                }
        );

        // 生成所有movie对组合
        JavaPairRDD<Tuple2<String, String>, Tuple7<Integer,Integer,Integer,Integer,Integer,Integer,Integer>> moviePairs = filteredRDD.mapToPair(s -> {
            Tuple3<String, Integer, Integer> movie1 = s._2._1;
            Tuple3<String, Integer, Integer> movie2 = s._2._2;
            Tuple2<String, String> m1m2Name = new Tuple2<>(movie1._1(), movie2._1());
            int ratingProduct = movie1._2() * movie2._2();
            int rating1Squared = movie1._2() * movie1._2();
            int rating2Squared = movie2._2() * movie2._2();
            Tuple7<Integer,Integer,Integer,Integer,Integer,Integer,Integer> t7 = new Tuple7<>(
                    movie1._2(), //movie1评分
                    movie1._3(), //movie1评分人数
                    movie2._2(), //movie2评分
                    movie2._3(), //movie2评分人数
                    ratingProduct, //movie1.rating * movie2.rating
                    rating1Squared, //movie1.rating**2
                    rating2Squared //movie2.rating**2
            );
            return new Tuple2<>(m1m2Name, t7);
        });

        // movie groupbykey
        JavaPairRDD<Tuple2<String, String>, Iterable<Tuple7<Integer, Integer, Integer, Integer, Integer, Integer, Integer>>> corrRDD = moviePairs.groupByKey();

        //相关度
        JavaPairRDD<Tuple2<String, String>, Tuple3<Double, Double, Double>> movieSimilarity = corrRDD.mapValues( t7 ->
            calculateCorrelations(t7)
        );

        //print
        List<Tuple2<Tuple2<String, String>, Tuple3<Double, Double, Double>>> result = movieSimilarity.collect();
        for(Tuple2<Tuple2<String, String>, Tuple3<Double, Double, Double>> t2:result) {
            System.out.println(t2._1 + " => " + t2._2);
        }

    }

    // 计算电影间的相关度
    static Tuple3<Double,Double,Double> calculateCorrelations(Iterable<Tuple7<Integer,Integer,Integer,Integer,Integer,Integer,Integer>> values) {
        int groupSize = 0; // 每个vector长度
        int dotProduct = 0; // movie1和movie2评分点积和
        int rating1Sum = 0; // movie1评分和
        int rating2Sum = 0; // movie2评分和
        int rating1NormSq = 0; // movie1评分平方和
        int rating2NormSq = 0; // movie2评分平方和
        int maxNumOfumRaters1 = 0;  // movie1的最大评分人数
        int maxNumOfumRaters2 = 0;  // movie2的最大评分人数
        for (Tuple7<Integer, Integer, Integer, Integer, Integer, Integer, Integer> t7 : values) {
            groupSize++;
            dotProduct += t7._5();
            rating1Sum += t7._1();
            rating2Sum += t7._3();
            rating1NormSq += t7._6();
            rating2NormSq += t7._7();
            int numOfRaters1 = t7._2(); // movie1评分人数
            if (numOfRaters1 > maxNumOfumRaters1) {
                maxNumOfumRaters1 = numOfRaters1;
            }
            int numOfRaters2 = t7._4(); //movie2评分人数
            if (numOfRaters2 > maxNumOfumRaters2) {
                maxNumOfumRaters2 = numOfRaters2;
            }
        }

        double pearson = calculatePearsonCorrelation(
                groupSize,
                dotProduct,
                rating1Sum,
                rating2Sum,
                rating1NormSq,
                rating2NormSq);

        double cosine = calculateCosineCorrelation(dotProduct,
                Math.sqrt(rating1NormSq),
                Math.sqrt(rating2NormSq));

        double jaccard = calculateJaccardCorrelation(groupSize,
                maxNumOfumRaters1,
                maxNumOfumRaters2);

        return  new Tuple3<>(pearson, cosine, jaccard);
    }

    static double calculatePearsonCorrelation(
        double size,
        double dotProduct,
        double rating1Sum,
        double rating2Sum,
        double rating1NormSq,
        double rating2NormSq)  {
    double numerator = size * dotProduct - rating1Sum * rating2Sum;
    double denominator = Math.sqrt(size * rating1NormSq - rating1Sum * rating1Sum) * Math.sqrt(size * rating2NormSq - rating2Sum * rating2Sum);
    return numerator / denominator;
}

    // dotProduct(A, B) / (norm(A) * norm(B))
    static double calculateCosineCorrelation(double dotProduct,
                                             double rating1Norm,
                                             double rating2Norm) {
        return dotProduct / (rating1Norm * rating2Norm);
    }

    // |Intersection(A, B)| / |Union(A, B)|
    static double calculateJaccardCorrelation(double inCommon,
                                              double totalA,
                                              double totalB) {
        double union = totalA + totalB - inCommon;
        return inCommon / union;
    }

}
