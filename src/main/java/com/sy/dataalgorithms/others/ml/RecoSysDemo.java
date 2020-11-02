package com.sy.dataalgorithms.others.ml;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.Serializable;

/**
 * @Author Shi Yan
 * @Date 2020/8/28 22:46
 */
public class RecoSysDemo {

    /**
     * 推荐
     * @param session
     * @param filePath
     */
    public static void alsRecsys(SparkSession session, String filePath) {
        JavaRDD<Rating> javaRDD = session.read()
                .textFile(filePath)
                .toJavaRDD()
                .map(Rating::parseLine);
        Dataset<Row> ratings = session.createDataFrame(javaRDD, Rating.class);
        Dataset<Row>[] splits = ratings.randomSplit(new double[] {0.8, 0.2});
        Dataset<Row> trainData = splits[0];
        Dataset<Row> testData = splits[1];
        //model
        ALS als = new ALS()
                .setMaxIter(5) //迭代次数
                .setRank(10) //矩阵分解R(m*n) = P(m*r)*Q(r*n)
                .setRegParam(0.01) //正则项
                .setUserCol("userId")
                .setItemCol("movieId")
                .setRatingCol("rating");
        ALSModel model = als.fit(trainData);
        //评估rmse
        model.setColdStartStrategy("drop");
        Dataset<Row> predictions = model.transform(testData);
        RegressionEvaluator evaluator = new RegressionEvaluator()
                .setMetricName("rmse")
                .setLabelCol("rating")
                .setPredictionCol("prediction");
        double rmse = evaluator.evaluate(predictions);
        System.out.println("test data rmse: " + rmse);
        //为指定user推荐topn movie
        Dataset<Row> users = ratings.select(als.getUserCol()).distinct().limit(5);
        users.show();
        Dataset<Row> userRecsItems = model.recommendForUserSubset(users, 10);
        userRecsItems.show();
    }

    //user对movie评分
    public static class Rating implements Serializable {
        private int userId;
        private int movieId;
        private float rating;
        private long timestamp;

        public Rating(int userId, int movieId, float rating, long timestamp) {
            this.userId = userId;
            this.movieId = movieId;
            this.rating = rating;
            this.timestamp = timestamp;
        }

        public int getUserId() {
            return userId;
        }

        public int getMovieId() {
            return movieId;
        }

        public float getRating() {
            return rating;
        }

        public long getTimestamp() {
            return timestamp;
        }

        public static Rating parseLine(String line) {
            String[] fields = line.split("::");
            return new Rating(Integer.parseInt(fields[0]), Integer.parseInt(fields[1]), Float.parseFloat(fields[2]), Long.parseLong(fields[3]));
        }
    }


}
