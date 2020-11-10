package com.sy.dataalgorithms.intermediate;

import com.sy.dataalgorithms.intermediate.customervalue.RFM;
import com.sy.init.InitSpark;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.ml.linalg.SQLDataTypes;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.*;
import org.apache.spark.sql.catalyst.encoders.RowEncoder;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.util.ArrayList;
import java.util.List;

import static org.apache.spark.sql.functions.col;


/**
 *
 * 什么是AHP---------------> https://baike.baidu.com/item/%E5%B1%82%E6%AC%A1%E5%88%86%E6%9E%90%E6%B3%95/1672?fr=aladdin）以及 （https://tellyouwhat.cn/p/ahp-users-value-score/）
 *
 * AHP(the analytic hierarchy process)，层级分析法
 *  为每个用户计算AHP得分，并根据RFM分群结果进行同类中的客户排序
 *  1.建立层次结构模型
 *  2.构造成对比较矩阵
 *  3.计算权向量并做一致性检验
 *
 *  目标：
 *      针针RFM中同类价值顾客排名
 *      利用RFM模型中的指标R、F、M
 *      为每一个用户计算AHP得分(根据AHP得分对同类价值顾客进行排名)
 *
 * @Author Shi Yan
 * @Date 2020/11/10 21:18
 */
public class AHP {

    public static void main(String[] args) {
        SparkSession sparkSession = InitSpark.getSparkSession();
        sparkSession.sparkContext().setLogLevel("ERROR");
        Dataset<Row> rfmDataset= RFM.rfmStatistics(sparkSession);
        Dataset<Row> predictionDataset = RFM.customersCluster(rfmDataset);
        List<Double> weightVector = ahpWeightVector(sparkSession);
        ahpScore(predictionDataset, weightVector);
        InitSpark.closeSparkSession();
    }

    /**
     *计算权重向量并作一致性检验
     * RFM比较矩阵假设如下(假设认为重要性M>F>R)：，以下矩阵的定义请查看百度对AHP的定义：（https://baike.baidu.com/item/%E5%B1%82%E6%AC%A1%E5%88%86%E6%9E%90%E6%B3%95/1672?fr=aladdin）以及 （https://tellyouwhat.cn/p/ahp-users-value-score/）
     *      R     F      M
     *
     * R    1    1/5    1/7
     *
     * F    5     1     1/3
     *
     * M    7     3      1
     *
     */
    public static List<Double> ahpWeightVector(SparkSession session) {

        //1.构建RFM两两比较矩阵
        List<Row> listRow = new ArrayList<>();//第一列为行号，使用行号的目的是按照行号排序的矩阵(行号分别对应R、F、M)(因为spark对数据处理不是有序的)
        listRow.add(RowFactory.create(1, 1.0, 1.0/5, 1.0/7));
        listRow.add(RowFactory.create(2, 5.0, 1.0, 1.0/3));
        listRow.add(RowFactory.create(3, 7.0, 3.0, 1.0));

        /**
         * +------+---+---+-------------------+
         * |rownum|  r|  f|                  m|
         * +------+---+---+-------------------+
         * |     1|1.0|0.2|0.14285714285714285|
         * |     2|5.0|1.0| 0.3333333333333333|
         * |     3|7.0|3.0|                1.0|
         * +------+---+---+-------------------+
         */
        Dataset<Row> rawDM = session.createDataFrame(listRow, new StructType(new StructField[]{
                new StructField("rownum", DataTypes.IntegerType, false, Metadata.empty()),//为行号
                new StructField("r", DataTypes.DoubleType,false, Metadata.empty()),
                new StructField("f", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("m", DataTypes.DoubleType, false, Metadata.empty())
        }));

        //a.rawDM每列对应的和
        //列r的和
        List<Row> rSumRow = rawDM.select(functions.sum(col("r"))).collectAsList();
        double rSum = rSumRow.get(0).getDouble(0);
        //列f的和
        List<Row> fSumRow = rawDM.select(functions.sum(col("f"))).collectAsList();
        double fSum = fSumRow.get(0).getDouble(0);
        //列m的和
        List<Row> mSumRow = rawDM.select(functions.sum(col("m"))).collectAsList();
        double mSum = mSumRow.get(0).getDouble(0);


        /**
         * b.rawDM每列除以对应列的和
         * +------+-------------------+--------------------+-------------------+
         * |rownum|                 nr|                  nf|                 nm|
         * +------+-------------------+--------------------+-------------------+
         * |     1|0.07692307692307693|0.047619047619047616|0.09677419354838708|
         * |     2|0.38461538461538464| 0.23809523809523808| 0.2258064516129032|
         * |     3| 0.5384615384615384|  0.7142857142857143| 0.6774193548387096|
         * +------+-------------------+--------------------+-------------------
         */
        Dataset<Row> normalizedDM = rawDM.map((MapFunction<Row, Row>) row ->
            RowFactory.create(row.getInt(0), row.getDouble(1)/rSum, row.getDouble(2)/fSum, row.getDouble(3)/mSum)
        , RowEncoder.apply(new StructType(new StructField[]{
                        new StructField("rownum", DataTypes.IntegerType, false, Metadata.empty()),//为行号
                new StructField("nr", DataTypes.DoubleType,false, Metadata.empty()),
                new StructField("nf", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("nm", DataTypes.DoubleType, false, Metadata.empty())
        })));

        /**
         * c.计算权向量，为normalizedDM每行数据的均值
         * +------+-------------------+
         * |rownum|             weight|
         * +------+-------------------+
         * |     1|0.07377210603017054|
         * |     2| 0.2828390247745087|
         * |     3| 0.6433888691953208|
         * +------+-------------------+
         *
         */
        Dataset<Row> criteriaWeights = normalizedDM.map((MapFunction<Row, Row>) row -> {
            double rowSum = 0.0;
            int rowNum = row.getInt(0); //行号
            for(int i = 1; i < row.size(); i ++) {
                rowSum += row.getDouble(i);
            }
            return RowFactory.create(rowNum, rowSum/(row.size() - 1 ));
        }, RowEncoder.apply(new StructType(new StructField[] {
                new StructField("rownum", DataTypes.IntegerType, false, Metadata.empty()),//为行号
                new StructField("weight", DataTypes.DoubleType,false, Metadata.empty())
        })));

        /**按行号升序排序，以对应R、F、M
         * +------+-------------------+
         * |rownum|             weight|
         * +------+-------------------+
         * |     1|0.07377210603017054|
         * |     2| 0.2828390247745087|
         * |     3| 0.6433888691953208|
         * +------+-------------------+
         */
        criteriaWeights = criteriaWeights.sort(col("rownum").asc());//按行号升序排序
        List<Row> criteriaList = criteriaWeights.collectAsList();
        List<Double> criteriaDouble = new ArrayList<>(); //权值向量
        for(Row row : criteriaList) {
            criteriaDouble.add(row.getDouble(1));
        }

        //2.验证权向量criteriaDouble的一致性
        /**
         * a.原始矩阵rawDM的第一行与权向量criteriaDouble相乘
         * +------+-------------------+-------------------+-------------------+
         * |rownum|                  r|                  f|                  m|
         * +------+-------------------+-------------------+-------------------+
         * |     1|0.07377210603017054|0.05656780495490174|0.09191269559933153|
         * |     2| 0.3688605301508527| 0.2828390247745087|0.21446295639844026|
         * |     3| 0.5164047422111937|  0.848517074323526| 0.6433888691953208|
         * +------+-------------------+-------------------+-------------------+
         */
        Dataset<Row> weightNormalizedM = rawDM.map((MapFunction<Row, Row>) row ->
            RowFactory.create(row.getInt(0), row.getDouble(1) * criteriaDouble.get(0), row.getDouble(2) * criteriaDouble.get(1), row.getDouble(3) * criteriaDouble.get(2))
        , RowEncoder.apply(new StructType(new StructField[]{
                new StructField("rownum", DataTypes.IntegerType, false, Metadata.empty()),//为行号
                new StructField("r", DataTypes.DoubleType,false, Metadata.empty()),
                new StructField("f", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("m", DataTypes.DoubleType, false, Metadata.empty())
        })));

        /**
         * b.weightNormalizedM每行分别求和
         * +------+-------------------+
         * |rownum|             rowsum|
         * +------+-------------------+
         * |     1|0.22225260658440382|
         * |     2| 0.8661625113238016|
         * |     3| 2.0083106857300406|
         * +------+-------------------+
         */
        Dataset<Row> rowSumWeights = weightNormalizedM.map((MapFunction<Row, Row>) row -> {
            double rowSum = 0.0;
            int rowNum = row.getInt(0); //行号
            for(int i = 1; i < row.size(); i ++) {
                rowSum += row.getDouble(i);
            }
            return RowFactory.create(rowNum, rowSum);
        }, RowEncoder.apply(new StructType(new StructField[] {
                new StructField("rownum", DataTypes.IntegerType, false, Metadata.empty()),//为行号
                new StructField("rowsum", DataTypes.DoubleType,false, Metadata.empty())
        })));

        rowSumWeights = rowSumWeights.sort(col("rownum").asc());//按行号升序排序
        List<Row> rowSumList = rowSumWeights.collectAsList();
        List<Double> rowSumDouble = new ArrayList<>(); // 每行求和的结果
        for(Row row : rowSumList) {
            rowSumDouble.add(row.getDouble(1));
        }

        /**
         * c.计算λ=3.065511827120929
         */
        double λ = 0.0;
        for(int i = 0; i < rowSumDouble.size(); i ++) {
            λ += (rowSumDouble.get(i) / criteriaDouble.get(i));
        }
        λ /= 3.0; //3为矩阵rawDM行数

        /**
         * d.计算CI一致性指标
         * CI=(λ - n)/(n - 1),n为矩阵行数
         * CI=0，有完全的一致性；CI 接近于0，有满意的一致性；CI 越大，不一致越严重。
         */
        double CI = (λ - 3)/(3 - 1);

        /**
         * e.计算CR一致性比率
         * 为衡量CI 的大小，引入随机一致性指标 RI
         * 其中，随机一致性指标RI和判断矩阵的阶数有关，一般情况下，矩阵阶数越大，则出现一致性随机偏离的可能性也越大，其对应关系如表2：
         *      平均随机一致性指标RI标准值(不同的标准不同，RI的值也会有微小的差异)
         *      阶数	1	2	3	     4	 5	       6	7	      8	       9	10	      11	12	      13	14	    15
         *      RI  	0	0	0.58	0.9	1.12	1.24	1.32	1.41	1.45	1.49	1.51	1.54	1.56	1.57	1.58
         *
         * 考虑到一致性的偏离可能是由于随机原因造成的，因此在检验判断矩阵是否具有满意的一致性时，还需将CI和随机一致性指标RI进行比较，得出检验系数CR，公式如下：
         *      CR=CI/RI
         * 一般，如果CR<0.1 ，则认为该判断矩阵通过一致性检验，否则就不具有满意一致性。
         */
        double CR = CI / 0.58; //0.056475713035283585 < 0.1，通过一致性检验
        if(CR < 0.1) {
            return criteriaDouble;
        } else {
            return null;
        }
    }

    /**
     * RFM聚类可以分为高价值用户、一般用户、低价值用户等。
     * 对于RFM中的同类用户的排序则使用AHP权向量给每个用户计算最终得分：利用每个用户的RFM向量与权值向量点乘得出AHP分数
     * @param dataset 经过RFM聚类后的数据
     * @param weightVector 权重向量
     */
    public static void ahpScore(Dataset<Row> dataset, List<Double> weightVector) {

        /**
         * 计算每个用户的AHP分值:
         *+----------+------------------+--------------------+----------+--------------------+
         * |customerid|          features|      scaledfeatures|prediction|            ahpscore|
         * +----------+------------------+--------------------+----------+--------------------+
         * |     12940| [46.0,4.0,876.29]|[0.12332439678284...|         1|0.024241021827781713|
         * |     13285|[23.0,4.0,2709.12]|[0.06166219839142...|         1|0.023847531248595018|
         * |     13623| [30.0,7.0,672.44]|[0.08042895442359...|         1|0.024049650279212683|
         * |     13832|  [17.0,2.0,40.95]|[0.04557640750670...|         1|0.014321280782467466|
         * |     14450|[180.0,3.0,483.25]|[0.48257372654155...|         0| 0.04870738944845504|
         * +----------+------------------+--------------------+----------+--------------------+
         */
        dataset = dataset.map((MapFunction<Row, Row>) row -> {
            int customerID = row.getInt(0);
            Vector featureVec = (Vector) row.get(1);
            Vector scaledFeatureVec = (Vector) row.get(2);
            int prediction = row.getInt(3);
            double aphScore = 0.0;
            for(int i = 0; i < weightVector.size(); i++) {
                aphScore += weightVector.get(i) * scaledFeatureVec.apply(i);
            }
            return RowFactory.create(customerID, Vectors.dense(new double[]{featureVec.apply(0), featureVec.apply(1), featureVec.apply(2)}), Vectors.dense(new double[]{scaledFeatureVec.apply(0), scaledFeatureVec.apply(1), scaledFeatureVec.apply(2)}), prediction, aphScore);
        }, RowEncoder.apply(new StructType(new StructField[]{
                new StructField("customerid", DataTypes.IntegerType, false, Metadata.empty()),//为行号
                new StructField("features", SQLDataTypes.VectorType(),false, Metadata.empty()),
                new StructField("scaledfeatures", SQLDataTypes.VectorType(), false, Metadata.empty()),
                new StructField("prediction", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("ahpscore", DataTypes.DoubleType, false, Metadata.empty())
        })));

        /**
         * 在同类价值用户中根据ahpscore排序
         * +----------+--------------------+--------------------+----------+------------------+----+
         * |customerid|            features|      scaledfeatures|prediction|          ahpscore|rank|
         * +----------+--------------------+--------------------+----------+------------------+----+
         * |     14646|[1.0,77.0,279489.02]|[0.00268096514745...|         1|0.7306140418787522|   1|
         * |     18102|[0.0,62.0,256438.49]|[0.0,0.2469635627...|         1|0.6609787921304062|   2|
         * |     14911|[1.0,248.0,132572...|[0.00268096514745...|         1|0.5933314030496094|   3|
         * |     17450|[8.0,55.0,187482.17]|[0.02144772117962...|         1|0.4982050472344627|   4|
         * |     14156|[9.0,66.0,113384.14]|[0.02412868632707...|         1|0.3430011157923704|   5|
         * +----------+--------------------+--------------------+----------+------------------+----+
         */
        dataset = dataset.withColumn("rank", functions.rank().over(Window.partitionBy("prediction").orderBy(col("ahpscore").desc())));
        dataset.show(5);
    }

}
