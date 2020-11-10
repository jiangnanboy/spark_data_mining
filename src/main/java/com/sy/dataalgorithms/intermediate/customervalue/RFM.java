package com.sy.dataalgorithms.intermediate.customervalue;

import com.sy.init.InitSchema;
import com.sy.init.InitSpark;
import com.sy.util.PropertiesReader;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.ml.feature.MinMaxScaler;
import org.apache.spark.ml.feature.MinMaxScalerModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.linalg.SQLDataTypes;
import org.apache.spark.sql.*;
import org.apache.spark.sql.catalyst.encoders.RowEncoder;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.sql.Timestamp;
import java.util.List;
import java.util.Random;

import static org.apache.spark.sql.functions.*;


/**
 * RFM是一种用来衡量当前客户价值和潜在客户价值的重要工具和手段
 * 在面向客户制定运营策略、营销策略时，我们希望能够针对不同的客户推行不同的策略，实现精准化运营，以期获取最大的转化率。精准化运营的前提是客户关系管理，而客户关系管理的核心是客户分类。
 * 通过客户分类，对客户群体进行细分，区别出低价值客户、高价值客户，对不同的客户群体开展不同的个性化服务，将有限的资源合理地分配给不同价值的客户，实现效益最大化。
 * 在客户分类中，RFM模型是一个经典的分类模型，模型利用通用交易环节中最核心的三个维度：
 *       1.最近消费(Recency)自上次购买以来的持续时间
 *       2.消费频率(Frequency)购买总数
 *       3.消费金额(Monetary)该客户花费的总金额
 *
 * @Author Shi Yan
 * @Date 2020/11/5 20:01
 */
public class RFM {
    public static void main(String[] args) {
        SparkSession sparkSession = InitSpark.getSparkSession();
        sparkSession.sparkContext().setLogLevel("ERROR");
        Dataset<Row> rfmDataset= rfmStatistics(sparkSession);
        customersCluster(rfmDataset);
        InitSpark.closeSparkSession();
    }

    /**
     * @param session
     */
    public static Dataset<Row> rfmStatistics(SparkSession session) {
        /**
         * 以下是数据集[online_retail.csv]的属性描述(数据集来自https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))：
         *
         * InvoiceNo: Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation.
         * StockCode: Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product.
         * Description: Product (item) name. Nominal.
         * Quantity: The quantities of each product (item) per transaction. Numeric.
         * InvoiceDate: Invice Date and time. Numeric, the day and time when each transaction was generated.
         * UnitPrice: Unit price. Numeric, Product price per unit in sterling.
         * CustomerID: Customer number. Nominal, a 5-digit integral number uniquely assigned to each customer.
         * Country: Country name. Nominal, the name of the country where each customer resides.
         */
        String path = PropertiesReader.get("customer_value_csv");
        Dataset<Row> dataset = session.read()
                .option("sep", ",")
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(path);

        /**统计每列值的数量
         * +----------------+---------------+-----------------+--------------+-----------------+---------------+----------------+-------------+
         * |InvoiceNo_count |StockCode_count|Description_count|Quantity_count|InvoiceDate_count|UnitPrice_count|CustomerID_count|Country_count|
         * +----------------+---------------+-----------------+--------------+-----------------+---------------+----------------+-------------+
         * |          541909|         541909|           540455|        541909|           541909|         541909|          406829|       541909|
         * +----------------+---------------+-----------------+--------------+-----------------+---------------+----------------+-------------+
         */
        //datasetColumnCount(dataset);

        //以上统计发现CustomerID有空值，进行过滤去除含有null/NAN的行,“any”为只要有缺失值就删除这一行
        dataset = dataset.na().drop("any");

        /**过滤空值后，统计每列值的数量
         * +----------------+---------------+-----------------+--------------+-----------------+---------------+----------------+-------------+
         * |InvoiceNo_count |StockCode_count|Description_count|Quantity_count|InvoiceDate_count|UnitPrice_count|CustomerID_count|Country_count|
         * +----------------+---------------+-----------------+--------------+-----------------+---------------+----------------+-------------+
         * |          406829|         406829|           406829|        406829|           406829|         406829|          406829|       406829|
         * +----------------+---------------+-----------------+--------------+-----------------+---------------+----------------+-------------+
         */
        //datasetColumnCount(dataset);

        //将InvoiceDate列中的非标准日期转为标准格式
        dataset = dataset.map((MapFunction<Row, Row>) row -> {

                StringBuilder sb = new StringBuilder();
                String invoiceDate = row.getString(4);
                String[] tokens = invoiceDate.trim().split("\\s+"); // 2011/7/27 15:12

                // year/month/day
                String yearMonthDay = tokens[0];
                String[] yMD = yearMonthDay.split("\\/");
                sb.append(yMD[0]).append("/");
                if(1 == yMD[1].length()) {
                    sb.append("0").append(yMD[1]).append("/");
                } else {
                    sb.append(yMD[1]).append("/");
                }
                if(1 == yMD[2].length()) {
                    sb.append("0").append(yMD[2]).append(" ");
                } else {
                    sb.append(yMD[2]).append(" ");
                }

                // hour/min
                String hourMin = tokens[1];
                String[] hm = hourMin.split(":");
                if(1 == hm[0].length()) {
                    sb.append("0").append(hm[0]).append(":");
                } else {
                    sb.append(hm[0]).append(":");
                }
                if(1 == hm[1].length()) {
                    sb.append("0").append(hm[1]);
                } else {
                    sb.append(hm[1]);
                }
                return RowFactory.create(row.getString(0), row.getString(1), row.getString(2), row.getInt(3), sb.toString(), row.getDouble(5), row.getInt(6), row.getString(7));

        }, RowEncoder.apply(InitSchema.initOnlineRetailSchema()));

        /**
         * 将InvoiceDate列转为时间戳，新增一列时间戳NewInvoiceDate
         * +---------+---------+--------------------+--------+----------------+---------+----------+--------------+-------------------+
         * |InvoiceNo|StockCode|         Description|Quantity|     InvoiceDate|UnitPrice|CustomerID|       Country|     NewInvoiceDate|
         * +---------+---------+--------------------+--------+----------------+---------+----------+--------------+-------------------+
         * |   536365|   85123A|WHITE HANGING HEA...|       6|2010/12/01 08:26|     2.55|     17850|United Kingdom|2010-12-01 08:26:00|
         * |   536365|    71053| WHITE METAL LANTERN|       6|2010/12/01 08:26|     3.39|     17850|United Kingdom|2010-12-01 08:26:00|
         * |   536365|   84406B|CREAM CUPID HEART...|       8|2010/12/01 08:26|     2.75|     17850|United Kingdom|2010-12-01 08:26:00|
         * |   536365|   84029G|KNITTED UNION FLA...|       6|2010/12/01 08:26|     3.39|     17850|United Kingdom|2010-12-01 08:26:00|
         * |   536365|   84029E|RED WOOLLY HOTTIE...|       6|2010/12/01 08:26|     3.39|     17850|United Kingdom|2010-12-01 08:26:00|
         * +---------+---------+--------------------+--------+----------------+---------+----------+--------------+-------------------+
         */
        //dataset = dataset.withColumn("NewInvoiceDate",functions.to_utc_timestamp(functions.unix_timestamp(col("InvoiceDate"), "yyyy/MM/dd HH:mm").cast("timestamp"), "UTC"));
        dataset = dataset.withColumn("NewInvoiceDate", functions.unix_timestamp(col("InvoiceDate"),"yyyy/MM/dd HH:mm").cast(DataTypes.TimestampType));

        /**
         * 计算总额：=》 Quantity*UnitPrice作为新列TotalPrice
         * +---------+---------+--------------------+--------+----------------+---------+----------+--------------+-------------------+----------+
         * |InvoiceNo|StockCode|         Description|Quantity|     InvoiceDate|UnitPrice|CustomerID|       Country|     NewInvoiceDate|TotalPrice|
         * +---------+---------+--------------------+--------+----------------+---------+----------+--------------+-------------------+----------+
         * |   536365|   85123A|WHITE HANGING HEA...|       6|2010/12/01 08:26|     2.55|     17850|United Kingdom|2010-12-01 08:26:00|      15.3|
         * |   536365|    71053| WHITE METAL LANTERN|       6|2010/12/01 08:26|     3.39|     17850|United Kingdom|2010-12-01 08:26:00|     20.34|
         * |   536365|   84406B|CREAM CUPID HEART...|       8|2010/12/01 08:26|     2.75|     17850|United Kingdom|2010-12-01 08:26:00|      22.0|
         * |   536365|   84029G|KNITTED UNION FLA...|       6|2010/12/01 08:26|     3.39|     17850|United Kingdom|2010-12-01 08:26:00|     20.34|
         * |   536365|   84029E|RED WOOLLY HOTTIE...|       6|2010/12/01 08:26|     3.39|     17850|United Kingdom|2010-12-01 08:26:00|     20.34|
         * +---------+---------+--------------------+--------+----------------+---------+----------+--------------+-------------------+----------+
         */
        dataset = dataset.withColumn("TotalPrice", functions.round(col("Quantity").multiply(col("UnitPrice")), 2));

        //获取NewInvoiceDate列中最大时间戳
        List<Row> maxInvoiceRow = dataset.select(functions.max(col("NewInvoiceDate")).as("MaxInvoiceDate")).collectAsList();
        Timestamp maxTimeStamp = maxInvoiceRow.get(0).getTimestamp(0);

        /**
         * 计算时间差=NewInvoiceDate列中最大时间 - 每列的时间，新增时间差列Duration为相差天数
         * +---------+---------+--------------------+--------+----------------+---------+----------+--------------+-------------------+----------+--------+
         * |InvoiceNo|StockCode|         Description|Quantity|     InvoiceDate|UnitPrice|CustomerID|       Country|     NewInvoiceDate|TotalPrice|Duration|
         * +---------+---------+--------------------+--------+----------------+---------+----------+--------------+-------------------+----------+--------+
         * |   536365|   85123A|WHITE HANGING HEA...|       6|2010/12/01 08:26|     2.55|     17850|United Kingdom|2010-12-01 08:26:00|      15.3|     373|
         * |   536365|    71053| WHITE METAL LANTERN|       6|2010/12/01 08:26|     3.39|     17850|United Kingdom|2010-12-01 08:26:00|     20.34|     373|
         * |   536365|   84406B|CREAM CUPID HEART...|       8|2010/12/01 08:26|     2.75|     17850|United Kingdom|2010-12-01 08:26:00|      22.0|     373|
         * |   536365|   84029G|KNITTED UNION FLA...|       6|2010/12/01 08:26|     3.39|     17850|United Kingdom|2010-12-01 08:26:00|     20.34|     373|
         * |   536365|   84029E|RED WOOLLY HOTTIE...|       6|2010/12/01 08:26|     3.39|     17850|United Kingdom|2010-12-01 08:26:00|     20.34|     373|
         * +---------+---------+--------------------+--------+----------------+---------+----------+--------------+-------------------+----------+--------+
         */
        dataset = dataset.withColumn("Duration", functions.datediff(functions.lit(maxTimeStamp), col("NewInvoiceDate")));

        /**
         * 计算 RFM => Recency, Frequency, Monetary
         *      最近-客户最近购买了多少？即自上次购买以来的持续时间
         *      频率——他们多久购买一次？即购买总数
         *      货币价值——他们花了多少钱？即该客户花费的总金额
         *
         * +----------+-------+
         * |CustomerID|Recency|
         * +----------+-------+
         * |     17420|     50|
         * |     16861|     59|
         * |     16503|    106|
         * |     15727|     16|
         * |     17389|      0|
         * +----------+-------+
         *
         * +----------+---------+
         * |CustomerID|Frequence|
         * +----------+---------+
         * |     15619|        1|
         * |     17389|       43|
         * |     12940|        4|
         * |     13623|        7|
         * |     14450|        3|
         * +----------+---------+
         *
         * +----------+--------+
         * |CustomerID|Monetary|
         * +----------+--------+
         * |     17420|  598.83|
         * |     16861|  151.65|
         * |     16503| 1421.43|
         * |     15727| 5178.96|
         * |     17389|31300.08|
         * +----------+--------+
         *
         */
        Dataset<Row> recencyDataset = dataset.groupBy("CustomerID").agg(functions.min(col("Duration")).as("Recency"));
        Dataset<Row> frequencyDataset = dataset.groupBy("CustomerID", "InvoiceNo").count().groupBy("CustomerID").agg(functions.count("*").as("Frequence"));
        Dataset<Row> monetaryDataset = dataset.groupBy("CustomerID").agg(functions.round(functions.sum("TotalPrice"), 2).as("Monetary"));

        /**
         * 连接recencyDataset、frequencyDataset、monetaryDataset，获得RFM的统计
         * +----------+-------+---------+--------+
         * |CustomerID|Recency|Frequence|Monetary|
         * +----------+-------+---------+--------+
         * |     12940|     46|        4|  876.29|
         * |     13285|     23|        4| 2709.12|
         * |     13623|     30|        7|  672.44|
         * |     13832|     17|        2|   40.95|
         * |     14450|    180|        3|  483.25|
         * +----------+-------+---------+--------+
         */
        Dataset<Row> rfmDataset = recencyDataset.join(frequencyDataset, "CustomerID").join(monetaryDataset, "CustomerID");

        return rfmDataset;
    }

    /**
     * 统计每列下值的数量
     * @param dataset
     */
    public static void datasetColumnCount(Dataset dataset) {
        dataset.agg(functions.count("InvoiceNo").as("InvoiceNo_count"),
                functions.count("StockCode").as("StockCode_count"),
                functions.count("Description").as("Description_count"),
                functions.count("Quantity").as("Quantity_count"),
                functions.count("InvoiceDate").as("InvoiceDate_count"),
                functions.count("UnitPrice").as("UnitPrice_count"),
                functions.count("CustomerID").as("CustomerID_count"),
                functions.count("Country").as("Country_count")).show();
    }

    /**
     * rfmStatistics()是对rfm的统计
     * 接下来需要对统计后的rfm数据进行分隔，以划分和分析不同的客户价值，即对客户分群，参考资料中提出3种方案对客户分群：
     *          1.根据经验，熟悉业务的人进行定义划分标准，需要不断修正
     *          2.统计每列的4分位数，根据分位数进行划分(spark中没有现在的4分位统计函数，可利用python)
     *          3.利用聚类自动划分(r、f、m作为特征，可统计更多特征)
     * 以下利用聚类对客户分类
     */
    public static Dataset<Row> customersCluster(Dataset<Row> rfmDataset) {

        /**
         * 转为以下形式：
         * +----------+------------------+
         * |CustomerID|       features   |
         * +----------+------------------+
         * |     12940| [46.0,4.0,876.29]|
         * |     13285|[23.0,4.0,2709.12]|
         * |     13623| [30.0,7.0,672.44]|
         * |     13832|  [17.0,2.0,40.95]|
         * |     14450|[180.0,3.0,483.25]|
         * +----------+------------------+
         */
        Dataset<Row> transData = rfmDataset.map((MapFunction<Row, Row>) row -> {
            int customerID = row.getInt(0);
            double recency = row.getInt(1);
            double frequence = (double)row.getLong(2);
            double monetary = row.getDouble(3);
            return RowFactory.create(customerID, Vectors.dense(new double[]{recency, frequence, monetary}));
        }, RowEncoder.apply(new StructType(new StructField[]{
                new StructField("CustomerID", DataTypes.IntegerType,false, Metadata.empty()),
                new StructField("features", SQLDataTypes.VectorType(), false, Metadata.empty())
        })));

        /**
         * 对数据标准化
         * +----------+------------------+--------------------+
         * |CustomerID|          features|      scaledFeatures|
         * +----------+------------------+--------------------+
         * |     12940| [46.0,4.0,876.29]|[0.12332439678284...|
         * |     13285|[23.0,4.0,2709.12]|[0.06166219839142...|
         * |     13623| [30.0,7.0,672.44]|[0.08042895442359...|
         * |     13832|  [17.0,2.0,40.95]|[0.04557640750670...|
         * |     14450|[180.0,3.0,483.25]|[0.48257372654155...|
         * +----------+------------------+--------------------+
         */
        MinMaxScalerModel featureScalar = new MinMaxScaler()
                .setInputCol("features")
                .setOutputCol("scaledFeatures")
                .fit(transData);
        Dataset<Row> scaledData = featureScalar.transform(transData);

        /**
         * 使用kmeans怎么确定中心个数：
         *      1.肘部法则：曲线中类似人的手肘处，但并不是所有函数曲线都有明显的“肘关节”
         *      2.轮廓系数[-1,1]：结合类内聚合度以及类间分离度两种指标来计算得到
         *              越接近1， 则说明样本 i 聚类合理。
         *              越接近-1，说明样本 i 更适合聚到其他类
         *              越接近0，则说明样本 i 在两个簇的边界上
         */

        /**
         *利用轮廓系数获取最佳聚类簇数，对数据进行聚类。后面根据聚类结果进行特征分析，完成对客户数据进行分群
         *          * +---+------------------+
         *          * |  k|        silhouette|
         *          * +---+------------------+
         *          * |  2|0.8045154385557953|
         *          * |  3|0.6993528775512052|
         *          * |  4|0.6689286654221447|
         *          * |  5|0.6356184024841809|
         *          * |  6|0.7174102265711756|
         *          * |  7|0.6720861758298997|
         *          * |  8| 0.601771359881241|
         *          * |  9|0.6292447334578428|
         */
        //int k = selectOptimalK(scaledData, 9, 20);

        //model
        KMeans kMeans = new KMeans().setFeaturesCol("scaledFeatures")
                .setK(2) //簇数
                .setSeed(1);
        KMeansModel model = kMeans.fit(scaledData);
        /**
         * 打印结果
         * +----------+------------------+--------------------+----------+
         * |CustomerID|          features|      scaledFeatures|prediction|
         * +----------+------------------+--------------------+----------+
         * |     12940| [46.0,4.0,876.29]|[0.12332439678284...|         1|
         * |     13285|[23.0,4.0,2709.12]|[0.06166219839142...|         1|
         * |     13623| [30.0,7.0,672.44]|[0.08042895442359...|         1|
         * |     13832|  [17.0,2.0,40.95]|[0.04557640750670...|         1|
         * |     14450|[180.0,3.0,483.25]|[0.48257372654155...|         0|
         * +----------+------------------+--------------------+----------+
         */
        Dataset<Row> predictions  = model.transform(scaledData);
        predictions.show(5);

        /**聚类中心
         * 0 => [0.6646001408433307,0.003600813448012091,0.01679317521229563]
         * 1 => [0.10719456205329142,0.020757763684444562,0.023451563060011567]
         */
        Vector[] centers = model.clusterCenters();
        for(Vector center : centers) {
            System.out.println(center);
        }

        /**
         * 通过以上可以看出数据分成了2类，怎么对这2类数据进行细分成 =》 重要客户；一般客户？(当然可以聚多个类，划分更加精细)
         * 一种方法是将聚类中心的三个列求出中位数(均值)，根据每个类的列值和求出中位数(均值)后列的值进行比较，分析出结果
         * 其中：
         *      Recency越小越好
         *      Frequency越大越好
         *      Monetary越大越好
         *
         *      请查看：customer_value_identification.ipynb
         */
        return predictions;
    }

    /**
     * 利用轮廓系数获取最佳聚类簇数
     *
     * @param scaledData
     * @param kClusters
     * @param numRuns
     * @return
     */
    public static int selectOptimalK(Dataset<Row> scaledData, int kClusters, int numRuns) {
        Random random = new Random();
        int optimalK = 2;
        double maxSilhouette = -1;
        for(int k = 2; k <= kClusters; k++) {
            double silhouette = 0.0;
            for(int run = 1; run <= numRuns; run++) {
                //model
                KMeans kMeans = new KMeans()//迭代次数默认为20
                        .setFeaturesCol("scaledFeatures")
                        .setK(k) //簇数
                        .setSeed(random.nextInt(100));
                //训练
                KMeansModel model = kMeans.fit(scaledData);
                //预测
                Dataset<Row> predictions  = model.transform(scaledData);
                //评估，轮廓系数
                ClusteringEvaluator evaluator = new ClusteringEvaluator();
                double sc = evaluator.evaluate(predictions);
                silhouette += sc;
            }
            silhouette /= numRuns;
            if(silhouette > maxSilhouette) {
                maxSilhouette = silhouette;
                optimalK = k;
            }
            System.out.println("k: " + k + " silhouette: " + silhouette);
        }
        return optimalK;
    }

}