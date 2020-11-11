package com.sy.dataalgorithms.intermediate;

import com.sy.init.InitSpark;
import com.sy.util.PropertiesReader;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.sql.*;
import org.apache.spark.sql.catalyst.encoders.RowEncoder;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.util.ArrayList;
import java.util.List;

import static org.apache.spark.sql.functions.col;

/**
 * @Author Shi Yan
 * @Date 2020/11/11 22:11
 */
public class StockRec {
    public static void main(String[] args) {
        SparkSession sparkSession = InitSpark.getSparkSession();
        sparkSession.sparkContext().setLogLevel("ERROR");
        alsRec(sparkSession);
        InitSpark.closeSparkSession();
    }

    public static void alsRec(SparkSession session) {

        /**
         * 以下是数据集[online_retail.csv]的属性描述：
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

        /**数据集schema:
         *  |-- InvoiceNo: string (nullable = true)
         *  |-- StockCode: string (nullable = true)
         *  |-- Description: string (nullable = true)
         *  |-- Quantity: integer (nullable = true)
         *  |-- InvoiceDate: string (nullable = true)
         *  |-- UnitPrice: double (nullable = true)
         *  |-- CustomerID: integer (nullable = true)
         *  |-- Country: string (nullable = true)
         *
         * +---------+---------+--------------------+--------+--------------+---------+----------+--------------+
         * |InvoiceNo|StockCode|         Description|Quantity|   InvoiceDate|UnitPrice|CustomerID|       Country|
         * +---------+---------+--------------------+--------+--------------+---------+----------+--------------+
         * |   536365|   85123A|WHITE HANGING HEA...|       6|2010/12/1 8:26|     2.55|     17850|United Kingdom|
         * |   536365|    71053| WHITE METAL LANTERN|       6|2010/12/1 8:26|     3.39|     17850|United Kingdom|
         * |   536365|   84406B|CREAM CUPID HEART...|       8|2010/12/1 8:26|     2.75|     17850|United Kingdom|
         * |   536365|   84029G|KNITTED UNION FLA...|       6|2010/12/1 8:26|     3.39|     17850|United Kingdom|
         * |   536365|   84029E|RED WOOLLY HOTTIE...|       6|2010/12/1 8:26|     3.39|     17850|United Kingdom|
         * +---------+---------+--------------------+--------+--------------+---------+----------+--------------+
         */
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

        //增加一列Asset=Quantity*UnitPrice
        dataset = dataset.withColumn("Asset", functions.round(col("Quantity").multiply(col("UnitPrice")),2));
        Dataset<Row> filterDataset = dataset.withColumnRenamed("StockCode", "Cusip")
                .select("CustomerID", "Cusip", "Quantity", "UnitPrice", "Asset");
        /**
         * 各列数量：
         * +-----------------+------------+---------------+----------------+------------+
         * |count(CustomerID)|count(Cusip)|count(Quantity)|count(UnitPrice)|count(Asset)|
         * +-----------------+------------+---------------+----------------+------------+
         * |           397924|      397924|         397924|          397924|      397924|
         * +-----------------+------------+---------------+----------------+------------+
         * 显示：
         * +----------+------+--------+---------+-----+
         * |CustomerID| Cusip|Quantity|UnitPrice|Asset|
         * +----------+------+--------+---------+-----+
         * |     17850|85123A|       6|     2.55| 15.3|
         * |     17850| 71053|       6|     3.39|20.34|
         * |     17850|84406B|       8|     2.75| 22.0|
         * |     17850|84029G|       6|     3.39|20.34|
         * |     17850|84029E|       6|     3.39|20.34|
         */
        filterDataset = filterDataset.filter(col("Asset").geq(0));
        //datasetColumnCount2(filterDataset);

        /**
         * 按股票代码Cusip分组，统计topn股票
         * +------+---------+----------+
         * | Cusip|Customers|TotalAsset|
         * +------+---------+----------+
         * |85123A|     2035|  100603.5|
         * | 22423|     1724| 142592.95|
         * |85099B|     1618|  85220.78|
         * | 84879|     1408|  56580.34|
         * | 47566|     1397|  68844.33|
         * +------+---------+----------+
         */
        Dataset<Row> popDataset = filterDataset.groupBy("Cusip")
                .agg(functions.count("CustomerID").as("Customers"), functions.round(functions.sum("Asset"), 2).as("TotalAsset"))
                .sort(functions.col("Customers").desc(), functions.col("TotalAsset").desc());


        //获取topn股票代码列表
        Row[] rows = (Row[])popDataset.select("Cusip").head(10);
        String[] cusipStr = new String[11];
        cusipStr[0] = "CustomerID";
        int index = 1;
        for(Row row:rows) {
            cusipStr[index++] = row.getString(0);
        }

        //用pivot进行行列转换（透视）
        Dataset<Row> pivotDataset = filterDataset.groupBy("CustomerID").pivot("Cusip").sum("Asset");
        pivotDataset = pivotDataset.na().fill(0.0);//空值填充为0

        /**选择列CustomerID及topn股票代码，用户下的每个topn代码的Asset和
         * +----------+------+-----+------+-----+-----+-----+-----+-----+----+-----+
         * |CustomerID|85123A|22423|85099B|84879|47566|20725|22720|20727|POST|23203|
         * +----------+------+-----+------+-----+-----+-----+-----+-----+----+-----+
         * |     16503|   0.0|  0.0|   0.0|  0.0|  0.0|  0.0|  0.0| 33.0| 0.0|  0.0|
         * |     15727| 123.9| 25.5|   0.0|  0.0|  0.0| 33.0| 99.0|  0.0| 0.0|  0.0|
         * |     14570|   0.0|  0.0|   0.0|  0.0|  0.0|  0.0|  0.0|  0.0| 0.0|  0.0|
         * |     14450|   0.0|  0.0|  8.32|  0.0|  0.0|  0.0| 49.5|  0.0| 0.0|  0.0|
         * |     13285|   0.0|12.75|  41.6|  0.0|  0.0| 33.0|  0.0| 33.0| 0.0| 40.3|
         * +----------+------+-----+------+-----+-----+-----+-----+-----+----+-----+
         */
        Dataset<Row> selectedDataset = pivotDataset.select(cusipStr[0],cusipStr[1],cusipStr[2],cusipStr[3],cusipStr[4],cusipStr[5],cusipStr[6],cusipStr[7],cusipStr[8],cusipStr[9],cusipStr[10]);

        /**
         * 每个客户的行向量进行归一化，行向量中的每个属性/行的值和，结果如下：
         *+----------+------------------+-------------------+-------------------+-----+-----+-------------------+------------------+------------------+----+------------------+
         * |CustomerID|            85123A|              22423|             85099B|84879|47566|              20725|             22720|             20727|POST|             23203|
         * +----------+------------------+-------------------+-------------------+-----+-----+-------------------+------------------+------------------+----+------------------+
         * |     16503|               0.0|                0.0|                0.0|  0.0|  0.0|                0.0|               0.0|               1.0| 0.0|               0.0|
         * |     15727|0.4402985074626866|0.09061833688699361|                0.0|  0.0|  0.0|0.11727078891257997|0.3518123667377399|               0.0| 0.0|               0.0|
         * |     14570|               0.0|                0.0|                0.0|  0.0|  0.0|                0.0|               0.0|               0.0| 0.0|               0.0|
         * |     14450|               0.0|                0.0|0.14389484607402284|  0.0|  0.0|                0.0|0.8561051539259772|               0.0| 0.0|               0.0|
         * |     13285|               0.0|  0.105941005400914| 0.3456584960531783|  0.0|  0.0| 0.2742002492729539|               0.0|0.2742002492729539| 0.0|0.3348566680515164|
         * +----------+------------------+-------------------+-------------------+-----+-----+-------------------+------------------+------------------+----+------------------+
         */
        Dataset<Row> ratingDataset = selectedDataset.map((MapFunction<Row,Row>) row ->{
            int customerID = row.getInt(0);
            double rowSum = 0.0;
            for(int i = 1;i < row.size() - 1;i++) {
                rowSum += row.getDouble(i);
            }
            double row1 = row.getDouble(1);double row2 = row.getDouble(2);
            double row3 = row.getDouble(3);double row4 = row.getDouble(4);
            double row5 = row.getDouble(5);double row6 = row.getDouble(6);
            double row7 = row.getDouble(7);double row8 = row.getDouble(8);
            double row9 = row.getDouble(9);double row10 = row.getDouble(10);
            if(rowSum > 0) {
                row1 /= rowSum;row2 /= rowSum;
                row3 /= rowSum;row4 /= rowSum;
                row5 /= rowSum;row6 /= rowSum;
                row7 /= rowSum;row8 /= rowSum;
                row9 /= rowSum;row10 /= rowSum;
            }
            return RowFactory.create(customerID,row1,row2,row3,row4,row5,row6,row7,row8,row9,row10);
        },RowEncoder.apply(selectedDataset.schema()));

        String[] columnName = ratingDataset.columns();//列名

        /**
         * 继续转换，结果如下：
         * +----------+------+------+
         * |CustomerID| Cusip|rating|
         * +----------+------+------+
         * |     16503|85123A|   0.0|
         * |     16503| 22423|   0.0|
         * |     16503|85099B|   0.0|
         * |     16503| 84879|   0.0|
         * |     16503| 47566|   0.0|
         * +----------+------+------+
         */
        JavaRDD<Row> ratingRDD = ratingDataset.toJavaRDD().map(row -> {
            List<Row> rowList = new ArrayList<>();
            int customerID = row.getInt(0);
            for(int i = 1; i < row.size(); i++) {
                rowList.add(RowFactory.create(customerID, columnName[i], row.getDouble(i)));
            }
          return rowList;
        }).flatMap(rows1 -> rows1.iterator());
        Dataset<Row> ratingSet = session.createDataFrame(ratingRDD, new StructType(new StructField[]{
                        new StructField("CustomerID", DataTypes.IntegerType,false, Metadata.empty()),
                        new StructField("Cusip", DataTypes.StringType, false, Metadata.empty()),
                        new StructField("rating", DataTypes.DoubleType, false, Metadata.empty())
                }));

        /**
         * 列Cusip到数字索引，结果如下：
         * +----------+------+------+------------+
         * |CustomerID| Cusip|rating|indexedCusip|
         * +----------+------+------+------------+
         * |     16503|85123A|   0.0|         8.0|
         * |     16503| 22423|   0.0|         2.0|
         * |     16503|85099B|   0.0|         7.0|
         * |     16503| 84879|   0.0|         6.0|
         * |     16503| 47566|   0.0|         5.0|
         * +----------+------+------+------------+
         * only showing top 5 rows
         *
         * root
         *  |-- CustomerID: integer (nullable = false)
         *  |-- Cusip: string (nullable = false)
         *  |-- rating: double (nullable = false)
         *  |-- indexedCusip: double (nullable = false)
         *
         *
         * Process finished with exit code 0
         *
         */
        StringIndexerModel labelIndexer = new StringIndexer()
                .setInputCol("Cusip")
                .setOutputCol("indexedCusip")
                .fit(ratingSet);
        ratingSet = labelIndexer.transform(ratingSet);
        ratingSet.show(5);
        ratingSet.printSchema();

    }

    /**
     * 统计每列下值的数量
     * @param dataset
     */
    public static void datasetColumnCount1(Dataset dataset) {
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
     * 统计每列下值的数量
     * @param dataset
     */
    public static void datasetColumnCount2(Dataset dataset) {
        dataset.agg(functions.count("CustomerID"),
                    functions.count("Cusip"),
                    functions.count("Quantity"),
                    functions.count("UnitPrice"),
                    functions.count("Asset")).show();
    }

}