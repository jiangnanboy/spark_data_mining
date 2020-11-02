package com.sy.dataalgorithms.others.streaming;

import com.sy.init.InitSpark;
import com.sy.util.PropertiesReader;
import org.apache.spark.api.java.function.FilterFunction;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.streaming.OutputMode;
import org.apache.spark.sql.streaming.StreamingQuery;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

import static org.apache.spark.sql.functions.col;

/**
 * @Author Shi Yan
 * @Date 2020/8/27 23:11
 */
public class StreamDemo {
    public static void main(String[] args) throws Exception{
        SparkSession session = InitSpark.getSparkSession();
        session.sparkContext().setLogLevel("ERROR");
        readAndCountData(session, PropertiesReader.get("streaming_csv"));
        InitSpark.closeSparkSession();
    }

    /**
     * 利用windows进行分组统计滑动窗口内的数据行数
     * @param session
     * @param filePath 为监听目录
     */
    public static void readAndCountData(SparkSession session, String filePath) {
        StructType userSchema = new StructType()
                .add("userId", DataTypes.LongType)
                .add("itemId", DataTypes.LongType)
                .add("categoryId", DataTypes.StringType)
                .add("behavior", DataTypes.StringType)
                .add("timestamp", DataTypes.TimestampType);

        Dataset<Row> dataset = session
                .readStream().format("csv")
                .option("sep", ",")
                .option("header", "true")
                .schema(userSchema)
                .load(filePath).repartition(10);

        //滑动窗口内统计行数
        Dataset<Row> windowedCounts = dataset.filter(new FilterFunction<Row>() {
            @Override
            public boolean call(Row row) throws Exception {
                return row.getAs("behavior").equals("pv");
            }
        }).select(col("itemId"), col("timestamp"));

        Dataset<Row> groupDataset = windowedCounts.withWatermark("timestamp", "1 minutes")
                .groupBy(functions.window(windowedCounts.col("timestamp"), "1 minutes", "5 seconds"))
                .count()
                .orderBy("window");

        try {
            //聚合用complete或update
            StreamingQuery query = groupDataset.writeStream()
                    .outputMode(OutputMode.Complete())
                    .format("console")
                    .option("truncate", "false") //记录显示不会被截断
                    .start();
            query.awaitTermination();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * 测试流数据处理
     * @param session
     * @param filePath
     */
    public static void testStream(SparkSession session, String filePath) {

    }


}
