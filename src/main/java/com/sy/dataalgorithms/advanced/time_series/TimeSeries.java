package com.sy.dataalgorithms.advanced.time_series;

import com.sy.init.InitSpark;
import com.sy.util.PropertiesReader;
import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.sql.*;
import org.apache.spark.sql.catalyst.encoders.RowEncoder;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

/**
 * @author sy
 * @date 2022/8/22 21:59
 */
public class TimeSeries {
    static SparkSession sparkSession;
    static {
        sparkSession = InitSpark.getSparkSession();
        sparkSession.sparkContext().setLogLevel("ERROR");
    }
    public static void main(String...args) {
        analysisData(sparkSession);
        InitSpark.closeSparkSession();
    }

    /**
     * 分析和挖掘数据
     * @param session
     */
    public static void analysisData(SparkSession session) {

        // 一.数据集

        /*  1.这里是历史销量sales_train_validation数据
                    +--------------------+-------------+---------+-------+--------+--------+---+---+---+---+---+---+---+---+-
            |                  id|      item_id|  dept_id| cat_id|store_id|state_id|d_1|d_2|d_3|d_4|d_5|d_6|d_7|d_8|d_9|d_10|...
            +--------------------+-------------+---------+-------+--------+--------+---+---+---+---+---+---+---+---+---+----+
            |HOBBIES_1_001_CA_...|HOBBIES_1_001|HOBBIES_1|HOBBIES|    CA_1|      CA|  0|  0|  0|  0|  0|  0|  0|  0|  0|   0|...
            |HOBBIES_1_002_CA_...|HOBBIES_1_002|HOBBIES_1|HOBBIES|    CA_1|      CA|  0|  0|  0|  0|  0|  0|  0|  0|  0|   0|...
            |HOBBIES_1_003_CA_...|HOBBIES_1_003|HOBBIES_1|HOBBIES|    CA_1|      CA|  0|  0|  0|  0|  0|  0|  0|  0|  0|   0|...
            +--------------------+-------------+---------+-------+--------+--------+---+---+---+---+---+---+---+---+---+----+

         schema:
         |-- id: string (nullable = true)
         |-- item_id: string (nullable = true)
         |-- dept_id: string (nullable = true)
         |-- cat_id: string (nullable = true)
         |-- store_id: string (nullable = true)
         |-- state_id: string (nullable = true)
         |-- d_1: integer (nullable = true)
         |-- d_2: integer (nullable = true)
         |-- d_3: integer (nullable = true)
         |-- d_4: integer (nullable = true)
         |-- ......

         */
        String salesTrainValidationPath = TimeSeries.class.getClassLoader().getResource(PropertiesReader.get("advanced_timeseries_sales_train_validation_csv")).getPath().replaceFirst("/", "");
        Dataset<Row> salesTVDataset = session.read()
                .option("sep", ",")
                .option("header", true)
                .option("inferSchema", true)
                .csv(salesTrainValidationPath);

        /*首先，我们只留下salesTVDataset中的历史特征值，删去其他列。
            +---+---+---+---+---+---+---+---+---+----+
            |d_1|d_2|d_3|d_4|d_5|d_6|d_7|d_8|d_9|d_10|
            +---+---+---+---+---+---+---+---+---+----+
            |  0|  0|  0|  0|  0|  0|  0|  0|  0|   0|...
            |  0|  0|  0|  0|  0|  0|  0|  0|  0|   0|...
            |  0|  0|  0|  0|  0|  0|  0|  0|  0|   0|...
            +---+---+---+---+---+---+---+---+---+----+
         */
        Column[]  columns = new Column[1913];
        int index = 0;
        for(String column : salesTVDataset.columns()) {
            if(column.contains("d_")) {
                columns[index] = functions.col(column);
                index++;
            }
        }
        Dataset<Row> xDataset = salesTVDataset.select(columns);

        /* 2.这里是日历信息calendar数据
                        +----------+--------+--------+----+-----+----+---+------------+------------+------------+------------+-------+-------+-------+
            |      date|wm_yr_wk| weekday|wday|month|year|  d|event_name_1|event_type_1|event_name_2|event_type_2|snap_CA|snap_TX|snap_WI|
            +----------+--------+--------+----+-----+----+---+------------+------------+------------+------------+-------+-------+-------+
            |2011-01-29|   11101|Saturday|   1|    1|2011|d_1|        null|        null|        null|        null|      0|      0|      0|
            |2011-01-30|   11101|  Sunday|   2|    1|2011|d_2|        null|        null|        null|        null|      0|      0|      0|
            |2011-01-31|   11101|  Monday|   3|    1|2011|d_3|        null|        null|        null|        null|      0|      0|      0|
            +----------+--------+--------+----+-----+----+---+------------+------------+------------+------------+-------+-------+-------+

            schema:
             |-- date: string (nullable = true)
             |-- wm_yr_wk: integer (nullable = true)
             |-- weekday: string (nullable = true)
             |-- wday: integer (nullable = true)
             |-- month: integer (nullable = true)
             |-- year: integer (nullable = true)
             |-- d: string (nullable = true)
             |-- event_name_1: string (nullable = true)
             |-- event_type_1: string (nullable = true)
             |-- event_name_2: string (nullable = true)
             |-- event_type_2: string (nullable = true)
             |-- snap_CA: integer (nullable = true)
             |-- snap_TX: integer (nullable = true)
             |-- snap_WI: integer (nullable = true)
         */
        String calendarPath = TimeSeries.class.getClassLoader().getResource(PropertiesReader.get("advanced_timeseries_calendar_csv")).getPath().replaceFirst("/", "");
        Dataset<Row> calendarDataset = session.read()
                .option("sep", ",")
                .option("header", true)
                .option("inferSchema", true)
                .csv(calendarPath);

        /* 3.商品每周的价格信息sell_prices
            +--------+-------------+--------+----------+
            |store_id|      item_id|wm_yr_wk|sell_price|
            +--------+-------------+--------+----------+
            |    CA_1|HOBBIES_1_001|   11325|      9.58|
            |    CA_1|HOBBIES_1_001|   11326|      9.58|
            |    CA_1|HOBBIES_1_001|   11327|      8.26|
            +--------+-------------+--------+----------+

            schema:
             |-- store_id: string (nullable = true)
             |-- item_id: string (nullable = true)
             |-- wm_yr_wk: integer (nullable = true)
             |-- sell_price: double (nullable = true)
         */
//        String sellPricesPath = TimeSeries.class.getClassLoader().getResource(PropertiesReader.get("advanced_timeseries_sell_prices_csv")).getPath().replaceFirst("/", "");
//        Dataset<Row> sellPricesDataset = session.read()
//                .option("sep", ",")
//                .option("header", true)
//                .option("inferSchema", true)
//                .csv(sellPricesPath);

        // (1).测试集,我们只是计算了第1914天的数据的特征。这只些特征只能用来预测1914天的销量，也就是说，实际上是我们的测试数据。
        int targetDay = 1914;
        // 使用历史数据中最后的7天构造特征
        int localRange = 7;
        // 由于使用前1913天的数据预测第1914天，历史数据与预测目标的距离只有1天，因此predictDistance=1
        // 如果使用前1913天的数据预测第1915天，则历史数据与预测目标的距离有2天，因此predictDistance=2，以此类推
        int predictDistance = 1;

        Dataset<Row> testDataset = getTestDataset(salesTVDataset, calendarDataset, xDataset, targetDay, predictDistance);

        // (2).训练集,为了构造训练数据，我们对1914天之前的日期进行同样的特征计算操作，并附上它们的当天销量作为数据标签。
        int trainingDataDays = 7; // 为了简便，现只取7天的数据作训练集
        Dataset<Row> trainDataset = getTrainDataset(salesTVDataset, calendarDataset, xDataset, trainingDataDays, targetDay, predictDistance);

        String salesTrainEvaluationPath = TimeSeries.class.getClassLoader().getResource(PropertiesReader.get("advanced_timeseries__sales_train_evaluation_csv")).getPath().replaceFirst("/", "");
        Dataset<Row> labelDataset = session.read()
                .option("sep", ",")
                .option("header", true)
                .option("inferSchema", true)
                .csv(salesTrainEvaluationPath);

        // (3).测试集的label
        Dataset<Row> testLabelDataset = getTestDatasetLabel(labelDataset, targetDay);
        // (4).训练集的label
        Dataset<Row> trainLabelDataset = getTrainDatasetLabel(labelDataset, targetDay, trainingDataDays, predictDistance);

        // (5).保存为csv文件，供python lightgbm训练
        // 保存test dataset
        String testDatasetCsvPath = "E:\\idea_project\\spark_data_mining\\src\\main\\resources\\dataalgorithms\\advanced\\timeseries_data\\testdata.csv";
        saveCsv(testDataset, testDataset.columns(), testDatasetCsvPath);

        // 保存train dataset
        String trainDatasetCsvPath = "E:\\idea_project\\spark_data_mining\\src\\main\\resources\\dataalgorithms\\advanced\\timeseries_data\\traindata.csv";
        saveCsv(trainDataset, trainDataset.columns(), trainDatasetCsvPath);

        // 保存test label
        String testLabelCsvPath = "E:\\idea_project\\spark_data_mining\\src\\main\\resources\\dataalgorithms\\advanced\\timeseries_data\\testlabel.csv";
        saveCsv(testLabelDataset, testLabelDataset.columns(), testLabelCsvPath);

        // 保存train label
        String trainLabelCsvPath = "E:\\idea_project\\spark_data_mining\\src\\main\\resources\\dataalgorithms\\advanced\\timeseries_data\\trainlabel.csv";
        saveCsv(trainLabelDataset, trainLabelDataset.columns(), trainLabelCsvPath);
    }

    /**
     * 一行一行保存为csv格式
     * @param dataset
     * @param columnName
     * @param csvPath
     */
    private static void saveCsv(Dataset<Row> dataset, String[] columnName, String csvPath) {
        try(BufferedWriter bw = Files.newBufferedWriter(Paths.get(csvPath), StandardCharsets.UTF_8)) {
            StringBuilder colunmSB = new StringBuilder();
            for(int i=0;i<columnName.length - 1;i++) {
                colunmSB.append(columnName[i]).append(",");
            }
            colunmSB.append(columnName[columnName.length - 1]);
            bw.write(colunmSB.toString());
            bw.newLine();
           List<Row> rowList = dataset.collectAsList();
           for(Row row : rowList) {
               StringBuilder sb = new StringBuilder();
               for(int i=0;i<columnName.length - 1;i++) {
                   sb.append(row.get(i)).append(",");
               }
               sb.append(row.get(columnName.length - 1));
               bw.write(sb.toString());
               bw.newLine();
           }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * 得到所有features
     * @param salesTVDataset
     * @param calendarDataset
     * @param xDataset
     * @param targetDay
     * @param predictDistance
     * @return
     */
    private static Dataset<Row> getAllFeatures(Dataset<Row> salesTVDataset, Dataset<Row> calendarDataset, Dataset<Row> xDataset, int targetDay, int predictDistance) {
        // 1.构建延迟特征
        Dataset<Row> historyFeatures = getDelayFeatures(xDataset, targetDay, predictDistance);

        // 2.构建日历特征
        Dataset<Row> calendarFeatures = getCalendarFeatures(salesTVDataset, calendarDataset, xDataset, targetDay, predictDistance - 1);

        // 3.将延迟特征和日历特征进行合并
        Dataset<Row> features = historyFeatures.join(calendarFeatures, "rn");
        features = features.sort(functions.col("rn"));
        return features;
    }
    /**
     * 测试集,我们只是计算了第1914天的数据的特征。这只些特征只能用来预测1914天的销量，也就是说，实际上是我们的测试数据。
     * @param salesTVDataset
     * @param calendarDataset
     * @param xDataset
     * @param targetDay
     * @param predictDistance
     * @return
     */
    private static Dataset<Row> getTestDataset(Dataset<Row> salesTVDataset, Dataset<Row> calendarDataset, Dataset<Row> xDataset, int targetDay, int predictDistance) {
        /*
        +---+----+----+----+----+----+----+----+-----+-----+-----+-----+-----+----+----+----+----+----+----+----+----+---------------+-----------------+-----------------+-----------------+-----------------+----------+
        | rn|la_1|la_2|la_3|la_4|la_5|la_6|la_7|la_14|la_21|la_28|la_42|la_56| p_1| p_2| p_3| p_4| p_5| p_6| p_7| p_8|same_month_mean|event_name_1_mean|event_name_2_mean|event_type_1_mean|event_type_2_mean| snap_mean|
        +---+----+----+----+----+----+----+----+-----+-----+-----+-----+-----+----+----+----+----+----+----+----+----+---------------+-----------------+-----------------+-----------------+-----------------+----------+
        |  1| 1.0| 2.0| 2.0| 5.0| 6.0| 7.0| 8.0| 13.0| 25.0| 27.0| 45.0| 55.0| 1.0| 1.0| 1.0| 2.0| 3.0| 3.0| 3.0| 3.0|      0.3218391|              NaN|              NaN|              NaN|              NaN|0.31644583|
        |  2| 0.0| 0.0| 0.0| 0.0| 1.0| 1.0| 1.0|  1.0|  1.0|  2.0|  6.0| 11.0| 0.0| 0.0| 0.0| 1.0| 1.0| 1.0| 2.0| 2.0|     0.11494253|              NaN|              NaN|              NaN|              NaN| 0.2548714|
        |  3| 1.0| 2.0| 3.0| 3.0| 4.0| 5.0| 6.0| 14.0| 15.0| 16.0| 19.0| 31.0| 1.0| 1.0| 1.0| 1.0| 1.0| 1.0| 2.0| 2.0|     0.16091955|              NaN|              NaN|              NaN|              NaN|0.15354638|
        |  4| 2.0| 9.0|12.0|13.0|13.0|14.0|18.0| 29.0| 45.0| 51.0| 90.0|104.0| 4.0| 6.0| 7.0| 7.0|11.0|14.0|17.0|17.0|      1.7183908|              NaN|              NaN|              NaN|              NaN| 1.7139517|
        |  5| 4.0| 6.0| 8.0|10.0|11.0|12.0|12.0| 20.0| 24.0| 38.0| 55.0| 69.0| 0.0| 2.0| 2.0| 3.0| 5.0| 7.0| 9.0|10.0|      0.6666667|              NaN|              NaN|              NaN|              NaN|0.99610287|
        |  6| 0.0| 0.0| 2.0| 2.0| 2.0| 2.0| 3.0|  9.0| 13.0| 17.0| 33.0| 48.0| 1.0| 2.0| 2.0| 2.0| 2.0| 2.0| 2.0| 2.0|      0.6896552|              NaN|              NaN|              NaN|              NaN| 0.8363211|
        |  7| 1.0| 2.0| 2.0| 2.0| 3.0| 3.0| 4.0|  5.0|  7.0| 11.0| 16.0| 23.0| 1.0| 1.0| 1.0| 2.0| 3.0| 3.0| 3.0| 4.0|     0.15517241|              NaN|              NaN|              NaN|              NaN|0.22837101|
        |  8| 1.0| 3.0| 6.0|12.0|16.0|19.0|56.0| 76.0|138.0|228.0|289.0|370.0|37.0|42.0|44.0|44.0|46.0|49.0|49.0|49.0|      7.5747128|              NaN|              NaN|              NaN|              NaN|  7.090413|
        |  9| 0.0| 0.0| 0.0| 0.0| 0.0| 6.0| 7.0| 15.0| 17.0| 25.0| 31.0| 34.0| 1.0| 1.0| 1.0| 1.0| 1.0| 7.0| 8.0| 8.0|       1.362069|              NaN|              NaN|              NaN|              NaN| 1.1855028|
        | 10| 2.0| 2.0| 4.0| 4.0| 4.0| 4.0| 4.0|  6.0| 12.0| 19.0| 25.0| 30.0| 0.0| 1.0| 3.0| 3.0| 3.0| 4.0| 4.0| 5.0|      0.7241379|              NaN|              NaN|              NaN|              NaN| 0.7077163|
        +---+----+----+----+----+----+----+----+-----+-----+-----+-----+-----+----+----+----+----+----+----+----+----+---------------+-----------------+-----------------+-----------------+-----------------+----------+

        schema:
         |-- rn: long (nullable = false)
                |-- la_1: float (nullable = false)
                |-- la_2: float (nullable = false)
                |-- la_3: float (nullable = false)
                |-- la_4: float (nullable = false)
                |-- la_5: float (nullable = false)
                |-- la_6: float (nullable = false)
                |-- la_7: float (nullable = false)
                |-- la_14: float (nullable = false)
                |-- la_21: float (nullable = false)
                |-- la_28: float (nullable = false)
                |-- la_42: float (nullable = false)
                |-- la_56: float (nullable = false)
                |-- p_1: float (nullable = false)
                |-- p_2: float (nullable = false)
                |-- p_3: float (nullable = false)
                |-- p_4: float (nullable = false)
                |-- p_5: float (nullable = false)
                |-- p_6: float (nullable = false)
                |-- p_7: float (nullable = false)
                |-- p_8: float (nullable = false)
                |-- same_month_mean: float (nullable = false)
                |-- event_name_1_mean: float (nullable = false)
                |-- event_name_2_mean: float (nullable = false)
                |-- event_type_1_mean: float (nullable = false)
                |-- event_type_2_mean: float (nullable = false)
                |-- snap_mean: float (nullable = false)
                */
        Dataset<Row> testDataset = getAllFeatures( salesTVDataset,  calendarDataset,  xDataset, targetDay, predictDistance);
        testDataset = testDataset.drop(functions.col("rn"));
        return testDataset;
    }

    /**
     *
     构造训练数据
     在时间序列的问题中，训练数据的构造一般使用滑窗的方式构造。以本案例为例，训练数据提供了长度为1913的历史序列。
     假设我们建模了一个单步模型，只预测下一天，那么，基于原始数据中的每一条序列，我们最多可以构造出1913条训练样本。但在这里面，时间靠前的那些样本，可以利用的历史信息就很少。因此，我们一般还要预留一个特征窗口，保证每条样本都可以抽取足够的信息。假设我们要保证每条样本至少有56天的历史信息，那么我们就需要从第57天开始滑窗。这一共可以构造出1857条样本。
     如果要需要预测的步长变大，比如预测后天的销量，那可以构造的样本也会变少一条。综上，对于一条时间序列，我们可以构造出
     的样本，其中，H是序列的长度(本案例中为1913)，f是每条样本的最小历史信息（本案例中为56），l是预测的步长（即predict_distance）。
     下面，我们来构造训练数据集。为了节约时间和内存，这里我们只选用第1914的前50天构造训练数据。读者可根据自己的情况选取更多的日期进行构造。
     * @param salesTVDataset
     * @param calendarDataset
     * @param xDataset
     * @param trainingDataDays
     * @param targetDay
     * @param predictDistance
     * @return
     */
    private static Dataset<Row> getTrainDataset(Dataset<Row> salesTVDataset, Dataset<Row> calendarDataset, Dataset<Row> xDataset, int trainingDataDays,
                                                int targetDay, int predictDistance) {
        /*
        day -> 1907
        day -> 1908
        day -> 1909
        day -> 1910
        day -> 1911
        day -> 1912
        day -> 1913
        +----+----+----+----+----+----+----+-----+-----+-----+-----+-----+---+---+---+---+----+----+----+----+---------------+-----------------+-----------------+-----------------+-----------------+----------+
        |la_1|la_2|la_3|la_4|la_5|la_6|la_7|la_14|la_21|la_28|la_42|la_56|p_1|p_2|p_3|p_4| p_5| p_6| p_7| p_8|same_month_mean|event_name_1_mean|event_name_2_mean|event_type_1_mean|event_type_2_mean| snap_mean|
        +----+----+----+----+----+----+----+-----+-----+-----+-----+-----+---+---+---+---+----+----+----+----+---------------+-----------------+-----------------+-----------------+-----------------+----------+
        | 0.0| 3.0| 4.0| 5.0| 5.0| 5.0| 5.0| 17.0| 19.0| 27.0| 42.0| 59.0|0.0|0.0|1.0|2.0| 2.0| 2.0| 2.0| 6.0|     0.28742516|              NaN|              NaN|              NaN|              NaN|0.31191224|
        | 0.0| 0.0| 0.0| 0.0| 0.0| 0.0| 0.0|  0.0|  1.0|  5.0|  8.0| 11.0|0.0|0.0|1.0|1.0| 1.0| 2.0| 2.0| 2.0|     0.11377245|              NaN|              NaN|              NaN|              NaN| 0.2554859|
        | 2.0| 3.0| 5.0| 7.0| 8.0| 8.0| 8.0|  9.0| 10.0| 12.0| 18.0| 25.0|0.0|0.0|0.0|0.0| 0.0| 1.0| 1.0| 1.0|     0.13173653|              NaN|              NaN|              NaN|              NaN|0.14968652|
        | 5.0| 5.0| 6.0| 6.0| 9.0| 9.0|11.0| 27.0| 33.0| 58.0| 79.0|101.0|2.0|3.0|3.0|7.0|10.0|13.0|13.0|15.0|      1.6826347|              NaN|              NaN|              NaN|              NaN| 1.7092477|
        | 1.0| 2.0| 4.0| 5.0| 6.0| 6.0| 8.0| 12.0| 26.0| 32.0| 52.0| 63.0|2.0|2.0|3.0|5.0| 7.0| 9.0|10.0|10.0|      0.6227545|              NaN|              NaN|              NaN|              NaN|  0.992163|
        | 0.0| 1.0| 1.0| 1.0| 1.0| 5.0| 6.0| 10.0| 14.0| 23.0| 38.0| 55.0|1.0|1.0|1.0|1.0| 1.0| 1.0| 1.0| 1.0|      0.7005988|              NaN|              NaN|              NaN|              NaN|  0.838558|
        | 0.0| 0.0| 0.0| 0.0| 0.0| 1.0| 1.0|  3.0|  7.0|  9.0| 16.0| 21.0|0.0|0.0|1.0|2.0| 2.0| 2.0| 3.0| 3.0|     0.13772455|              NaN|              NaN|              NaN|              NaN|0.22648902|
        | 1.0| 1.0| 1.0| 5.0|13.0|15.0|20.0| 82.0|172.0|199.0|314.0|391.0|5.0|7.0|7.0|9.0|12.0|12.0|12.0|46.0|       7.556886|              NaN|              NaN|              NaN|              NaN|  7.085423|
        | 1.0| 1.0| 1.0| 1.0| 1.0| 8.0| 8.0| 10.0| 18.0| 18.0| 25.0| 27.0|0.0|0.0|0.0|0.0| 6.0| 7.0| 7.0| 7.0|      1.3772455|              NaN|              NaN|              NaN|              NaN| 1.1865203|
        | 0.0| 0.0| 1.0| 1.0| 1.0| 1.0| 2.0|  8.0| 15.0| 15.0| 23.0| 31.0|1.0|3.0|3.0|3.0| 4.0| 4.0| 5.0| 5.0|      0.7305389|              NaN|              NaN|              NaN|              NaN|0.70846397|
        +----+----+----+----+----+----+----+-----+-----+-----+-----+-----+---+---+---+---+----+----+----+----+---------------+-----------------+-----------------+-----------------+-----------------+----------+

         */
        Dataset<Row> trainDataset = null;
        for(int day=targetDay-trainingDataDays-predictDistance+1;day<targetDay-predictDistance+1;day++) {
            System.out.println("day -> " + day);
            Dataset<Row> dataset = getAllFeatures(salesTVDataset, calendarDataset, xDataset, day, predictDistance);
            dataset = dataset.drop(functions.col("rn"));
            if(null == trainDataset) {
                trainDataset = dataset;
            } else {
                trainDataset = trainDataset.unionAll(dataset);
            }
        }
        return trainDataset;
    }

    /**
     * 获取test的label
     * @param saleTrainEvaluation
     * @param targetDay
     * @return
     */
    private static Dataset<Row> getTestDatasetLabel(Dataset<Row> saleTrainEvaluation, int targetDay) {
        /*
        +-----+
        |label|
        +-----+
        |    0|
        |    0|
        |    0|
        |    0|
        |    1|
        |    0|
        |    0|
        |   19|
        |    0|
        |    0|
        +-----+

        schema:
         |-- label: integer (nullable = true)
         */
        Dataset<Row> testLabel = saleTrainEvaluation.select(functions.col("d_" + targetDay).as("label"));
        return testLabel;
    }

    /**
     * 获取train的label
     * @param saleTrainEvaluation
     * @param targetDay
     * @param trainingDataDays
     * @param predictDistance
     * @return
     */
    private static Dataset<Row> getTrainDatasetLabel(Dataset<Row> saleTrainEvaluation, int targetDay, int trainingDataDays, int predictDistance) {
        /*
        day -> 1907
        day -> 1908
        day -> 1909
        day -> 1910
        day -> 1911
        day -> 1912
        day -> 1913
        +-----+
        |label|
        +-----+
        |    1|
        |    0|
        |    1|
        |    4|
        |    0|
        |    1|
        |    1|
        |   37|
        |    1|
        |    0|
        +-----+

        root
         |-- label: integer (nullable = true)
         */
        Dataset<Row> trainLabel = null;
        for(int day=targetDay-trainingDataDays-predictDistance+1;day<targetDay-predictDistance+1;day++) {
            System.out.println("day -> " + day);
            Dataset<Row> dataset = saleTrainEvaluation.select(functions.col("d_" + day).as("label"));
            if(null == trainLabel) {
                trainLabel = dataset;
            } else {
                trainLabel = trainLabel.unionAll(dataset);
            }
        }
        return trainLabel;
    }

    /**
     * 时间序列的延迟特征：
     * 首先，这是一个对时间序列的预测任务。在时间序列预测任务中，基于历史信息抽取的特征是很重要的一部分，这些特征又常常被称为延迟特征（Lag Feature)。
     * 延迟特征的抽取需要考虑两个主要因素。首先是局部性，即距离预测目标时间越近的数据影响越大，如预测t_(k+1)的时候，历史序列的值t_k到t_1的影响越来越小。
     * 其次是周期性，大多数实际任务中的时间序列，受到人类活动的影响，基本都会以星期（七天）作为一个周期。
     * 利用这个，我们可以聚合与预测时间点同个周期的历史值，并抽取特征。
     * @param xDataset
     * @return
     */
    private static Dataset<Row> getDelayFeatures(Dataset<Row> xDataset, int targetDay, int predictDistance) {
        /*
            +---+---+---+---+---+---+---+
            |l_1|l_2|l_3|l_4|l_5|l_6|l_7|
            +---+---+---+---+---+---+---+
            |1.0|1.0|0.0|3.0|1.0|1.0|1.0|
            |0.0|0.0|0.0|0.0|1.0|0.0|0.0|
            |1.0|1.0|1.0|0.0|1.0|1.0|1.0|
            +---+---+---+---+---+---+---+

            schema:
             |-- l_1: float (nullable = false)
             |-- l_2: float (nullable = false)
             |-- l_3: float (nullable = false)
             |-- l_4: float (nullable = false)
             |-- l_5: float (nullable = false)
             |-- l_6: float (nullable = false)
             |-- l_7: float (nullable = false)
         */
//        Dataset<Row> localFeatures = getLocalFeatures(xDataset, targetDay, predictDistance, localRange);

        /*
            l_1 表示预测目标前一天的历史值，这里我们抽取了七天的的历史值。 对于历史值的聚合，我们还可以用一个小技巧得到更稳定的特征。
            对于单天的历史值，或多或少都有些随机因素，具有较大的不确定性，例如某天天气不好，销量突然下降。
            实际上，我们可以用连续几天的加和（或均值），用于减缓不确定性带来的影响。
            更具体来说，我们可以用前一天的历史值、前面两天的历史值的和、等等来作为局部特征。
         */

        /*
            +----+----+----+----+----+----+----+
            |la_1|la_2|la_3|la_4|la_5|la_6|la_7|
            +----+----+----+----+----+----+----+
            | 1.0| 2.0| 2.0| 5.0| 6.0| 7.0| 8.0|
            | 0.0| 0.0| 0.0| 0.0| 1.0| 1.0| 1.0|
            | 1.0| 2.0| 3.0| 3.0| 4.0| 5.0| 6.0|
            +----+----+----+----+----+----+----+

            schame:
             |-- la_1: float (nullable = false)
             |-- la_2: float (nullable = false)
             |-- la_3: float (nullable = false)
             |-- la_4: float (nullable = false)
             |-- la_5: float (nullable = false)
             |-- la_6: float (nullable = false)
             |-- la_7: float (nullable = false)
         */
//        Dataset<Row> localAccumulatedFeature = getLocalAccumulatedFeature(xDataset, targetDay, predictDistance, localRange);

        /*
                +----+----+----+----+----+----+----+-----+-----+-----+-----+-----+
                |la_1|la_2|la_3|la_4|la_5|la_6|la_7|la_14|la_21|la_28|la_42|la_56|
                +----+----+----+----+----+----+----+-----+-----+-----+-----+-----+
                | 1.0| 2.0| 2.0| 5.0| 6.0| 7.0| 8.0| 13.0| 25.0| 27.0| 45.0| 55.0|
                | 0.0| 0.0| 0.0| 0.0| 1.0| 1.0| 1.0|  1.0|  1.0|  2.0|  6.0| 11.0|
                | 1.0| 2.0| 3.0| 3.0| 4.0| 5.0| 6.0| 14.0| 15.0| 16.0| 19.0| 31.0|
                +----+----+----+----+----+----+----+-----+-----+-----+-----+-----+

                schame:
                 |-- la_1: float (nullable = false)
                 |-- la_2: float (nullable = false)
                 |-- la_3: float (nullable = false)
                 |-- la_4: float (nullable = false)
                 |-- la_5: float (nullable = false)
                 |-- la_6: float (nullable = false)
                 |-- la_7: float (nullable = false)
                 |-- la_14: float (nullable = false)
                 |-- la_21: float (nullable = false)
                 |-- la_28: float (nullable = false)
                 |-- la_42: float (nullable = false)
                 |-- la_56: float (nullable = false)
         */
        Dataset<Row> longTermFeatures = getAccumulatedFeatures(xDataset, targetDay, predictDistance);
        // 增加行号
        longTermFeatures = zipWithIndex(longTermFeatures, "rn");

        // 2.周期特征
        /*
                +---+---+---+---+---+---+---+---+
                |p_1|p_2|p_3|p_4|p_5|p_6|p_7|p_8|
                +---+---+---+---+---+---+---+---+
                |1.0|0.0|0.0|1.0|1.0|0.0|0.0|0.0|
                |0.0|0.0|0.0|1.0|0.0|0.0|1.0|0.0|
                |1.0|0.0|0.0|0.0|0.0|0.0|1.0|0.0|
                +---+---+---+---+---+---+---+---+

                schame:
                 |-- p_1: float (nullable = false)
                 |-- p_2: float (nullable = false)
                 |-- p_3: float (nullable = false)
                 |-- p_4: float (nullable = false)
                 |-- p_5: float (nullable = false)
                 |-- p_6: float (nullable = false)
                 |-- p_7: float (nullable = false)
                 |-- p_8: float (nullable = false)
         */
        Dataset<Row> periodSale = getPeriodSale(xDataset, targetDay, predictDistance);

        // 使用累计的历史值，来提高稳定性。
        /*
            +---+---+---+---+---+---+---+---+
            |p_1|p_2|p_3|p_4|p_5|p_6|p_7|p_8|
            +---+---+---+---+---+---+---+---+
            |1.0|1.0|1.0|2.0|3.0|3.0|3.0|3.0|
            |0.0|0.0|0.0|1.0|1.0|1.0|2.0|2.0|
            |1.0|1.0|1.0|1.0|1.0|1.0|2.0|2.0|
            +---+---+---+---+---+---+---+---+

            schema:
             |-- p_1: float (nullable = false)
             |-- p_2: float (nullable = false)
             |-- p_3: float (nullable = false)
             |-- p_4: float (nullable = false)
             |-- p_5: float (nullable = false)
             |-- p_6: float (nullable = false)
             |-- p_7: float (nullable = false)
             |-- p_8: float (nullable = false)
         */
        Dataset<Row> txPeriod = getPeriodFeatures(periodSale);
        txPeriod = zipWithIndex(txPeriod, "rn");
        // 综上，以下是基于历史数据构造出的所有特征。整合longTermFeatures与txPeriod
        /*
            +---+----+----+----+----+----+----+----+-----+-----+-----+-----+-----+----+----+----+----+----+----+----+----+
            | rn|la_1|la_2|la_3|la_4|la_5|la_6|la_7|la_14|la_21|la_28|la_42|la_56| p_1| p_2| p_3| p_4| p_5| p_6| p_7| p_8|
            +---+----+----+----+----+----+----+----+-----+-----+-----+-----+-----+----+----+----+----+----+----+----+----+
            |  1| 1.0| 2.0| 2.0| 5.0| 6.0| 7.0| 8.0| 13.0| 25.0| 27.0| 45.0| 55.0| 1.0| 1.0| 1.0| 2.0| 3.0| 3.0| 3.0| 3.0|
            |  2| 0.0| 0.0| 0.0| 0.0| 1.0| 1.0| 1.0|  1.0|  1.0|  2.0|  6.0| 11.0| 0.0| 0.0| 0.0| 1.0| 1.0| 1.0| 2.0| 2.0|
            |  3| 1.0| 2.0| 3.0| 3.0| 4.0| 5.0| 6.0| 14.0| 15.0| 16.0| 19.0| 31.0| 1.0| 1.0| 1.0| 1.0| 1.0| 1.0| 2.0| 2.0|
            |  4| 2.0| 9.0|12.0|13.0|13.0|14.0|18.0| 29.0| 45.0| 51.0| 90.0|104.0| 4.0| 6.0| 7.0| 7.0|11.0|14.0|17.0|17.0|
            |  5| 4.0| 6.0| 8.0|10.0|11.0|12.0|12.0| 20.0| 24.0| 38.0| 55.0| 69.0| 0.0| 2.0| 2.0| 3.0| 5.0| 7.0| 9.0|10.0|
            |  6| 0.0| 0.0| 2.0| 2.0| 2.0| 2.0| 3.0|  9.0| 13.0| 17.0| 33.0| 48.0| 1.0| 2.0| 2.0| 2.0| 2.0| 2.0| 2.0| 2.0|
            |  7| 1.0| 2.0| 2.0| 2.0| 3.0| 3.0| 4.0|  5.0|  7.0| 11.0| 16.0| 23.0| 1.0| 1.0| 1.0| 2.0| 3.0| 3.0| 3.0| 4.0|
            |  8| 1.0| 3.0| 6.0|12.0|16.0|19.0|56.0| 76.0|138.0|228.0|289.0|370.0|37.0|42.0|44.0|44.0|46.0|49.0|49.0|49.0|
            |  9| 0.0| 0.0| 0.0| 0.0| 0.0| 6.0| 7.0| 15.0| 17.0| 25.0| 31.0| 34.0| 1.0| 1.0| 1.0| 1.0| 1.0| 7.0| 8.0| 8.0|
            | 10| 2.0| 2.0| 4.0| 4.0| 4.0| 4.0| 4.0|  6.0| 12.0| 19.0| 25.0| 30.0| 0.0| 1.0| 3.0| 3.0| 3.0| 4.0| 4.0| 5.0|
            +---+----+----+----+----+----+----+----+-----+-----+-----+-----+-----+----+----+----+----+----+----+----+----+

            schema:
             |-- la_1: float (nullable = false)
             |-- la_2: float (nullable = false)
             |-- la_3: float (nullable = false)
             |-- la_4: float (nullable = false)
             |-- la_5: float (nullable = false)
             |-- la_6: float (nullable = false)
             |-- la_7: float (nullable = false)
             |-- la_14: float (nullable = false)
             |-- la_21: float (nullable = false)
             |-- la_28: float (nullable = false)
             |-- la_42: float (nullable = false)
             |-- la_56: float (nullable = false)
             |-- p_1: float (nullable = false)
             |-- p_2: float (nullable = false)
             |-- p_3: float (nullable = false)
             |-- p_4: float (nullable = false)
             |-- p_5: float (nullable = false)
             |-- p_6: float (nullable = false)
             |-- p_7: float (nullable = false)
             |-- p_8: float (nullable = false)

         */
        Dataset<Row> historyFeatures = longTermFeatures.join(txPeriod, "rn");
        historyFeatures = historyFeatures.sort(functions.col("rn"));
        return historyFeatures;
    }

    /**
     * 日历信息构造特征:
     * 接下来，我们考虑如何使用日历信息构造特征。首先，日历信息本身就是周期性的一种表达。
     * 从中我们可以挖掘出更多的周期性特征。例如，某些商品的销量可能随着月份呈现周期性规律
     * @param salesTVDataset
     * @param calendarDataset
     * @param xDataset
     * @param targetDay 现在假设要预测的目标是第1914天的销量d_1914。先抽取局部特征。
     * @param predictDistance
     * @return
     */
    private static Dataset<Row> getCalendarFeatures(Dataset<Row> salesTVDataset, Dataset<Row> calendarDataset, Dataset<Row> xDataset, int targetDay, int predictDistance) {
        /*
        商品的销量变化呈现出一定的周期性规律。这提醒我们在预测销量时，可以参考历史数据中与预测目标日期位于同一月份的日期的销量。
        因此，我们挑出历史数据中所有与目标预测日期所在月份相同的日期，并计算他们的销量平均值。
         */
        Dataset<Row> sameMonthMeanFeatures = getSameMonthMeanFeature(calendarDataset, xDataset, targetDay, predictDistance);
        sameMonthMeanFeatures = zipWithIndex(sameMonthMeanFeatures, "rn");
        Dataset<Row> eventFeatures = getEventFeatures(calendarDataset, xDataset, targetDay, predictDistance);
        eventFeatures = zipWithIndex(eventFeatures, "rn");
        Dataset<Row> snapFeatures = getSnapFeature(salesTVDataset, calendarDataset, targetDay, predictDistance);
        snapFeatures = zipWithIndex(snapFeatures, "rn");

        /*
        +---+---------------+-----------------+-----------------+-----------------+-----------------+----------+
        | rn|same_month_mean|event_name_1_mean|event_name_2_mean|event_type_1_mean|event_type_2_mean| snap_mean|
        +---+---------------+-----------------+-----------------+-----------------+-----------------+----------+
        |  1|      0.3218391|              NaN|              NaN|              NaN|              NaN|0.31644583|
        |  2|     0.11494253|              NaN|              NaN|              NaN|              NaN| 0.2548714|
        |  3|     0.16091955|              NaN|              NaN|              NaN|              NaN|0.15354638|
        |  4|      1.7183908|              NaN|              NaN|              NaN|              NaN| 1.7139517|
        |  5|      0.6666667|              NaN|              NaN|              NaN|              NaN|0.99610287|
        |  6|      0.6896552|              NaN|              NaN|              NaN|              NaN| 0.8363211|
        |  7|     0.15517241|              NaN|              NaN|              NaN|              NaN|0.22837101|
        |  8|      7.5747128|              NaN|              NaN|              NaN|              NaN|  7.090413|
        |  9|       1.362069|              NaN|              NaN|              NaN|              NaN| 1.1855028|
        | 10|      0.7241379|              NaN|              NaN|              NaN|              NaN| 0.7077163|
        +---+---------------+-----------------+-----------------+-----------------+-----------------+----------+
         */
        // 合并
        Dataset<Row> calendarFeatures= sameMonthMeanFeatures.join(eventFeatures, "rn").join(snapFeatures, "rn");
        // 按列"rn"排序
        calendarFeatures = calendarFeatures.orderBy(functions.col("rn"));

        return calendarFeatures;
    }

    /**
     * 挑出历史数据中所有与目标预测日期所在月份相同的日期，并计算他们的销量平均值。
     * @param calendarDataset
     * @param xDataset
     * @param targetDay
     * @param predictDistance
     * @return
     */
    private static Dataset<Row> getSameMonthMeanFeature(Dataset<Row> calendarDataset, Dataset<Row> xDataset, int targetDay, int predictDistance) {
        // 增加行号
        calendarDataset = calendarDataset.withColumn("rn", functions.row_number().over(Window.orderBy(functions.col("date"))));
        // 小于目标日期targetDay的历史数据
        Dataset<Row> calendarHistory = calendarDataset.where(functions.col("rn").lt(targetDay - predictDistance));
        // 目标日期tagetDay的数据
        Dataset<Row> targetDayDataset = calendarDataset.where(functions.col("rn").equalTo(targetDay));
        // 目标日期targetDay的月份
        int tagetDateMonth = targetDayDataset.head().getAs("month");
        Dataset<Row> sameMonthDateDataset = calendarHistory.where(functions.col("month").equalTo(tagetDateMonth));
        List<Row> rowList = sameMonthDateDataset.select(functions.col("d")).collectAsList();
        Column[] columns = new Column[rowList.size()];
        int index = 0;
        for(Row row : rowList) {
            String dN = row.getAs(0);
            columns[index++] = functions.col(dN);
        }
        /*
        +----+----+----+----+----+----+----+----+----+----+
        |d_63|d_64|d_65|d_66|d_67|d_68|d_69|d_70|d_71|d_72|
        +----+----+----+----+----+----+----+----+----+----+
        |   0|   0|   0|   0|   0|   0|   0|   0|   0|   0|
        |   0|   0|   0|   0|   0|   0|   0|   0|   0|   0|
        |   0|   0|   0|   0|   0|   0|   0|   0|   0|   0|
        |   0|   0|   0|   0|   1|   0|   0|   1|   1|   0|
         */
        Dataset<Row> sameMonthDataset = xDataset.select(columns);

        /*
        +---------------+
        |same_month_mean|
        +---------------+
        |      0.3218391|
        |     0.11494253|
        |     0.16091955|
        ...
        +---------------+
         */
        Dataset<Row> sameMonthMeanDataset = sameMonthDataset.map((MapFunction<Row, Row>) row -> {
            Object[] values = new Object[1];
            int sum=0;
            for(int i=0;i<row.size();i++){
                sum+=row.getInt(i);
            }
            values[0] = (float)sum/row.size();
            return RowFactory.create(values);
        }, RowEncoder.apply(new StructType(new StructField[]{new StructField("same_month_mean", DataTypes.FloatType, false, Metadata.empty())})));

        return sameMonthMeanDataset;
    }

    /**
     * 此外，日历信息还包含了特殊事件。例如，当天是否在NBA总决赛期间、当天是否父亲节、所在州当天是否允许SNAP的食品消费券等等。
     * 这些特殊事件对商品的销量也会有显著影响。例如，我们发现FOODS_3_015_CA_1这一商品在NBA决赛期间平均销量略高于其他日期的平均销量。
     *
     * 故如果目标预测日期发生了某一特殊事件（event_name不为null），则我们计算历史数据中发生相同事件的日期的销量平均值。
     * 如果目标预测日期没有特殊事件，则这些特征记为null。
     * @param calendarDataset
     * @param xDataset
     * @param targetDay
     * @param predictDistance
     * @return
     */
    private static Dataset<Row> getEventFeatures(Dataset<Row> calendarDataset, Dataset<Row> xDataset, int targetDay, int predictDistance) {
        // event_name_1
        Dataset<Row> allEventName1Dataset = calendarDataset.select(functions.col("event_name_1"));
        List<String> allEventName1List = allEventName1Dataset.distinct().collectAsList().stream().filter(row -> StringUtils.isNotBlank(row.getString(0)))
                .map(row -> row.getString(0)).collect(Collectors.toList());
        // event_name_2
        Dataset<Row> allEventName2Dataset = calendarDataset.select(functions.col("event_name_2"));
        List<String> allEventName2List = allEventName2Dataset.distinct().collectAsList().stream().filter(row -> StringUtils.isNotBlank(row.getString(0)))
                .map(row -> row.getString(0)).collect(Collectors.toList());
        // event_type_1
        Dataset<Row> allEventType1Dataset = calendarDataset.select(functions.col("event_type_1"));
        List<String> allEventType1List = allEventType1Dataset.distinct().collectAsList().stream().filter(row -> StringUtils.isNotBlank(row.getString(0)))
                .map(row -> row.getString(0)).collect(Collectors.toList());
        // event_type_2
        Dataset<Row> allEventType2Dataset = calendarDataset.select(functions.col("event_type_2"));
        List<String> allEventType2List = allEventType2Dataset.distinct().collectAsList().stream().filter(row -> StringUtils.isNotBlank(row.getString(0)))
                .map(row -> row.getString(0)).collect(Collectors.toList());
        // all_event_list
        List<String> allEventList = new ArrayList<>();
        allEventList.addAll(allEventName1List);
        allEventList.addAll(allEventName2List);
        allEventList.addAll(allEventType1List);
        allEventList.addAll(allEventType2List);

        // 增加行号
        calendarDataset = calendarDataset.withColumn("rn", functions.row_number().over(Window.orderBy(functions.col("date"))));
        // 小于目标日期targetDay的历史数据
        Dataset<Row> calendarHistory = calendarDataset.where(functions.col("rn").lt(targetDay - predictDistance));

        // 定义structfield
        StructField[] structField = new StructField[allEventList.size()];
        StructType structType = new StructType(structField);
        for(int i=0; i<allEventList.size(); i++) {
            structField[i] = new StructField(allEventList.get(i) + "_mean_price", DataTypes.FloatType, false, Metadata.empty());
        }

        List<Integer> eventNameSizeList = new ArrayList<>();
        List<Column> columnList = new ArrayList<>();
        // 1.event_name_1
        for(int i=0;i<allEventName1List.size();i++) {
            String eventName = allEventName1List.get(i);
            Dataset<Row> equalToEventNameDataset =  calendarHistory.where(functions.col("event_name_1").equalTo(eventName));
            List<Row> eventName1List = equalToEventNameDataset.select(functions.col("d")).collectAsList();
            Column[] columns = new Column[eventName1List.size()];
            int index = 0;
            for(Row row : eventName1List) {
                String dN = row.getAs(0);
                columns[index++] = functions.col(dN);
            }
            eventNameSizeList.add(columns.length);
            columnList.addAll(Arrays.asList(columns));
        }


        // 2.event_name_2
        for(int i=0;i<allEventName2List.size();i++) {
            String eventName = allEventName2List.get(i);
            Dataset<Row> equalToEventNameDataset =  calendarHistory.where(functions.col("event_name_2").equalTo(eventName));
            List<Row> eventName1List = equalToEventNameDataset.select(functions.col("d")).collectAsList();
            Column[] columns = new Column[eventName1List.size()];
            int index = 0;
            for(Row row : eventName1List) {
                String dN = row.getAs(0);
                columns[index++] = functions.col(dN);
            }
            eventNameSizeList.add(columns.length);
            columnList.addAll(Arrays.asList(columns));
        }

        // 3.event_type_1
        for(int i=0;i<allEventType1List.size();i++) {
            String eventName = allEventType1List.get(i);
            Dataset<Row> equalToEventNameDataset =  calendarHistory.where(functions.col("event_type_1").equalTo(eventName));
            List<Row> eventName1List = equalToEventNameDataset.select(functions.col("d")).collectAsList();
            Column[] columns = new Column[eventName1List.size()];
            int index = 0;
            for(Row row : eventName1List) {
                String dN = row.getAs(0);
                columns[index++] = functions.col(dN);
            }
            eventNameSizeList.add(columns.length);
            columnList.addAll(Arrays.asList(columns));
        }

        // 4.event_type_2
        for(int i=0;i<allEventType2List.size();i++) {
            String eventName = allEventType2List.get(i);
            Dataset<Row> equalToEventNameDataset =  calendarHistory.where(functions.col("event_type_2").equalTo(eventName));
            List<Row> eventName1List = equalToEventNameDataset.select(functions.col("d")).collectAsList();
            Column[] columns = new Column[eventName1List.size()];
            int index = 0;
            for(Row row : eventName1List) {
                String dN = row.getAs(0);
                columns[index++] = functions.col(dN);
            }
            eventNameSizeList.add(columns.length);
            columnList.addAll(Arrays.asList(columns));
        }
        Column[] columns = new Column[columnList.size()];
        for(int i=0;i<columnList.size();i++) {
            columns[i] = columnList.get(i);
        }
        Dataset<Row> sameMonthDataset = xDataset.select(columns);

        /*
        +-----------------+----------------------+-----------------------+-----------------------+--------------------------+-----------------------+-------------------------+--------------------+------------------+--------------------+----------------------+--------------------+-----------------------+-------------------------+----------------------------+-------------------------+------------------------+---------------------+------------------------+--------------------+----------------------+------------------------+--------------------+--------------------+------------------------+-----------------------+------------------------------+-------------------+----------------------+--------------------+-----------------+-----------------------+-------------------------+------------------------+-------------------+-------------------+--------------------+-------------------+-------------------+--------------------+
        |Easter_mean_price|MemorialDay_mean_price|Father's day_mean_price|Mother's day_mean_price|IndependenceDay_mean_price|NBAFinalsEnd_mean_price|NBAFinalsStart_mean_price|EidAlAdha_mean_price|NewYear_mean_price|LentWeek2_mean_price|VeteransDay_mean_price|LentStart_mean_price|Chanukah End_mean_price|Ramadan starts_mean_price|OrthodoxChristmas_mean_price|OrthodoxEaster_mean_price|ValentinesDay_mean_price|Pesach End_mean_price|Cinco De Mayo_mean_price|Christmas_mean_price|Eid al-Fitr_mean_price|StPatricksDay_mean_price|SuperBowl_mean_price|Purim End_mean_price|PresidentsDay_mean_price|Thanksgiving_mean_price|MartinLutherKingDay_mean_price|LaborDay_mean_price|ColumbusDay_mean_price|Halloween_mean_price|Easter_mean_price|Father's day_mean_price|OrthodoxEaster_mean_price|Cinco De Mayo_mean_price|National_mean_price|Cultural_mean_price|Religious_mean_price|Sporting_mean_price|Cultural_mean_price|Religious_mean_price|
        +-----------------+----------------------+-----------------------+-----------------------+--------------------------+-----------------------+-------------------------+--------------------+------------------+--------------------+----------------------+--------------------+-----------------------+-------------------------+----------------------------+-------------------------+------------------------+---------------------+------------------------+--------------------+----------------------+------------------------+--------------------+--------------------+------------------------+-----------------------+------------------------------+-------------------+----------------------+--------------------+-----------------+-----------------------+-------------------------+------------------------+-------------------+-------------------+--------------------+-------------------+-------------------+--------------------+
        |              0.2|                   0.0|                   0.25|                    0.2|                       0.6|                    0.4|                      0.2|                 0.6|               0.2|                 0.0|                   1.0|                 0.0|                    0.0|                      0.6|                         0.0|                      0.0|                     0.5|                  0.2|                     0.0|                 0.0|                   0.4|              0.16666667|                 0.0|          0.16666667|                     0.0|                    0.6|                           0.2|                1.4|                   0.4|                 0.0|              0.0|                    2.0|                      0.0|                     0.0|         0.43137255|                0.2|           0.1923077|             0.1875|          0.6666667|                 0.0|
        |              0.6|                   0.2|                   0.25|                    0.2|                       0.6|                    0.2|                      0.0|                 0.2|               0.2|                 0.0|                   0.8|          0.33333334|                    1.4|                      0.4|                         0.0|                     0.25|              0.16666667|                  0.4|                     0.5|                 0.0|                   0.2|                     0.0|                 0.0|          0.16666667|                     0.0|                    0.2|                           0.2|                0.4|                   0.2|                 0.0|              0.0|                    0.0|                      0.0|                     1.0|         0.27450982|         0.22857143|          0.32692307|             0.0625|         0.33333334|                 0.0|
        |              0.0|                   0.6|                    0.0|                    0.0|                       0.0|                    0.0|                      0.0|                 0.4|               0.2|          0.16666667|                   0.0|          0.16666667|                    0.2|                      0.2|                         0.0|                      0.0|              0.16666667|                  0.2|                    0.25|                 0.0|                   0.2|                     0.0|          0.16666667|                 0.0|                     0.0|                    0.2|                           0.0|                0.0|                   0.0|                 0.4|              0.0|                    0.0|                      0.0|                     0.0|         0.09803922|        0.114285715|          0.15384616|             0.0625|                0.0|                 0.0|
        |              4.0|                   1.4|                   2.25|                    2.0|                       1.4|                    1.0|                      1.4|                 1.4|               1.4|           0.6666667|                   1.4|           1.3333334|                    1.8|                      1.6|                         1.8|                     2.75|               0.8333333|                  1.4|                     1.5|                 0.0|                   1.6|               1.3333334|           2.6666667|                 2.5|               1.8333334|                    2.8|                           3.2|                1.8|                   1.4|                 1.0|              0.0|                    3.0|                      4.0|                     1.0|          1.6666666|                1.8|           1.6538461|               1.75|          1.3333334|                 4.0|
        |              0.8|                   0.8|                   0.25|                    0.4|                       1.0|                    0.4|                      0.4|                 1.8|               1.2|                 0.5|                   2.0|           1.1666666|                    1.4|                      1.2|                         1.6|                     0.75|                     0.5|                  0.4|                     0.0|                 0.0|                   1.0|               0.8333333|                 1.0|           0.8333333|               0.6666667|                    0.0|                           1.2|                1.2|                   0.4|                 1.2|              0.0|                    0.0|                      3.0|                     0.0|         0.84313726|                0.6|           1.0576923|              0.625|                0.0|                 3.0|
        |              1.2|                   3.4|                    0.0|                    0.2|                       0.4|                    0.2|                      0.4|                 0.0|               0.0|           0.8333333|                   0.6|                 0.5|                    1.4|                      0.6|                         0.0|                     1.25|                     1.5|                  0.8|                     0.5|                 0.0|                   1.2|                     1.5|          0.33333334|           0.6666667|               0.8333333|                    0.4|                           0.4|                2.0|                   1.8|                 0.2|              0.0|                    1.0|                      0.0|                     4.0|         0.98039216|                0.8|          0.71153843|             0.3125|          1.6666666|                 0.0|
        |              0.0|                   0.0|                   0.25|                    0.4|                       0.0|                    0.4|                      0.0|                 0.2|               0.2|          0.33333334|                   0.6|                 0.0|                    0.0|                      0.4|                         0.4|                      0.0|              0.16666667|                  0.2|                     0.0|                 0.0|                   0.0|                     0.5|          0.16666667|          0.16666667|                     0.0|                    0.0|                           0.2|                0.2|                   0.2|                 0.4|              0.0|                    0.0|                      0.0|                     0.0|         0.13725491|         0.25714287|          0.17307693|             0.1875|                0.0|                 0.0|
        |              4.2|                  10.2|                    3.5|                    4.6|                       6.4|                   10.4|                      3.6|                 6.4|               2.0|           7.6666665|                   4.0|                 5.0|                    4.2|                      6.4|                        20.2|                     4.75|                     9.0|                 11.2|                    13.0|                 0.0|                   4.8|               6.3333335|           3.1666667|           4.3333335|               6.1666665|                    0.2|                          16.0|                2.6|                  11.8|                15.8|              9.0|                    4.0|                      3.0|                     1.0|          5.9411764|           8.028571|           7.4423075|             5.5625|          4.6666665|                 3.0|
        |              1.4|                   0.6|                   0.75|                    0.8|                       0.6|                    0.4|                      0.4|                 1.2|               1.2|                 0.5|                   0.8|           0.8333333|                    2.2|                      1.4|                         1.4|                      3.0|               0.8333333|                  0.2|                    2.25|                 0.0|                   0.4|               1.6666666|           1.3333334|           1.1666666|                     1.0|                    1.6|                           0.8|                0.0|                   0.4|                 1.2|              9.0|                    0.0|                      4.0|                     1.0|          0.7058824|          1.2571429|           1.1730769|               0.75|          3.3333333|                 4.0|
        |              1.2|                   1.0|                   1.25|                    0.2|                       0.6|                    0.6|                      0.6|                 0.6|               1.2|                 0.5|                   0.0|                 0.5|                    0.8|                      1.2|                         0.4|                     0.75|                     0.5|                  0.6|                    0.25|                 0.0|                   1.4|               0.6666667|           1.3333334|                 1.0|               1.3333334|                    0.4|                           0.2|                0.6|                   0.8|                 0.4|              0.0|                    2.0|                      0.0|                     1.0|           0.627451|         0.62857145|           0.7692308|              0.875|                1.0|                 0.0|
        +-----------------+----------------------+-----------------------+-----------------------+--------------------------+-----------------------+-------------------------+--------------------+------------------+--------------------+----------------------+--------------------+-----------------------+-------------------------+----------------------------+-------------------------+------------------------+---------------------+------------------------+--------------------+----------------------+------------------------+--------------------+--------------------+------------------------+-----------------------+------------------------------+-------------------+----------------------+--------------------+-----------------+-----------------------+-------------------------+------------------------+-------------------+-------------------+--------------------+-------------------+-------------------+--------------------+
         */
        Dataset<Row> sameMonthMeanDataset = sameMonthDataset.map((MapFunction<Row, Row>) row -> {
            Object[] values = new Object[eventNameSizeList.size()];
            List<Integer> rowList = new ArrayList<>();
            for(int j=0;j<row.size();j++){
                if(Optional.ofNullable(row.getInt(j)).isPresent()) {
                    rowList.add(row.getInt(j));
                } else {
                    rowList.add(0);
                }
            }
            int startIndex = 0;
            for(int index=0; index< eventNameSizeList.size(); index++) {
                int size = eventNameSizeList.get(index);
                List<Integer> subList = rowList.subList(startIndex, size + startIndex);
                int sum = subList.stream().mapToInt(Integer::intValue).sum();
                values[index] = (float)sum/subList.size();
                startIndex += size;
            }
            return RowFactory.create(values);
        }, RowEncoder.apply(structType));

        // 目标日期的eventname和eventtype
        List<Row> targetDayEventName1List = calendarDataset.where(functions.col("rn").equalTo(targetDay)).select(functions.col("event_name_1")).collectAsList();
        String targetDayEventName1 = targetDayEventName1List.get(0).getString(0);
        List<Row> targetDayEventName2List = calendarDataset.where(functions.col("rn").equalTo(targetDay)).select(functions.col("event_name_2")).collectAsList();
        String targetDayEventName2 = targetDayEventName2List.get(0).getString(0);
        List<Row> targetDayEventType1List = calendarDataset.where(functions.col("rn").equalTo(targetDay)).select(functions.col("event_type_1")).collectAsList();
        String targetDayEventType1 = targetDayEventType1List.get(0).getString(0);
        List<Row> targetDayEventType2List = calendarDataset.where(functions.col("rn").equalTo(targetDay)).select(functions.col("event_type_2")).collectAsList();
        String targetDayEventType2 = targetDayEventType2List.get(0).getString(0);

        /*
        +-----------------+-----------------+-----------------+-----------------+
        |event_name_1_mean|event_name_2_mean|event_type_1_mean|event_type_2_mean|
        +-----------------+-----------------+-----------------+-----------------+
        |              NaN|              NaN|              NaN|              NaN|
        |              NaN|              NaN|              NaN|              NaN|
        |              NaN|              NaN|              NaN|              NaN|
        |              NaN|              NaN|              NaN|              NaN|
        |              NaN|              NaN|              NaN|              NaN|
        |              NaN|              NaN|              NaN|              NaN|
        |              NaN|              NaN|              NaN|              NaN|
        |              NaN|              NaN|              NaN|              NaN|
        |              NaN|              NaN|              NaN|              NaN|
        |              NaN|              NaN|              NaN|              NaN|
        +-----------------+-----------------+-----------------+-----------------+
         */
        Dataset<Row> eventFeature = sameMonthMeanDataset.map((MapFunction<Row, Row>) row -> {
            Object[] values = new Object[4];
            if(StringUtils.isBlank(targetDayEventName1)) {
                values[0] = Float.NaN;
            } else {
                values[0] = row.getAs(targetDayEventName1 + "_mean_price");
            }

            if(StringUtils.isBlank(targetDayEventName2)) {
                values[1] = Float.NaN;
            } else {
                values[1] = row.getAs(targetDayEventName2 + "_mean_price");
            }

            if(StringUtils.isBlank(targetDayEventType1)) {
                values[2] = Float.NaN;
            } else {
                values[2] = row.getAs(targetDayEventType1 + "_mean_price");
            }

            if(StringUtils.isBlank(targetDayEventType2)) {
                values[3] = Float.NaN;
            } else {
                values[3] = row.getAs(targetDayEventType2 + "_mean_price");
            }
            return RowFactory.create(values);
        }, RowEncoder.apply(new StructType(new StructField[]{
                new StructField( "event_name_1_mean", DataTypes.FloatType, false, Metadata.empty()),
                new StructField( "event_name_2_mean", DataTypes.FloatType, false, Metadata.empty()),
                new StructField( "event_type_1_mean", DataTypes.FloatType, false, Metadata.empty()),
                new StructField( "event_type_2_mean", DataTypes.FloatType, false, Metadata.empty())
        })));

        return eventFeature;
    }

    /**
     * 计算历史当中与目标预测日期使用SNAP消费券情况相同的所有日期的销量平均值
     * @param salesTVDataset
     * @param calendarDataset
     * @param targetDay
     * @param predictDistance
     * @return
     */
    private static Dataset<Row> getSnapFeature(Dataset<Row> salesTVDataset, Dataset<Row> calendarDataset, int targetDay, int predictDistance) {
        String[] states = new String[]{"CA", "TX", "WI"};
        // 增加行号
        calendarDataset = calendarDataset.withColumn("rn", functions.row_number().over(Window.orderBy(functions.col("date"))));
        // 小于目标日期targetDay的历史数据
        Dataset<Row> calendarHistory = calendarDataset.where(functions.col("rn").lt(targetDay-predictDistance));

        Map<String, Pair<List<String>, List<String>>> stateMap = new HashMap<>();
        Map<String, Integer> snapStateMap = new HashMap<>();
        for(String state : states) {
            String snapCol = "snap_" + state;

            List<Row> snapDColsList = calendarHistory.where(functions.col(snapCol).equalTo(1)).select(functions.col("d")).collectAsList();
            List<String> snapDCols = new ArrayList<>();
            for(Row row : snapDColsList) {
                snapDCols.add(row.getString(0));
            }

            List<Row> noSnapDColsList = calendarHistory.where(functions.col(snapCol).equalTo(0)).select(functions.col("d")).collectAsList();
            List<String> noSnapDCols = new ArrayList<>();
            for(Row row : noSnapDColsList) {
                noSnapDCols.add(row.getString(0));
            }

            stateMap.put(state, Pair.of(snapDCols, noSnapDCols));

            List<Row> rowList = calendarDataset.where(functions.col("rn").equalTo(targetDay)).select(functions.col(snapCol)).collectAsList();
            snapStateMap.put(state, rowList.get(0).getInt(0));
        }

        /*
        +----------+
        | snap_mean|
        +----------+
        |0.31644583|
        | 0.2548714|
        |0.15354638|
        | 1.7139517|
        |0.99610287|
        | 0.8363211|
        |0.22837101|
        |  7.090413|
        | 1.1855028|
        | 0.7077163|
        +----------+
         */
        Dataset<Row> snap_feature = salesTVDataset.map((MapFunction<Row, Row>) row -> {
            Object[] values = new Object[1];
            if(StringUtils.equals(states[0], row.getString(0).split("_")[3])) {
                Pair<List<String>, List<String>> pair = stateMap.get(states[0]);
                List<String> snapDCols = pair.getLeft();
                List<String> noSnapDCols = pair.getRight();
                if(snapStateMap.get(states[0]) == 1) {
                    int sum = 0;
                    for(String dName : snapDCols) {
                        sum += (int)row.getAs(dName);
                    }
                    values[0] = (float)sum/snapDCols.size();
                } else {
                    int sum = 0;
                    for(String dName : noSnapDCols) {
                        sum += (int)row.getAs(dName);
                    }
                    values[0] = (float)sum/noSnapDCols.size();
                }

            } else if(StringUtils.equals(states[1], row.getString(0).split("_")[3])) {
                Pair<List<String>, List<String>> pair = stateMap.get(states[1]);
                List<String> snapDCols = pair.getLeft();
                List<String> noSnapDCols = pair.getRight();
                if(snapStateMap.get(states[1]) == 1) {
                    int sum = 0;
                    for(String dName : snapDCols) {
                        sum += (int)row.getAs(dName);
                    }
                    values[0] = (float)sum/snapDCols.size();
                } else {
                    int sum = 0;
                    for(String dName : noSnapDCols) {
                        sum += (int)row.getAs(dName);
                    }
                    values[0] = (float)sum/noSnapDCols.size();
                }
            } else if(StringUtils.equals(states[2], row.getString(0).split("_")[3])) {
                Pair<List<String>, List<String>> pair = stateMap.get(states[2]);
                List<String> snapDCols = pair.getLeft();
                List<String> noSnapDCols = pair.getRight();
                if(snapStateMap.get(states[2]) == 1) {
                    int sum = 0;
                    for(String dName : snapDCols) {
                        sum += (int)row.getAs(dName);
                    }
                    values[0] = (float)sum/snapDCols.size();
                } else {
                    int sum = 0;
                    for(String dName : noSnapDCols) {
                        sum += (int)row.getAs(dName);
                    }
                    values[0] = (float)sum/noSnapDCols.size();
                }
            }
            return RowFactory.create(values);
        }, RowEncoder.apply(new StructType(new StructField[]{
                new StructField("snap_mean", DataTypes.FloatType, false, Metadata.empty())
        })));

        return snap_feature;
    }


    /**
     * 使用历史数据中最后的7天构造特征
     * @param xDataset
     * @param targetDay
     * @param predictDistance
     * @param localRange
     * @return
     */
    private static Dataset<Row> getLocalFeatures(Dataset<Row> xDataset, int targetDay, int predictDistance, int localRange) {
        StructField[] structField = new StructField[localRange];
        StructType structType = new StructType(structField);
        for(int i=0; i<localRange; i++) {
            structField[i] = new StructField("l_" + (i + 1), DataTypes.FloatType, false, Metadata.empty());
        }
        Dataset<Row> localFeatures = xDataset.map((MapFunction<Row, Row>) row -> {
            Object[] values = new Object[localRange];
            for(int i=0; i<localRange; i++) {
                int v = row.getAs("d_" + (targetDay - i - predictDistance));
                values[i] = Float.parseFloat(String.valueOf(v));
            }
            return RowFactory.create(values);
        }, RowEncoder.apply(structType));
        return localFeatures;
    }

    /**
     * 距离目标日期的7天内滚动累加值
     * 对于历史值的聚合，我们还可以用一个小技巧得到更稳定的特征。
     * 对于单天的历史值，或多或少都有些随机因素，具有较大的不确定性，例如某天天气不好，销量突然下降。
     * 实际上，我们可以用连续几天的加和（或均值），用于减缓不确定性带来的影响。
     * 更具体来说，我们可以用前一天的历史值、前面两天的历史值的和、等等来作为局部特征。
     * @param xDataset
     * @param targetDay
     * @param predictDistance
     * @param localRange
     * @return
     */
    private static Dataset<Row> getLocalAccumulatedFeature(Dataset<Row> xDataset, int targetDay, int predictDistance, int localRange) {
        StructField[] structField = new StructField[localRange];
        StructType structType = new StructType(structField);
        for(int i=0; i<localRange; i++) {
            structField[i] = new StructField("la_" + (i + 1), DataTypes.FloatType, false, Metadata.empty());
        }
        Dataset<Row> localAccumulatedFeature = xDataset.map((MapFunction<Row, Row>) row -> {
            Object[] values = new Object[localRange];
            int sum =0 ;
            for(int i=0; i<localRange; i++) {
                int v1 = row.getAs("d_" + (targetDay - i - predictDistance));
                sum += v1;
                float value = Float.parseFloat(String.valueOf(sum));
                values[i] = value;
            }
            return RowFactory.create(values);
        }, RowEncoder.apply(structType));
        return localAccumulatedFeature;
    }

    /**
     * 我们从历史序列里的最近的56天，构造出12个特征。
     * @param xDataset
     * @param targetDay
     * @param predictDistance
     * @return
     */
    private static Dataset<Row> getAccumulatedFeatures(Dataset<Row> xDataset, int targetDay, int predictDistance) {
        List<Integer> usedHistoryDistances = Arrays.asList(new Integer[] {1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 42, 56});
        StructField[] structField = new StructField[usedHistoryDistances.size()];
        StructType structType = new StructType(structField);
        int index = 0;
        for(int i : usedHistoryDistances) {
            structField[index] = new StructField("la_" + i, DataTypes.FloatType, false, Metadata.empty());
            index++;
        }
        Dataset<Row> longTermFeatures  = xDataset.map((MapFunction<Row, Row>) row -> {
            Object[] values = new Object[usedHistoryDistances.size()];
            int sum =0 ;
            int valueIndex = 0;
            for(int i=0; i<usedHistoryDistances.get(usedHistoryDistances.size() - 1); i++) {
                int v1 = row.getAs("d_" + (targetDay - i - predictDistance));
                sum += v1;
                if(usedHistoryDistances.contains(i+1)) {
                    float value = Float.parseFloat(String.valueOf(sum));
                    values[valueIndex] = value;
                    valueIndex++;
                }
            }
            return RowFactory.create(values);
        }, RowEncoder.apply(structType));

        return longTermFeatures;
    }

    /**
     * 现在我们来看周期特征。我们主要考虑以星期作为周期。
     * 并且，我们选用56天历史值，也就是过去8周的数据，构造周期特征。
     * 因此，我们先取得和目标预测值同周期的历史数据。
     * @param xDataset
     * @param targetDay
     * @param predictDistance
     * @return
     */
    private static Dataset<Row> getPeriodSale(Dataset<Row> xDataset, int targetDay, int predictDistance) {
        int localRange = 8;
        StructField[] structField = new StructField[localRange];
        StructType structType = new StructType(structField);
        for(int i=0; i<localRange; i++) {
            structField[i] = new StructField("p_" + (i + 1), DataTypes.FloatType, false, Metadata.empty());
        }
        int period = 7;
        int iStart = (predictDistance + period - 1) / period;
        Dataset<Row> peroidSale = xDataset.map((MapFunction<Row, Row>) row -> {
            Object[] values = new Object[localRange];
            for(int i=0; i<localRange; i++) {
                int curDay = targetDay - (i + iStart) * period;
                int v1 = row.getAs("d_" + curDay);
                float value = Float.parseFloat(String.valueOf(v1));
                values[i] = value;
            }
            return RowFactory.create(values);
        }, RowEncoder.apply(structType));
        return peroidSale;
    }

    /**
     * 使用累计的历史值，来提高稳定性
     * @param periodSaleDataset
     * @return
     */
    private static Dataset<Row> getPeriodFeatures(Dataset<Row> periodSaleDataset) {
        int localRange = 8;
        StructField[] structField = new StructField[localRange];
        StructType structType = new StructType(structField);
        for(int i=0; i<localRange; i++) {
            structField[i] = new StructField("p_" + (i + 1), DataTypes.FloatType, false, Metadata.empty());
        }
        Dataset<Row> txPeriod = periodSaleDataset.map((MapFunction<Row, Row>) row -> {
            Object[] values = new Object[localRange];
            float sum =0.0f ;
            for(int i=0; i<localRange; i++) {
                float v1 = row.getFloat(i);
                sum += v1;
                values[i] = sum;
            }
            return RowFactory.create(values);
        }, RowEncoder.apply(structType));
        return txPeriod;
    }

    /**
     * 为dataset增加行号
     * @param df
     * @param indexName
     * @return
     */
    private static Dataset<Row> zipWithIndex(Dataset<Row> df, String indexName) {
        JavaRDD<Row> javaRDD = df.javaRDD().zipWithIndex().map(t -> {
            Row r = t._1;
            long index = t._2 + 1;
            Object[] values = new Object[r.size() + 1];
            for(int i=0;i<r.size();i++) {
                values[i] = r.getFloat(i);
            }
            values[values.length - 1] = index;
            return RowFactory.create(values);
        });
        StructType newSchema = df.schema()
                .add(new StructField(indexName, DataTypes.LongType, false, Metadata.empty()));
        return df.sparkSession().createDataFrame(javaRDD, newSchema);
    }

}

