package com.sy.dataalgorithms.basics;

import com.sy.init.InitSpark;
import com.sy.util.PropertiesReader;
import org.apache.spark.api.java.function.FilterFunction;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.sql.*;
import org.apache.spark.sql.catalyst.encoders.RowEncoder;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.storage.StorageLevel;

import java.util.ArrayList;
import java.util.List;

import static org.apache.spark.sql.functions.*;

/**
 *
 * 参考【https://blog.csdn.net/weixin_45758323/article/details/107722810】
 * 数据集【taobao_persona.csv】来自和鲸社区 https://www.kesci.com/home/dataset/5ef7024363975d002c9235d3，
 * 记录了 2014 年 11 月 18 日至 2014 年 12 月 18 日的电商 APP 部分用户行为记录，原始数据共 1048575 条，共有 6 个属性。
 *
 *      属性名	                属性值
 *      user_id	                用户 ID,一个 ID 允许触发多个行为
 *      item_id	                商品 ID
 *      behavior_type	        用户行为类型，包括浏览=1，收藏=2，加入购物车=3，购买=4
 *      user_geohash	        用户地理位置
 *      item_category	        商品所属类目的 ID
 *      time	                用户行为发生的日期和时刻，值域为 2014/11/18-2014/12/18
 *
 * @Author Shi Yan
 * @Date 2020/11/18 21:26
 */
public class TrafficOperationAnalysis {
    public static void main(String[] args) {
        SparkSession sparkSession = InitSpark.getSparkSession();
        sparkSession.sparkContext().setLogLevel("ERROR");
        analysis(sparkSession);
        InitSpark.closeSparkSession();
    }

    /**
     * 1.app流量分析
     * 2.用户行为分析
     * 3.商品分析
     * [4.用户价值分析,5.时间序列分析] -》 这两个暂时未做
     *
     * 从 30 天里 APP 的总浏览量、总独立访客量、访问深度、每天访客量及成交量、每个时段访客数及成交量、各环节的流失率分析近 30 天的 APP 流量，及时发现流量运营异常。
     *
     * 从用户浏览活跃时段、用户购买活跃时段、用户浏览或收藏或加购或购买最多的类目、最近 30 天购买频次、最近 7 天活跃天数、复购率分析用户行为，了解用户喜好，以便进行后续的用户分类和商品推荐。
     *
     * 从商品类目的浏览量、收藏量、加购量、成交量分析不同商品类目的受欢迎程度，定位主要盈利和较为冷门的商品类目。
     *
     * 利用传统的 RFM 模型分析用户价值，由于原始数据缺少用户的消费金额，因此本项目只从 RF 划分用户群体。
     *
     * 利用 ARIMA 模型预测未来七天（2014/12/19-2014/12/25）的访客量，包括平稳性检验、差分法平稳序列、拟合预测。
     *
     * @param session
     */
    public static void analysis(SparkSession session) {
        String path = PropertiesReader.get("taobao_persona_csv");

        /**
         * 加载数据查看schema，样本数量为：23291027
         * +--------+---------+-------------+------------+-------------+-------------+
         * | user_id|  item_id|behavior_type|user_geohash|item_category|         time|
         * +--------+---------+-------------+------------+-------------+-------------+
         * |10001082|285259775|            1|     97lk14c|         4076|2014-12-08 18|
         * |10001082|  4368907|            1|        null|         5503|2014-12-12 12|
         * |10001082|  4368907|            1|        null|         5503|2014-12-12 12|
         * |10001082| 53616768|            1|        null|         9762|2014-12-02 15|
         * |10001082|151466952|            1|        null|         5232|2014-12-12 11|
         * +--------+---------+-------------+------------+-------------+-------------+
         *
         *  |-- user_id: integer (nullable = true)
         *  |-- item_id: integer (nullable = true)
         *  |-- behavior_type: integer (nullable = true)
         *  |-- user_geohash: string (nullable = true)
         *  |-- item_category: integer (nullable = true)
         *  |-- time: string (nullable = true)
         *
         */
        Dataset<Row> dataset = session.read()
                .option("sep", ",")
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(path);

        /**
         * 将time拆为两列date和time
         * +--------+---------+-------------+------------+-------------+----+----------+
         * | user_id|  item_id|behavior_type|user_geohash|item_category|time|      date|
         * +--------+---------+-------------+------------+-------------+----+----------+
         * |10001082|285259775|            1|     97lk14c|         4076|  18|2014-12-08|
         * |10001082|  4368907|            1|        null|         5503|  12|2014-12-12|
         * |10001082|  4368907|            1|        null|         5503|  12|2014-12-12|
         * |10001082| 53616768|            1|        null|         9762|  15|2014-12-02|
         * |10001082|151466952|            1|        null|         5232|  11|2014-12-12|
         * +--------+---------+-------------+------------+-------------+----+----------+
         */
        dataset = dataset.withColumn("date", functions.substring(col("time"),0, 10));
        dataset =dataset.withColumn("time", functions.regexp_replace(col("time"), col("time"), functions.substring(col("time"),12, 2)));

        /**
         * 查看每列缺失值，可以看出user_geohash缺失较多，该列数据暂时用不上，故可删除
         * +-------+-------+-------------+------------+-------------+----+----+
         * |user_id|item_id|behavior_type|user_geohash|item_category|time|date|
         * +-------+-------+-------------+------------+-------------+----+----+
         * |      0|      0|            0|    15911010|            0|   0|   0|
         * +-------+-------+-------------+------------+-------------+----+----+
         */
        String[] columnsName = dataset.columns();
        Column[] columns = new Column[columnsName.length];
        for(int index = 0;index < columnsName.length; index++) {
            columns[index] = functions.count(functions.when(functions.isnull(col(columnsName[index])), columnsName[index])).as(columnsName[index]);
        }
        //dataset.select(columns).show();

        /**删除列user_geohash
         * +--------+---------+-------------+-------------+----+----------+
         * | user_id|  item_id|behavior_type|item_category|time|      date|
         * +--------+---------+-------------+-------------+----+----------+
         * |10001082|285259775|            1|         4076|  18|2014-12-08|
         * |10001082|  4368907|            1|         5503|  12|2014-12-12|
         * |10001082|  4368907|            1|         5503|  12|2014-12-12|
         * |10001082| 53616768|            1|         9762|  15|2014-12-02|
         * |10001082|151466952|            1|         5232|  11|2014-12-12|
         * +--------+---------+-------------+-------------+----+----------+
         *  |-- user_id: integer (nullable = true)
         *  |-- item_id: integer (nullable = true)
         *  |-- behavior_type: integer (nullable = true)
         *  |-- item_category: integer (nullable = true)
         *  |-- time: string (nullable = true)
         *  |-- date: string (nullable = true)
         *
         */
        dataset = dataset.drop("user_geohash");

        /**
         * 增加一列，将一天24小时划分为：'凌晨','上午','中午','下午','晚上'
         *      '凌晨'：0-5
         *      '上午'：5-10
         *      '中午'：10-13
         *      '下午'：13-18
         *      '晚上'：18-24
         *
         * +--------+---------+-------------+-------------+----+----------+----------+
         * | user_id|  item_id|behavior_type|item_category|time|      date|time_slice|
         * +--------+---------+-------------+-------------+----+----------+----------+
         * |10001082|285259775|            1|         4076|  18|2014-12-08|      晚上|
         * |10001082|  4368907|            1|         5503|  12|2014-12-12|      中午|
         * |10001082|  4368907|            1|         5503|  12|2014-12-12|      中午|
         * |10001082| 53616768|            1|         9762|  15|2014-12-02|      下午|
         * |10001082|151466952|            1|         5232|  11|2014-12-12|      中午|
         * +--------+---------+-------------+-------------+----+----------+----------+
         */

        dataset = dataset.map((MapFunction<Row, Row>) row -> {
            String timeSlice;
            int time = Integer.parseInt(row.getString(4).trim());
            if(time >= 0 && time < 5) {
                timeSlice = "凌晨";
            } else if (time >= 5 && time < 10) {
                timeSlice = "上午";
            } else if (time >= 10 && time < 13) {
                timeSlice = "中午";
            } else if (time >= 13 && time < 18) {
                timeSlice = "下午";
            } else {
                timeSlice = "晚上";
            }
            return RowFactory.create(row.getInt(0), row.getInt(1), row.getInt(2), row.getInt(3), row.getString(4), row.getString(5), timeSlice);
        }, RowEncoder.apply(new StructType(new StructField[]{
                new StructField("user_id", DataTypes.IntegerType,false, Metadata.empty()),
                new StructField("item_id", DataTypes.IntegerType,false, Metadata.empty()),
                new StructField("behavior_type", DataTypes.IntegerType,false, Metadata.empty()),
                new StructField("item_category", DataTypes.IntegerType,false, Metadata.empty()),
                new StructField("time", DataTypes.StringType,false, Metadata.empty()),
                new StructField("date", DataTypes.StringType,false, Metadata.empty()),
                new StructField("time_slice", DataTypes.StringType,false, Metadata.empty())
        })));

        //app流量分析析
        //appTrafficAnalysis(dataset);

        //用户行为分析
        //userBehaviorAnalysis(session, dataset);

        //商品分析
        productAnalysis(dataset);
    }

    /**
     * app流量分析
     *      1.pv页面浏览量
     *      2.uv独立访客
     *      3.访问深度
     *      4.每天访客数和成交量
     *      5.不同时段的访客数及成交量
     *      6.流失率
     *
     * @param dataset
     */
    public static void appTrafficAnalysis(Dataset<Row> dataset) {
        /**
         * 1.pv页面浏览量
         * 每次页面被加载，pv增加1，衡量app页面曝光量
         */
        long pv_30 =  dataset.where(col("behavior_type").equalTo(1)).count();

        /**
         * 2.uv独立访客
         * 在指定时间内app的用户数量
         */
        long uv_30 = dataset.select("user_id").distinct().count();

        /**
         * 3.访问深度
         * 人均页面浏览量为，pv/uv，每个用户平均访问了多少个页面，衡量用户访问质量的重要指标。
         * 访问深度越大，意味着每个用户来到对应渠道之后流量页面增多，用户对网站的内容越有兴趣。
         * 但访问深度并不是越高越好，过大的访问深度也可能说明网站的功能引导方面较为薄弱，导致用户在站内迷失方向而无法找到目标内容。
         * 与访问深度相关的流量质量指标还有用户在页面的停留时间，如果该页面较为简单，但用户的停留时间过长，
         * 这可能意味着用户没有注意到页面的关键信息或没有注意到引导按钮，从而降低了该页面在引导方面带来的贡献，并弱化了用户体验。
         */
        float page_depth = (float)pv_30 / uv_30;

        // 21940520 : 20000 : 1097.026
        System.out.println(pv_30 + " : " + uv_30 + " : " + page_depth);

        /**
         * 4.每天访客数和成交量
         *
         * 每天访客数
         * +----------+--------------+
         * |      date|count(user_id)|
         * +----------+--------------+
         * |2014-12-13|         13318|
         * |2014-12-11|         13764|
         * |2014-12-05|         12605|
         * |2014-11-27|         12365|
         * |2014-11-19|         12591|
         * +----------+--------------+
         *
         * 每天成交量
         * +----------+--------------+
         * |      date|count(user_id)|
         * +----------+--------------+
         * |2014-12-13|          6965|
         * |2014-12-11|          6771|
         * |2014-12-05|          6050|
         * |2014-11-27|          6785|
         * |2014-11-19|          6329|
         * +----------+--------------+
         */
        dataset.where(col("behavior_type").equalTo(1)).groupBy("date").agg(countDistinct("user_id")).show(5);
        dataset.where(col("behavior_type").equalTo(4)).groupBy("date").agg(count("user_id")).show(5);


        /**
         * 5.不同时段的访客数及成交量
         *
         * 不同时段访客数
         *+----+--------------+
         * |time|count(user_id)|
         * +----+--------------+
         * |  23|         14859|
         * |  12|         16385|
         * |  11|         16233|
         * |  16|         16205|
         * |  08|         13855|
         * +----+--------------+
         *
         * 不同时段成交量
         * +----+--------------+
         * |time|count(user_id)|
         * +----+--------------+
         * |  23|         12404|
         * |  12|         13228|
         * |  11|         13464|
         * |  16|         13535|
         * |  08|          6575|
         * +----+--------------+
         *
         */
        dataset.where(col("behavior_type").equalTo(1)).groupBy("time").agg(countDistinct("user_id")).show(24);
        dataset.where(col("behavior_type").equalTo(4)).groupBy("time").agg(count("user_id")).show(24);

        /**
         * 6.流失率
         *      (1)按时间分析
         *      (2)按日期分析
         *      (3)按时间段分析
         * 假定该 APP 用户完成一笔购物订单需要经过四个步骤，浏览→收藏→加入购物车→购买，则用户量将随着环节的进行而越来越少。
         * 通过分析本环节与上一环节的访客量之比，可以得知各环节之间的转化率或者流失率，比如有加入购物车行为的访客量 / 有收藏行为的访客量衡量了加入购物车的转化率，
         * 该比率越高，意味着具有购买意向的用户数量越多，产品越吸引人。长期追踪业务的流量转化情况，如若发现异动，则说明购物流程中有环节的漏斗转化出现问题，
         * 通过定位具体的环节可进行原因的追踪。
         *
         * 收藏转化率 = 收藏访客量 / 浏览访客量
         * 加入购物车转化率 = 加入购物车访客量 / 收藏访客量
         * 购买转化率 = 购买访客量 / 加入购物车访客量
         *
         * 收藏流失率 = 1.0 - 收藏转化率
         * 加购流失率 = 1.0 - 加入购物车转化率
         * 购买流失率 = 1.0 - 购买转化率
         *
         * 流失率与转化率对应，其值等于 【1-转化率】，衡量了各个环节转化过程中用户的流失情况。
         */

        /**
         * (1).按时间计算收藏流失率，加购流失率，购买流失率
         *
         * +----+-------------+-----+
         *          * |time|behavior_type|total|
         *          * +----+-------------+-----+
         *          * |  05|            4|  813|
         *          * |  03|            4|  994|
         *          * |  09|            4|10344|
         *          * |  11|            4|13464|
         *          * |  11|            2|20329|
         *          * +----+-------------+-----+
         *          |-- time: string (nullable = true)
         *          *  |-- behavior_type: integer (nullable = true)
         *          *  |-- total: long (nullable = false)
         */
        Dataset<Row> timeChurnRate = dataset.groupBy("time", "behavior_type").agg(count("user_id").as("total"));
        List<Double> timeChurnRateList = calculateChurnRate(timeChurnRate, "time", 01);
        timeChurnRateList.stream().forEach(System.out::println);

        /**
         * (2).按日期计算收藏流失率，加购流失率，购买流失率
         *
         * +----------+-------------+------+
         * |      date|behavior_type| total|
         * +----------+-------------+------+
         * |2014-12-05|            2| 14569|
         * |2014-12-02|            4|  7285|
         * |2014-12-03|            2| 16429|
         * |2014-11-26|            4|  6626|
         * |2014-11-18|            1|644879|
         * +----------+-------------+------+
         *  |-- date: string (nullable = true)
         *  |-- behavior_type: integer (nullable = true)
         *  |-- total: long (nullable = false)
         */
        Dataset<Row> dateChurnRate = dataset.groupBy("date", "behavior_type").agg(count("user_id").as("total"));
        List<Double> dateChurnRateList = calculateChurnRate(dateChurnRate, "date", "2014-11-19");
        dateChurnRateList.stream().forEach(System.out::println);

        /**
         * (3).按时间段计算收藏流失率，加购流失率，购买流失率
         *+----------+-------------+-------+
         * |time_slice|behavior_type|  total|
         * +----------+-------------+-------+
         * |      上午|            4|  23067|
         * |      凌晨|            3|  52899|
         * |      凌晨|            2|  43881|
         * |      中午|            3|  90537|
         * |      下午|            1|5184968|
         * +----------+-------------+-------+
         *  |-- time_slice: string (nullable = false)
         *  |-- behavior_type: integer (nullable = false)
         *  |-- total: long (nullable = false)
         */
        Dataset<Row> timeSliceChurnRate = dataset.groupBy("time_slice", "behavior_type").agg(count("user_id").as("total"));
        List<Double> timeSliceChurnRateList = calculateChurnRate(timeSliceChurnRate, "time_slice", "下午");
        timeSliceChurnRateList.stream().forEach(System.out::println);
    }

    /**
     * 按时间段计算收藏流失率，加购流失率，购买流失率
     * @param dataset
     * @param dateOrTime
     * @return
     */
    public static List<Double> calculateChurnRate(Dataset<Row> dataset, String groupByColName, Object dateOrTime) {
        Dataset<Row> time_1_dataset = dataset.select(groupByColName, "behavior_type", "total").where(col(groupByColName).equalTo(dateOrTime));
        time_1_dataset.show();
        List<Row> time1List = time_1_dataset.collectAsList();
        long browseTotal = 0;
        long collectionTotal = 0;
        long addCartTotal = 0;
        long purchaseTotal = 0;

        for(Row row : time1List) {
            if(1 == row.getInt(1)) {
                browseTotal = row.getLong(2);
            } else if(2 == row.getInt(1)) {
                collectionTotal = row.getLong(2);
            } else if(3 == row.getInt(1)) {
                addCartTotal = row.getLong(2);
            } else {
                purchaseTotal = row.getLong(2);
            }
        }

        double collectionChurnRate = 1.0 - collectionTotal / (double)browseTotal;
        double addCartChurnRate = 1.0 - addCartTotal / (double)collectionTotal;
        double purchaseChurnRate = 1.0 - purchaseTotal / (double)addCartTotal;
        List<Double> churnRateList = new ArrayList<>();
        churnRateList.add(collectionChurnRate);
        churnRateList.add(addCartChurnRate);
        churnRateList.add(purchaseChurnRate);
        return churnRateList;
    }

    /**
     * 用户行为分析
     *      1.用户浏览活跃时段
     *      2.用户购买活跃时段
     *      3.用户浏览最多的类目
     *      4.用户收藏最多的类目
     *      5.用户加购最多的类目
     *      6.用户购买最多的类目
     *      7.最近 30 天购买次数
     *      8.最近 7 天的活跃天数
     *      9.复购率
     * @param dataset
     */
    public static void userBehaviorAnalysis(SparkSession session, Dataset<Row> dataset) {
        dataset.persist(StorageLevel.MEMORY_AND_DISK());
        Dataset<Row> userIDDataset = dataset.select("user_id").distinct(); //用来保存用户相关标签

        /**
         * 1.用户浏览活跃时段
         */

        //每个用户浏览时段
        Dataset<Row> timeBrowse = dataset.where(col("behavior_type").equalTo(1)).groupBy("user_id", "time_slice").agg(count("item_id").as("item_id_count"));
        //每个用户浏览次数最多时段
        Dataset<Row> timeBrowseMax = timeBrowse.groupBy("user_id").agg(functions.max("item_id_count").as("browse_max"));

        /**
         * 对timeBrowse与timeBrowseMax按user_id进行join，保留用户浏览的全部时段和次数最多的时段
         * +-------+----------+-------------+----------+
         * |user_id|time_slice|item_id_count|browse_max|
         * +-------+----------+-------------+----------+
         * | 256830|      晚上|          145|       145|
         * | 256830|      上午|            5|       145|
         * | 256830|      中午|           44|       145|
         * | 256830|      下午|           74|       145|
         * |2418415|      凌晨|           42|       134|
         * +-------+----------+-------------+----------+
         */
        timeBrowse = timeBrowse.join(timeBrowseMax, timeBrowse.col("user_id").equalTo(timeBrowseMax.col("user_id")), "left").select(timeBrowse.col("user_id"), timeBrowse.col("time_slice"), timeBrowse.col("item_id_count"), timeBrowseMax.col("browse_max"));
        /**
         * 每个用户浏览次数最多的时段，如有并列最多的时段，放在一起用逗号隔开
         * +--------+----------+
         * | user_id|time_slice|
         * +--------+----------+
         * |  256830|      晚上|
         * | 2418415|      下午|
         * |10699161|      晚上|
         * |11727915|      晚上|
         * |13600137|      晚上|
         * +--------+----------+
         */
        timeBrowse = timeBrowse.filter((FilterFunction<Row>) row -> row.getLong(2) == row.getLong(3));
        timeBrowse.createOrReplaceTempView("time_browser");
        Dataset<Row> timeBrowseTimeSlice = session.sql("select user_id, collect_list(time_slice) as time_slice from time_browser group by user_id");
        timeBrowseTimeSlice = timeBrowseTimeSlice.map((MapFunction<Row, Row>) row -> {
            StringBuilder sb = new StringBuilder();
            List<String> list = row.getList(1);
            for(int i = 0; i < list.size() - 1; i++) {
                sb.append(list.get(i)).append(",");
            }
            sb.append(list.get(list.size()-1));
            return RowFactory.create(row.getInt(0), sb.toString());
        }, RowEncoder.apply(new StructType(new StructField[]{
                new StructField("user_id", DataTypes.IntegerType,false, Metadata.empty()),
                new StructField("time_slice", DataTypes.StringType,false, Metadata.empty())
        })));

        userIDDataset = userIDDataset.join(timeBrowseTimeSlice, userIDDataset.col("user_id").equalTo(timeBrowseTimeSlice.col("user_id")), "left").select(userIDDataset.col("user_id"), timeBrowseTimeSlice.col("time_slice"));


        /**
         * 2.用户购买活跃时段
         * +--------+----------+
         * | user_id|time_slice|
         * +--------+----------+
         * |  256830|      晚上|
         * | 2418415|      下午|
         * |10699161|      晚上|
         * |11727915|      中午|
         * |13600137|      晚上|
         * +--------+----------+
         */
        //每个用户购买时段
        Dataset<Row> timeBuy = dataset.where(col("behavior_type").equalTo(4)).groupBy("user_id", "time_slice").agg(count("item_id").as("item_id_count"));
        //每个用户购买次数最多时段
        Dataset<Row> timeBuyMax = timeBuy.groupBy("user_id").agg(functions.max("item_id_count").as("buy_max"));
        timeBuy = timeBuy.join(timeBuyMax, timeBuy.col("user_id").equalTo(timeBuyMax.col("user_id")), "left").select(timeBuy.col("user_id"), timeBuy.col("time_slice"), timeBuy.col("item_id_count"), timeBuyMax.col("buy_max"));
        timeBuy = timeBuy.filter((FilterFunction<Row>) row -> row.getLong(2) == row.getLong(3));
        timeBuy.createOrReplaceTempView("time_buy");
        Dataset<Row> timeBuyTimeSlice = session.sql("select user_id, collect_list(time_slice) as time_slice from time_buy group by user_id");
        timeBuyTimeSlice = timeBuyTimeSlice.map((MapFunction<Row, Row>) row -> {
            StringBuilder sb = new StringBuilder();
            List<String> list = row.getList(1);
            for(int i = 0; i < list.size() - 1; i++) {
                sb.append(list.get(i)).append(",");
            }
            sb.append(list.get(list.size()-1));
            return RowFactory.create(row.getInt(0), sb.toString());
        }, RowEncoder.apply(new StructType(new StructField[]{
                new StructField("user_id", DataTypes.IntegerType,false, Metadata.empty()),
                new StructField("time_slice", DataTypes.StringType,false, Metadata.empty())
        })));
        //timeBuyTimeSlice.show(5);
        userIDDataset = userIDDataset.join(timeBuyTimeSlice, userIDDataset.col("user_id").equalTo(timeBuyTimeSlice.col("user_id")), "left").select(userIDDataset.col("user_id"), timeBuyTimeSlice.col("time_slice"));

        /**
         * 3.用户浏览最多的类目
         * +--------+-------------+
         * | user_id|item_category|
         * +--------+-------------+
         * |  256830|         6228|
         * | 2418415|         2993|
         * |10699161|          279|
         * |11727915|         5468|
         * |13600137|         3660|
         * +--------+-------------+
         */
        Dataset<Row> categoryBrowser = dataset.where(col("behavior_type").equalTo(1)).select("user_id", "item_id", "item_category");
        Dataset<Row> categoryMostBrowser = categoryBrowser.groupBy("user_id", "item_category").agg(count("item_id").as("item_category_counts"));
        Dataset<Row> categoryMostBrowserMax = categoryMostBrowser.groupBy("user_id").agg(functions.max("item_category_counts").as("item_category_counts_max"));
        /**
         * +-------+-------------+--------------------+------------------------+
         * |user_id|item_category|item_category_counts|item_category_counts_max|
         * +-------+-------------+--------------------+------------------------+
         * | 256830|        11406|                  12|                      71|
         * | 256830|         9765|                   2|                      71|
         * | 256830|         3064|                   1|                      71|
         * | 256830|        12090|                   6|                      71|
         * | 256830|        12553|                   3|                      71|
         * +-------+-------------+--------------------+------------------------+
         *
         * |-- user_id: integer (nullable = false)
         *  |-- item_category: string (nullable = false)
         *  |-- item_category_counts: long (nullable = false)
         *  |-- item_category_counts_max: long (nullable = true)
         */
        categoryMostBrowser = categoryMostBrowser.join(categoryMostBrowserMax, categoryMostBrowser.col("user_id").equalTo(categoryMostBrowserMax.col("user_id")), "left").select(categoryMostBrowser.col("user_id"), categoryMostBrowser.col("item_category"), categoryMostBrowser.col("item_category_counts"), categoryMostBrowserMax.col("item_category_counts_max"));
        categoryMostBrowser = categoryMostBrowser.select(col("user_id"), col("item_category").cast(DataTypes.StringType),col("item_category_counts"),col("item_category_counts_max"));
        categoryMostBrowser = categoryMostBrowser.filter((FilterFunction<Row>) row -> row.getLong(2) == row.getLong(3));
        categoryMostBrowser.createOrReplaceTempView("category_browser");
        Dataset<Row> categoryBrowserItemCategory = session.sql("select user_id, collect_list(item_category) as item_category from category_browser group by user_id");
        categoryBrowserItemCategory = categoryBrowserItemCategory.map((MapFunction<Row, Row>) row -> {
            StringBuilder sb = new StringBuilder();
            List<String> list = row.getList(1);
            for(int i = 0; i < list.size() - 1; i++) {
                sb.append(list.get(i)).append(",");
            }
            sb.append(list.get(list.size()-1));
            return RowFactory.create(row.getInt(0), sb.toString());
        }, RowEncoder.apply(new StructType(new StructField[]{
                new StructField("user_id", DataTypes.IntegerType,false, Metadata.empty()),
                new StructField("item_category", DataTypes.StringType,false, Metadata.empty())
        })));
        userIDDataset = userIDDataset.join(categoryBrowserItemCategory, userIDDataset.col("user_id").equalTo(categoryBrowserItemCategory.col("user_id")), "left").select(userIDDataset.col("user_id"), categoryBrowserItemCategory.col("item_category"));

        /**
         * 4.用户收藏最多类目
         */
        Dataset<Row> categoryCollect = dataset.where(col("behavior_type").equalTo(2)).select("user_id", "item_id", "item_category");
        Dataset<Row> categoryMostCollect = categoryCollect.groupBy("user_id", "item_category").agg(count("item_id").as("item_category_counts"));
        Dataset<Row> categoryMostCollectMax = categoryMostCollect.groupBy("user_id").agg(functions.max("item_category_counts").as("item_category_counts_max"));
        categoryMostCollect = categoryMostCollect.join(categoryMostCollectMax, categoryMostCollect.col("user_id").equalTo(categoryMostCollectMax.col("user_id")), "left").select(categoryMostCollect.col("user_id"), categoryMostCollect.col("item_category"), categoryMostCollect.col("item_category_counts"), categoryMostCollectMax.col("item_category_counts_max"));
        categoryMostCollect = categoryMostCollect.select(col("user_id"), col("item_category").cast(DataTypes.StringType),col("item_category_counts"),col("item_category_counts_max"));
        categoryMostCollect = categoryMostCollect.filter((FilterFunction<Row>) row -> row.getLong(2) == row.getLong(3));
        categoryMostCollect.createOrReplaceTempView("category_collect");
        Dataset<Row> categoryCollectItemCategory = session.sql("select user_id, collect_list(item_category) as item_category from category_collect group by user_id");
        categoryCollectItemCategory = categoryCollectItemCategory.map((MapFunction<Row, Row>) row -> {
            StringBuilder sb = new StringBuilder();
            List<String> list = row.getList(1);
            for(int i = 0; i < list.size() - 1; i++) {
                sb.append(list.get(i)).append(",");
            }
            sb.append(list.get(list.size()-1));
            return RowFactory.create(row.getInt(0), sb.toString());
        }, RowEncoder.apply(new StructType(new StructField[]{
                new StructField("user_id", DataTypes.IntegerType,false, Metadata.empty()),
                new StructField("item_category", DataTypes.StringType,false, Metadata.empty())
        })));
        userIDDataset = userIDDataset.join(categoryCollectItemCategory, userIDDataset.col("user_id").equalTo(categoryCollectItemCategory.col("user_id")), "left").select(userIDDataset.col("user_id"), categoryCollectItemCategory.col("item_category"));
        /**
         * 5.用户加购最多类目
         */
        Dataset<Row> categoryCart = dataset.where(col("behavior_type").equalTo(3)).select("user_id", "item_id", "item_category");
        Dataset<Row> categoryMostCart = categoryCart.groupBy("user_id", "item_category").agg(count("item_id").as("item_category_counts"));
        Dataset<Row> categoryMostCartMax = categoryMostCart.groupBy("user_id").agg(functions.max("item_category_counts").as("item_category_counts_max"));
        categoryMostCart = categoryMostCart.join(categoryMostCartMax, categoryMostCart.col("user_id").equalTo(categoryMostCartMax.col("user_id")), "left").select(categoryMostCart.col("user_id"), categoryMostCart.col("item_category"), categoryMostCart.col("item_category_counts"), categoryMostCartMax.col("item_category_counts_max"));
        categoryMostCart = categoryMostCart.select(col("user_id"), col("item_category").cast(DataTypes.StringType),col("item_category_counts"),col("item_category_counts_max"));
        categoryMostCart = categoryMostCart.filter((FilterFunction<Row>) row -> row.getLong(2) == row.getLong(3));
        categoryMostCart.createOrReplaceTempView("category_cart");
        Dataset<Row> categoryCartItemCategory = session.sql("select user_id, collect_list(item_category) as item_category from category_cart group by user_id");
        categoryCartItemCategory = categoryCartItemCategory.map((MapFunction<Row, Row>) row -> {
            StringBuilder sb = new StringBuilder();
            List<String> list = row.getList(1);
            for(int i = 0; i < list.size() - 1; i++) {
                sb.append(list.get(i)).append(",");
            }
            sb.append(list.get(list.size()-1));
            return RowFactory.create(row.getInt(0), sb.toString());
        }, RowEncoder.apply(new StructType(new StructField[]{
                new StructField("user_id", DataTypes.IntegerType,false, Metadata.empty()),
                new StructField("item_category", DataTypes.StringType,false, Metadata.empty())
        })));
        userIDDataset = userIDDataset.join(categoryCartItemCategory, userIDDataset.col("user_id").equalTo(categoryCartItemCategory.col("user_id")), "left").select(userIDDataset.col("user_id"), categoryCartItemCategory.col("item_category"));


        /**
         * 6.用户购买最多类目
         */
        Dataset<Row> categoryBuy = dataset.where(col("behavior_type").equalTo(4)).select("user_id", "item_id", "item_category");
        Dataset<Row> categoryMostBuy = categoryBuy.groupBy("user_id", "item_category").agg(count("item_id").as("item_category_counts"));
        Dataset<Row> categoryMostBuyMax = categoryMostBuy.groupBy("user_id").agg(functions.max("item_category_counts").as("item_category_counts_max"));
        categoryMostBuy = categoryMostBuy.join(categoryMostBuyMax, categoryMostBuy.col("user_id").equalTo(categoryMostBuyMax.col("user_id")), "left").select(categoryMostBuy.col("user_id"), categoryMostBuy.col("item_category"), categoryMostBuy.col("item_category_counts"), categoryMostBuyMax.col("item_category_counts_max"));
        categoryMostBuy = categoryMostBuy.select(col("user_id"), col("item_category").cast(DataTypes.StringType),col("item_category_counts"),col("item_category_counts_max"));
        categoryMostBuy = categoryMostBuy.filter((FilterFunction<Row>) row -> row.getLong(2) == row.getLong(3));
        categoryMostBuy.createOrReplaceTempView("category_buy");
        Dataset<Row> categoryBuyItemCategory = session.sql("select user_id, collect_list(item_category) as item_category from category_buy group by user_id");
        categoryBuyItemCategory = categoryBuyItemCategory.map((MapFunction<Row, Row>) row -> {
            StringBuilder sb = new StringBuilder();
            List<String> list = row.getList(1);
            for(int i = 0; i < list.size() - 1; i++) {
                sb.append(list.get(i)).append(",");
            }
            sb.append(list.get(list.size()-1));
            return RowFactory.create(row.getInt(0), sb.toString());
        }, RowEncoder.apply(new StructType(new StructField[]{
                new StructField("user_id", DataTypes.IntegerType,false, Metadata.empty()),
                new StructField("item_category", DataTypes.StringType,false, Metadata.empty())
        })));
        userIDDataset = userIDDataset.join(categoryBuyItemCategory, userIDDataset.col("user_id").equalTo(categoryBuyItemCategory.col("user_id")), "left").select(userIDDataset.col("user_id"), categoryBuyItemCategory.col("item_category"));

        /**
         * 7.最近 30 天购买次数
         *
         * +--------+-------------+
         * | user_id|item_id_count|
         * +--------+-------------+
         * |64869447|          571|
         * |33538654|          372|
         * |49776801|          307|
         * |42873856|          237|
         * |46993829|          221|
         * +--------+-------------+
         */
        Dataset<Row> counts30Buy = dataset.where(col("behavior_type").equalTo(4)).groupBy("user_id").agg(count("item_id").as("item_id_count"));
        userIDDataset = userIDDataset.join(counts30Buy, userIDDataset.col("user_id").equalTo(counts30Buy.col("user_id")), "left").select(userIDDataset.col("user_id"), counts30Buy.col("item_id_count"));
        counts30Buy = counts30Buy.sort(col("item_id_count").desc());
        /**
         * 8.最近 7 天的活跃天数
         *
         * 通过总结只活跃一天的用户的规律，定位这部分用户活跃天数少的原因，
         * 进而有针对性的制定运营策略，尽量提升这部分用户群的活跃天数。而对于每日都活跃的用户，可以拆解指标观察平台的忠实用户特征以及忠诚度的比例，
         * 或通过研究这部分用户的其他指标，比如访问活跃时段、最近一周购物次数等，结合聚类算法进行用户群的分类。
         * +---------+---------------+
         * |  user_id|counts_7_active|
         * +---------+---------------+
         * |138287418|              7|
         * | 58281309|              7|
         * | 41356531|              6|
         * |105482435|              5|
         * | 29968142|              4|
         * +---------+---------------+
         */

        Dataset<Row> near7 = dataset.where(col("date").gt("2014-12-11"));
        Dataset<Row> near7Active = near7.groupBy("user_id").agg(functions.countDistinct(col("date")).as("counts_7_active"));
        //最近7天每日活跃用户数占比
        double active7Pent = near7Active.where(col("counts_7_active").equalTo(7)).count() / (double)near7Active.count();
        //最近7天只有一日活跃用户数占比
        double active1Pent = near7Active.filter(col("counts_7_active").equalTo(1)).count() / (double)near7Active.count();

        userIDDataset = userIDDataset.join(near7Active, userIDDataset.col("user_id").equalTo(near7Active.col("user_id")), "left")
                .select(userIDDataset.col("user_id"), near7Active.col("counts_7_active"));

        /**
         * 9.复购率
         */

        Dataset<Row> rebuy30 = dataset.where(col("behavior_type").equalTo(4)).groupBy("user_id").agg(count("item_id").as("reybuy30_item_id_count"));
        userIDDataset = userIDDataset.join(rebuy30, userIDDataset.col("user_id").equalTo(rebuy30.col("user_id")), "left")
                .select(userIDDataset.col("user_id"), rebuy30.col("reybuy30_item_id_count"));
        Dataset<Row> rebuy7 = near7.where(col("behavior_type").equalTo(4)).groupBy("user_id").agg(count("item_id").as("rebuy7_item_id_count"));
        userIDDataset = userIDDataset.join(rebuy7, userIDDataset.col("user_id").equalTo(rebuy7.col("user_id")), "left")
                .select(userIDDataset.col("user_id"), rebuy7.col("rebuy7_item_id_count"));
        //最近30天复购率
        double rebuy30Rate = rebuy30.where(col("reybuy30_item_id_count").geq(2)).count() / (double) rebuy30.count();
        //最近7天复购率
        double rebuy7Rate = rebuy7.where(col("rebuy7_item_id_count").geq(2)).count() / (double)rebuy7.count();



        userIDDataset = userIDDataset.na().fill(-1);
        userIDDataset.show(10);

    }

    /**
     * 商品分析
     * 分别查看浏览量、收藏量、加购量、成交量最多和最少的五个类目，可以精准定位到具体的商品类目，并针对问题进行优化改进。
     * 对于流量异常的商品类目，可以结合业务实境考虑以下因素：商品所属的类目分类模糊、类目下属商品的需求（例如大型家电电器，用户一个月内复购同类商品的概率极低；而牙刷等生活日用品，一个月内复购同类商品的概率较高）、
     * 类目下属商品的均价高于人均消费水平、页面设计吸引力等。
     *
     * 1.浏览量最多和最少的 TOP5 类目
     * 2.收藏量最多和最少的 TOP5 类目
     * 3.加购量最多和最少的 TOP5 类目
     * 4.购买量最多和最少的 TOP5 类目
     *
     * @param dataset
     */
    public static void productAnalysis(Dataset<Row> dataset) {
        /**
         * 浏览
         * 1863 : 746826
         * 13230 : 715855
         * 5027 : 632255
         * 5894 : 629911
         * 6513 : 577970
         * 13276 : 1
         * 6799 : 1
         * 10857 : 1
         * 1107 : 1
         * 6905 : 1
         * ----------------
         * 收藏
         * 1863 : 18355
         * 5894 : 15890
         * 13230 : 15615
         * 6513 : 12797
         * 5027 : 11968
         * 12682 : 1
         * 6153 : 1
         * 3050 : 1
         * 3836 : 1
         * 2310 : 1
         * -----------------
         * 加购
         * 1863 : 18042
         * 6513 : 13514
         * 5894 : 13287
         * 13230 : 12343
         * 5027 : 11341
         * 1413 : 1
         * 6153 : 1
         * 3836 : 1
         * 3050 : 1
         * 13100 : 1
         * -------------------
         * 购买
         * 6344 : 4454
         * 1863 : 3647
         * 5232 : 3049
         * 7957 : 2256
         * 6513 : 2195
         * 10403 : 1
         * 13100 : 1
         * 89 : 1
         * 5806 : 1
         * 6438 : 1
         */
        /**
         * 1.浏览量最多和最少的 TOP5 类目
         */
        topTailItemCategoryAnalysis(dataset, 1);

        /**
         * 2.收藏量最多和最少的 TOP5 类目
         */
        topTailItemCategoryAnalysis(dataset, 2);

        /**
         * 3.加购量最多和最少的 TOP5 类目
         */

        topTailItemCategoryAnalysis(dataset, 3);
        /**
         * 4.购买量最多和最少的 TOP5 类目
         */
        topTailItemCategoryAnalysis(dataset, 4);

    }

    public static void topTailItemCategoryAnalysis(Dataset<Row> dataset, int behavior) {
        Dataset<Row> item5Category = dataset.where(col("behavior_type").equalTo(behavior)).groupBy("item_category").agg(count("user_id").as("item_category_count"));
        item5Category = item5Category.sort(col("item_category_count").desc());
        Row[] topRow = (Row[])item5Category.head(5);

        for(Row row : topRow) {
            System.out.println(row.getInt(0) + " : " + row.getLong(1));
        }
        Row[] tailRow = (Row[])item5Category.tail(5);
        for(Row row : tailRow) {
            System.out.println(row.getInt(0) + " : " + row.getLong(1));
        }
    }

}

