package com.sy.dataalgorithms.intermediate;

import com.sy.init.InitSpark;
import com.sy.util.PropertiesReader;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.SQLDataTypes;
import org.apache.spark.ml.linalg.SparseVector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
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
 * 模拟预测一个应用程序的用户流失
 * 参考：
 *      https://towardsdatascience.com/predicting-customer-churn-with-spark-4d093907b2dc
 *      https://github.com/celestinhermez/sparkify_customer_churn/blob/master/Sparkify.ipynb（数据集名mini_sparkify_event_data.json，123MB，要自行下载）
 *      https://www.kaggle.com/yukinagae/sparkify-project-churn-prediction
 * @Author Shi Yan
 * @Date 2020/11/12 21:28
 */
public class CustomerChurnAnalysis {
    public static void main(String[] args) {
        SparkSession sparkSession = InitSpark.getSparkSession();
        sparkSession.sparkContext().setLogLevel("ERROR");
        churnAnalysis(sparkSession);
        InitSpark.closeSparkSession();
    }

    public static void churnAnalysis(SparkSession session) {
        /**
         * 加载数据
         * 这是日志数据，包含226个不同用户的信息，从为一首歌点赞到更改账户设置，都有详细的操作。
         *
         * 数据描述
         * artist: Artist name (ex. Daft Punk)
         * auth: User authentication status (ex. Logged)
         * firstName: User first name (ex. Colin)
         * gender: Gender (ex. F or M)
         * itemInSession: Item count in a session (ex. 52)
         * lastName: User last name (ex. Freeman)
         * length: Length of song (ex. 223.60771)
         * level: User plan (ex. paid)
         * location: User's location (ex. Bakersfield)
         * method: HTTP method (ex. PUT)
         * page: Page name (ex. NextSong)
         * registration: Registration timestamp (unix timestamp) (ex. 1538173362000)
         * sessionId: Session ID (ex. 29)
         * song: Song (ex. Harder Better Faster Stronger)
         * status: HTTP status (ex. 200)
         * ts: Event timestamp(unix timestamp) (ex. 1538352676000)
         * userAgent: User's browswer agent (ex. Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0)
         * userId: User ID (ex. 30)
         *
         * +----------------+---------+---------+------+-------------+--------+---------+-----+--------------------+------+--------+-------------+---------+--------------------+------+-------------+--------------------+------+
         * |          artist|     auth|firstName|gender|itemInSession|lastName|   length|level|            location|method|    page| registration|sessionId|                song|status|           ts|           userAgent|userId|
         * +----------------+---------+---------+------+-------------+--------+---------+-----+--------------------+------+--------+-------------+---------+--------------------+------+-------------+--------------------+------+
         * |  Martha Tilston|Logged In|    Colin|     M|           50| Freeman|277.89016| paid|     Bakersfield, CA|   PUT|NextSong|1538173362000|       29|           Rockpools|   200|1538352117000|Mozilla/5.0 (Wind...|    30|
         * |Five Iron Frenzy|Logged In|    Micah|     M|           79|    Long|236.09424| free|Boston-Cambridge-...|   PUT|NextSong|1538331630000|        8|              Canada|   200|1538352180000|"Mozilla/5.0 (Win...|     9|
         * |    Adam Lambert|Logged In|    Colin|     M|           51| Freeman| 282.8273| paid|     Bakersfield, CA|   PUT|NextSong|1538173362000|       29|   Time For Miracles|   200|1538352394000|Mozilla/5.0 (Wind...|    30|
         * |          Enigma|Logged In|    Micah|     M|           80|    Long|262.71302| free|Boston-Cambridge-...|   PUT|NextSong|1538331630000|        8|Knocking On Forbi...|   200|1538352416000|"Mozilla/5.0 (Win...|     9|
         * |       Daft Punk|Logged In|    Colin|     M|           52| Freeman|223.60771| paid|     Bakersfield, CA|   PUT|NextSong|1538173362000|       29|Harder Better Fas...|   200|1538352676000|Mozilla/5.0 (Wind...|    30|
         * +----------------+---------+---------+------+-------------+--------+---------+-----+--------------------+------+--------+-------------+---------+--------------------+------+-------------+--------------------+------+
         */
        String path = PropertiesReader.get("mini_sparkify_event_data"); //数据集自行下载
        Dataset<Row> dataset = session.read().json(path);
        dataset.persist(StorageLevel.MEMORY_AND_DISK());
        dataset.show(5);
        /**
         *
         */

        //创建视图
        dataset.createOrReplaceTempView("userlogs");

        String[] columnsName = dataset.columns();
        Column[] columns = new Column[columnsName.length];
        for(int index = 0;index < columnsName.length; index++) {
            columns[index] = functions.count(functions.when(functions.isnull(col(columnsName[index])), columnsName[index])).as(columnsName[index]);
        }
        /**
         * 查看每个列为null的数量
         *
         * +------+----+---------+------+-------------+--------+------+-----+--------+------+----+------------+---------+-----+------+---+---------+------+
         * |artist|auth|firstName|gender|itemInSession|lastName|length|level|location|method|page|registration|sessionId| song|status| ts|userAgent|userId|
         * +------+----+---------+------+-------------+--------+------+-----+--------+------+----+------------+---------+-----+------+---+---------+------+
         * | 58392|   0|     8346|  8346|            0|    8346| 58392|    0|    8346|     0|   0|        8346|        0|58392|     0|  0|     8346|     0|
         * +------+----+---------+------+-------------+--------+------+-----+--------+------+----+------------+---------+-----+------+---+---------+------+
         */
        dataset.select(columns);

        /**
         * 列“firstName”中的缺失值
         * +------+----------+---------+------+-------------+--------+------+-----+--------+------+-----+------------+---------+----+------+-------------+---------+------+
         * |artist|      auth|firstName|gender|itemInSession|lastName|length|level|location|method| page|registration|sessionId|song|status|           ts|userAgent|userId|
         * +------+----------+---------+------+-------------+--------+------+-----+--------+------+-----+------------+---------+----+------+-------------+---------+------+
         * |  null|Logged Out|     null|  null|          100|    null|  null| free|    null|   GET| Home|        null|        8|null|   200|1538355745000|     null|      |
         * |  null|Logged Out|     null|  null|          101|    null|  null| free|    null|   GET| Help|        null|        8|null|   200|1538355807000|     null|      |
         * |  null|Logged Out|     null|  null|          102|    null|  null| free|    null|   GET| Home|        null|        8|null|   200|1538355841000|     null|      |
         * |  null|Logged Out|     null|  null|          103|    null|  null| free|    null|   PUT|Login|        null|        8|null|   307|1538355842000|     null|      |
         * |  null|Logged Out|     null|  null|            2|    null|  null| free|    null|   GET| Home|        null|      240|null|   200|1538356678000|     null|      |
         * |  null|Logged Out|     null|  null|            3|    null|  null| free|    null|   PUT|Login|        null|      240|null|   307|1538356679000|     null|      |
         * |  null|Logged Out|     null|  null|            0|    null|  null| free|    null|   PUT|Login|        null|      100|null|   307|1538358102000|     null|      |
         * |  null|Logged Out|     null|  null|            0|    null|  null| free|    null|   PUT|Login|        null|      241|null|   307|1538360117000|     null|      |
         * |  null|Logged Out|     null|  null|           14|    null|  null| free|    null|   GET| Home|        null|      187|null|   200|1538361527000|     null|      |
         * |  null|Logged Out|     null|  null|           15|    null|  null| free|    null|   PUT|Login|        null|      187|null|   307|1538361528000|     null|      |
         * +------+----------+---------+------+-------------+--------+------+-----+--------+------+-----+------------+---------+----+------+-------------+---------+------+
         */
        dataset.where(col("firstName").isNaN()).show(10);

        /**
         * 列"artist"中的缺失值
         * +------+----------+---------+------+-------------+--------+------+-----+--------------------+------+---------------+-------------+---------+----+------+-------------+--------------------+------+
         * |artist|      auth|firstName|gender|itemInSession|lastName|length|level|            location|method|           page| registration|sessionId|song|status|           ts|           userAgent|userId|
         * +------+----------+---------+------+-------------+--------+------+-----+--------------------+------+---------------+-------------+---------+----+------+-------------+--------------------+------+
         * |  null| Logged In|    Colin|     M|           54| Freeman|  null| paid|     Bakersfield, CA|   PUT|Add to Playlist|1538173362000|       29|null|   200|1538352905000|Mozilla/5.0 (Wind...|    30|
         * |  null| Logged In|    Micah|     M|           84|    Long|  null| free|Boston-Cambridge-...|   GET|    Roll Advert|1538331630000|        8|null|   200|1538353150000|"Mozilla/5.0 (Win...|     9|
         * |  null| Logged In|    Micah|     M|           86|    Long|  null| free|Boston-Cambridge-...|   PUT|      Thumbs Up|1538331630000|        8|null|   307|1538353376000|"Mozilla/5.0 (Win...|     9|
         * |  null| Logged In|    Alexi|     F|            4|  Warren|  null| paid|Spokane-Spokane V...|   GET|      Downgrade|1532482662000|       53|null|   200|1538354749000|Mozilla/5.0 (Wind...|    54|
         * |  null| Logged In|    Alexi|     F|            7|  Warren|  null| paid|Spokane-Spokane V...|   PUT|      Thumbs Up|1532482662000|       53|null|   307|1538355255000|Mozilla/5.0 (Wind...|    54|
         * |  null| Logged In|    Micah|     M|           95|    Long|  null| free|Boston-Cambridge-...|   PUT|    Thumbs Down|1538331630000|        8|null|   307|1538355306000|"Mozilla/5.0 (Win...|     9|
         * |  null| Logged In|    Micah|     M|           97|    Long|  null| free|Boston-Cambridge-...|   GET|           Home|1538331630000|        8|null|   200|1538355504000|"Mozilla/5.0 (Win...|     9|
         * |  null| Logged In|    Micah|     M|           99|    Long|  null| free|Boston-Cambridge-...|   PUT|         Logout|1538331630000|        8|null|   307|1538355687000|"Mozilla/5.0 (Win...|     9|
         * |  null| Logged In|  Ashlynn|     F|            9|Williams|  null| free|     Tallahassee, FL|   PUT|      Thumbs Up|1537365219000|      217|null|   307|1538355711000|"Mozilla/5.0 (Mac...|    74|
         * |  null|Logged Out|     null|  null|          100|    null|  null| free|                null|   GET|           Home|         null|        8|null|   200|1538355745000|                null|      |
         * +------+----------+---------+------+-------------+--------+------+-----+--------------------+------+---------------+-------------+---------+----+------+-------------+--------------------+------+
         */
        dataset.where(col("artist").isNaN()).show(10);

        /**
         * 过滤注销的用户
         */
        dataset = dataset.where(col("auth").notEqual("Logged Out"));

        /**通过以上分析可知：
         * 虽然在较高的级别上userId或sessionId列中没有丢失的值，但进一步查看firstName列中丢失的值，可以发现注销的用户拥有空(但不是null)的用户ID。排除这些用户。
         * 列artist中也有缺失的值，但这些值对应于与音乐无关的动作的日志(如“Add to Playlist”、“Roll Advert”等)。只要这些用户是登录的，我们就希望保留这个activity，因为他们可能是分类的重要行为标记。
         */


        /**
         *  |-- artist: string (nullable = true)
         *  |-- auth: string (nullable = true)
         *  |-- firstName: string (nullable = true)
         *  |-- gender: string (nullable = true)
         *  |-- itemInSession: long (nullable = true)
         *  |-- lastName: string (nullable = true)
         *  |-- length: double (nullable = true)
         *  |-- level: string (nullable = true)
         *  |-- location: string (nullable = true)
         *  |-- method: string (nullable = true)
         *  |-- page: string (nullable = true)
         *  |-- registration: long (nullable = true)
         *  |-- sessionId: long (nullable = true)
         *  |-- song: string (nullable = true)
         *  |-- status: long (nullable = true)
         *  |-- ts: long (nullable = true)
         *  |-- userAgent: string (nullable = true)
         *  |-- userId: string (nullable = true)
         */
        dataset.printSchema();

        /**
         * 对用户事件动作group,count
         *
         * +--------------------+------+
         * |                Page| count|
         * +--------------------+------+
         * |              Cancel|    52|
         * |    Submit Downgrade|    63|
         * |         Thumbs Down|  2546|
         * |                Home| 10118|
         * |           Downgrade|  2055|
         * |         Roll Advert|  3933|
         * |              Logout|  3226|
         * |       Save Settings|   310|
         * |Cancellation Conf...|    52|
         * |               About|   509|
         * | Submit Registration|     5|
         * |            Settings|  1514|
         * |            Register|    18|
         * |     Add to Playlist|  6526|
         * |          Add Friend|  4277|
         * |            NextSong|228108|
         * |           Thumbs Up| 12551|
         * |                Help|  1477|
         * |             Upgrade|   499|
         * |               Error|   253|
         * +--------------------+------+
         */
        dataset.groupBy("Page").count().show();

        /**
         * 不同的用户数量
         *+--------+
         * |nb_users|
         * +--------+
         * |     226|
         * +--------+
         */
        session.sql("select count(distinct userId) as nb_users from userlogs").show();

        /**
         * 创建数据集，包括用户id和标签(是否流失)
         *
         * 定义：列Page中值为"Cancellation Confirmation"为流失，其它非流失
         */

        Dataset<Row> churnDataset = session.sql("select distinct userId, 1 as churn from userlogs where Page='Cancellation Confirmation'");
        Dataset<Row> noChurnDataset = session.sql("select distinct userId, 0 as churn from userlogs where userId not in (select distinct userId from userlogs where Page='Cancellation Confirmation')");

        /**
         * union churnDataset  noChurnDataset,shuffling the rows
         */
        Dataset<Row> unionDataset = churnDataset.union(noChurnDataset);
        unionDataset.createOrReplaceTempView("churn");
        unionDataset = session.sql("select * from churn order by rand()");
        unionDataset.createOrReplaceTempView("churn");

        /**
         * churn	userId
         *  0	     174
         *  1	     52
         */
        unionDataset.groupBy(col("churn")).count().show();

        /**
         * 以上通过定义page为“Cancellation Confirmation”为可以流失用户。
         * 这种方法使我们能够在用户流失之前研究其行为，尝试建立预测模型并提取表明将来有流失风险的行为。
         * 从以上可知建立的数据集不平衡，在建立模型前可进行采样缓解这种不平衡带来的不准确问题。
         */


        /**
         * 增加数据
         */
        Dataset<Row> joinedDataset = dataset.join(unionDataset, "userId");
        /**
         *
         +-----+-----+------+
         |churn|level| count|
         +-----+-----+------+
         |    0| free| 43430|
         |    0| paid|189957|
         |    1| paid| 32476|
         |    1| free| 12388|
         +-----+-----+------+
         */
        joinedDataset.groupBy(col("churn"), col("level")).count().show(5);

        /**
         * 统计列length
         *
         * +-----+-----------------+
         * |churn|      avg(length)|
         * +-----+-----------------+
         * |    1|248.6327956440622|
         * |    0|249.2091353888082|
         * +-----+-----------------+
         */
        joinedDataset.groupBy("churn").avg("legnth").show(5);

        /**
         * itemInSession变量，它表示歌曲在当前会话中的排名
         *
         *
         +-----+------------------+
         |churn|avg(itemInSession)|
         +-----+------------------+
         |    1|109.23299304564907|
         |    0|115.94533542999396|
         +-----+------------------+
         */
        joinedDataset.groupBy("churn").avg("itemInSession").show(5);

        /**
         * 增加一列，表示state
         */
        joinedDataset = joinedDataset.withColumn("state", functions.substring(col("location"),-2, 3));

        /**
         * 查看流失信息
         *
         * +-----+-----+-----+
         * |churn|state|count|
         * +-----+-----+-----+
         * |    0|   CA|39158|
         * |    0|   PA|23708|
         * |    0|   TX|22200|
         * |    0|   NH|18637|
         * |    0|   FL|11427|
         * +-----+-----+-----+
         * only showing top 5 rows
         *
         * +-----+-----+-----+
         * |churn|state|count|
         * +-----+-----+-----+
         * |    1|   CA| 7613|
         * |    1|   CO| 4317|
         * |    1|   MS| 3839|
         * |    1|   WA| 3526|
         * |    1|   OH| 3173|
         * +-----+-----+-----+
         * only showing top 5 rows
         */
        joinedDataset.groupBy(col("churn"),col("state")).count().where(col("churn").equalTo(0)).sort(col("count").desc()).show(5);
        joinedDataset.groupBy(col("churn"),col("state")).count().where(col("churn").equalTo(1)).sort(col("count").desc()).show(5);

        /**
         *Study potential differences based on ts
         * +-----+--------------------+
         * |churn|             avg(ts)|
         * +-----+--------------------+
         * |    1|1.539919263874465E12|
         * |    0|1.541159010797734...|
         * +-----+--------------------+
         */
        joinedDataset.groupBy("churn").avg("ts").show(5);

        /**
         * 将列ts转为时间
         */
        joinedDataset = joinedDataset.withColumn("date", functions.from_unixtime((col("ts").divide(1000)).cast(DataTypes.TimestampType)));

        /**
         * 创建几个基于时间的特征
         * 首冼，从列date中抽取day与month
         */
        joinedDataset = joinedDataset.withColumn("day", dayofmonth(col("date"))).withColumn("month", month(col("date")));

        /**
         * 然后针对给定用户，获得每天和每月不同会话的平均数量
         */
        Dataset<Row> dayDataset = joinedDataset.groupBy("userId", "day")
                .agg(countDistinct("sessionId"))
                .groupBy("userId")
                .avg("count(DISTINCT sessionId)")
                .withColumnRenamed("avg(count(DISTINCT sessionId))", "daily_sessions");
        Dataset<Row> monthDataset = joinedDataset.groupBy("userId", "month")
                .agg(countDistinct("sessionId"))
                .groupBy("userId")
                .avg("count(DISTINCT sessionId)")
                .withColumnRenamed("avg(count(DISTINCT sessionId))", "monthly_sessions");

        /**
         * 对于给定的用户id，将dayDataset与monthDataset连接到原始的joinedDataset中的第一行，后面会用到
         */
        joinedDataset = joinedDataset.join(dayDataset, "userId").join(monthDataset, "userId");

        /**
         * 比较两组用户的每日和每月会话次数
         *
         * +-----+-------------------+
         * |churn|avg(daily_sessions)|
         * +-----+-------------------+
         * |    1| 1.3486546923719596|
         * |    0| 1.6796452897398528|
         * +-----+-------------------+
         *
         * +-----+---------------------+
         * |churn|avg(monthly_sessions)|
         * +-----+---------------------+
         * |    1|    9.296020565858312|
         * |    0|   14.264002136651344|
         * +-----+---------------------+
         */
        joinedDataset.groupBy("churn").avg("daily_sessions").show();
        joinedDataset.groupBy("churn").avg("monthly_sessions").show();


        /**
         * 列registration表示用户注册时间，转为时间列
         */
        joinedDataset = joinedDataset.withColumn("registration_date", functions.from_unixtime((col("registration").divide(1000)).cast(DataTypes.TimestampType)));
        /**
         * 创建一个用户注册后的天数的新特征
         */
        joinedDataset = joinedDataset.withColumn("days_since_registration", datediff(current_date(), col("registration_date")));
        /**
         * 创建一个新特征，记录他们注册的月份，以考虑潜在的促销活动
         */
        joinedDataset = joinedDataset.withColumn("month_registration", month(col("registration_date")));

        /**
         * 基于user agent研究潜在差异
         *
         * +-----+--------------------+-----+
         * |churn|           userAgent|count|
         * +-----+--------------------+-----+
         * |    0|"Mozilla/5.0 (Win...|18226|
         * |    0|"Mozilla/5.0 (Mac...|16298|
         * |    0|"Mozilla/5.0 (Mac...|15914|
         * |    0|"Mozilla/5.0 (Win...|15237|
         * |    0|Mozilla/5.0 (Wind...|15224|
         * +-----+--------------------+-----+
         * only showing top 5 rows
         *
         * +-----+--------------------+-----+
         * |churn|           userAgent|count|
         * +-----+--------------------+-----+
         * |    1|"Mozilla/5.0 (Mac...| 4736|
         * |    1|"Mozilla/5.0 (Win...| 4525|
         * |    1|Mozilla/5.0 (Wind...| 3437|
         * |    1|"Mozilla/5.0 (Mac...| 2534|
         * |    1|Mozilla/5.0 (Maci...| 2462|
         * +-----+--------------------+-----+
         */
        joinedDataset.groupBy("churn", "userAgent").count().where(col("churn").equalTo(0)).sort(col("count").desc()).show(5);
        joinedDataset.groupBy("churn", "userAgent").count().where(col("churn").equalTo(1)).sort(col("count").desc()).show(5);


        /**
         * 对每个用户Page下的action(事件)类型数量 =》 downgrades, upgrades, thumbs up, thumbs down, add friend, add to playlist, roll advert
         * 对以上每一个action创建一个新列，每次事件(action)发生时用1表示
         */
        List<String> actionType = new ArrayList<>();
        actionType.add("Downgrade");
        actionType.add("Roll Advert");
        actionType.add("Thumbs Down");
        actionType.add("Add to Playlist");
        actionType.add("Add Friend");
        actionType.add("Thumbs Up");
        for(String type:actionType) {
            joinedDataset.withColumn(type, (col("Page").equalTo(type)).cast(DataTypes.IntegerType));
        }

        /**
         * 构建特征数据
         */
        Dataset<Row> featuresDataset = joinedDataset.groupBy("userId").agg(
                avg("itemInSession"),
                avg("length"),
                min("daily_sessions"),
                min("monthly_sessions"),
                min("days_since_registration"),
                min("month_registration"),
                max("level"),
                max("userAgent"),
                max("state"),
                sum("Downgrade"),
                sum("Roll Advert"),
                sum("Thumbs Down"),
                sum("Add to Playlist"),
                sum("Add Friend"),
                sum("Thumbs Up"),
                max("churn")
        );

        featuresDataset = featuresDataset.select(
                col("avg(itemInSession)").as("itemInSession"),
                col("avg(length)").as("length"),
                col("min(daily_sessions)").as("daily_sessions"),
                col("min(monthly_sessions)").as("monthly_sessions"),
                col("min(days_since_registration)").as("days_since_registration"),
                col("min(month_registration)").as("month_registration"),
                col("max(level)").as("level"),
                col("max(userAgent)").as("userAgent"),
                col("max(state)").as("state"),
                col("sum(Downgrade)").as("downgrade"),
                col("sum(Roll Advert)").as("rollAdvert"),
                col("sum(Thumbs Down)").as("thumbsDown"),
                col("sum(Add to Playlist)").as("addToPlaylist"),
                col("sum(Add Friend)").as("addFriend"),
                col("sum(Thumbs Up)").as("thumbsUp"),
                col("max(churn)").as("label"));

        /**
         * 查看null 值
         * +-------------+------+--------------+----------------+-----------------------+------------------+-----+---------+-----+---------+----------+----------+-------------+---------+--------+-----+
         * |itemInSession|length|daily_sessions|monthly_sessions|days_since_registration|month_registration|level|userAgent|state|downgrade|rollAdvert|thumbsDown|addToPlaylist|addFriend|thumbsUp|label|
         * +-------------+------+--------------+----------------+-----------------------+------------------+-----+---------+-----+---------+----------+----------+-------------+---------+--------+-----+
         * |            0|     1|             0|               0|                      1|                 1|    0|        1|    1|        0|         0|         0|            0|        0|       0|    0|
         * +-------------+------+--------------+----------------+-----------------------+------------------+-----+---------+-----+---------+----------+----------+-------------+---------+--------+-----+
         */
        String[] featureColumnsName = featuresDataset.columns();
        Column[] featureColumns = new Column[featureColumnsName.length];
        for(int index = 0;index < featureColumnsName.length; index++) {
            featureColumns[index] = functions.count(functions.when(functions.isnull(col(featureColumnsName[index])), featureColumnsName[index])).as(featureColumnsName[index]);
        }
        featuresDataset.select(featureColumns).show();

        //移除Null值
        featuresDataset = featuresDataset.where(col("userAgent").isNotNull()).where(col("state").isNotNull());


        /**
         * 以下为构建模型流程
         */
        StringIndexer indexerState = new StringIndexer()
                .setInputCol("state")
                .setOutputCol("state_index");
        StringIndexer indexerLevel = new StringIndexer()
                .setInputCol("level")
                .setOutputCol("level_index");
        StringIndexer indexerUA = new StringIndexer()
                .setInputCol("userAgent")
                .setOutputCol("userAgent_index");
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{
                        "itemInSession",
                        "length",
                        "daily_sessions",
                        "monthly_sessions",
                        "days_since_registration",
                        "month_registration",
                        "level_index",
                        "state_index",
                        "userAgent_index",
                        "downgrade",
                        "rollAdvert",
                        "thumbsDown",
                        "addToPlaylist",
                        "addFriend",
                        "thumbsUp"})
                .setOutputCol("features");

        Pipeline featuresPipeline = new Pipeline()
                .setStages(new PipelineStage[]{indexerState, indexerLevel, indexerUA, assembler});
        Dataset<Row> modelFeaturesDataset = featuresPipeline.fit(featuresDataset).transform(featuresDataset);

        /**
         * 在通过VectorAssembler将多列特征值和并后，由于spark存储格式的原因，会将含有很多0值的一行转为稀疏向量sparseVector
         * 进行存储。然而在后续计算过程中，我们需要的是密集向量，所以需要将稀疏向量转为密集向量。
         */
        Dataset<Row> dataset2 = modelFeaturesDataset.select("label", "features");


        /*JavaRDD<Row> javaRDD = dataset2.toJavaRDD().map(row ->  {
            if(((SparseVector)row.getAs("features")).size() > 1) {
                DenseVector denseVector = (DenseVector)row.getAs("features");
                return RowFactory.create(row.getInt(0), denseVector);
            } else {
                return RowFactory.create(null, Vectors.dense(new double[]{}));
            }
        });
        Dataset<Row> modelDataset = session.createDataFrame(javaRDD, new StructType(new StructField[]{
                new StructField("label", DataTypes.IntegerType,false, Metadata.empty()),
                new StructField("features", SQLDataTypes.VectorType(), false, Metadata.empty())
        }));*/


        Dataset<Row> modelDataset = dataset2.map((MapFunction<Row, Row>) row -> {

                if(((SparseVector)row.getAs("features")).size() > 1) {
                    DenseVector denseVector = (DenseVector)row.getAs("features");
                    return RowFactory.create(row.getInt(0), denseVector);
                } else {
                    return RowFactory.create(null, Vectors.dense(new double[]{}));
                }

        }, RowEncoder.apply(new StructType(new StructField[]{
                new StructField("label", DataTypes.IntegerType,false, Metadata.empty()),
                new StructField("features", SQLDataTypes.VectorType(), false, Metadata.empty())
        })));

        //分为train与test
        Dataset<Row>[] trainTestDataset = modelDataset.randomSplit(new double[]{0.8, 0.2});
        //因为样本类别不平衡，所以对label为流失的类别样本进行上采样
        System.out.println("训练集中流失的用户有：" + trainTestDataset[0].where(col("label").equalTo(1)).count());
        System.out.println("训练集中不是流失的用户有：" + trainTestDataset[0].where(col("label").equalTo(0)).count());
        Dataset<Row> trainChurnDataset = trainTestDataset[0].where(col("label").equalTo(1))
                //有放回采样，采样比例
                .sample(true, trainTestDataset[0].where(col("label").equalTo(0)).count()/trainTestDataset[0].where(col("label").equalTo(1)).count());
        Dataset<Row> trainNoChurnDataset = trainTestDataset[0].where(col("label").equalTo(0));
        Dataset<Row> trainDataset = trainChurnDataset.unionAll(trainNoChurnDataset);

        /**
         * 构建模型
         */
        /*GBTClassifier gbtClassifier = new GBTClassifier()
                .setLabelCol("label")
                .setFeaturesCol("features")
                .setMaxIter(10);*/

        /**
        RandomForestClassifier rf = new RandomForestClassifier()
                .setLabelCol("label")
                .setFeaturesCol("features");

        //训练和评估
        Dataset<Row> predictions = rf.fit(trainDataset).transform(trainTestDataset[1]);
        //accuracy评估
        MulticlassClassificationEvaluator accuracyEvaluator = new MulticlassClassificationEvaluator()
                .setMetricName("accuracy");
        double accuracy = accuracyEvaluator.evaluate(predictions.select("label", "prediction"));

        //f1评估
        MulticlassClassificationEvaluator f1Evaluator = new MulticlassClassificationEvaluator()
                .setMetricName("f1");
        double f1Score = f1Evaluator.evaluate(predictions.select("label", "prediction"));

        System.out.println("准确率：" + accuracy);
        System.out.println("f1值：" + f1Score);
        **/

        /**
         * 优化随机森林参数
         */
        RandomForestClassifier rfModel = new RandomForestClassifier()
                .setLabelCol("label")
                .setFeaturesCol("features");
        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(rfModel.minInfoGain(), new double[]{0,1})
                .addGrid(rfModel.numTrees(), new int[]{20, 50})
                .addGrid(rfModel.maxDepth(), new int[]{5, 10})
                .build();
        CrossValidator crossValidator = new CrossValidator()
                .setEstimator(new Pipeline().setStages(new PipelineStage[]{rfModel}))
                .setEstimatorParamMaps(paramGrid)
                .setEvaluator(new MulticlassClassificationEvaluator().setMetricName("f1"))
                .setNumFolds(3);
        CrossValidatorModel classifyModel = crossValidator.fit(trainDataset);
        Dataset<Row> predictions = classifyModel.transform(trainTestDataset[1]);

        //accuracy评估
        MulticlassClassificationEvaluator accuracyEvaluator = new MulticlassClassificationEvaluator()
                .setMetricName("accuracy");
        double accuracy = accuracyEvaluator.evaluate(predictions.select("label", "prediction"));

        //f1评估
        MulticlassClassificationEvaluator f1Evaluator = new MulticlassClassificationEvaluator()
                .setMetricName("f1");
        double f1Score = f1Evaluator.evaluate(predictions.select("label", "prediction"));

        System.out.println("准确率：" + accuracy);
        System.out.println("f1值：" + f1Score);

        //特征重要性
        Pipeline bestPipeline = (Pipeline)classifyModel.bestModel().parent();
        PipelineStage[] stage = bestPipeline.getStages();
        RandomForestClassificationModel rfc = (RandomForestClassificationModel) stage[0];
        System.out.println(rfc.featureImportances());
    }

}

