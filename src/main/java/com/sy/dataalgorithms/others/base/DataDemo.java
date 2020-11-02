package com.sy.dataalgorithms.others.base;

import com.sy.init.InitMysql;
import com.sy.init.InitSpark;
import com.sy.util.PropertiesReader;
import org.apache.commons.lang.StringUtils;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.FilterFunction;
import static org.apache.spark.sql.functions.col;

import org.apache.spark.sql.*;
import scala.Tuple2;

import java.util.Arrays;

/**
 * 基本的操作，包括读写jdbc,txt,csv,json格式数据
 * @Author Shi Yan
 * @Date 2020/8/26 19:22
 */
public class DataDemo {

    public static void main(String[] args) {
        SparkSession session = InitSpark.getSparkSession();
        session.sparkContext().setLogLevel("ERROR");
        //readTable(session, "(select id, press from book) as BOOK");
        //joinTable(session, "course", "course_stage");
        //readTxt(session, PropertiesReader.get("demo_txt"));
        readJson(session, PropertiesReader.get("others_base_demo_json"));
        InitSpark.closeSparkSession();
    }

    /**
     * 读取表book，按字段press进行分组统计与排序
     * @param session
     * @param table or sql
     * @return
     */
    public static void readTable(SparkSession session, String table) {
        Dataset<Row> datasetBook = session.read().format("jdbc")
                .option("url", "jdbc:mysql://" + InitMysql.mysql_hostname + ":" + InitMysql.mysql_port + "/" + InitMysql.mysql_database + "?tinyInt1isBit=false&useUnicode=true&characterEncoding=utf-8")
                .option("driver", "com.mysql.cj.jdbc.Driver")
                .option("user", InitMysql.mysql_username)
                .option("password", InitMysql.mysql_password)
                .option("dbtable", table) //或者option("dbtable", "database.tablename")
                .load();
        datasetBook.show();
        datasetBook.filter(new FilterFunction<Row>() {
            @Override
            public boolean call(Row row) throws Exception {
                return StringUtils.isNotBlank(row.getAs("press"));
            }
        }).select(col("press")).groupBy("press").count().orderBy(col("count").desc()).show();
    }

    /**
     * 两表join
     * @param session
     * @param table1
     * @param table2
     */
    public static void joinTable(SparkSession session, String table1, String table2) {
        DataFrameReader reader = session.read().format("jdbc")
                .option("url", "jdbc:mysql://" + InitMysql.mysql_hostname + ":" + InitMysql.mysql_port + "/" + InitMysql.mysql_database+"?tinyInt1isBit=false&useUnicode=true&characterEncoding=utf-8")
                .option("driver", "com.mysql.cj.jdbc.Driver")
                .option("user", InitMysql.mysql_username)
                .option("password", InitMysql.mysql_password);
        Dataset<Row> datasetTable1 = reader.option("dbtable", table1).load().select("id", "name");
        Dataset<Row> datasetTable2 = reader.option("dbtable", table2).load().select("stage", "course_id", "course_name");
        datasetTable1.createOrReplaceTempView("table1");
        datasetTable2.createOrReplaceTempView("table2");

        //1.通过sql方式join
        Dataset<Row> joinSqlDataset = session.sql("select t1.id, t1.name, t2.stage from table1 t1 join table2 t2 on t1.id = t2.course_id");
        joinSqlDataset.show();

        //2.通过关键字join
        Dataset<Row> joinDataset = datasetTable1.join(datasetTable2, datasetTable1.col("id").equalTo(datasetTable2.col("course_id")))
                .select(datasetTable1.col("id"), datasetTable1.col("name"), datasetTable2.col("stage"));
        joinDataset.show();
    }

    /**
     * 读取txt文件，并进行单词统计
     * @param session
     * @param filePath
     * @return
     */
    public static void readTxt(SparkSession session, String filePath) {
        JavaPairRDD<String, Integer> countJavaPairRDD = session.read().textFile(filePath).javaRDD()
                .flatMap(line -> Arrays.asList(line.split(" ")).iterator())
                .mapToPair(word -> new Tuple2<>(word, 1))
                .reduceByKey((a, b) -> a + b);
        countJavaPairRDD.foreach(t -> System.out.println(t._1 + " : " + t._2));
    }

    /**
     * 读取json文件，过滤出age>25的数据
     * @param session
     * @param filePath
     * @return
     */
    public static void readJson(SparkSession session, String filePath) {
        Dataset<Row> dataset = session.read().json(filePath);
        dataset.filter(col("age").gt(25)).show();
    }

    /**
     * 读取csv文件
     * @param session
     * @param filePath
     * @return
     */
    public static void readCsv(SparkSession session, String filePath) {
        Dataset<Row> peopleDFCsv = session.read().format("csv")
                .option("sep", ";")
                .option("inferSchema", "true")
                .option("header", "true")
                .load(filePath);
    }

    /**
     * 数据写入表中
     * @param dataset
     * @param table
     */
    public static void writeJdbc(Dataset<Row> dataset, String table) {
        dataset.write()
                .format("jdbc")
                .option("url", "jdbc:mysql://" + InitMysql.mysql_hostname + ":" + InitMysql.mysql_port + "/" + InitMysql.mysql_database+"?tinyInt1isBit=false&useUnicode=true&characterEncoding=utf-8")
                .option("driver", "com.mysql.cj.jdbc.Driver")
                .option("user", InitMysql.mysql_username)
                .option("password", InitMysql.mysql_password)
                .option("dbtable", table)
                .save();
    }

    /**
     * 数据写入txt中
     * @param dataset
     * @param filePath
     */
    public static void writeTxt(Dataset<Row> dataset, String filePath) {
        dataset.toJavaRDD().map(row -> row.toString()).saveAsTextFile(filePath);
    }

    /**
     * 数据写入json中
     * @param dataset
     * @param filePath
     */
    public static void writeJson(Dataset<Row> dataset, String filePath) {
        dataset.write().format("json").save(filePath);
    }

    /**
     * 数据写入到csv中
     * @param dataset
     * @param filePath
     */
    public static void writeCsv(Dataset<Row> dataset, String filePath) {
        dataset.write()
                .format("CSV")
                .option("header", "true")
                .option("delimiter", ",")
                .save(filePath);
    }

}
