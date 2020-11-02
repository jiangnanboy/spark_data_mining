package com.sy.dataalgorithms.others.ml;

import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.stat.Correlation;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.*;
import org.apache.spark.sql.SparkSession;

import java.util.Arrays;
import java.util.List;


/**
 * @Author Shi Yan
 * @Date 2020/8/28 22:47
 */
public class StatisticsDemo {

    /**
     * 相关性统计
     * @param session
     */
    public static void correlationDemo(SparkSession session) {
        List<Row> data = Arrays.asList(
                RowFactory.create(Vectors.sparse(4, new int[] {0, 3}, new double[]{1.0, -2.0})),
                RowFactory.create(Vectors.dense(4.0, 5.0, 0.0, 3.0)),
                RowFactory.create(Vectors.dense(6.0, 7.0, 0.0, 8.0)),
                RowFactory.create(Vectors.sparse(4, new int[]{0, 3}, new double[]{9.0, 1.0}))
        );
        StructType schema = new StructType(new StructField[] {
                new StructField("features", new VectorUDT(), false, Metadata.empty()),
        });
        Dataset<Row> df = session.createDataFrame(data, schema);
        Row r1 = Correlation.corr(df, "features").head();
        System.out.println("Pearson 相关系数矩阵:\n" + r1.get(0).toString());
        Row r2 = Correlation.corr(df, "features", "spearman").head();
        System.out.println("Spearman 相关系数矩阵:\n" + r2.get(0).toString());
    }

}

