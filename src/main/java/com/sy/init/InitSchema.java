package com.sy.init;

import org.apache.spark.sql.Encoder;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.util.ArrayList;
import java.util.List;

/**
 * @Author Shi Yan
 * @Date 2020/8/11 21:42
 */
public class InitSchema {

    public static Encoder<String> stringEncoder() {
        return Encoders.STRING();
    }

    public static StructType initStateSchema() {
        StructType schema = new StructType(new StructField[]{
                new StructField("name", DataTypes.StringType,false, Metadata.empty()),
                new StructField("time", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("value", DataTypes.IntegerType, false, Metadata.empty())
        });
        return schema;
    }

    public static StructType initSchema() {
        StructType schema = new StructType(new StructField[]{
                new StructField("name", DataTypes.StringType,false, Metadata.empty()),
                new StructField("time", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("value", DataTypes.IntegerType, false, Metadata.empty())
        });
        return schema;
    }

    public static StructType initSchema(String[] filedName) {
        List<StructField> fileds = new ArrayList<>();
        for(String filed:filedName) {
            fileds.add(DataTypes.createStructField(filed, DataTypes.DoubleType, false, Metadata.empty()));
        }
        StructType schema = DataTypes.createStructType(fileds);
        return schema;
    }

    public static StructType intitSchema(String otherName, String[] filedName) {
        List<StructField> fileds = new ArrayList<>();
        fileds.add(DataTypes.createStructField(otherName, DataTypes.StringType, false, Metadata.empty()));
        for(String filed:filedName) {
            fileds.add(DataTypes.createStructField(filed, DataTypes.DoubleType, false, Metadata.empty()));
        }
        StructType schema = DataTypes.createStructType(fileds);
        return schema;
    }


}
