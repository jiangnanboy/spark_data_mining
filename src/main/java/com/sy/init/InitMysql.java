package com.sy.init;

import com.sy.util.PropertiesReader;

import java.util.Properties;

/**
 * @Author Shi Yan
 * @Date 2020/8/13 21:37
 */
public class InitMysql {
    public static final String mysql_hostname;
    public static final String mysql_username;
    public static final String mysql_password;
    public static final String mysql_database;
    public static final String mysql_port;
    public static final Properties connectionProperties = new Properties();
    static {
        mysql_hostname = PropertiesReader.get("mysql_hostname");
        mysql_username = PropertiesReader.get("mysql_username");
        mysql_password = PropertiesReader.get("mysql_password");
        mysql_database = PropertiesReader.get("mysql_database");
        mysql_port = PropertiesReader.get("mysql_port");

        connectionProperties.put("user", mysql_username);
        connectionProperties.put("password", mysql_password);
        connectionProperties.put("driver", "com.mysql.cj.jdbc.Driver");
    }
}
