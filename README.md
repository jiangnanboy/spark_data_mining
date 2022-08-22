This repository provides tutorial code for big data mining to learn [spark](https://github.com/apache/spark).
本库利用java spark实现的数据挖掘项目，包括一些数据的常规分析与挖掘，也包括了一些机器学习算法。这些项目都可以直接运行在所指定的数据集上。未来如果看到有好的数据挖掘项目或者有自己不错的想法都会随时更新实现。

<br/>

##Contents

#### 1. basics
* [TrafficOperationAnalysis淘宝APP一个月数据的流量运营分析](https://github.com/jiangnanboy/spark_tutorial/blob/master/src/main/java/com/sy/dataalgorithms/basics/TrafficOperationAnalysis.java)
* [AssociationRules关联规则](https://github.com/jiangnanboy/spark_tutorial/blob/master/src/main/java/com/sy/dataalgorithms/basics/AssociationRules.java)
* [FindCommonFriends共同好友](https://github.com/jiangnanboy/spark_tutorial/blob/master/src/main/java/com/sy/dataalgorithms/basics/FindCommonFriends.java)
* [FriendRecom推荐朋友](https://github.com/jiangnanboy/spark_tutorial/blob/master/src/main/java/com/sy/dataalgorithms/basics/FriendRecom.java)
* [ItermCFMovieRecom推荐电影](https://github.com/jiangnanboy/spark_tutorial/blob/master/src/main/java/com/sy/dataalgorithms/basics/ItermCFMovieRecom.java)
* [KNN最近邻](https://github.com/jiangnanboy/spark_tutorial/blob/master/src/main/java/com/sy/dataalgorithms/basics/KNN.java)
* [NB贝叶斯](https://github.com/jiangnanboy/spark_tutorial/blob/master/src/main/java/com/sy/dataalgorithms/basics/NB.java)

#### 2. intermediate
* [ALS股票组合推荐](https://github.com/jiangnanboy/spark_tutorial/blob/master/src/main/java/com/sy/dataalgorithms/intermediate/StockRec.java)
* [Markov马尔可夫智能邮件预测](https://github.com/jiangnanboy/spark_tutorial/blob/master/src/main/java/com/sy/dataalgorithms/intermediate/smartemail)
* [SparkSmote样本采样](https://github.com/jiangnanboy/spark_tutorial/blob/master/src/main/java/com/sy/dataalgorithms/intermediate/SparkSmote.java)
* [RFM客户价值分群挖掘](https://github.com/jiangnanboy/spark_tutorial/blob/master/src/main/java/com/sy/dataalgorithms/intermediate/customervalue)
* [手机基站定位数据的商圈分析](https://github.com/jiangnanboy/spark_tutorial/blob/master/src/main/java/com/sy/dataalgorithms/intermediate/BusinessCircle.java)
* [AHP层次分析顾客价值得分](https://github.com/jiangnanboy/spark_tutorial/blob/master/src/main/java/com/sy/dataalgorithms/intermediate/AHP.java)
* [RF用户流失预测](https://github.com/jiangnanboy/spark_tutorial/blob/master/src/main/java/com/sy/dataalgorithms/intermediate/CustomerChurnAnalysis.java)

#### 3.advanced
* [时间序列预测(商品销量预测)](https://github.com/jiangnanboy/spark_tutorial/blob/master/src/main/java/com/sy/dataalgorithms/advanced/time_series)

#### requirements
* [java1.8]
* [spark3.0]
* [python3.6]
* [pandas1.1.4]
* [numpy1.18.5]
* [lightgbm2.3.x]

#### references
* [data algorithms](https://github.com/mahmoudparsian/data-algorithms-book)
* [spark](https://github.com/apache/spark/tree/master/examples/src/main/java/org/apache/spark/examples)
* [spark_smote](https://github.com/jiangnanboy/spark-smote)
* [LearningApacheSpark](https://github.com/runawayhorse001/LearningApacheSpark)
