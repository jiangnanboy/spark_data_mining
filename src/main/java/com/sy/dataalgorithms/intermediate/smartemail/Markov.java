package com.sy.dataalgorithms.intermediate.smartemail;

import com.sy.init.InitSpark;
import com.sy.util.PropertiesReader;
import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.time.DateUtils;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.Optional;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.sql.*;
import scala.Serializable;
import scala.Tuple2;

import java.util.*;

/**
 * 马尔可夫智能邮件营销：
 * 1.初始化状态矩阵
 * 2.统计状态转移概率矩阵
 * 3.对统计后的状态转移概率矩阵进行规范化，使每行的概率和为“1”
 * 4.结合初始的状态矩阵和统计后的转移矩阵，输出最终的状态转移概率矩阵(保存为txt格式，后面用python加载进行预测)
 *  *
 *  *   | SL   SE   SG   ...
 *  *---+-----------------------
 *  *SL |
 *  *   |
 *  *SE |
 *  *   |
 *  *SG |
 *  *   |
 *  *...|
 *
 * @Author Shi Yan
 * @Date 2020/10/29 10:06
 */
public class Markov {
    public static void main(String[] args) {

        SparkSession sparkSession = InitSpark.getSparkSession();
        sparkSession.sparkContext().setLogLevel("ERROR");

        //定义的状态及索引号
        Map<String, String> states = new HashMap<>();
        states.put("SL", "0");
        states.put("SE", "1");
        states.put("SG", "2");
        states.put("ML", "3");
        states.put("ME", "4");
        states.put("MG", "5");
        states.put("LL", "6");
        states.put("LE", "7");
        states.put("LG", "8");
        Broadcast<Map<String, String>> broadcastStatesMap = JavaSparkContext.fromSparkContext(sparkSession.sparkContext()).broadcast(states);

        //初始化状态矩阵，值为 : 1.0/size
        List<Tuple2<String, List<Double>>> initStateList = initState(states.size());
        Broadcast<List<Tuple2<String, List<Double>>>> broadcastInitStateList = JavaSparkContext.fromSparkContext(sparkSession.sparkContext()).broadcast(initStateList);

        //创建和统计状态转移矩阵
        buildStateTransitionMatrix(sparkSession, broadcastStatesMap, broadcastInitStateList);

        InitSpark.closeSparkSession();
    }

    /**
     * 建立状态转移概率矩阵
     * @param session
     */
    public static void buildStateTransitionMatrix(SparkSession session, Broadcast<Map<String, String>> broadcastStatesMap, Broadcast<List<Tuple2<String, List<Double>>>> broadcastInitStateList) {
        //customerID,transactionID,purchaseDate,amount（顾客ID,交易ID，交易日期，金额）
        String path = PropertiesReader.get("intermediate_smart_email_txt");
        JavaRDD<String> javaRDD = session.read().textFile(path).toJavaRDD().coalesce(10);

        //key=customerID,v=(purchaseDate,amount)
        JavaPairRDD<String, Tuple2<Long, Integer>> javaPairRDD = javaRDD.mapToPair(line -> {
            String[] tokens = StringUtils.split(line, ",");
            if(4 != tokens.length) {
                return null;
            }
            long date = DateUtils.parseDate(tokens[2], "yyyy-MM-dd").getTime();
            int amount = Integer.parseInt(tokens[3]);
            Tuple2<Long, Integer> t2 = new Tuple2<>(date, amount);
            return new Tuple2<>(tokens[0], t2);
        });

        //group by customerID
        JavaPairRDD<String, Iterable<Tuple2<Long, Integer>>> customerRDD = javaPairRDD.groupByKey();

        //创建状态序列
        JavaPairRDD<String, List<String>> stateSequence = customerRDD.mapValues(dateAndAmount -> {
            List<Tuple2<Long, Integer>> list = toList(dateAndAmount);
            Collections.sort(list, TupleComparatorAscending.INSTANCE);//对list按日期排序
            return toStateSequence(list);
        });

        /**
         * customerID, List<State>
         * 所有状态的频率为1 =》（（fromState, toState）,1）
         *   | S1   S2   S3   ...
         *---+-----------------------
         *S1 | <probability-value>
         *   |
         *S2 |
         *   |
         *S3 |
         *   |
         *...|
         */
        JavaPairRDD<Tuple2<String, String>, Integer> model = stateSequence.flatMapToPair(s -> {
            List<String> states = s._2;
            List<Tuple2<Tuple2<String, String>, Integer>> mapOut = new ArrayList<>();
            if((null == states) || (states.size() < 2)) {
                return Collections.emptyIterator();
            }
            for(int i = 0; i < (states.size() - 1); i++) {
                String fromState = states.get(i);
                String toState = states.get(i+1);
                Tuple2<String, String> t2 = new Tuple2<>(fromState, toState);
                mapOut.add(new Tuple2<>(t2, 1));
            }
            return mapOut.iterator();
        });

        // 统计所有状态频率：  ((fromState, toState), frequence)
        JavaPairRDD<Tuple2<String, String>, Integer> fromStateToStateFrequence1 = model.reduceByKey((i1, i2) -> i1 + i2);

        // ((fromState, toState), frequence) =》 (fromState, (toState, frequence))
        JavaPairRDD<String, Tuple2<String, Integer>> fromStateToStateFrequence2 = fromStateToStateFrequence1.mapToPair(s -> {
            String key = s._1._1;
            Tuple2<String, Integer> value = new Tuple2<>(s._1._2, s._2);
            return new Tuple2<>(key, value);
        });

        // group by fromState =》 fromState，List<Tuple2<toState, frequence>> => rowNumber,List<Tuple2<toState, frequence>>
        JavaPairRDD<String, Iterable<Tuple2<String, Integer>>> groupState = fromStateToStateFrequence2.groupByKey().mapToPair(st2 -> {
            String rowNumber = broadcastStatesMap.getValue().get(st2._1);
            return new Tuple2<>(rowNumber, st2._2);
        });

        //初始化矩阵状态，value = 1.0 / size
        //List<Tuple2<String, List<Double>>> initStateList = initState(broadcastStatesMap.getValue().size());
        JavaPairRDD<String, List<Double>> initStatePairRDD = JavaSparkContext.fromSparkContext(session.sparkContext()).parallelizePairs(broadcastInitStateList.getValue());

        //initStatePairRDD.leftOuterJoin(groupState)
        JavaPairRDD<String, Tuple2<List<Double>, Optional<Iterable<Tuple2<String, Integer>>>>> joinPairRDD = initStatePairRDD.leftOuterJoin(groupState);

        //规范化转移矩阵，使行的概率和为“1”
        JavaPairRDD<String, List<Double>> resultJavaPairRDD = joinPairRDD.mapValues(lot2 -> {
            int size = broadcastStatesMap.getValue().size();
            List<Double> listDouble = lot2._1;
            Optional<Iterable<Tuple2<String, Integer>>> option = lot2._2;
            if(option.isPresent()) {
                Iterable<Tuple2<String, Integer>> toStateFrequence = option.get();
                Iterator<Tuple2<String, Integer>> iter = toStateFrequence.iterator();
                List<Tuple2<String, Integer>> iterList = new ArrayList<>();
                int sum = 0;
                while(iter.hasNext()) {
                    Tuple2<String, Integer> t2 = iter.next();
                    iterList.add(t2);
                    sum += t2._2;
                }
                //加入平滑，防止概率为0
                if(iterList.size() < size) {
                    sum += size;
                    for(int i = 0; i < listDouble.size(); i ++) {
                        listDouble.set(i, 1.0/sum);
                    }
                }

                for(int i = 0; i < iterList.size(); i++) {
                    String stateNumber = broadcastStatesMap.getValue().get(iterList.get(i)._1);
                    double numalizeValue = iterList.get(i)._2 / (double)sum;
                    listDouble.set(Integer.parseInt(stateNumber), numalizeValue);
                }

            } else {
                return listDouble;
            }
            return listDouble;
        });

        //1.利用sortByKey对转移状态排序，最终的状态转移概率矩阵
        //List<Tuple2<String, List<Double>>> stateResult = resultJavaPairRDD.sortByKey().collect();

        //2.利用takeOrdered对转移状态排序，最终的状态转移概率矩阵
        List<Tuple2<String, List<Double>>> stateResult = resultJavaPairRDD.takeOrdered(broadcastStatesMap.getValue().size(), StateTupleComparatorAscending.INSTANCE);

        //打印转移概率矩阵
        for(Tuple2<String, List<Double>> s : stateResult) {
            StringBuilder sb = new StringBuilder();
            sb.append(s._1).append(",");
            for(int i = 0; i < (s._2.size() - 1); i ++) {
                sb.append(s._2.get(i)).append(" ");
            }
            sb.append(s._2.get(s._2.size() - 1));
            System.out.println(sb.toString());
        }

    }

    /**
     * 初始化矩阵状态,value=1.0/size
     * @param size
     * @return
     */
    public static List<Tuple2<String, List<Double>>> initState(int size) {
        List<Tuple2<String, List<Double>>> initStateList = new ArrayList<>();
        for(int row = 0; row < size; row ++) {
            List<Double> listDouble = new ArrayList<>();
            for(int col = 0; col < size; col ++) {
                listDouble.add(1.0 / size);
            }
            initStateList.add(new Tuple2<>(String.valueOf(row), listDouble));
        }
        return initStateList;
    }

    static List<Tuple2<Long, Integer>> toList(Iterable<Tuple2<Long, Integer>> iterable) {
        List<Tuple2<Long, Integer>> list = new ArrayList<>();
        for(Tuple2<Long, Integer> element:iterable) {
            list.add(element);
        }
        return list;
    }

    /**
     *  1.定义状态含义：
     *  上一次交易后的经过时间     与前次交易相比的交易额
     *          S：小                 L：显著小于
     *          M：中                 E：基本相同
     *          L：大                 G：显著大于
     *
     *  2.定义状态（9*9转移矩阵）：
     *  状态名                   上一次交易后的经过时间；与前次交易相比的交易额
     *  SL                            小：显著小于
     *  SE                            小：基本相同
     *  SG                            小：显著大于
     *  ML                            中：显著小于
     *  ME                            中：基本相同
     *  MG                            中：显著大于
     *  LL                            大：显著小于
     *  LE                            大：基本相同
     *  LG                            大：显著大于
     *
     * @param list
     * @return
     */
    static List<String> toStateSequence(List<Tuple2<Long, Integer>> list) {
        //list = [(Date1,Amount1),(Date2,Amount2),(Date3,Amount3),...],Date1<=Date2<=Date3...
        //至少每个用户有两次交易历史
        if(list.size() < 2) {
            return null;
        }
        List<String> stateSequence = new ArrayList<>();

        Tuple2<Long, Integer> prior = list.get(0);
        for(int i = 1; i < list.size(); i++) {
            Tuple2<Long, Integer> current = list.get(i);

            long pariorDate = prior._1;
            long date = current._1;
            //1天=86400000毫秒
            long daysDiff = (date - pariorDate) / 86400000;

            int priorAmount = prior._2;
            int amount = current._2;
            int amountDiff = amount - priorAmount;

            //两次相邻交易日期相隔天数
            String dd = null;
            if(daysDiff < 30) {
                dd = "S";
            } else if(daysDiff < 60) {
                dd = "M";
            } else {
                dd = "L";
            }

            //两次相邻交易的金额差距
            String ad = null;
            if(priorAmount < 0.9 * amount) {
                ad = "L";
            } else if(priorAmount < 1.1 * amount) {
                ad = "E";
            } else {
                ad = "G";
            }

            String element = dd + ad;
            stateSequence.add(element);
            prior = current;
        }
        return stateSequence;
    }

}

class TupleComparatorAscending implements Comparator<Tuple2<Long, Integer>>, Serializable {
    static public TupleComparatorAscending INSTANCE = new TupleComparatorAscending();
    @Override
    public int compare(Tuple2<Long, Integer> t1, Tuple2<Long, Integer> t2) {
        return t1._1.compareTo(t2._1);
    }
}

class StateTupleComparatorAscending implements Comparator<Tuple2<String, List<Double>>>, Serializable {
    static StateTupleComparatorAscending INSTANCE = new StateTupleComparatorAscending();
    @Override
    public int compare(Tuple2<String, List<Double>> o1, Tuple2<String, List<Double>> o2) {
        return o1._1.compareTo(o2._1);
    }
}

