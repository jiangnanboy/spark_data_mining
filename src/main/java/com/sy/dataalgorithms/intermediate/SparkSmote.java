package com.sy.dataalgorithms.intermediate;

/**
 * @author YanShi
 * @date 2020/11/2 20:11
 */
import com.sy.init.InitSpark;
import com.sy.util.PropertiesReader;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.ml.linalg.*;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.*;
import com.google.common.collect.Lists;
import org.apache.commons.lang3.RandomUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;

import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder;
import org.apache.spark.sql.catalyst.encoders.RowEncoder;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import static org.apache.spark.sql.functions.max;
import static org.apache.spark.sql.functions.count;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

import scala.Tuple2;

/**
 * @Author Shi Yan
 * @Date 2020/11/2 9:46
 */
public class SparkSmote {

    public static void main(String[] args) {
        SparkSession sparkSession = InitSpark.getSparkSession();
        sparkSession.sparkContext().setLogLevel("ERROR");

        JavaSparkContext.fromSparkContext(sparkSession.sparkContext()).setLogLevel("WARN");

        String smote_path = PropertiesReader.get("intermediate_smote_txt");
        Dataset<String> dataset = sparkSession.read().textFile(smote_path);

        int[] indices = {0,1,2,3};

        JavaRDD<Row> javaRDD = dataset.javaRDD()
                .filter(str -> !StringUtils.isBlank(str))//过滤空行
                .map(s -> {
                    String[] str = StringUtils.split(s, "\\s++");
                    double[] values = new double[indices.length];
                    for (int p = 1; p < str.length; p++) {
                        values[p] = Double.parseDouble(str[p].split(":")[1].trim());
                    }
                    Vector sparseVec = Vectors.sparse(4, indices, values);//18为向量维度，indices为向量中非零位置

                    return RowFactory.create(str[0], sparseVec); //转为Row(label, feature)
                });

        Dataset<Row> rowDataset = sparkSession.createDataset(javaRDD.rdd(), EncoderInit.getlabelFeaturesRowEncoder());
        smote(sparkSession, rowDataset, 5, 0.5, 0.5).show(50); //利用smote进行过采样
        InitSpark.closeSparkSession();
    }

    /**
     *  (1) 对于少数类(X)中每一个样本x，计算它到少数类样本集(X)中所有样本的距离，得到其k近邻。
     *  (2) 根据样本不平衡比例设置一个采样比例以确定采样倍率sampling_rate，对于每一个少数类样本x，
     *      从其k近邻中随机选择sampling_rate个近邻，假设选择的近邻为 x(1),x(2),...,x(sampling_rate)。
     *  (3) 对于每一个随机选出的近邻 x(i)(i=1,2,...,sampling_rate)，分别与原样本按照如下的公式构建新的样本
     *      xnew=x+rand(0,1)?(x(i)?x)
     *
     *  http://sofasofa.io/forum_main_post.php?postid=1000817
     *  http://sci2s.ugr.es/multi-imbalanced
     * @param session
     * @param labelFeatures
     * @param knn 样本相似近邻
     * @param samplingRate 近邻采样率 (knn * samplingRate),从knn中选择几个近邻
     * @parm rationToMax 采样比率(与最多类样本数的比率) 0.1表示与最多样本的比率是 -> (1:10),即达到最多样本的比率
     * @return
     */
    public static Dataset<Row> smote(SparkSession session, Dataset<Row> labelFeatures, int knn, double samplingRate, double rationToMax) {

        Dataset<Row> labelCountDataset = labelFeatures.groupBy("label").agg(count("label").as("keyCount"));
        List<Row> listRow = labelCountDataset.collectAsList();
        ConcurrentMap<String, Long> keyCountConMap = new ConcurrentHashMap<>(); //每个label对应的样本数
        for(Row row : listRow)
            keyCountConMap.put(row.getString(0), row.getLong(1));
        Row maxSizeRow = labelCountDataset.select(max("keyCount").as("maxSize")).first();
        long maxSize = maxSizeRow.getAs("maxSize");//最大样本数

        JavaPairRDD<String, SparseVector> sparseVectorJPR = labelFeatures.toJavaRDD().mapToPair(row -> {
            String label = row.getString(0);
            SparseVector features = (SparseVector) row.get(1);
            return new Tuple2<String, SparseVector>(label, features);
        });

        JavaPairRDD<String, List<SparseVector>> combineByKeyPairRDD = sparseVectorJPR.combineByKey(sparseVector -> {
                    List<SparseVector> list = new ArrayList<>();
                    list.add(sparseVector);
                    return list;
                }, (list, sparseVector) -> {list.add(sparseVector);return list;},
                (list_A, list_B) -> {list_A.addAll(list_B);return list_A;});


        JavaSparkContext jsc = JavaSparkContext.fromSparkContext(session.sparkContext());
        final Broadcast<ConcurrentMap<String, Long>> keyCountBroadcast = jsc.broadcast(keyCountConMap);
        final Broadcast<Long> maxSizeBroadcast = jsc.broadcast(maxSize);
        final Broadcast<Integer> knnBroadcast = jsc.broadcast(knn);
        final Broadcast<Double> samplingRateBroadcast = jsc.broadcast(samplingRate);
        final Broadcast<Double> rationToMaxBroadcast = jsc.broadcast(rationToMax);

        /**
         * JavaPairRDD<String, List<SparseVector>>
         * JavaPairRDD<String, String>
         * JavaRDD<Row>
         */
        JavaPairRDD<String, List<SparseVector>> pairRDD = combineByKeyPairRDD
                .filter(slt -> {
                    return slt._2().size() > 1;
                })
                .mapToPair(slt -> {
                    String label = slt._1();
                    ConcurrentMap<String, Long> keySizeConMap = keyCountBroadcast.getValue();
                    long oldSampleSize = keySizeConMap.get(label);
                    long max = maxSizeBroadcast.getValue();
                    double ration = rationToMaxBroadcast.getValue();
                    int Knn = knnBroadcast.getValue();
                    double rate = samplingRateBroadcast.getValue();
                    if (oldSampleSize < maxSize * rationToMax) {
                        int needSampleSize = (int) (max * ration - oldSampleSize);
                        List<SparseVector> list = generateSample(slt._2(), needSampleSize, Knn, rate);
                        return new Tuple2<String, List<SparseVector>>(label, list);
                    } else {
                        return slt;
                    }
                });

        JavaRDD<Row> javaRowRDD = pairRDD.flatMapToPair(slt -> {
            List<Tuple2<String, SparseVector>> floatPairList = new ArrayList<>();
            String label = slt._1();
            for(SparseVector sv : slt._2())
                floatPairList.add(new Tuple2<String, SparseVector>(label, sv));
            return floatPairList.iterator();
        }).map(svt->{
            return RowFactory.create(svt._1(), svt._2());
        });

        Dataset<Row> resultDataset = session.createDataset(javaRowRDD.rdd(), EncoderInit.getlabelFeaturesRowEncoder());
        return resultDataset;
    }

    /**
     * @param sparseVectorList
     * @param needSampleSize 需要增加多少样本
     * @param knn 样本的k近邻
     * @param samplingRate 从k近邻中随机选择 (samplingRate * knn) 个
     * @return sample
     */
    private static List<SparseVector> generateSample(List<SparseVector> sparseVectorList, int needSampleSize, int knn, double samplingRate) {
        if(sparseVectorList.size() < knn) {
            knn = (int) (sparseVectorList.size() * 0.5);
        }
        List<SparseVector> listAll = new ArrayList<>();
        listAll.addAll(sparseVectorList);
        Map<String, SparseVector> vectorMap = new LinkedHashMap<>();
        String prefix = "key";
        int num = 0;
        for(SparseVector sv : sparseVectorList) {
            vectorMap.put(prefix + num, sv);
            num++;
        }
        int generateSmapleSize = 0;//产生了多少样本
        boolean fullSample = false;
        for(Map.Entry<String, SparseVector> entry : vectorMap.entrySet()) {
            String key = entry.getKey();
            SparseVector sv_1 = entry.getValue();
            Queue<Item> topItems = new PriorityQueue<Item>(knn);
            boolean full = false;
            double lowestTopValue = Double.NEGATIVE_INFINITY;
            for(Map.Entry<String, SparseVector> entry_2 : vectorMap.entrySet()) {
                String key_2 = entry_2.getKey();
                if(key.equals(key_2))
                    continue;
                SparseVector sv_2 = entry_2.getValue();
                double sim = sv_1.indices().length < sv_2.indices().length ? calculateDistance(sv_1, sv_2) : calculateDistance(sv_2, sv_1);
                if (!full || sim > lowestTopValue) {
                    topItems.add(new Item(key_2, sim));
                    if (full) {
                        topItems.poll();
                    } else if (topItems.size() > knn) {
                        full = true;
                        topItems.poll();
                    }
                    lowestTopValue = topItems.peek().getSim();
                }
            }
            int size = topItems.size();
            List<Item> result = Lists.newArrayListWithCapacity(size);
            result.addAll(topItems);
            Collections.sort(result);
            int randomSize = (int) Math.round(knn * samplingRate);//knn中随机产生样本个数
            Set<Integer> randomSet = new HashSet<>();
            while(true) {
                int number = RandomUtils.nextInt(0, result.size());
                randomSet.add(number);
                if(randomSet.size() >= randomSize)
                    break;
            }
            for(int index : randomSet) {
                Item item = result.get(index);
                String itemKey = item.getKey();
                SparseVector sv = vectorMap.get(itemKey);
                SparseVector randomSV = generateRandomSparseVector(sv_1, sv);
                listAll.add(randomSV);
                generateSmapleSize ++ ;
                if(generateSmapleSize >= needSampleSize) {
                    fullSample = true;
                    break;
                }
            }
            if(fullSample)
                break;
        }
        return listAll;
    }

    /**
     * xnew = x + rand(0,1) ? (x(i) ? x)
     * @param sv_1 -> x
     * @param sv_2 -> x(i)
     * @return
     */
    private static SparseVector generateRandomSparseVector(SparseVector sv_1, SparseVector sv_2) {
        int[] sv_1_indices = sv_1.indices();//有序
        int[] sv_2_indices = sv_2.indices();//有序
        int sv_1_size = sv_1_indices.length;
        int sv_2_size = sv_2_indices.length;
        Map<Integer, Double> sv_1_map = new LinkedHashMap<>();
        Map<Integer, Double> sv_2_map = new LinkedHashMap<>();
        for(int sv_1_idx = 0; sv_1_idx < sv_1_size; sv_1_idx++) {
            sv_1_map.put(sv_1_indices[sv_1_idx], sv_1.values()[sv_1_idx]);
        }
        for(int sv_2_idx = 0; sv_2_idx < sv_2_size; sv_2_idx++) {
            sv_2_map.put(sv_2_indices[sv_2_idx], sv_2.values()[sv_2_idx]);
        }
        int[] commonInt = getCommon(sv_1_indices, sv_2_indices);//相同元素
        int common = commonInt.length;
        double[] generate_new = new double[sv_1_size + sv_2_size - common];//new value
        int[] generate_indices = sortTwoSortedArray(sv_1_indices, sv_2_indices, common);//new indices
        int i = 0;
        for(int indice : generate_indices) {
            double new_value = 0.0;
            if(sv_1_map.containsKey(indice) && sv_2_map.containsKey(indice)) {
                new_value = sv_1_map.get(indice) + Math.random() * (sv_2_map.get(indice) - sv_1_map.get(indice));
            } else if(sv_1_map.containsKey(indice)) {
                new_value = sv_1_map.get(indice) + Math.random() * (0.0 - sv_1_map.get(indice));
            } else {
                new_value = 0.0 + Math.random() * (sv_2_map.get(indice) - 0.0);
            }
            generate_new[i++] = new_value;
        }

        return (SparseVector) Vectors.sparse(sv_1.size(), generate_indices, generate_new);
    }

    /**
     * 共同元素
     * @param array1
     * @param array2
     * @return
     */
    private static int[] getCommon(int []array1, int []array2) {
        int[] array3;
        int i=0,j=0,k=0;
        //Arrays.sort(array1);//排序
        //Arrays.sort(array2);
        if(array1.length < array2.length){//定义数组3，长度尽量小
            array3 = new int[array1.length];
        }else{
            array3 = new int[array2.length];
        }
        while( i< array1.length && j<array2.length){//求公共元素，并存到数组3
            if(array1[i] > array2[j]){
                j++;
            }else if(array1[i] < array2[j]){
                i++;
            }else{
                array3[k] = array1[i];
                i++;
                j++;
                k++;
            }
        }
        array3 = Arrays.copyOf(array3, k);//祛除公共元素以外的初始值0
        return array3;
    }

    /**
     * 两个有序数组排序,去除重复元素
     * @param a
     * @param b
     * @param common
     */
    private static int[] sortTwoSortedArray(int[] a, int[] b, int common) {
        int length1 = a.length;
        int length2 = b.length;
        int newArrayLength = length1 + length2 - common;
        int[] result = new int[newArrayLength];
        int i = 0, j = 0, k = 0;   //i:用于标示a数组    j：用来标示b数组  k：用来标示传入的数组
        while (i < length1 && j < length2) {
            /* 元素不管重复与否，直接给合并到一起 */
            //if (a[i] <= b[j]) {
            //    result[k++] = a[i++];
            //} else {
            //    result[k++] = b[j++];
            //}
            /* 去重复元素，但是空间利用率还是浪费啦，看结果后面有默认的2个0显示 */
            if (a[i] < b[j]) {
                result[k++] = a[i++];
            } else if (a[i] == b[j]) {
                result[k++] = a[i];
                //在某个位置上2个值相等的话，取哪个都一样，
                // 然后这个相等的位置的2个值都可以不用比啦，都直接向后移动1，继续比较
                j++;
                i++;
            } else {
                result[k++] = b[j++];
            }
        }
        /* 后面while循环是用来保证两个数组比较完之后剩下的一个数组里的元素能顺利传入结果数组 */
        while (i < a.length) {
            result[k++] = a[i++];
        }
        while (j < b.length) {
            result[k++] = b[j++];
        }
        return result;
    }

    /**
     * cosine similarity
     * @param sv_1
     * @param sv_2
     * @return
     */
    private static double calculateDistance(SparseVector sv_1, SparseVector sv_2) {
        double dot = BLAS.dot(sv_1, sv_2);
        double sv_1_norm = Vectors.norm(sv_1, 2.0);
        double sv_2_norm = Vectors.norm(sv_2, 2.0);
        if((sv_1_norm != 0.0) && (sv_2_norm != 0.0))
            return dot / (sv_1_norm * sv_2_norm);
        else
            return 0.0;
    }

    /**
     * item包括 key,similarity
     */
    static class Item implements Comparable<Item>{
        private String key = null;
        private double sim = 0.0;
        public Item(String key, double sim) {
            this.key = key;
            this.sim = sim;
        }
        public String getKey() {
            return key;
        }
        public double getSim() {
            return sim;
        }
        @Override
        public int compareTo(Item o) {
            return this.getSim() > o.getSim() ? -1 : 1;
        }
    }
}

class EncoderInit {

    /**
     * 初始化row
     * label -> String
     * features -> Vector
     * @return
     */
    public static ExpressionEncoder<Row> getlabelFeaturesRowEncoder() {
        List<StructField> fieldList = new ArrayList<>();
        fieldList.add(DataTypes.createStructField("label", DataTypes.StringType, false, Metadata.empty()));
        fieldList.add(DataTypes.createStructField("features", new VectorUDT(), false, Metadata.empty()));
        StructType rowSchema = DataTypes.createStructType(fieldList);
        ExpressionEncoder<Row> rowEnocder = RowEncoder.apply(rowSchema);
        return rowEnocder;
    }

}