此项目将围绕一个时间序列预测任务展开。该任务是Kaggle上的一个比赛，M5 Forecasting - Accuarcy（https://www.kaggle.com/c/m5-forecasting-accuracy/notebooks ）。M5的赛题目标是预测沃尔玛各种商品在未来28天的销量。本案例使用前1913天的数据作为训练数据，来预测1914天到1941天的销量。并且，我们只对最细粒度的30490条序列进行预测。
训练数据从kaggle中自行下载：

- calendar.csv - Contains information about the dates on which the products are sold.
- sales_train_validation.csv - Contains the historical daily unit sales data per product and store [d_1 - d_1913]
- sample_submission.csv - The correct format for submissions. Reference the Evaluation tab for more info.
- sell_prices.csv - Contains information about the price of the products sold per store and date.
- sales_train_evaluation.csv - Includes sales [d_1 - d_1941] (labels used for the Public leaderboard)

以上数据下载后放入resources/advanced下，并在properties.properties中配置一下文件名和路径，以供程序读取和处理数据。

1.数据处理以及特征提取利用spark进行操作，见TimeSeries.java。

2.模型的训练及预测利用lightgbm进行操作，见time_series.ipynb，data下是spark处理好的数据。
