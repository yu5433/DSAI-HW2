# DSAI-HW2
### Environment
* Excute 
```bash=
$ py trader.py 
```
* requrements (python version: 3.9.5)
```
numpy==1.19.5
pandas==1.4.1
scikit-learn==1.0.2
tensorflow==2.8.0
statsmodels==0.13.2
matplotlib==3.5.1
```
### 程式流程
* 訓練模型
    * 資料打包
    * 丟入LSTM
    * 資料驗證
        * 以training data後30筆資料作為驗證資料
    ![](https://i.imgur.com/YaC54Fv.png)
* 預測股票與買賣操作
    * 將當天以及前5天的data打包做預測產生10天的開盤價預測
    * 取這10天的平均與當天價比較大小判斷買進賣出
### 程式開發
* 使用模型 LSTM：
    * Data Preprocessing：觀察pacf、acf plot中，最具相關為前六期。
    * Input Data：取前六期的開、高、低、收盤，共24筆資料作為輸入資料。
    * Output Data：預測後10天股票開盤價
* 判斷方式
    * 我們原本打算採用短期長期均線的判斷方式，當短期均線由下往上和長期均線交叉時買進，反之賣出。但是發現這個方式在20天內操作次數相當少，並且**大賠特賠**，於是改用其他方法
    * 另外一個方法是以預測的後10天做平均，若大於本日的價格就買進，反之賣出。
