# ML-Stock-price-forecast

# I. 研究目的

如何選擇一隻會讓自己賺錢的股票，可說是長久以來的難題，且市面上充斥著不同的投資策略和技術指摽，
讓人無所適從。我們期望可以利用機器學習的方法，在眾多的技術指標中，預測未來股票漲或跌的可能性，提升投資的報酬率。

# II.	資料描述

1.	收盤價： 
    為個股當天收盤的價格。
2.	相對強弱指標(RSV)：
    (今日收盤價-九日內最低價)/(九日內最高價-九日內最低價)\*100，該指標值介於0到100之間。
3.	隨機指標(KD)：
    可分為K值與D值。當K值由下而上穿越D值，為黃金交叉，行情看好；當K值由上而下跌破D值，為死亡交叉，行情看壞。
    K值：前日K值*(2/3) + 當日RSV*(1/3)，該指標值介於0到100間。
    D值：前日D值*(2/3) + 當日K值*(1/3) ，該指標值介於0到100間。
4.	動量指標(Mom)：
    為當日股價及前九天的股價差值，由此可以看出股價在其中波段漲跌幅度。
5.	移動平均線(SMA)：
    N日收盤價總和/N。這邊我們採用了五日，十日及二十日，若股票向上突破SMA則代表股價走強，為買進訊號，反之亦然。
6.	加權移動平均線(WMA)：
    (十日前收盤價\*1 +九日前收盤價\*2+…+今日收盤價\*10) / (1+2+…+10)。我們採用十日的WMA，為十日的股價加權平均。
7.	相對強弱指標(RSI)：
    九日股價上漲幅度的加總/（九日股價上漲幅度的加總 + 九日股價下跌幅度的加總））\*100。越高時代表市場越熱絡，越低時越冷清。其中RSI值大於80時，股價下跌的機率大；RSI值小於20時，上漲的機率高。
8.	三大法人：
    分別為外資，投信，自營商，所佔的持股比例。一般而言，若外資持股比例在股票市場中佔有較大，則股價的漲跌會容易受到外資的買賣而影響。
    
* 資料蒐集：
   - 台灣證卷交易所網站看盤軟體之交易股價資訊

# III.資料檢視&初步分析

![image](https://github.com/oicjacky/ML-Stock-price-forecast/blob/master/data.png)
![image](https://github.com/oicjacky/ML-Stock-price-forecast/blob/master/pic01.png)

* 股價動態圖 : <http://rpubs.com/skyking363/499515>

# IV.	研究方法

每日收盤價的漲跌作為反應變數(Y)，各種技術指標(RSV、Mom、SMA、WMA、RSI)以及三大法人資訊當作解釋變數(X<sub>1</sub>,X<sub>2</sub>,…,X<sub>p</sub>)。

#### 目標:
 > 透過技術指標建立模型，並預測未來股價的漲跌。

![image](https://github.com/oicjacky/ML-Stock-price-forecast/blob/master/pic02.PNG)

* 資料處理：

將每日收盤價與前一日做比較得到當日股價的漲跌。(Y=+1代表漲，Y=-1代表跌。)
當日的技術指標及三大法人資訊(X<sub>1</sub>,X<sub>2</sub>,…,X<sub>p</sub>)值轉換為±1。(+1代表看漲，-1代表看跌。)

[Presentation](https://github.com/oicjacky/ML-Stock-price-forecast/blob/master/ML%20presentation%200613.pdf)

# V. 問題討論:

|   |Random Forest|AR       |LSTM|LSTM+Random Forest|
|---|-------------|---------|----|--------------|
|訓練速度|快       |最快     |慢   |最慢|
|準確率/MSE|52.26%/33.41 |\* /10.99|\* /19.07|54.41%/38.30|


  - 不論是做哪個模型，先將資料標準化對預測的結果差異相當大。
  - 其中我們有將技術指標轉換為1,-1，再用Random Forest做預測，發現到這個方法並不可行，準確率不高。也有嘗試用過LSTM + Random Forest的方法，但是預測結果和單純使用Random Forest差不多。
  - AR 模型預測結果最佳，研判可能原因在於資料包含時間性，因此機器學習方法預測誤差較大，利用時間序列模型會得到較好結果，未來也可以試著利用時間序列模型結合機器學習方法，或許會得到更好的結果。 由於我們目前只預測隔天的股價，期望未來可以預測長期趨勢，或是預測出最佳的投資策略。
  
  [Detail report](https://github.com/oicjacky/ML-Stock-price-forecast/blob/master/ML_report-final.pdf)


