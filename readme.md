# FCM

聚类算法FCM(Fuzzy C-means)实现

## 依赖
>numpy\
pandas

## 参数

>m: m过大时，聚类效果不好，m过小时，类似HCM\
n: 分为n类\
j: 迭代收敛条件\
count: 最大迭代次数

## 用法



```python
import pandas as pd
import FCM
```

In [1]:
```python
data = pd.read_csv("Iris.csv")
data.head()
```
Out [1]:



<div>
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Sepal_length</th>
      <th>Speal_width</th>
      <th>Petal_lenth</th>
      <th>Petal_width</th>
      <th>test_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>5.7</td>
      <td>2.8</td>
      <td>4.1</td>
      <td>1.3</td>
      <td>Iris-versicolor</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>5.2</td>
      <td>3.5</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>5.7</td>
      <td>4.4</td>
      <td>1.5</td>
      <td>0.4</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>5.4</td>
      <td>3.4</td>
      <td>1.5</td>
      <td>0.4</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>6.9</td>
      <td>3.1</td>
      <td>5.1</td>
      <td>2.3</td>
      <td>Iris-virginica</td>
    </tr>
  </tbody>
</table>
</div>


接受Dataframe型数据，labels为参数名

In [2]:

```python
fcm = FCM.FCM(m=4, n=3, j=0.0001, count=100)
labels = ["Sepal_length", "Speal_width", "Petal_lenth", "Petal_width"]
fcm.get_data(data, labels)
fcm.cluster()
```
Out [2]:

    error: 24.928166
    error: 23.645085
    error: 20.755189
    error: 16.210627
    error: 12.842908
    error: 11.973991
    error: 11.710713
    error: 11.614152
    error: 11.575814
    error: 11.559651
    error: 11.552498
    error: 11.549213
    error: 11.547659
    error: 11.546907
    error: 11.546538
    error: 11.546354
    error: 11.546261
    error: 11.546214
    error: 11.546190
    error: 11.546178
    

对应的聚类结果会添加到新列train_label中

In [3]:

```python
data.head()
```
Out [3]:



<div>
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Sepal_length</th>
      <th>Speal_width</th>
      <th>Petal_lenth</th>
      <th>Petal_width</th>
      <th>test_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>5.7</td>
      <td>2.8</td>
      <td>4.1</td>
      <td>1.3</td>
      <td>Iris-versicolor</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>5.2</td>
      <td>3.5</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>5.7</td>
      <td>4.4</td>
      <td>1.5</td>
      <td>0.4</td>
      <td>Iris-setosa</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>5.4</td>
      <td>3.4</td>
      <td>1.5</td>
      <td>0.4</td>
      <td>Iris-setosa</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>6.9</td>
      <td>3.1</td>
      <td>5.1</td>
      <td>2.3</td>
      <td>Iris-virginica</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



如果训练集已有标记，可计算聚类准确度和标记到聚类结果的映射

In [4]:

```python
map_dict = fcm.kind_map("test_label")
print map_dict
```

Out [4]:

    accuracy: 0.906666666667
    {'Iris-virginica': '2', 'Iris-setosa': '0', 'Iris-versicolor': '1'}


