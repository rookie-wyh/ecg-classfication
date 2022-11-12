# ecg_classfication

此项目对心率失常数据集进行分类，分别实现了CNN, RNN, LSTM, GRU, MLP模型进行训练

## 1. 数据下载

MIT心率失常数据集，[下载地址](https://www.physionet.org/content/mitdb/1.0.0/)

## 2. 数据预处理

对原始数据进行信息去噪，并按心率节拍切分数据，保存为txt文件。以下操作在当前目录下生成preprocessing文件夹，包括train.txt和val.txt文件

```python
python preprocessing.py --data_path mit-bih-arrhythmia-database-1.0.0
```

## 3. 训练模型

输入以下命令训练分类模型，更多指定参数可在main.py中查看，其中--model参数可指定为 `lstm`,`cnn`,`rnn`,`gru`,`mlp`其中任一模型

```python
python main.py --model lstm \
--batch_size 128 \
--epochs 20 \
--device cpu
```