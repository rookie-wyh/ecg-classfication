import wfdb
import pywt
import os
import numpy as np
import json
import argparse

# 小波去噪预处理
def denoise(data):
    # 小波变换
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    # 阈值去噪
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    # 小波反变换,获取去噪后的信号
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata

def getDataSet(number, X_data, Y_data, args):

    assert not os.path.exists(args.data_path), "data_path '{}' not exists".format(args.data_path)
    cgClassSet = ['N', 'A', 'V', 'L', 'R']
    index_to_class = {}
    for i in range(len(cgClassSet)):
        index_to_class[i] = cgClassSet[i]
    json_str = json.dumps(index_to_class, indent=4)
    with open("class_indices.json", "w") as f:
        f.write(json_str)

    # 读取心电数据记录
    # print("正在读取 " + number + " 号心电数据...")
    record = wfdb.rdrecord(args.data_path + number, channel_names=['MLII'])
    data = record.p_signal.flatten()
    rdata = denoise(data)
    # 获取心电数据记录中R波的位置和对应的标签
    annotation = wfdb.rdann(args.data_path + number, 'atr')
    Rlocation = annotation.sample  #对应位置
    Rclass = annotation.symbol  #对应标签
    start = 10
    end = 5
    i = start
    j = len(Rclass) - end
    while i < j:
        try:
            label = cgClassSet.index(Rclass[i])
            x_train = rdata[Rlocation[i] - 99 : Rlocation[i] + 201]
            X_data.append(x_train)
            Y_data.append(label)
            i += 1
        except ValueError:
            i += 1
    return

# 加载数据集并进行预处理
def loadData(args):
    numberSet = ['100', '101', '103', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115',
                 '116', '117', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '208',
                 '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230',
                 '231', '232', '233', '234']
    dataSet = []
    labelSet = []
    for n in numberSet:
        getDataSet(n, dataSet, labelSet, args)

    dataSet = np.array(dataSet).reshape((-1, 300))
    labelSet = np.array(labelSet).reshape((-1, 1))
    print(dataSet.shape)
    print(labelSet.shape)

    all_data = np.hstack((dataSet, labelSet))
    shuffle = np.random.permutation(all_data.shape[0])

    val_len = int(0.2 * dataSet.shape[0])
    val_index = shuffle[:val_len]
    train_index = shuffle[val_len:]
    train_data = all_data[train_index]
    val_data = all_data[val_index]
    if not os.path.exists("preprocessing/"):
        os.mkdir("preprocessing/")
    np.savetxt("preprocessing/train.txt", train_data, fmt="%.3f")
    np.savetxt("preprocessing/val.txt", val_data, fmt="%.3f")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default='mit-bih-arrhythmia-database-1.0.0/', type=str)
    args = parser.parse_args()

    loadData(args)