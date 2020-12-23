# KE-MCW
Source codes of A Keyphrase Extraction method based on Multi-size Convolution Windows

## Preparation
You need to prepare  the pre-trained word vectors.
* Pre-trained word vectors. Download [GoogleNews-vectors-negative300.bin.gz](https://code.google.com/archive/p/word2vec/)
* Our data is provided [here](https://drive.google.com/file/d/1z1JGWMnQkkWw_4tjptgO-dxXD0OeTfuP/view) for you. Unzip and put these data into data/.


## Details
Multi-size CNN + Joint RNN model + attention

* data文件夹存储数据集

* checkpoints文件夹存储模型训练得到的参数

* mymain.py是训练程序，mypredict.py测试程序

* models/mymodel_mutisize_CNN_LSTM_attention.py定义了我们的模型

* load.py用于加载数据集

* tools.py定义了一些工具函数

## Requirement
tensorflow1.14.0 + nltk

