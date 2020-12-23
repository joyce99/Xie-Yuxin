# import os
# # from tensorflow.python import pywrap_tensorflow
# #
# # model_dir = './mycheckpoints'
# # checkpoint_path = os.path.join(model_dir,'model-12.data-00000-of-00001') # 保存的ckpt文件名，不一定是这个
# # # Read data from checkpoint file
# # reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
# # var_to_shape_map = reader.get_variable_to_shape_map()
# # # Print tensor name and values
# # for key in var_to_shape_map:
# #     print("tensor_name: ", key)
# #     print(reader.get_tensor(key)) # 打印变量的值，对我们查找问题没啥影响，打印出来反而影响找问题
from matplotlib import pyplot as plt
import numpy as np

x = np.linspace(1, 100, num= 25, endpoint = True)
style_list = ["g+-", "r*-", "b.-", "yo-"]
def y_subplot(x,i):
    return np.cos(i * np.pi *x)
plt.figure(figsize=(8,8))
for i in range(1,5):
    plt.subplot(2,2,i)
    plt.plot(x, y_subplot(x,i), marker='o')
    plt.xlabel('x%d' % i)
    plt.ylabel('y%d' % i)
    plt.title('%d' % i)
    plt.tight_layout()
    # plt.axis('off') #  关掉x y轴的刻度
plt.subplots_adjust(wspace=0,hspace=0)
plt.show()
