# PSENet-Tensorflow2
使用tensorflow2实现PSENet  

这边我遇到了问题，不知道是我代码写的有问题，还是什么，我这边训练了10W个step之后，然后进行预测，但是预测的结果有问题，如果有小伙伴能帮忙解决，我这边进行有偿，因为学生党，钱不多。当然更好的是一起进行探讨了。邮箱: m18888170129@163.com  

本人尝试使用tensorlfow2实现PSENET的模型，此处参考了 https://github.com/looput/PSENet-Tensorflow 的实现  
中途需要用到tfrecord，我这边将需要的训练数据集放到网盘上了 ，有需要的可以下载直接训练，也可以自己生成数据集  
所有的配置文件在config中，需要的同学自己对应修改下  
****
python write_tfrecord.py 将会在dataset下面生成对应的训练tfrecord文件  
运行 python train.py 进行训练  
tensorboard --logdir=logs 查看损失函数的变化
****
预测：（存在问题，预测不准确）  
运行 python predict.py 进行预测  
