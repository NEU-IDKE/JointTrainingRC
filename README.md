# JointTrainingRC
一种联合训练框架用于学习更加可辩别的关系特征
## 准备事项
1、创建pretrained_model目录，下载相应的预训练权重
2、新建outputs/best_checkpoint目录，用来保存断点
3、因为实验用到了对比学习，所以需要将数据集处理成对比学习的格式，在datasets目录中解压即可
4、执行train.py即可
