# -*- coding: utf-8 -*-
import caffe
from caffe import layers as L, params as P, proto, to_proto

# 设定文件的保存路径
root = '/home/liangw/work/caffe/'  # 根目录
#train_list = root + 'mnist/train/train.txt'  # 训练图片列表
#test_list = root + 'mnist/test/test.txt'  # 测试图片列表
train_lmdb = root + 'examples/compact_bilinear/train_lmdb' # 训练集lmdb
test_lmdb = root + 'examples/compact_bilinear/val_lmdb'
#train_meanfile = root + 'data/cifar100_train_mean.binaryproto' #训练集均值文件 
#test_meanfile = root + 'data/cifar100_test_mean.binaryproto'

train_proto = root + 'examples/attentive_cb/attentive_cb_train.prototxt'  # 训练配置文件
test_proto = root + 'examples/attentive_cb/attentive_cb_test.prototxt'  # 测试配置文件
solver_proto = root + 'examples/attentive_cb/attentive_cb_solver.prototxt'  # 参数文件
weight = root + 'examples/attentive_cb/VGG_ILSVRC_16_layers.caffemodel'

def Network(lmdb, batch_size, include_acc=False, finetune_last=False):
    n = caffe.NetSpec()
    if include_acc:
        n.data,n.label = L.Data(source=lmdb, name='data', backend=P.Data.LMDB, batch_size=batch_size, ntop=2, transform_param=dict(mirror=False, crop_size=448, mean_value=[104,117,123]))
    else:
        n.data,n.label = L.Data(source=lmdb, name='data', backend=P.Data.LMDB, batch_size=batch_size, ntop=2, transform_param=dict(mirror=True, crop_size=448, mean_value=[104,117,124]))
    
    #basenet_VGG
    if (finetune_last == True):
        w_lr_mult = 0.0
        w_decay_mult = 0.0
        b_lr_mult = 0.0
        b_decay_mult = 0.0
    else:
        w_lr_mult = 1.0
        w_decay_mult = 1.0
        b_lr_mult = 2.0
        b_decay_mult = 0.0

    n.conv1_1 = L.Convolution(n.data, name='conv1_1', kernel_size=3, pad=1, num_output=64, param=[dict(lr_mult=w_lr_mult, decay_mult=w_decay_mult),\
                    dict(lr_mult=b_lr_mult, decay_mult=b_decay_mult)])
    n.relu1_1 = L.ReLU(n.conv1_1, name='relu1_1')
    n.conv1_2 = L.Convolution(n.relu1_1, name='conv1_2', kernel_size=3, pad=1, num_output=64, param=[dict(lr_mult=w_lr_mult, decay_mult=w_decay_mult),\
                    dict(lr_mult=b_lr_mult, decay_mult=b_decay_mult)])    
    n.relu1_2 = L.ReLU(n.conv1_2, name='relu1_2')
    n.pool1 = L.Pooling(n.relu1_2, name='pool1', pool=P.Pooling.MAX, kernel_size=2, stride=2)


    n.conv2_1 = L.Convolution(n.pool1, name='conv2_1', kernel_size=3, pad=1, num_output=128, param=[dict(lr_mult=w_lr_mult, decay_mult=w_decay_mult),\
                    dict(lr_mult=b_lr_mult, decay_mult=b_decay_mult)])    
    n.relu2_1 = L.ReLU(n.conv2_1, name='relu2_1')
    n.conv2_2 = L.Convolution(n.relu2_1, name='conv2_2', kernel_size=3, pad=1, num_output=128, param=[dict(lr_mult=w_lr_mult, decay_mult=w_decay_mult),\
                    dict(lr_mult=b_lr_mult, decay_mult=b_decay_mult)])          
    n.relu2_2 = L.ReLU(n.conv2_2, name='relu2_2')
    n.pool2 = L.Pooling(n.relu2_2, name='pool2', pool=P.Pooling.MAX, kernel_size=2, stride=2)


    n.conv3_1 = L.Convolution(n.pool2, name='conv3_1', kernel_size=3, pad=1, num_output=256, param=[dict(lr_mult=w_lr_mult, decay_mult=w_decay_mult),\
                    dict(lr_mult=b_lr_mult, decay_mult=b_decay_mult)])    
    n.relu3_1 = L.ReLU(n.conv3_1, name='relu3_1')
    n.conv3_2 = L.Convolution(n.relu3_1, name='conv3_2', kernel_size=3, pad=1, num_output=256, param=[dict(lr_mult=w_lr_mult, decay_mult=w_decay_mult),\
                    dict(lr_mult=b_lr_mult, decay_mult=b_decay_mult)])          
    n.relu3_2 = L.ReLU(n.conv3_2, name='relu3_2')
    n.conv3_3 = L.Convolution(n.relu3_2, name='conv3_3', kernel_size=3, pad=1, num_output=256, param=[dict(lr_mult=w_lr_mult, decay_mult=w_decay_mult),\
                    dict(lr_mult=b_lr_mult, decay_mult=b_decay_mult)])    
    n.relu3_3 = L.ReLU(n.conv3_3, name='relu3_3')
    n.pool3 = L.Pooling(n.relu3_3, name='pool3', pool=P.Pooling.MAX, kernel_size=2, stride=2)    


    n.conv4_1 = L.Convolution(n.pool3, name='conv4_1', kernel_size=3, pad=1, num_output=512, param=[dict(lr_mult=w_lr_mult, decay_mult=w_decay_mult),\
                    dict(lr_mult=b_lr_mult, decay_mult=b_decay_mult)])    
    n.relu4_1 = L.ReLU(n.conv4_1, name='relu4_1')
    n.conv4_2 = L.Convolution(n.relu4_1, name='conv4_2', kernel_size=3, pad=1, num_output=512, param=[dict(lr_mult=w_lr_mult, decay_mult=w_decay_mult),\
                    dict(lr_mult=b_lr_mult, decay_mult=b_decay_mult)])          
    n.relu4_2 = L.ReLU(n.conv4_2, name='relu4_2')
    n.conv4_3 = L.Convolution(n.relu4_2, name='conv4_3', kernel_size=3, pad=1, num_output=512, param=[dict(lr_mult=w_lr_mult, decay_mult=w_decay_mult),\
                    dict(lr_mult=b_lr_mult, decay_mult=b_decay_mult)])    
    n.relu4_3 = L.ReLU(n.conv4_3, name='relu4_3')
    n.pool4 = L.Pooling(n.relu4_3, name='pool4', pool=P.Pooling.MAX, kernel_size=2, stride=2)


    n.conv5_1 = L.Convolution(n.pool4, name='conv5_1', kernel_size=3, pad=1, num_output=512, param=[dict(lr_mult=w_lr_mult, decay_mult=w_decay_mult),\
                    dict(lr_mult=b_lr_mult, decay_mult=b_decay_mult)])    
    n.relu5_1 = L.ReLU(n.conv5_1, name='relu5_1')
    n.conv5_2 = L.Convolution(n.relu5_1, name='conv5_2', kernel_size=3, pad=1, num_output=512, param=[dict(lr_mult=w_lr_mult, decay_mult=w_decay_mult),\
                    dict(lr_mult=b_lr_mult, decay_mult=b_decay_mult)])          
    n.relu5_2 = L.ReLU(n.conv5_2, name='relu5_2')
    n.conv5_3 = L.Convolution(n.relu5_2, name='conv5_3', kernel_size=3, pad=1, num_output=512, param=[dict(lr_mult=w_lr_mult, decay_mult=w_decay_mult),\
                    dict(lr_mult=b_lr_mult, decay_mult=b_decay_mult)])    
    n.relu5_3 = L.ReLU(n.conv5_3, name='relu5_3') 

    #bilinear
    n.bilinear_layer = L.CompactBilinear(n.relu5_3, n.relu5_3, name='bilinear_layer', compact_bilinear_param=dict(num_output=8192,sum_pool=False))

    #sqrt+l2
    n.signed_sqrt = L.SignedSqrt(n.bilinear_layer, name='signed_sqrt')
    n.l2_normalization= L.L2Normalize(n.signed_sqrt, name='l2_normalization')

    #FC
    n.fc = L.InnerProduct(n.l2_normalization, name='fc', num_output=200, param=[dict(lr_mult=1, decay_mult=1),\
            dict(lr_mult=2, decay_mult=0)], weight_filler=dict(type='gaussian',std=0.0001), bias_filler=dict(type='constant',value=0.0))
    n.loss = L.SoftmaxWithLoss(n.fc, n.label, name='loss')
    if include_acc:  # test阶段需要有accuracy层
        n.accuracy = L.Accuracy(n.fc, n.label, name='accuracy')
        return to_proto(n.loss, n.accuracy)
    else:
        return to_proto(n.loss)    

def write_net(finetune_last=False):
    # 写入train.prototxt
    with open(train_proto, 'w') as f:
        f.write(str(Network(train_lmdb, batch_size=8, finetune_last=finetune_last)))

    # 写入test.prototxt
    with open(test_proto, 'w') as f:
        f.write(str(Network(test_lmdb, batch_size=4, include_acc=True, finetune_last=finetune_last)))

# 编写一个函数，生成参数文件
def gen_solver(solver_file, train_net, test_net):
    s = proto.caffe_pb2.SolverParameter()
    s.train_net = train_net
    s.test_net.append(test_net)
    s.test_interval = 4000  # 每训练多少次进行一次测试
    s.test_iter.append(1450)  # 测试样本数/test_batch_size,如10000/100=100
    s.test_initialization=False
    s.average_loss=100
    s.max_iter = 60000  # 最大训练次数
    s.base_lr = 1  # 基础学习率
    s.momentum = 0.9  # 动量
    s.weight_decay = 0.000005  # 权值衰减项
    s.lr_policy = 'multistep'  # 学习率变化规则
    s.stepvalue.append(20000)
    s.stepvalue.append(30000) 
    s.stepvalue.append(40000)
    s.stepvalue.append(50000)

    #s.stepsize = 3000  # 学习率变化频率
    s.gamma = 0.25  # 学习率变化指数
    s.display = 100  # 屏幕显示间隔
    s.snapshot = 10000  # 保存caffemodel的间隔
    s.snapshot_prefix = root + 'examples/attentive_cb/snapshot/att_cb'  # caffemodel前缀
    s.type = 'SGD'  # 优化算法
    s.solver_mode = proto.caffe_pb2.SolverParameter.GPU  # 加速
    # 写入solver.prototxt
    with open(solver_file, 'w') as f:
        f.write(str(s))


# 开始训练
def training(solver_proto):
    caffe.set_device(2) #设置GPU
    caffe.set_mode_gpu()
    solver = caffe.SGDSolver(solver_proto)
    solver.net.copy_from(weight) # finetune预训练模型
    solver.solve()

#主函数
if __name__ == '__main__':
    write_net(finetune_last=True)
    gen_solver(solver_proto, train_proto, test_proto)
    training(solver_proto)
