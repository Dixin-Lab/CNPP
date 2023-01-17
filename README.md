# TPP Alignment

For data generation

```bash
cd ./data_generation
python synthetic_graph.py
```

For training

```bash
cd ./Joint_sahp
sh test.sh	
```





总结

10点 90边  seq_num=4000(3200 train，800 test/dev) end time=150 有向图，mu_i=d_i/sum(d_i)生成数据

### **思路1**：**无监督** 参数化双随机矩阵P, P=gumbel_sinkhorn(f(X1)) ,X2=P@X1

f(x)是对P的参数化，gumbel_sinkhorn将他变成一个可学习的双随机矩阵。



f(x)=F1(x)

sigma(F1(x))

F2(sigma(F1(x)))

sigma(F2(sigma(F1(x))))

其中，F1，F2为线性变换(乘矩阵)，sigma为激活函数(这里用relu)



以上参数化方式效果均不理想。

核心代码 sahp.py

![image-20230112200529403](https://github.com/Zephyr-29/TPP-Align/blob/main/image-20230112200529403.png)

![image-20230112200651099](https://github.com/Zephyr-29/TPP-Align/blob/main/image-20230112200651099.png)



loss function=nll **无正则**



当左图的tpp seq来时，我们从X1取emb，当右图的tpp seq来时，我们取从X2=P@X1作为事件的emb。

试过X2=P@X1.detach() ,即对P和X1交替优化，效果与X2=P@X1相似。



### **思路2**

在思路1的基础上加正则

loss function function=nll + 系数*||mu2-P@mu1||_F^2



### **思路3**

学两组事件emb X1，X2，计算X1和X2之间的相似度 or gumbel_sinkhorn(X1@X2.T) 作为匹配矩阵

loss function=nll



以上思路效果均**不好**，表现是，在seed 固定时，不同epoch top1 3 5 匹配acc数据极不稳定（不如或略大于0.1，0.3，0.5 ），左图和右图的F1 score 较为稳定，不同epoch有一定波动。

 **F1 score 与seed 的选取关系极大**，会从0.01->0.28,但当seed 固定时，F1 score 不同epoch变化较小。top1 3 5不同epoch 不管什么seed，都波动很大，前一个top1=0.5，后一个可能就是0。



数据量增加(num_seq,len_seq)会增加F1 score 的学习效果。



### **思考**：

1. X1 是否可以加 encoder 

2. loss=nll+正则 
2.  是否可以只学tpp，loss=nll，得到X1或X1  X2作为节点的初始化，然后作为初始化在送到什么地方做后续的图匹配。但感觉，用tpp学出的emb做初始化应该不如目标明确的那个deep walk，node2vec 专门提取图的结构特征的方法好。

4. gumbel_sinkhorn 两个参水 tau(温度)，iter(迭代次数)。控制这两个参数可以使得P硬或软。硬只里面接近只有01元素，软指里面全是0.几 0.几 0.几（懂我意思 /doge) 考虑到我们将P作为最终的prediction 结果。在硬的时候，top1有意义，感觉top3 5 没意义，都是10(-20)了，在软的时候，top1 3 5都有

### For许老师debug：

![image-20230112202421518](https://github.com/Zephyr-29/TPP-Align/blob/main/image-20230112202421518.png)

与然哥上传的第一版代码相比，只有这5个文件是新增or变化的。

tick_simulate.py 用tick 生成并保存数据

sinkhorn.py Gumbel_sinkorn的实现（我直接粘贴的 错不了 也看过生成的双随机矩阵的矩阵效果 贼好）

sahp.py  主要看init()  forward()  get_P() 三个函数就行

train_sahp.py 加正则/测试acc

metric.py 根据ground truth 测试匹配结果acc



跨文件种子？？
