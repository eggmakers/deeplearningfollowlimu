import torch


#标量只有一个元素的张量表示
x = torch.tensor([3.0])
y = torch.tensor([2.0])

print(x, '\n', y, '\n', x + y, '\n',  x * y, '\n',  x / y, '\n', x ** y)


#可以将向量视为标量值组成的列表
x = torch.arange(4)
print(x)


#通过访问张量的索引来访问任意元素
print(x[3])


#创建m*n的矩阵
A = torch.arange(20).reshape(5, 4)
print(A)


#转置
print(A.T)


#对称矩阵
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(B == B.T)#其转置等于其本身


#矩阵是向量的推广
X = torch.arange(24).reshape(2, 3, 4)
print(X)


#给定相同形状的任何两个张量，二元运算的结果是相同的
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()
print(A.shape, '\n', A + B)


#哈达玛积
print(A * B)
a = 2
X = torch.arange(24).reshape(2, 3, 4)
print(a + X)
print((a * X).shape)


#求和
print(X.sum())


C = A
#任意形状张量的元素和
A = torch.arange(20.0 * 2).reshape(2, 5, 4)
print('A = ', A)
print(A.shape, '\n', A.sum())


#制定球和汇总张量的轴
A_sum_axis0 = A.sum(axis=0)
print(A_sum_axis0, '\n', A_sum_axis0.shape)
print(A.sum(axis=[0, 1]))


#平均值
print(A.mean())
print(A.sum() / A.numel())
print(A.mean(axis=0))
print(A.sum(axis=0) / A.shape[0])


#计算总和或均值是保持轴数不变
sum_A = A.sum(axis=1, keepdim=True)
print(sum_A)
print(A / sum_A)


#累计总和
print(A.cumsum(axis=0))


#点积
y = torch.ones(4, dtype=torch.float32)
print(x)
print(y)
# print(torch.dot(x, y))
print(torch.sum(x * y))

A = C
print(A.shape)
print(x.shape)
# print(torch.mv(A, x))


#矩阵拼接
B = torch.ones(4, 3)
print(torch.mm(A, B))


#范数
u = torch.tensor([3.0, -4.0])
print(torch.norm(u))

#范数之和
print(torch.abs(u).sum())


#矩阵的F范数
print(torch.norm(torch.ones((4, 9))))


#按特定轴求和
a = torch.ones((2, 5, 4))
print(a.shape)
print(a.sum(axis=1).shape)
print(a.sum(axis=1))
print(a.sum(axis=1, keepdim=True).shape)
print(a.sum(axis=1, keepdim=True))