import torch

def conjugate_gradient(Ax, b, x=None, tol=1e-10, max_iter=1000):
    """
    用共軛梯度法求解 Ax = b，其中 A 是一個對稱正定矩陣。

    參數:
    Ax: callable，計算矩陣 A 與向量 x 的乘積函數
    b: PyTorch Tensor，目標向量
    x: 初始向量，如果不提供則默認為 0 向量
    tol: 誤差容忍度
    max_iter: 最大迭代次數

    返回:
    x: PyTorch Tensor，近似解
    """
    if x is None:
        x = torch.zeros_like(b)

    r = b - Ax(x)
    p = r.clone()
    rs_old = torch.dot(r, r)

    for i in range(max_iter):
        Ap = Ax(p)
        alpha = rs_old / torch.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap

        rs_new = torch.dot(r, r)
        if torch.sqrt(rs_new) < tol:
            print(f"Converged after {i+1} iterations.")
            break

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x

# 示例使用 假設我們有對稱正定矩陣 A 和向量 b
n = 10
A = torch.randn(n, n)
A = A.t().mm(A)  # 搆造對稱正定矩陣
b = torch.randn(n)

# 定義矩陣乘法 Ax
Ax = lambda x: A.mm(x.unsqueeze(1)).squeeze(1)

# 使用共軛梯度法求解 Ax = b
x = conjugate_gradient(Ax, b)

# 檢查結果
print("Residual:", torch.norm(A.mm(x.unsqueeze(1)).squeeze(1) - b))
**********************************************
from torch import randn as trchRandn, randint as trchRndnint
from torch.nn import CrossEntropyLoss, Module, Linear, BatchNorm1d, ReLU
from torch.optim import Adam

class SimpleNet(Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = Linear(784, 256)
        self.bn1 = BatchNorm1d(256)  # 批歸一化層
        self.fc2 = Linear(256, 10)
        self.relu = ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)  # 批歸一化
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNet() # 實例化模型、損失函數和優化器
criterion = CrossEntropyLoss()
優化器 = Adam(model.parameters(), lr=0.001)

input_data = trchRandn(64, 784)     #假設是批次大小為64的輸入
labels = trchRndnint(0, 10, (64,))  #隨機生成標簽

output = model(input_data) # 訓練步驟示例
loss = criterion(output, labels)

優化器.zero_grad() # 反向傳播和優化
loss.backward()
優化器.step()
