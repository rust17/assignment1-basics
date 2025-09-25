from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1

        return loss

if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1e1)

    print("初始参数形状:", weights.shape)
    print("初始参数值:", weights)

    for t in range(10):
        opt.zero_grad()
        loss = (weights**2).mean()

        # 打印详细信息
        print(f"\n=== 第 {t} 次迭代 ===")
        print(f"损失值: {loss.item():.6f}")
        if t == 0:
            print("初始参数示例:", weights[0, 0].item())

        loss.backward()

        print(f"梯度示例: {weights.grad[0, 0].item():.6f}")
        print(f"参数更新前: {weights[0, 0].item():.6f}")

        opt.step()

        print(f"参数更新后: {weights[0, 0].item():.6f}")
