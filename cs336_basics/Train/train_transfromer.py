import torch
import logging

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # 实现公式：ℓ = log(∑exp(o[i] - m)) - o[y] + m
    # 其中 o 是单个样本的 logits 向量，y 是正确答案索引，m 是 o 中的最大值
    # 首先要找到输入向量 logits 每一行的最大值
    max_val = torch.max(logits, dim=1, keepdim=True).values

    # 首先，计算 exp(...)
    # 然后，计算 ∑(...)，也就是求和
    # 最后，计算 log(...)
    # torch.log(torch.sum(torch.exp(logits - max_val), dim=1, keepdim=True)) 等价于
    # torch.logsumexp(logits - max_val, dim=1, keepdim=True)
    logsumexp_val = torch.logsumexp(logits - max_val, dim=1, keepdim=True)

    # 根据 targets 来精确地从 logits (也就是 o) 中挑出正确答案对应的那个 logit 值 o[y]
    # 首先需要把 targets 向量从形状 (batch_size,) 转变成 (batch_size, 1)
    correct_logits = torch.gather(logits, 1, targets.unsqueeze(1))

    # 返回平均损失
    return (logsumexp_val - correct_logits + max_val).mean()

from collections.abc import Callable
from typing import Optional
import math
# 如果需要展示详细日志，请执行： uv run pytest -k test_adamw --log-cli-level=INFO
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        super().__init__(params, {
            'lr': lr, # 学习率
            'betas': betas, # 超参数
            'eps': eps, # 防除零参数
            'weight_decay': weight_decay, # 权重衰减率
        }) # 模型创建后，把所有的参数会传给基类的构造函数，基类会把它们整理成一个列表，叫做 self.param_groups，通常情况下，这个 self.param_groups 只包含一个元素，这个元素是一个字典，其中包含了 params 键，这个键所对应的值就是 θ（PyTorch 里又叫 p）

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                logging.info(f"=======开始=======")
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data) # 返回一个和 p 形状一样的张量
                    state['v'] = torch.zeros_like(p.data)

                state['step'] = state['step'] + 1
                beta1, beta2 = group['betas']
                # 当前梯度
                grad = p.grad.data
                logging.info(f"当前梯度{grad}")

                # 惯性小球
                # m ←β1m + (1−β1)g
                state['m'] = beta1 * state['m'] + (1 - beta1) * grad
                logging.info(f"惯性小球{state['m']}")

                # 环境探测器
                # v ←β2v + (1−β2)g2
                state['v'] = beta2 * state['v'] + (1 - beta2) * grad * grad
                logging.info(f"环境探测器{state['v']}")

                # 校准
                # 1 - β1^t
                beta1_correct = 1 - beta1 ** state['step']
                # 1 - β2^t
                beta2_correct = 1 - beta2 ** state['step']
                # α_t = α * sqrt(beta2_correct) / beta1_correct
                lr_correct = group['lr'] * math.sqrt(beta2_correct) / beta1_correct

                # 实际步长张量
                # step_size = α_t * (m / (√v + ϵ))
                step_size = lr_correct * (state['m'] / (torch.sqrt(state['v']) + group['eps']))
                logging.info(f"实际步长张量{step_size}")
                logging.info(f"下山前{p.data}")
                # θ ← θ - step_size
                p.data = p.data - step_size # 下山
                logging.info(f"下山后{p.data}")

                # 权重衰减，防止过拟合
                # θ ← θ - αλθ
                p.data = p.data - group['lr'] * group['weight_decay'] * p.data
                logging.info(f"防止过拟合{p.data}")
                logging.info(f"=======结束=======")

        return loss

def cosine_annealing(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    # t: 当前是第几步 (current iteration)
    # alpha_max: 最大学习率
    # alpha_min: 最小（最终）学习率
    # Tw: 预热阶段的步数 (Warm-up iterations)
    # Tc: 余弦退火阶段结束的步数 (Cosine annealing iterations)
    if it < warmup_iters:
        # 预热阶段 (Warm-up): 如果 t < Tw
        # α_t = (t / Tw) * alpha_max
        # 这是一个从 0 到 alpha_max 的线性增长。
        learn_rate = (it / warmup_iters) * max_learning_rate
    elif it >= warmup_iters and it <= cosine_cycle_iters:
        # 余弦退火阶段 (Cosine annealing): 如果 Tw <= t <= Tc
        # α_t = alpha_min + 0.5 * (1 + cos(...)) * (alpha_max - alpha_min)
        # 这里的 ... 是 (t - Tw) / (Tc - Tw) * π
        # 这是一个从 alpha_max 平滑下降到 alpha_min 的余弦曲线。
        x = (it - warmup_iters)/(cosine_cycle_iters - warmup_iters)*math.pi
        learn_rate = min_learning_rate + 0.5 * (1 + math.cos(x)) * (max_learning_rate - min_learning_rate)
    else:
        # 后退火阶段 (Post-annealing): 如果 t > Tc
        # α_t = alpha_min
        # 学习率保持在最小值不变。
        learn_rate = min_learning_rate
    return learn_rate

from collections.abc import Iterable

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    # 输入是一系列模型参数 (a list of parameters) 和一个最大范数 M (max_norm)。这个函数要原地 (in-place) 修改这些参数的梯度
    total_norm_sq = 0.0

    # 第一步：计算总范数
    # 你需要遍历所有传入的参数 p。对于每个 p，如果它有梯度 (p.grad 不是 None)，你就需要：
    # 计算这个梯度张量的 ℓ2-norm。torch.norm() 函数可以直接做到。
    # 由于我们要的是所有梯度的总范- 数，而不是每个张量的范数，所以更直接的方法是：
    # 把 p.grad 的所有元素求平方，然后加起来。
    # 把所有参数的这个“平方和”累加起来，得到一个 total_norm_sq。
    # 最后，对 total_norm_sq 开平方根 torch.sqrt()，就得到了 total_norm。
    for p in parameters:
        if p.grad is None:
            continue
        total_norm_sq += torch.sum(torch.norm(p.grad) ** 2)
    total_norm = torch.sqrt(total_norm_sq)

    # 第二步：进行裁剪
    # 计算裁剪系数 clip_coef = max_norm / (total_norm + 1e-6)。
    # 如果 clip_coef < 1 (这等价于 total_norm > max_norm)，就再次遍历所有参数 p。
    # 对于每个有梯度的 p，执行 p.grad.mul_(clip_coef)。.mul_() 是一个原地乘法操作。

    if total_norm >= max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in parameters:
                if p.grad is not None:
                    p.grad.mul_(clip_coef)

