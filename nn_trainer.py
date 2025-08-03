import torch
from torch import Tensor, nn
import matplotlib.pyplot as plt

from nn import DROptRunner

device = "cuda"
torch.set_default_device(device)

learn_span = 365
termination_prob = 1e-3

w = torch.tensor([
    0.212,
    1.2931,
    2.3065,
    8.2956,
    6.4133,
    0.8334,
    3.0194,
    0.001,
    1.8722,
    0.1666,
    0.796,
    1.4835,
    0.0614,
    0.2629,
    1.6483,
    0.6014,
    1.8729,
    0.5425,
    0.0912,
    0.0658,
    0.1542,
], device=device, requires_grad=True)

def stability_after_success(s, r, d, rating):
    hard_penalty = torch.where(rating == 2, w[15], 1.0)
    easy_bonus = torch.where(rating == 4, w[16], 1.0)
    return torch.clamp_min(
        s
        * (
            1
            + torch.exp(w[8])
            * (11 - d)
            * torch.pow(s, -w[9])
            * (torch.exp((1 - r) * w[10]) - 1)
            * hard_penalty
            * easy_bonus
        ),
        S_MIN,
    )


def stability_after_failure(s, r, d):
    return torch.clamp_min(
        torch.minimum(
            w[11]
            * torch.pow(d, -w[12])
            * (torch.pow(s + 1, w[13]) - 1)
            * torch.exp((1 - r) * w[14]),
            s / torch.exp(w[17] * w[18]),
        ),
        S_MIN,
    )


def init_d(rating):
    return w[4] - torch.exp(w[5] * (rating - 1)) + 1


def linear_damping(delta_d, old_d):
    return delta_d * (10 - old_d) / 9


def mean_reversion(init, current):
    return w[7] * init + (1 - w[7]) * current


def next_d(d, rating):
    delta_d = -w[6] * (rating - 3)
    new_d = d + linear_damping(delta_d, d)
    new_d = mean_reversion(init_d(4), new_d)
    return torch.clamp(new_d, 1, 10)


def power_forgetting_curve(t, s, decay: float = -w[20]):
    factor = 0.9 ** (1 / decay) - 1
    return (1 + factor * t / s) ** decay

def integral_power_forgetting_curve(t: torch.Tensor, s: torch.Tensor, decay: torch.Tensor):
    factor = 0.9 ** (1 / decay) - 1

    # Handle special case when decay == -1 to avoid division by zero
    if torch.isclose(decay, torch.tensor(-1.0, device=decay.device)):
        # Integral = (s / factor) * log|1 + factor * t / s|
        return (s / factor) * torch.log(torch.abs(1 + factor * t / s))
    else:
        numerator = s
        denominator = factor * (decay + 1)
        base = 1 + factor * t / s
        return numerator / denominator * torch.pow(base, decay + 1)


def next_interval(s, r, decay: float = -w[20]):
    factor = 0.9 ** (1 / decay) - 1
    ivl = s / factor * (r ** (1 / decay) - 1)
    return ivl.clamp(1, 365 * 10)

ratio2 = None

S_MIN = 0.001
class DROpt(DROptRunner):
    def expected_learn(self, acc_prob, stability, difficulty, day, tr, cost) -> Tensor:
        if day >= learn_span or acc_prob <= termination_prob:
            return torch.tensor([0.0], dtype=torch.float32, device=device, requires_grad=True)

        # print(stability > 0)
        if stability > 0:
            # print("here", stability, tr, difficulty)
            s_recall = stability_after_success(stability, tr, difficulty, torch.tensor([3.0], device=device, requires_grad=True))
            # print(s_recall)
            s_forget = stability_after_failure(stability, tr, difficulty)
            d_recall = next_d(difficulty, 3.0)
            d_forget = next_d(difficulty, 1.0)
        else:
            s_recall = w[2].unsqueeze(0)
            s_forget = w[0].unsqueeze(0)
            d_recall = init_d(torch.tensor([3.0]))
            d_forget = init_d(torch.tensor([1.0]))

        dr = self.lerp_s(stability, difficulty)
        
        # print(stability, s_recall)

        ivl_recall = next_interval(s_recall, dr).squeeze(0)
        tr_recall = power_forgetting_curve(ivl_recall, s_recall)
        # print(ivl_recall, s_recall, tr_recall)
        ivl_forget = next_interval(s_forget, dr).squeeze(0)
        tr_forget = power_forgetting_curve(ivl_forget, s_forget)

        cost_total = acc_prob * cost
        cost_total = cost_total + self.expected_learn(
            acc_prob * tr,
            s_recall,
            d_recall,
            day + ivl_recall,
            tr_recall,
            self.RECALL_COST,
        )
        cost_total = cost_total + self.expected_learn(
            acc_prob * (1 - tr),
            s_forget,
            d_forget,
            day + ivl_forget,
            tr_forget,
            self.FORGET_COST,
        )
        return cost_total

    def memorization(self) -> torch.Tensor:
        s = torch.arange(1, 100)
        interval = next_interval(s, self.lerp_s(s, init_d(torch.ones_like(s) * 3)))
        return integral_power_forgetting_curve(interval, s, -w[20]).mean()

    def forward(self):
        cost = self.expected_learn(
            torch.tensor(1.0, requires_grad=True, dtype=torch.float32),
            torch.tensor([0.0], requires_grad=True, dtype=torch.float32),
            torch.tensor([0.0], requires_grad=True, dtype=torch.float32),
            torch.tensor([0],   requires_grad=True, dtype=torch.float32),
            torch.tensor(0.8, requires_grad=True, dtype=torch.float32),
            self.LEARN_COST)

        memorised = self.memorization()

        return cost, memorised


model = DROpt().to(device)
optimizer = torch.optim.Adam(model.nn.parameters(), lr=0.01)

try:
    for epoch in range(100):
        optimizer.zero_grad()

        cost, memorised = model() 
        cost.retain_grad()
        memorised.retain_grad()
        loss = (1000 * cost / memorised)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch} Loss: {loss.item():.4f}, mem/min={(memorised / (cost * 60)).item()}, cost={cost.item()}, memorised={memorised.item()}")
finally:
    torch.save(model, "model.pth")

