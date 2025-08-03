import torch
from torch import nn
torch.set_default_device("cuda")
from script import DEVICE, plot_simulation, s_max_aware_next_interval, simulation_table, w

class DROpt(nn.Module):
    def __init__(self):
        super().__init__()

        self.nn = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(2, 100),
            nn.ReLU(),            
            nn.Linear(100, 1),
            nn.Sigmoid()  # Output in (0, 1)
        )
        self.FORGET_COST = torch.tensor(23.185, dtype=torch.float32)
        self.RECALL_COST = torch.tensor(7.8454, dtype=torch.float32)
        self.LEARN_COST = torch.tensor(19.4698, dtype=torch.float32)

    def lerp_s(self, s, d):
        # print(s, d)
        # import traceback

        # traceback.print_stack()
        s = s.to(torch.float32)
        d = d.to(torch.float32)
        out = self.nn(torch.stack([s, d], dim=-1)).squeeze(-1)
        out = torch.lerp(torch.ones_like(out) * 0.7,torch.ones_like(out) * 0.95, out)
        # print(out)
        return out


    def forward(self, s, d):
        return self.lerp_s(s, d)

def run_neural_network():
    model = torch.load("model.pth", weights_only=False).to(DEVICE)

    s = torch.arange(1, 100)
    r = model(s, torch.ones_like(s) * 4)
    print(r)

    plot_simulation(
        lambda s, d: s_max_aware_next_interval(s,d, model(s,d), -w[20]),
        f"nn",
    )

if __name__ == "__main__":

    run_neural_network()
    print("--------------------------------")

    print(
        "| Scheduling Policy | Average number of reviews per day | Average number of minutes per day | Total knowledge at the end | Knowledge per minute |"
    )
    print("| --- | --- | --- | --- | --- |")
    for (
        title,
        review_cnt_per_day,
        cost_per_day,
        memorized_cnt_at_end,
        knowledge_per_minute,
    ) in simulation_table:
        print(
            f"| {title} | {review_cnt_per_day:.1f} | {cost_per_day:.1f} | {memorized_cnt_at_end:.0f} | {knowledge_per_minute:.0f} |"
        )
