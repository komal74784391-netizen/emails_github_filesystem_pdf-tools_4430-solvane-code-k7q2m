import torch


class GradientSolver:
    def __init__(self, steps=500, lr=0.05):
        self.steps = steps
        self.lr = lr

    def solve(self, problem):
        x = problem.relaxed_init()
        optimiser = torch.optim.Adam([x], lr=self.lr)
        for _ in range(self.steps):
            loss = problem.constraint_loss(x)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        return problem.round(x)


class ProjectionSolver:
    def __init__(self, iterations=200):
        self.iterations = iterations

    def solve(self, problem):
        x = problem.relaxed_init()
        for _ in range(self.iterations):
            x = problem.project(x)
        return problem.round(x)
