from .solvers import GradientSolver, ProjectionSolver

TASKS = ["sudoku", "nqueens", "graph_coloring", "tsp", "scheduling",
         "portfolio", "circuit", "packing"]


def run_task(name, solver, seed=0):
    problem = make_problem(name, seed)
    solution = solver.solve(problem)
    return evaluate(name, solution)
