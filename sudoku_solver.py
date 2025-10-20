from typing import List

from ortools.sat.python import cp_model


NUM_FIGURE = 9

REGION_BLOCKS = {
    'TL': ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)), 
    'T': ((0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)), 
    'TR': ((0, 6), (0, 7), (0, 8), (1, 6), (1, 7), (1, 8), (2, 6), (2, 7), (2, 8)), 
    'L': ((3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (5, 0), (5, 1), (5, 2)), 
    'C': ((3, 3), (3, 4), (3, 5), (4, 3), (4, 4), (4, 5), (5, 3), (5, 4), (5, 5)), 
    'R': ((3, 6), (3, 7), (3, 8), (4, 6), (4, 7), (4, 8), (5, 6), (5, 7), (5, 8)), 
    'BL': ((6, 0), (6, 1), (6, 2), (7, 0), (7, 1), (7, 2), (8, 0), (8, 1), (8, 2)), 
    'B': ((6, 3), (6, 4), (6, 5), (7, 3), (7, 4), (7, 5), (8, 3), (8, 4), (8, 5)), 
    'BR': ((6, 6), (6, 7), (6, 8), (7, 6), (7, 7), (7, 8), (8, 6), (8, 7), (8, 8))
}


class VarMatrixSolutionReceiver(cp_model.CpSolverSolutionCallback):
    def __init__(self, variables: List[List[cp_model.IntVar]]):
        super().__init__()

        self.variables = variables

        self.solution_count = 0
        self.solutions = []

    def on_solution_callback(self):
        self.solution_count += 1

        solution = [[0 for _ in range(NUM_FIGURE)] for _ in range(NUM_FIGURE)]
        for i in range(NUM_FIGURE):
            for j in range(NUM_FIGURE):
                for n in range(NUM_FIGURE):
                    if self.Value(self.variables[i][j][n]):
                        solution[i][j] = n + 1
        self.solutions.append(solution)


class SudokuCPModel(object):
    def __init__(self, info: List[List[int]]):
        self.info = info

        self.solutions = []
        self.unique_solution = None

    def run(self):
        model = cp_model.CpModel()

        x = [[[model.NewBoolVar(name=f'x_{i + 1}_{j + 1}_{n + 1}') 
               for n in range(1, 10)] for j in range(1, 10)] for i in range(1, 10)]

        # initial information
        for i in range(NUM_FIGURE):
            for j in range(NUM_FIGURE):
                if self.info[i][j]:
                    model.Add(x[i][j][self.info[i][j] - 1] == 1)

        # each block has only 1 number
        for i in range(NUM_FIGURE):
            for j in range(NUM_FIGURE):
                model.Add(sum(x[i][j][n] for n in range(NUM_FIGURE)) == 1)

        # each number only appears once in each row
        for i in range(NUM_FIGURE):
            for n in range(NUM_FIGURE):
                model.Add(sum(x[i][j][n] for j in range(NUM_FIGURE)) == 1)

        # each number only appears once in each column
        for j in range(NUM_FIGURE):
            for n in range(NUM_FIGURE):
                model.Add(sum(x[i][j][n] for i in range(NUM_FIGURE)) == 1)

        # each number only appears once in each region
        for region in REGION_BLOCKS:
            blocks = REGION_BLOCKS[region]
            for n in range(NUM_FIGURE):
                model.Add(sum(x[i][j][n] for i, j in blocks) == 1)

        solver = cp_model.CpSolver()
        solution_receiver = VarMatrixSolutionReceiver(variables=x)
        status = solver.SearchForAllSolutions(model=model, callback=solution_receiver)
        print()
        print(f'status = {solver.StatusName(status=status)}', '\n')
        print(f'the number of solution found: {solution_receiver.solution_count}', '\n')

        self.solutions = solution_receiver.solutions
        if solution_receiver.solution_count == 1:
            self.unique_solution = self.solutions[0]
            print('the unique solution:')
            for row in self.unique_solution:
                print(row)
            print()


if __name__ == '__main__':
    # the info template of 9-figure sudoku
    info_template = [
        [0, 0, 0,  0, 0, 0,  0, 0, 0], 
        [0, 0, 0,  0, 0, 0,  0, 0, 0], 
        [0, 0, 0,  0, 0, 0,  0, 0, 0], 

        [0, 0, 0,  0, 0, 0,  0, 0, 0], 
        [0, 0, 0,  0, 0, 0,  0, 0, 0], 
        [0, 0, 0,  0, 0, 0,  0, 0, 0], 

        [0, 0, 0,  0, 0, 0,  0, 0, 0], 
        [0, 0, 0,  0, 0, 0,  0, 0, 0], 
        [0, 0, 0,  0, 0, 0,  0, 0, 0]
    ]

    info = [
        [7, 0, 9,  4, 0, 0,  0, 6, 8], 
        [0, 0, 0,  0, 2, 0,  0, 4, 0], 
        [0, 0, 3,  0, 0, 0,  0, 0, 0], 

        [0, 6, 0,  0, 0, 0,  0, 0, 0], 
        [0, 0, 0,  0, 0, 1,  5, 0, 0], 
        [8, 0, 4,  2, 0, 0,  0, 0, 9], 

        [0, 3, 0,  7, 0, 0,  0, 0, 0], 
        [0, 2, 0,  0, 0, 0,  0, 0, 6], 
        [6, 0, 7,  0, 5, 0,  9, 0, 0]
    ]

    sudoku_model = SudokuCPModel(info=info)
    sudoku_model.run()
