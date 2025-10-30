from datetime import datetime
from typing import List, Tuple, Set
import math
import random

from ortools.linear_solver import pywraplp


class TSP(object):
    '''
    TSP (traveling salesman problem)
    '''
    def __init__(self, spots: List[Tuple[int, int]]):
        '''
        TSP, initialise

        @param spots: the positions of the spots
        '''

        self.spots = spots

        self.distances = self._get_distances()

    @property
    def num_spot(self) -> int:
        '''
        get the number of the spots

        @return: the number of the spots
        '''

        return len(self.spots)

    def _get_distances(self, wave_factor: float = 0.3) -> List[List[float]]:
        '''
        get the distance matrix of the spots

        @param wave_factor: the factor of random fluctuation, default: 0.3

        @return distances: the distance matrix of the spots
        '''

        distances = [[round(math.sqrt(math.pow(self.spots[i][0] - self.spots[j][0], 2) + math.pow(
            self.spots[i][1] - self.spots[j][1], 2)), 3) * (random.random() * wave_factor * 2 + (1 - wave_factor)) 
            for j in range(self.num_spot)] for i in range(self.num_spot)]

        return distances
    
    def ip_model_mtz(self):
        '''
        IP model based on Miller-Tucker-Zemlin formulation
        '''

        # about problem type:
        # 'CBC_MIXED_INTEGER_PROGRAMMING': integer progamming
        # 'SCIP_MIXED_INTEGER_PROGRAMMING': integer progamming, using SCIP solver
        # 'GLOP_LINEAR_PROGRAMMING': linear progamming, cannot be used for integer variables
        solver = pywraplp.Solver('klee_minty_cube', problem_type=pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

        # basic variables: to choose the arcs
        x = [[solver.BoolVar(name=f'x_{i}_{j}') for j in range(self.num_spot)] for i in range(self.num_spot)]

        # assistant variables: to confirm the order of the spots
        z = [solver.NumVar(lb=0, ub=solver.infinity(), name=f'z_{i}') for i in range(self.num_spot)]

        # the out-degree constraints
        for i in range(self.num_spot):
            solver.Add(sum(x[i][j] for j in range(self.num_spot)) == 1)

        # the in-degree constraints
        for j in range(self.num_spot):
            solver.Add(sum(x[i][j] for i in range(self.num_spot)) == 1)

        # the constraints to avoid sub-loops
        # set spot 0 as the first spot
        for i in range(1, self.num_spot):
            for j in range(1, self.num_spot):
                solver.Add(z[j] >= z[i] + self.num_spot * (x[i][j] - 1) + 1)

        solver.Minimize(
            sum(self.distances[i][j] * x[i][j] for j in range(self.num_spot) for i in range(self.num_spot)))

        dts = datetime.now()
        status = solver.Solve()
        dte = datetime.now()
        tm = round((dte - dts).seconds + (dte - dts).microseconds / 1e6, 3)
        print(f'MTZ model solving time: {tm} s', '\n')

        if status == pywraplp.Solver.OPTIMAL:
            obj_opt = round(solver.Objective().Value(), 3)
            print('objective value:  {}'.format(obj_opt), '\n')

            # get the order of the spots
            z_ = sorted([(z[i].solution_value(), i) for i in range(self.num_spot)])
            orders = [tup[1] for tup in z_] + [0]

            # print the result
            total_distance = 0
            for i in range(len(orders) - 1):
                spot_from, spot_to = orders[i], orders[i + 1]
                distance = self.distances[spot_from][spot_to]
                total_distance += distance
                print(f'{spot_from} -> {spot_to}, distance: {round(distance, 3)}, total: {round(total_distance, 3)}')
            print()
        else:
            print('No optimum solutions found!')

        tm_solver = round(solver.wall_time() / 1000, 3)
        num_iteration = solver.iterations()
        print(f'the solving time got from the solver: {tm_solver} s')
        print(f'the number of iterations got from the solver: {num_iteration}', '\n')

    def ip_model_gg(self):
        '''
        IP model based on Gavish-Graves formulation
        '''

        solver = pywraplp.Solver('klee_minty_cube', problem_type=pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

        # basic variables: to choose the arcs
        x = [[solver.BoolVar(name=f'x_{i}_{j}') for j in range(self.num_spot)] for i in range(self.num_spot)]

        # assistant variables: to confirm the order of the chosen arcs
        y = [[solver.NumVar(lb=0, ub=solver.infinity(), name=f'y_{i}_{j}') for j in range(self.num_spot)] 
             for i in range(self.num_spot)]

        # the out-degree constraints
        for i in range(self.num_spot):
            solver.Add(sum(x[i][j] for j in range(self.num_spot)) == 1)

        # the in-degree constraints
        for j in range(self.num_spot):
            solver.Add(sum(x[i][j] for i in range(self.num_spot)) == 1)

        # the constraints to avoid sub-loops
        for i in range(1, self.num_spot):
            solver.Add(sum(y[i][j] for j in range(self.num_spot) if j != i) - sum(
                y[j][i] for j in range(self.num_spot) if j and j != i) == 1)

            for j in range(self.num_spot):
                if j != i:
                    solver.Add(y[i][j] <= (self.num_spot - 1) * x[i][j])

        solver.Minimize(
            sum(self.distances[i][j] * x[i][j] for j in range(self.num_spot) for i in range(self.num_spot)))

        dts = datetime.now()
        status = solver.Solve()
        dte = datetime.now()
        tm = round((dte - dts).seconds + (dte - dts).microseconds / 1e6, 3)
        print(f'GG model solving time: {tm} s', '\n')

        if status == pywraplp.Solver.OPTIMAL:
            obj_opt = round(solver.Objective().Value(), 3)
            print('objective value:  {}'.format(obj_opt), '\n')

            # get the order of the spots
            y_ = sorted([(y[i][j].solution_value(), i, j) for i in range(self.num_spot) for j in range(self.num_spot) 
                         if y[i][j].solution_value() > 0.9])
            orders = [(0, y_[0][1])] + [(tup[1], tup[2]) for tup in y_]

            # print the result
            total_distance = 0
            for spot_from, spot_to in orders:
                distance = self.distances[spot_from][spot_to]
                total_distance += distance
                print(f'{spot_from} -> {spot_to}, distance: {round(distance, 3)}, total: {round(total_distance, 3)}')
            print()
        else:
            print('No optimum solutions found!')

        tm_solver = round(solver.wall_time() / 1000, 3)
        num_iteration = solver.iterations()
        print(f'the solving time got from the solver: {tm_solver} s')
        print(f'the number of iterations got from the solver: {num_iteration}', '\n')

    def simulated_annealing(self, temperature_start: int = 5000, temperature_end: int = 100, 
                            coe_anneal: float = 0.99, num_iteration: int = 100, search_large_neighbour: bool = False):
        '''
        Simulated Annealing algorithm

        @param temperature_start: the staring temperature, default: 5000
        @param temperature_end: the staring temperature, default: 100
        @param coe_anneal: the coefficient of annealing, i.e. temperature decreasing, default: 0.99
        @param num_iteration: the number of iterations at each temperature, default: 100
        @param search_large_neighbour: if to search larger neighbourhood at each temperature (local search), 
            specifically, to change the order of more than 2 spots each time, default: False
        '''

        dts = datetime.now()

        # initial solution and objective value
        route = [i for i in range(self.num_spot)] + [0]
        objective = self._get_distance_route(route=route)
        route_opt, objective_opt = route, objective
        print(f'initial route: {route_opt}')
        print(f'distance: {round(objective_opt, 3)}', '\n')

        # the outer loop: annealing
        temperature = temperature_start
        num_anneal = 0
        while temperature > temperature_end:
            print(f'current temperature: {round(temperature, 4)}')
            print(f'current annealing times: {num_anneal}', '\n')

            # the inner loop: the iterations of each temperature
            for _ in range(num_iteration):
                route_old = route.copy()
                if search_large_neighbour:
                    route = self._generate_new_route(route=route)
                else:
                    route, _ = self._generate_new_route_swap(route=route)
                objective_old, objective = objective, self._get_distance_route(route=route)
                difference = objective - objective_old

                # Metropolis Rule
                if difference >= 0:
                    # random value too large, thus restore the old solution
                    if random.random() >= math.exp(-difference / temperature):
                        route = route_old.copy()

                # update the optimum solution and objective value
                if objective < objective_opt:
                    route_opt, objective_opt = route.copy(), objective
                    print(f'update the optimum route: {route_opt}')
                    print(f'distance: {round(objective_opt, 3)}', '\n')

            temperature *= coe_anneal
            num_anneal += 1

        dte = datetime.now()
        tm = round((dte - dts).seconds + (dte - dts).microseconds / 1e6, 3)
        print(f'Simulated Annealing algorithm running time: {tm} s', '\n')

        # print the result
        total_distance = 0
        for i in range(len(route_opt) - 1):
            spot_from, spot_to = route_opt[i], route_opt[i + 1]
            distance = self.distances[spot_from][spot_to]
            total_distance += distance
            print(f'{spot_from} -> {spot_to}, distance: {round(distance, 3)}, total: {round(total_distance, 3)}')
        print()

    def tabu_search(self, num_iteration: int = 100, size_neighbour: int = 50, len_tabu: int = 10):
        '''
        Tabu Search algorithm

        @param num_iteration: the number of iterations, default: 100
        @param size_neighbour: the number of searches of each neighbourhood, default: 50
        @param len_tabu: the length of the tabu list, default: 10
        '''

        dts = datetime.now()

        # initial solution and objective value
        route = [i for i in range(self.num_spot)] + [0]
        objective = self._get_distance_route(route=route)
        route_opt, objective_opt = route, objective
        print(f'initial route: {route_opt}')
        print(f'distance: {round(objective_opt, 3)}', '\n')

        tabu_list = []

        # the outer loop: iterations
        for i in range(num_iteration):
            print(f'current iteration: {i}', '\n')

            # the inner loop: local search
            neighbours = []
            for _ in range(size_neighbour):
                route_new, change = self._generate_new_route_swap(route=route)
                objective_new = self._get_distance_route(route=route_new)
                neighbours.append({
                    'route': route_new, 
                    'change': change, 
                    'distance': objective_new
                })
            neighbours.sort(key=lambda x: x['distance'])

            # update the solution
            for neighbour in neighbours:
                # case 1: a better solution
                if neighbour['distance'] < objective_opt:
                    route = neighbour['route']
                    route_opt, objective_opt = neighbour['route'], neighbour['distance']
                    print(f'update the route: {route_opt}')
                    print(f'distance: {round(objective_opt, 3)}', '\n')
                    tabu_list.append(neighbour['change'])
                    break

                # case 2: the change of current optimum solution is in the tabu list
                elif neighbour['change'] in tabu_list:
                    print(f'current spots change of local optimum solution is banned: {neighbour["change"]}')

                # case 3: accept a bad solution temporarily
                else:
                    route = neighbour['route']
                    print(f'accept a bad solution temporarily: {neighbour["route"]}')
                    print(f'spots change: {neighbour["change"]}, distance: {round(neighbour["distance"], 3)}', '\n')
                    tabu_list.append(neighbour['change'])
            print()

            # if the tabu list is too large, release the preceding solutions in it
            if len(tabu_list) > len_tabu:
                num_release = (math.ceil(len(tabu_list) / len_tabu) - 1) * len_tabu
                print(f'the tabu list is too large, release {num_release} solutions', '\n')
                tabu_list = tabu_list[num_release:]

        dte = datetime.now()
        tm = round((dte - dts).seconds + (dte - dts).microseconds / 1e6, 3)
        print(f'Tabu Search algorithm running time: {tm} s', '\n')

        # print the result
        total_distance = 0
        for i in range(len(route_opt) - 1):
            spot_from, spot_to = route_opt[i], route_opt[i + 1]
            distance = self.distances[spot_from][spot_to]
            total_distance += distance
            print(f'{spot_from} -> {spot_to}, distance: {round(distance, 3)}, total: {round(total_distance, 3)}')
        print()

    def _get_distance_route(self, route: List[int]) -> float:
        '''
        get the total distance of a route

        @param route: the order of the spots in a route

        @return: the total distance of a route
        '''

        return sum(self.distances[route[i]][route[i + 1]] for i in range(len(route) - 1))

    def _generate_new_route(self, route: List[int], coe_max_sub_sequence: float = 0.3) -> List[int]:
        '''
        generate a new route by changing the order of some spots in a specefic route

        @param route: a specefic route
        @param coe_max_sub_sequence: the coefficient of the maximum number of the spots to change order, 
            specifically, the coefficient is to be multiplied by the number of all the spots

        @return route_: the new route
        '''

        max_len_sub_sequence = math.ceil(self.num_spot * coe_max_sub_sequence)
        len_sub_sequence = random.randint(2, max_len_sub_sequence)
        sub_sequence = random.sample(
            route[1: -1], k=len_sub_sequence)  # keep the spot 0 as the starting and ending points

        route_ = route.copy()
        sub_positions = sorted([route_.index(spot) for spot in sub_sequence])
        for i in range(len(sub_sequence)):
            route_[sub_positions[i]] = sub_sequence[i]

        return route_

    def _generate_new_route_swap(self, route: List[int]) -> Tuple[List[int], Set]:
        '''
        generate a new route by swapping two spots in a specefic route, and record the swapped spots

        @param route: a specefic route

        @return route_: the new route
        @return change: the swapped spots
        '''

        [spot_1, spot_2] = random.sample(route[1: -1], k=2)

        route_ = route.copy()
        tmp, route_[spot_1] = route_[spot_1], route_[spot_2]
        route_[spot_2] = tmp

        change = {spot_1, spot_2}

        return route_, change


if __name__ == '__main__':
    random.seed(42)

    num_spot = 20
    ran_position = (1, 99)
    spots = [(random.randint(ran_position[0], ran_position[1]), random.randint(ran_position[0], ran_position[1])) 
              for _ in range(num_spot)]

    tsp = TSP(spots=spots)
    print()

    # method = 'mtz_model'
    # method = 'gg_model'
    # method = 'simulated_annealing'
    method = 'tabu_search'

    if method == 'mtz_model':
        tsp.ip_model_mtz()
    elif method == 'gg_model':
        tsp.ip_model_gg()
    elif method == 'simulated_annealing':
        tsp.simulated_annealing(temperature_start=5000, temperature_end=1000, coe_anneal=0.99, num_iteration=100, 
                                search_large_neighbour = False)
    else:
        size_neighbour = num_spot * 5
        len_tabu = num_spot
        tsp.tabu_search(num_iteration=100, size_neighbour=size_neighbour, len_tabu=len_tabu)
