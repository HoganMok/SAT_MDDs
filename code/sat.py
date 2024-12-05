import time as timer
import heapq
import random
import copy
from single_agent_planner import compute_heuristics, a_star, get_location, get_sum_of_cost
from mdd import MDD

class SATSolver(object):
    """The high-level search of CBS."""

    def __init__(self, my_map, starts, goals):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0

        self.open_list = []

        self.MDDs = []
        self.time_horizon = 0
        self.variable_ID = 1
        self.position_vars = {}
        self.transition_vars = {}
        self.clauses = []

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def push_node(self, node):
        heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))
        print("Generate node {}".format(self.num_of_generated))
        self.num_of_generated += 1

    def pop_node(self):
        _, _, id, node = heapq.heappop(self.open_list)
        print("Expand node {}".format(id))
        self.num_of_expanded += 1
        return node

    def position_variables(self):
        position_vars = {}

        for a in range(self.num_of_agents):
            for t in range(self.time_horizon):
                for node in self.MDDs[a][t]:
                    position_vars[(a, node, t)] = self.variable_ID
                    self.variable_ID += 1
        return position_vars

    def is_neighbour(self, u ,v):
        if u[0] == v[0]:
            if u[1] == v[1] + 1:
                return True
            elif u[1] == v[1] - 1:
                return True
        elif u[1] == v[1]:
            if u[0] == v[0] + 1:
                return True
            elif u[0] == v[0] - 1:
                return True
        return False

    def transition_variables(self):
        transition_vars = {}

        for a in range(self.num_of_agents):
            for t in range(self.time_horizon - 1):
                for parent in self.MDDs[a][t]:
                    for child in self.MDDs[a][t + 1]:
                        if self.is_neighbour(parent, child):
                            transition_vars[(a, parent, child, t)] = self.variable_ID
                            self.variable_ID += 1

        return transition_vars

    def position_validity(self):
        clauses = []

        for a in range(self.num_of_agents):
            for t in range(self.time_horizon):
                positions = []
                for key in self.position_vars:
                    agent, node, time_step = key
                    if agent == a and time_step == t:
                        positions.append(self.position_vars[key])
                clauses.append(positions)

                for i in range(len(positions)):
                    for j in range(i + 1, len(positions)):
                        clauses.append([-positions[i], -positions[j]])
        return clauses

    #Invalid transitions was pruned during constructing the MDDs
    def transition_validity(self):
        clauses = []
        # for var in self.position_vars.items():
        #     print("Position Variables: ", var)
        # for var in self.transition_vars.items():
        #     print("Transition Variables: ", var)
        for a in range(self.num_of_agents):
            for t in range(self.time_horizon - 1):
                for (agent, parent_node, child_node, time_step) in self.transition_vars:
                    if agent == a and time_step == t:
                        transition_var = self.transition_vars[(a, parent_node, child_node, time_step)]

                        #Start-to-Transition: Ensure that if the agent is at a position, it must take a valid transition
                        clauses.append([-self.position_vars[(agent, parent_node, time_step)], transition_var])

                        #Transition-to-End: Ensure that if a transition occurs, the agent ends up in the correct position at t + 1
                        #Be in the starting position at the beginning of the transition
                        #End up in the destination position at the conclusion of the transition
                        if (agent, parent_node, time_step + 1) in self.position_vars:
                            clauses.append([-transition_var, self.position_vars[(agent, parent_node, time_step)]])
                            clauses.append([-transition_var, self.position_vars[(agent, parent_node, time_step + 1)]])

                for (agent, parent_node, child_node, time_step) in self.transition_vars:
                    for (agent2, parent_node2, child_node2, time_step2) in self.transition_vars:
                        if (agent == agent2 and time_step == time_step2 and parent_node == parent_node2
                                and child_node != child_node2):
                            #Outgoing Transition Mutual Exclusivity: At most one transition can start from the same position
                            clauses.append([-self.transition_vars[(agent, parent_node, child_node, time_step)],
                                            -self.transition_vars[(agent, parent_node, child_node2, time_step2)]])
                        elif (agent == agent2 and time_step == time_step2 and parent_node != parent_node2
                                and child_node == child_node2):
                            #Incoming Transition Mutual Exclusivity: At most one transition can end at the same position
                            clauses.append([-self.transition_vars[(agent, parent_node, child_node, time_step)],
                                            -self.transition_vars[(agent, parent_node2, child_node, time_step2)]])

        return clauses

    def avoid_vertex_collisions(self):
        clauses = []

        for t in range(self.time_horizon):
            for a in range(self.num_of_agents):
                for a2 in range(a+1, self.num_of_agents):
                    position_a1 = []
                    position_a2 = []
                    if t < len(self.MDDs[a]):
                        position_a1 = self.MDDs[a][t]
                    if t < len(self.MDDs[a2]):
                        position_a2 = self.MDDs[a2][t]
                    # print("Position 1: ", position_a1)
                    # print("Position 2: ", position_a2)
                    for pos in position_a1:
                         if pos in position_a2:
                             clauses.append([-self.position_vars[(a, pos, t)], -self.position_vars[(a2, pos, t)]])
        return clauses

    def avoid_edge_collisions(self):
        clauses = []
        for t in range(self.time_horizon):
            for a in range(self.num_of_agents):
                for a2 in range(a+1, self.num_of_agents):
                    transitions_a = []
                    transitions_a2 = []
                    if t < len(self.MDDs[a]) -1:
                        for parent in self.MDDs[a][t]:
                            for child in self.MDDs[a][t+1]:
                                if (a, parent, child, t) in self.position_vars:
                                    transitions_a.append((parent,child))

                    if t < len(self.MDDs[a2]) - 1:
                        for parent in self.MDDs[a2][t]:
                            for child in self.MDDs[a2][t+1]:
                                if (a2, parent, child, t) in self.position_vars:
                                    transitions_a2.append((parent,child))

                    for (parent, child) in transitions_a:
                        for (parent2, child2) in transitions_a2:
                            if parent == child2 and parent2 == child:
                                clauses.append([-self.transition_vars[(a, parent, child, t)],
                                                -self.transition_vars[(a2, parent2, child2, t)]])
        return clauses

    def enforce_goal_constraints(self):
        clauses = []

        for a in range(self.num_of_agents):
            goal_position = self.goals[a]
            goal_clause = []
            for t in range(self.time_horizon):
                if (a, goal_position, t) in self.position_vars:
                    goal_clause.append(self.position_vars[(a, goal_position, t)])
            if goal_clause:
                clauses.append(goal_clause)

        return clauses

    def encode_to_cnf(self):
        self.clauses.extend(self.position_validity())
        self.clauses.extend(self.transition_validity())
        self.clauses.extend(self.avoid_vertex_collisions())
        self.clauses.extend(self.avoid_edge_collisions())
        self.clauses.extend(self.enforce_goal_constraints())

    def write_cnf_to_file(self, filename):
        with open(filename, 'w') as f:
            f.write(f"p cnf {self.variable_ID-1} {len(self.clauses)}\n")

            for clause in self.clauses:
                f.write(' '.join(map(str, clause)) + ' 0\n')

    def find_solution(self, disjoint=True):
        """ Finds paths for all agents from their start locations to their goal locations

        disjoint    - use disjoint splitting or not
        """

        self.start_time = timer.time()

        # Generate the root node
        # constraints   - list of constraints
        # paths         - list of paths, one for each agent
        #               [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        # collisions     - list of collisions in paths
        root = {'cost': 0,
                'constraints': [],
                'paths': [],
                'time_horizon': 0,
                'collisions': []}

        for i in range(self.num_of_agents):
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path)
            if len(path) > self.time_horizon:
                self.time_horizon = len(path)

        print(root['time_horizon'])
        self.push_node(root)

        for i in range(self.num_of_agents):
            mdd = MDD(self.my_map, self.starts, self.goals, self.time_horizon, i)
            mdd.build()
            self.MDDs.append(mdd.levels)
        self.position_vars = self.position_variables()
        self.transition_vars = self.transition_variables()

        self.encode_to_cnf()
        self.write_cnf_to_file("mapf_problem.cnf")
        # for mdd in self.MDDs:
        #     print(mdd)
        return None


    def print_results(self, node):
        print("\n Found a solution! \n")
        CPU_time = timer.time() - self.start_time
        print("CPU time (s):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Expanded nodes:  {}".format(self.num_of_expanded))
        print("Generated nodes: {}".format(self.num_of_generated))
