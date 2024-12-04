import time as timer
import heapq
import random
import copy
from single_agent_planner import compute_heuristics, a_star, get_location, get_sum_of_cost
from mdd import MDD


def detect_collision(path1, path2):
    ##############################
    # Task 3.1: Return the first collision that occurs between two robot paths (or None if there is no collision)
    #           There are two types of collisions: vertex collision and edge collision.
    #           A vertex collision occurs if both robots occupy the same location at the same timestep
    #           An edge collision occurs if the robots swap their location at the same timestep.
    #           You should use "get_location(path, t)" to get the location of a robot at time t.

    max_time = max(len(path1), len(path2))

    for t in range(max_time):
        loc1 = get_location(path1, t)
        loc2 = get_location(path2, t)

        if loc1 == loc2:
            return {'loc': [loc1], 'timestep': t}

        if t < max_time - 1:
            next_loc1 = get_location(path1, t + 1)
            next_loc2 = get_location(path2, t + 1)

            if loc1 == next_loc2 and loc2 == next_loc1:
                return {'loc': [loc1, next_loc1], 'timestep': t+1}

    return None


def detect_collisions(paths):
    ##############################
    # Task 3.1: Return a list of first collisions between all robot pairs.
    #           A collision can be represented as dictionary that contains the id of the two robots, the vertex or edge
    #           causing the collision, and the timestep at which the collision occurred.
    #           You should use your detect_collision function to find a collision between two robots.

    collisions = []

    for i in range(len(paths)):
        for j in range(i+1, len(paths)):
            collision = detect_collision(paths[i], paths[j])

            if collision:
                collisions.append({
                    'a1': i,
                    'a2': j,
                    'loc': collision['loc'],
                    'timestep': collision['timestep']
                })

    return collisions


def standard_splitting(collision):
    ##############################
    # Task 3.2: Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint prevents the first agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the second agent to be at the
    #                            specified location at the specified timestep.
    #           Edge collision: the first constraint prevents the first agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the second agent to traverse the
    #                          specified edge at the specified timestep

    constraints = []

    if len(collision['loc']) == 1:
        constraints.append({
            'agent': collision['a1'],
            'loc': collision['loc'],
            'timestep': collision['timestep']
        })

        constraints.append({
            'agent': collision['a2'],
            'loc': collision['loc'],
            'timestep': collision['timestep']
        })
    elif len(collision['loc']) == 2:
        constraints.append({
            'agent': collision['a1'],
            'loc': [collision['loc'][0], collision['loc'][1]],
            'timestep': collision['timestep']
        })

        constraints.append({
            'agent': collision['a2'],
            'loc': [collision['loc'][1], collision['loc'][0]],
            'timestep': collision['timestep']
        })
    return constraints


def disjoint_splitting(collision):
    ##############################
    # Task 4.1: Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint enforces one agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the same agent to be at the
    #                            same location at the timestep.
    #           Edge collision: the first constraint enforces one agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the same agent to traverse the
    #                          specified edge at the specified timestep
    #           Choose the agent randomly

    constraints = []

    chosen_agent = collision['a1'] if random.randint(0,1) == 0 else collision['a2']
    other_agent = collision['a2'] if chosen_agent == collision['a1'] else collision['a1']

    if len(collision['loc']) == 1:
        constraints.append({
            'agent': chosen_agent,
            'loc': collision['loc'],
            'timestep': collision['timestep'],
            'positive': True
        })
        constraints.append({
            'agent': other_agent,
            'loc': collision['loc'],
            'timestep': collision['timestep'],
            'positive': False
        })

    elif len(collision['loc']) == 2:
        constraints.append({
            'agent': chosen_agent,
            'loc': [collision['loc'][0], collision['loc'][1]],
            'timestep': collision['timestep'],
            'positive': True
        })
        constraints.append({
            'agent': other_agent,
            'loc': [collision['loc'][1], collision['loc'][0]],
            'timestep': collision['timestep'],
            'positive': False
        })

    return constraints

def paths_violate_constraint(constraint, paths):
    assert constraint['positive'] is True
    rst = []
    for i in range(len(paths)):
        if i == constraint['agent']:
            continue
        curr = get_location(paths[i], constraint['timestep'])
        prev = get_location(paths[i], constraint['timestep'] - 1)
        if len(constraint['loc']) == 1:  # vertex constraint
            if constraint['loc'][0] == curr:
                rst.append(i)
        else:  # edge constraint
            if constraint['loc'][0] == prev or constraint['loc'][1] == curr \
                    or constraint['loc'] == [curr, prev]:
                rst.append(i)
    return rst




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
        self.variable_ID = 0
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
        self.clauses.append(self.position_validity())
        self.clauses.append(self.transition_validity())
        self.clauses.append(self.avoid_vertex_collisions())
        self.clauses.append(self.avoid_edge_collisions())
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
