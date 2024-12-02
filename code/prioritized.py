import time as timer
from single_agent_planner import compute_heuristics, a_star, get_sum_of_cost


class PrioritizedPlanningSolver(object):
    """A planner that plans for each robot sequentially."""

    def __init__(self, my_map, starts, goals):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.CPU_time = 0

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def find_solution(self):
        """ Finds paths for all agents from their start locations to their goal locations."""

        start_time = timer.time()
        result = []
        # constraints = [{'agent': 0, 'loc': [(1,5)], 'timestep': 4}]
        # constraints = [{'agent': 1, 'loc': [(1,2), (1,3)], 'timestep': 1}]
        # constraints = [{'agent': 0, 'loc': [(1,5)], 'timestep': 10}]
        constraints = [
            {'agent': 1, 'loc': [(1, 3)], 'timestep': 2},
            {'agent': 1, 'loc': [(1, 2)], 'timestep': 2},
            {'agent': 1, 'loc': [(1, 3), (1,4)], 'timestep': 2}
        ]
        # constraints = []
        for i in range(self.num_of_agents):  # Find path for each agent
            L_prior = sum(len(path) for path in result)
            upper_bound = L_prior+len(self.my_map) + len(self.my_map[0])

            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, constraints)
            if path is None or len(path) > upper_bound:
                raise BaseException('No solutions')
            result.append(path)

            ##############################
            # Task 2: Add constraints here
            #         Useful variables:
            #            * path contains the solution path of the current (i'th) agent, e.g., [(1,1),(1,2),(1,3)]
            #            * self.num_of_agents has the number of total agents
            #            * constraints: array of constraints to consider for future A* searches

            goal_loc = self.goals[i]
            goal_reach_time = len(path) - 1

            for ts in range(goal_reach_time, goal_reach_time + 100):
                for future_agent in range(i+1, self.num_of_agents):
                    constraints.append({
                        'agent': future_agent,
                        'loc': [goal_loc],
                        'timestep': ts})

            for ts, location in enumerate(path):
                for future_agent in range(i+1, self.num_of_agents):
                    constraints.append({
                        'agent': future_agent,
                        'loc': [location],
                        'timestep': ts
                    })

                if ts+1 < len(path):
                    next_loc = path[ts+1]
                    for future_agent in range(i+1, self.num_of_agents):
                        constraints.append({
                            'agent': future_agent,
                            'loc': [location, next_loc],
                            'timestep': ts+1
                        })
            ##############################

        self.CPU_time = timer.time() - start_time

        print("\n Found a solution! \n")
        print("CPU time (s):    {:.2f}".format(self.CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(result)))
        print(result)
        return result
