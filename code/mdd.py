from single_agent_planner import move


class MDD:
    def __init__(self, graph, start, goal, time_horizon, agent):
        self.graph = graph
        self.start = start[agent]
        self.goal = goal[agent]
        self.time_horizon = time_horizon
        self.levels = {}

    def build(self):
        self.levels[0] = {self.start}
        for t in range(1, self.time_horizon):
            self.levels[t] = set()
            for node in self.levels[t-1]:
                neighbors = self.get_neighbors(node)

                for neighbor in neighbors:
                    if self.can_reach_goal(neighbor, self.time_horizon-t):
                        self.levels[t].add(neighbor)

    def get_neighbors(self, node):
        neighbors = list()
        for direction in range(4):
            move_node = move(node, direction)
            if 0 < move_node[0] < len(self.graph) - 1 and 0 < move_node[1] < len(self.graph[0]) - 1 and not self.graph[move_node[0]][move_node[1]]:
                neighbors.append(move_node)
        return neighbors


    def can_reach_goal(self, node, time_horizon):
        visited = set()
        queue = [(node, 0)]
        while queue:
            cur, time_step = queue.pop(0)
            if cur == self.goal:
                return True
            if time_step < time_horizon:
                for neighbor in self.get_neighbors(cur):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, time_step + 1))
        return False
