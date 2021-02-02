from grid import Node, NodeGrid
from math import inf
import heapq


class PathPlanner(object):
    """
    Represents a path planner, which may use Dijkstra, Greedy Search or A* to plan a path.
    """
    def __init__(self, cost_map):
        """
        Creates a new path planner for a given cost map.

        :param cost_map: cost used in this path planner.
        :type cost_map: CostMap.
        """
        self.cost_map = cost_map
        self.node_grid = NodeGrid(cost_map)

    @staticmethod
    def construct_path(goal_node):
        """
        Extracts the path after a planning was executed.

        :param goal_node: node of the grid where the goal was found.
        :type goal_node: Node.
        :return: the path as a sequence of (x, y) positions: [(x1,y1),(x2,y2),(x3,y3),...,(xn,yn)].
        :rtype: list of tuples.
        """
        node = goal_node
        # Since we are going from the goal node to the start node following the parents, we
        # are transversing the path in reverse
        reversed_path = []
        while node is not None:
            reversed_path.append(node.get_position())
            node = node.parent
        return reversed_path[::-1]  # This syntax creates the reverse list

    def dijkstra(self, start_position, goal_position):
        """
        Plans a path using the Dijkstra algorithm.

        :param start_position: position where the planning stars as a tuple (x, y).
        :type start_position: tuple.
        :param goal_position: goal position of the planning as a tuple (x, y).
        :type goal_position: tuple.
        :return: the path as a sequence of positions and the path cost.
        :rtype: list of tuples and float.
        """

        # The first return is the path as sequence of tuples (as returned by the method construct_path())
        # The second return is the cost of the path
        pq = []
        i, j = start_position
        cost_dijkstra = [[inf for x in range(self.cost_map.width)] for y in range(self.cost_map.height)]
        cost_dijkstra[i][j] = 0
        heapq.heappush(pq, (cost_dijkstra[i][j], start_position))
        while not len(pq) == 0:
            f, node = heapq.heappop(pq)
            i, j = node
            self.node_grid.get_node(i, j).closed = True
            if node == goal_position:
                while not len(pq) == 0:
                    heapq.heappop(pq)
            else:
                for successor in self.node_grid.get_successors(i, j):
                    i_suc, j_suc = successor
                    if not self.node_grid.get_node(i_suc, j_suc).closed:
                        dist = self.cost_map.get_edge_cost(node, successor)
                        if cost_dijkstra[i_suc][j_suc] > cost_dijkstra[i][j] + dist:
                            cost_dijkstra[i_suc][j_suc] = cost_dijkstra[i][j] + dist
                            self.node_grid.get_node(i_suc, j_suc).parent = self.node_grid.get_node(i, j)
                            heapq.heappush(pq, (cost_dijkstra[i_suc][j_suc], successor))
        i, j = goal_position
        cost = cost_dijkstra[i][j]
        path = self.construct_path(self.node_grid.get_node(i, j))
        self.node_grid.reset()
        return path, cost

    def greedy(self, start_position, goal_position):
        """
        Plans a path using greedy search.

        :param start_position: position where the planning stars as a tuple (x, y).
        :type start_position: tuple.
        :param goal_position: goal position of the planning as a tuple (x, y).
        :type goal_position: tuple.
        :return: the path as a sequence of positions and the path cost.
        :rtype: list of tuples and float.
        """

        # The first return is the path as sequence of tuples (as returned by the method construct_path())
        # The second return is the cost of the path
        pq = []
        i, j = start_position
        goal = self.node_grid.get_node(goal_position[0], goal_position[1])
        heapq.heappush(pq, (goal.distance_to(i, j), start_position))
        while not len(pq) == 0:
            f, node = heapq.heappop(pq)
            i, j = node
            self.node_grid.get_node(i, j).closed = True
            if node == goal_position:
                while not len(pq) == 0:
                    heapq.heappop(pq)
            else:
                for successor in self.node_grid.get_successors(i, j):
                    i_suc, j_suc = successor
                    if not self.node_grid.get_node(i_suc, j_suc).closed:
                        dist = goal.distance_to(i_suc, j_suc)
                        self.node_grid.get_node(i_suc, j_suc).parent = self.node_grid.get_node(i, j)
                        self.node_grid.get_node(i_suc, j_suc).closed = True
                        heapq.heappush(pq, (dist, successor))
        cost = 0
        path = self.construct_path(self.node_grid.get_node(goal_position[0], goal_position[1]))
        for n in range(len(path)-1):
            cost += self.cost_map.get_edge_cost(path[n], path[n+1])
        self.node_grid.reset()
        return path, cost

    def a_star(self, start_position, goal_position):
        """
        Plans a path using A*.

        :param start_position: position where the planning stars as a tuple (x, y).
        :type start_position: tuple.
        :param goal_position: goal position of the planning as a tuple (x, y).
        :type goal_position: tuple.
        :return: the path as a sequence of positions and the path cost.
        :rtype: list of tuples and float.
        """

        # The first return is the path as sequence of tuples (as returned by the method construct_path())
        # The second return is the cost of the path
        pq = []
        i, j = start_position
        g = [[inf for x in range(self.cost_map.width)] for y in range(self.cost_map.height)]
        f = [[inf for x in range(self.cost_map.width)] for y in range(self.cost_map.height)]
        goal = self.node_grid.get_node(goal_position[0], goal_position[1])
        g[i][j] = 0
        f[i][j] = goal.distance_to(i, j)
        heapq.heappush(pq, (f[i][j], start_position))
        while not len(pq) == 0:
            k, node = heapq.heappop(pq)
            i, j = node
            self.node_grid.get_node(i, j).closed = True
            if node == goal_position:
                while not len(pq) == 0:
                    heapq.heappop(pq)
            else:
                for successor in self.node_grid.get_successors(i, j):
                    i_suc, j_suc = successor
                    if not self.node_grid.get_node(i_suc, j_suc).closed:
                        cost = self.cost_map.get_edge_cost(node, successor)
                        h = goal.distance_to(i_suc, j_suc)
                        if f[i_suc][j_suc] > g[i][j] + cost + h:
                            g[i_suc][j_suc] = g[i][j] + cost
                            f[i_suc][j_suc] = g[i_suc][j_suc] + h
                            self.node_grid.get_node(i_suc, j_suc).parent = self.node_grid.get_node(i, j)
                            heapq.heappush(pq, (f[i_suc][j_suc], successor))
        path = self.construct_path(self.node_grid.get_node(goal_position[0], goal_position[1]))
        i, j = goal_position
        final_cost = g[i][j]
        self.node_grid.reset()
        return path, final_cost