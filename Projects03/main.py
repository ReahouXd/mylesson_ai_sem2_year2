import heapq

#define the graph as an adjacency list
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F', 'G'],
    'D': [],
    'E': [],
    'F': [],
    'G': ['H'],
    'H': []
}

#define heuristic values for each node
heuristics = {
    'A': 6,
    'B': 4,
    'C': 2,
    'D': 6,
    'E': 4,
    'F': 3,
    'G': 1,
    'H': 0
}
def greedy_best_first_search(praph, start, goal, heuistices):
    #Create a priority queue and enqueue the start node with its heuristic value
    queue = [(heuistices[start], start)]
    #set to keep track of visited nodes
    visited = set()
    
    while queue:  
        #Dequeue the node with the lowest heuristic value
        _, node = heapq.heappop(queue)
        #if this node is the goal , return True
        if node == goal:
            return True
        #if the node hasn't been visited yet
        if node not in visited:
            #mark it as visited
            visited.add(node)
            
            #Enqueue all adjacent nodes (neighbors) with their heuristic values
            for neighbor in graph[node]:
                if neighbor not in visited:
                    heapq.heappush(queue, (heuistices[neighbor], neighbor))
                    
    # if the loop complete without finding the goal, return False
    return False

# Example usage:
start_node = 'A'
goal_node = 'H'
path_exists = greedy_best_first_search(graph, start_node, goal_node, heuristics)
print(f"Path from {start_node} ot {goal_node} exists: {path_exists}")