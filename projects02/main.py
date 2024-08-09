from collections import deque
#define the graph as an adjacency list
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}
def bfs(graph, start, goal):
    #Create a queue for BFS and enqenue the start nod
    queue = deque([start])
        #Create a set for visited nodes
    visited = set()
        #Loop until the queue is empty
    while queue:
            #Dequeue a node from the queue
        node = queue.popleft()
        #Check if the node has been visited
        if node not in visited:
            #Mark the node as visited
            visited.add(node)
            #Check if the node is the goal
            if node == goal:
                return True
            #Else enqueue the unvisited neighbors
            for neighbour in graph[node]:
                if neighbour not in visited:
                    queue.append(neighbour)
        #If the goal was never reached
    return False

# Example usage:

start_node = 'A'
goal_node = 'F'
path_exists = bfs(graph, start_node, goal_node)
print(f"Path from {start_node} to {goal_node} exists : {path_exists}")