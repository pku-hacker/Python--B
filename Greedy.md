# Greedy

**Greedy algorithms** solve problems by making a series of choices, each of which looks best at the moment. The decision is made by choosing the most optimal solution in each step, with the hope that this local optimum will lead to a global optimum.

### Key Characteristics of Greedy Algorithms:
1. **Greedy Choice Property**:
   - A global optimal solution can be arrived at by making a series of locally optimal choices.

2. **Optimal Substructure**:
   - The problem can be broken down into smaller subproblems, and the optimal solution of the problem can be constructed from the optimal solutions of its subproblems.

### When to Use Greedy Algorithms:
- Greedy algorithms work when the problem has both the **Greedy Choice Property** and **Optimal Substructure**.
- Examples include problems related to optimization, scheduling, and graph traversal.

---

### Common Problems Solved Using Greedy Algorithms:

1. **Activity Selection Problem**:
   - Given `n` activities with start and finish times, select the maximum number of activities that don't overlap.

   **Approach**:
   - Sort activities by their finish times.
   - Select the activity that finishes first (greedily) and move to the next non-overlapping activity.
   
   **Example Code**:
   ```python
   def activity_selection(activities):
       # Sort activities based on their finish time
       activities.sort(key=lambda x: x[1])
       
       selected = [activities[0]]
       last_finish_time = activities[0][1]
       
       for i in range(1, len(activities)):
           if activities[i][0] >= last_finish_time:
               selected.append(activities[i])
               last_finish_time = activities[i][1]
       
       return selected

   # Example input: (start, finish)
   activities = [(1, 3), (2, 5), (4, 6), (6, 7)]
   print(activity_selection(activities))
   ```

2. **Huffman Encoding**:
   - A greedy algorithm is used to build an optimal binary tree for data compression.

3. **Fractional Knapsack Problem**:
   - Unlike the 0/1 Knapsack, where items must be taken or left, in this problem, fractions of items can be taken to maximize the total value within a given weight capacity.

   **Approach**:
   - Sort items by their value-to-weight ratio.
   - Take as much of the item with the highest ratio as possible until the knapsack is full.

4. **Minimum Spanning Tree (Prim’s/Kruskal’s Algorithm)**:
   - Both algorithms use a greedy approach to find the minimum spanning tree in a weighted graph.

5. **Dijkstra’s Algorithm**:
   - Finds the shortest path from a source node to all other nodes in a graph with non-negative weights.

---

### Example Problem: Fractional Knapsack
```python
def fractional_knapsack(values, weights, capacity):
    items = sorted(zip(values, weights), key=lambda x: x[0]/x[1], reverse=True)
    
    total_value = 0
    for value, weight in items:
        if capacity >= weight:
            capacity -= weight
            total_value += value
        else:
            total_value += value * (capacity / weight)
            break
    
    return total_value

values = [60, 100, 120]
weights = [10, 20, 30]
capacity = 50
print(fractional_knapsack(values, weights, capacity))  # Output: 240.0
```

---

### Advantages of Greedy Algorithms:
- **Efficient**: Often faster compared to dynamic programming (less overhead).
- **Simple**: Easy to implement as it follows a straightforward strategy.

### Limitations:
- Greedy algorithms don't always guarantee an optimal solution. For some problems (e.g., 0/1 Knapsack, TSP), a greedy approach might not work.

Would you like to try solving a specific greedy problem together?
