# Assignment #C: 五味杂陈



Updated 1148 GMT+8 Dec 10, 2024

2024 fall, Complied by 南宫圣光，元培

**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora [https://typoraio.cn](https://typoraio.cn/) ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。

## 1. 题目



### 1115. 取石子游戏



dfs, https://www.acwing.com/problem/content/description/1117/

思路：50min

1. 确保a始终是较大的数，b是较小的数。    
2. 如果a // b >= 2，说明先手可以取走足够的石子从而获胜，返回True。    
3. 如果a == b，游戏结束，先手必胜。    
4. 递归调用fsg(b, a - b)，模拟下一步状态，并反转结果，确保博弈最优解。

代码：

```
def fsg(a, b):
    if b > a:  
        a, b = b, a  # 确保a >= b
    if a // b >= 2 or a == b:  # 先手必胜的两种情况
        return True
    return not fsg(b, a - b)  # 反转下一次的结果

if __name__ == "__main__":
    while True:
        # 读取输入的两个数字a和b
        a, b = map(int, input().split())
        if a == 0 or b == 0:  # 结束条件
            break
        if a < b:  
            a, b = b, a  # 保证a >= b
        # 判断结果并输出
        if fsg(a, b):
            print("win")
        else:
            print("lose")

```

![image-20241217201933205](C:\Users\남궁성광\AppData\Roaming\Typora\typora-user-images\image-20241217201933205.png)

### 25570: 洋葱



Matrices, http://cs101.openjudge.cn/practice/25570

思路：30min

从最外层开始，通过遍历矩阵的外边缘计算这一层的元素之和。

然后，将矩阵缩小，进入下一层，重复步骤 1。

继续递归/迭代计算直到到达矩阵的最中心。

记录每一层的元素之和，并输出最大的层和。

代码：

```
def max_layer_sum(matrix, n):
    max_sum = float('-inf')  # 初始化最大层和
    layer = 0  # 记录当前层数

    # 逐层从外向内计算
    while layer < (n + 1) // 2:  # 一共最多 (n+1)//2 层
        current_sum = 0

        # 上边界，从左到右
        for j in range(layer, n - layer):
            current_sum += matrix[layer][j]
        
        # 右边界，从上到下
        for i in range(layer + 1, n - layer):
            current_sum += matrix[i][n - layer - 1]

        # 下边界，从右到左
        if n - layer - 1 > layer:  # 确保下边界不是上边界
            for j in range(n - layer - 2, layer - 1, -1):
                current_sum += matrix[n - layer - 1][j]

        # 左边界，从下到上
        if n - layer - 1 > layer:  # 确保左边界不是右边界
            for i in range(n - layer - 2, layer, -1):
                current_sum += matrix[i][layer]

        # 更新最大层和
        max_sum = max(max_sum, current_sum)
        layer += 1  # 进入下一层

    return max_sum

# 输入处理
n = int(input())
matrix = [list(map(int, input().split())) for _ in range(n)]

# 计算并输出结果
print(max_layer_sum(matrix, n))
```

![image-20241217195048687](C:\Users\남궁성광\AppData\Roaming\Typora\typora-user-images\image-20241217195048687.png)

### 1526C1. Potions(Easy Version)



greedy, dp, data structures, brute force, *1500, https://codeforces.com/problemset/problem/1526/C1

思路：30min

**从左到右遍历每个药水**：

- 假设喝下当前药水，更新健康值（`health`）。
- 记录喝下的药水，用最小堆（`min-heap`）来保存**负数药水**。

**如果健康值变负**：

- 从最小堆中移除一个**最小的负数药水**（相当于把最坏的药水“吐掉”）。
- 调整健康值，使健康值回到非负状态。

**结果**：

- 遍历结束时，喝下的药水总数就是最大能喝的药水数量。

代码：

```
import heapq

def max_potions(n, potions):
    health = 0
    heap = []
    count = 0 
    
    for p in potions:
        health += p  
        heapq.heappush(heap, p)  
        count += 1  
        
        if health < 0:  
            smallest_negative = heapq.heappop(heap) 
            health -= smallest_negative 
            count -= 1  
    
    return count


n = int(input())
potions = list(map(int, input().split()))
print(max_potions(n, potions))

```

![image-20241217195147080](C:\Users\남궁성광\AppData\Roaming\Typora\typora-user-images\image-20241217195147080.png)

### 22067: 快速堆猪



辅助栈，http://cs101.openjudge.cn/practice/22067/

思路：1h

**主栈** `stack`：用于存储所有的猪的重量。

**辅助栈** `min_stack`：用于存储当前最小的猪重量。

代码：

```
class PigStack:
    def __init__(self):
        self.stack = []  # 主栈，存所有的猪重量
        self.min_stack = []  # 辅助栈，存当前的最小重量

    def push(self, n):
        self.stack.append(n)
        # 如果min_stack为空，或者n <= 当前最小值，则压入辅助栈
        if not self.min_stack or n <= self.min_stack[-1]:
            self.min_stack.append(n)

    def pop(self):
        if self.stack:  # 如果主栈不为空
            top = self.stack.pop()
            if self.min_stack and top == self.min_stack[-1]:
                self.min_stack.pop()  # 同时弹出辅助栈栈顶元素

    def min(self):
        if self.min_stack:  # 如果辅助栈不为空
            print(self.min_stack[-1])

# 读取输入并处理
import sys
input = sys.stdin.read

def main():
    pig_stack = PigStack()
    commands = input().splitlines()
    
    for command in commands:
        if command.startswith("push"):
            _, n = command.split()
            pig_stack.push(int(n))
        elif command == "pop":
            pig_stack.pop()
        elif command == "min":
            pig_stack.min()

if __name__ == "__main__":
    main()
```

![image-20241217195402951](C:\Users\남궁성광\AppData\Roaming\Typora\typora-user-images\image-20241217195402951.png)

### 20106: 走山路



Dijkstra, http://cs101.openjudge.cn/practice/20106/

思路：1h

从出发点开始，使用优先队列（`heapq`）进行最短路径搜索。

每次移动到相邻的格子时，代价为当前高度与相邻格子高度差的绝对值。

在优先队列中，每次取出当前代价最小的点进行扩展。

如果到达目标点，则返回当前累积的代价。

代码：

```
import heapq

def min_energy_cost(m, n, p, terrain, queries):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上、下、左、右

    def dijkstra(start, end):
        """使用Dijkstra算法计算最小体力消耗"""
        sr, sc = start
        er, ec = end

        if terrain[sr][sc] == "#" or terrain[er][ec] == "#":
            return "NO"

        pq = [(0, sr, sc)]  # (当前消耗的体力, 当前行, 当前列)
        visited = [[False] * n for _ in range(m)]
        min_cost = [[float('inf')] * n for _ in range(m)]
        min_cost[sr][sc] = 0

        while pq:
            cost, r, c = heapq.heappop(pq)
            if visited[r][c]:
                continue
            visited[r][c] = True

            if (r, c) == (er, ec):  # 到达目标点
                return cost

            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n and not visited[nr][nc] and terrain[nr][nc] != "#":
                    next_cost = cost + abs(terrain[r][c] - terrain[nr][nc])
                    if next_cost < min_cost[nr][nc]:
                        min_cost[nr][nc] = next_cost
                        heapq.heappush(pq, (next_cost, nr, nc))

        return "NO"

    results = []
    for i in range(m):
        terrain[i] = [int(x) if x != "#" else "#" for x in terrain[i]]

    for query in queries:
        sr, sc, er, ec = query
        result = dijkstra((sr, sc), (er, ec))
        results.append(result)

    return results

# 输入处理
if __name__ == "__main__":
    m, n, p = map(int, input().split())
    terrain = [input().split() for _ in range(m)]
    queries = [tuple(map(int, input().split())) for _ in range(p)]

    results = min_energy_cost(m, n, p, terrain, queries)
    for res in results:
        print(res)
```

![image-20241217200849786](C:\Users\남궁성광\AppData\Roaming\Typora\typora-user-images\image-20241217200849786.png)

### 04129: 变换的迷宫



bfs, http://cs101.openjudge.cn/practice/04129/

思路：1h

**找到起点 `S` 和终点 `E`**，初始化 BFS。

**四个方向**(上、下、左、右)尝试扩展：

- 如果下一位置是 `.`：可以直接走。
- 如果下一位置是 `#`，需要检查 `time % K == 0`，只有在石头消失时才能走。

使用 **visited** 数组记录 `(x, y)` 位置在不同时间状态下是否访问过，避免重复搜索。

当到达出口 `E` 时，返回当前时间。

如果队列为空也无法到达出口，输出 "Oop!"。

代码：

```
from collections import deque

def solve_maze(t, test_cases):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上、下、左、右

    def bfs(grid, R, C, K, start, end):
        queue = deque([(start[0], start[1], 0)])  # 初始化队列 (x, y, time)
        visited = [[[False] * (K + 1) for _ in range(C)] for _ in range(R)]  # 记录访问状态
        
        while queue:
            x, y, time = queue.popleft()
            if (x, y) == end:
                return time  # 到达出口，返回时间

            next_time = time + 1
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                # 边界检查
                if 0 <= nx < R and 0 <= ny < C:
                    cell = grid[nx][ny]
                    
                    # 检查石头消失条件
                    if cell == "#" and next_time % K != 0:
                        continue  # 石头未消失，无法走
                    
                    # 只访问未访问过的状态
                    if not visited[nx][ny][next_time % K]:
                        visited[nx][ny][next_time % K] = True
                        queue.append((nx, ny, next_time))

        return "Oop!"  # 无法到达出口

    results = []
    for case in test_cases:
        R, C, K = case[0]
        grid = case[1]
        start, end = None, None

        # 查找起点 S 和终点 E
        for i in range(R):
            for j in range(C):
                if grid[i][j] == "S":
                    start = (i, j)
                elif grid[i][j] == "E":
                    end = (i, j)

        # 运行 BFS
        result = bfs(grid, R, C, K, start, end)
        results.append(result)
    
    return results


# 输入读取
if __name__ == "__main__":
    T = int(input())  # 组数
    test_cases = []
    
    for _ in range(T):
        R, C, K = map(int, input().split())
        grid = [list(input().strip()) for _ in range(R)]
        test_cases.append(((R, C, K), grid))
    
    # 执行并输出结果
    results = solve_maze(T, test_cases)
    for res in results:
        print(res)
```

![image-20241217200721223](C:\Users\남궁성광\AppData\Roaming\Typora\typora-user-images\image-20241217200721223.png)

## 2. 学习总结和收获

……难

如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。