# Assignment #B: Dec Mock Exam大雪前一天



Updated 1649 GMT+8 Dec 5, 2024

2024 fall, Complied by 南宫圣光、元培

**说明：**

1）⽉考： AC6（请改为同学的通过数） 。考试题⽬都在“题库（包括计概、数算题目）”⾥⾯，按照数字题号能找到，可以重新提交。作业中提交⾃⼰最满意版本的代码和截图。

2）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora [https://typoraio.cn](https://typoraio.cn/) ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。

## 1. 题目



### E22548: 机智的股民老张



http://cs101.openjudge.cn/practice/22548/

思路：用dp的思考方式，每次更新max_profit和min_price

花费时间：5min

代码：

```
##机智的股民老张

a = list(map(int, input().split()))

max_profit = 0
min_price = a[0]

for price in a[1:]:
    max_profit = max(max_profit, price - min_price)
    min_price = min(min_price, price)
    
print(max_profit)
```



![image-20241209152236052](C:\Users\남궁성광\AppData\Roaming\Typora\typora-user-images\image-20241209152236052.png)

### M28701: 炸鸡排



greedy, http://cs101.openjudge.cn/practice/28701/

思路：没太懂，参照了答案的解析

花费时间：1h

代码：

```
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    k = int(data[1])
    times = [int(data[2 + i]) for i in range(n)]
    
    # 计算总炸制时间
    total_time = sum(times)
    
    # 对炸制时间进行排序
    times.sort()
    
    # 初始最大持续时间为总炸制时间除以 k
    max_time = total_time / k
    
    # 如果最长的炸制时间大于或等于 max_time，则需要调整 k 的值
    if times[-1] > max_time:
        for i in range(n - 1, -1, -1):
            if times[i] <= max_time:
                break
            total_time -= times[i]
            k -= 1
            max_time = total_time / k
    
    # 输出结果，保留三位小数
    print(f"{max_time:.3f}")

if __name__ == "__main__":
    main()
```



![image-20241209173901973](C:\Users\남궁성광\AppData\Roaming\Typora\typora-user-images\image-20241209173901973.png)

### M20744: 土豪购物



dp, http://cs101.openjudge.cn/practice/20744/

思路：使用前缀和动态维护当前最大连续子数组的总和。

花费时间：1h

代码：

```
def max_value_with_removal(values):
    values = list(map(int, values.split(',')))
    n = len(values)
    
    if n == 1:
        return values[0]
    
    dp_no_removal = [0] * n
    dp_with_removal = [0] * n

    dp_no_removal[0] = values[0]
    dp_with_removal[0] = float('-inf')  
    

    for i in range(1, n):
        dp_no_removal[i] = max(values[i], dp_no_removal[i-1] + values[i])
        dp_with_removal[i] = max(dp_with_removal[i-1] + values[i], dp_no_removal[i-1])

    return max(max(dp_no_removal), max(dp_with_removal))

input_data = input()  
print(max_value_with_removal(input_data))
```



![image-20241209202231738](C:\Users\남궁성광\AppData\Roaming\Typora\typora-user-images\image-20241209202231738.png)

### T25561: 2022决战双十一



brute force, dfs, http://cs101.openjudge.cn/practice/25561/

思路：这道题一直写下去，一直报错……参照了答案

代码：

```
result = float("inf")
n, m = map(int, input().split())
store_prices = [input().split() for _ in range(n)]
you= [input().split() for _ in range(m)]
la=[0]*m
def dfs(i,sum1):
    global result
    if i==n:
        jian=0
        for i2 in range(m):
            store_j=0
            for k in you[i2]:
                a,b=map(int,k.split('-'))
                if la[i2]>=a:
                    store_j=max(store_j,b)
            jian+=store_j
        result=min(result,sum1-(sum1//300)*50-jian)
        return
    for i1 in store_prices[i]:
        idx,p=map(int,i1.split(':'))
        la[idx-1]+=p
        dfs(i+1,sum1+p)
        la[idx-1]-=p
dfs(0,0)
print(result)
```



![image-20241209204827691](C:\Users\남궁성광\AppData\Roaming\Typora\typora-user-images\image-20241209204827691.png)

### T20741: 两座孤岛最短距离



dfs, bfs, http://cs101.openjudge.cn/practice/20741/

思路：用 **DFS** 或 **BFS** 遍历整张地图，将属于第一个孤岛的所有点标记为一种颜色，再找出属于第二个孤岛的所有点标记为另一种颜色

花费时间：1h

代码：

```
from collections import deque

def min_bridge_steps(n, grid):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右方向

    def in_bounds(x, y):
        return 0 <= x < n and 0 <= y < n

    def dfs_mark_island(x, y, mark):
        """标记一个孤岛为 mark"""
        stack = [(x, y)]
        grid[x][y] = mark
        island = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if in_bounds(nx, ny) and grid[nx][ny] == 1:
                    grid[nx][ny] = mark
                    stack.append((nx, ny))
                    island.append((nx, ny))
        return island

    # 1. 找到两个孤岛并分别标记
    islands = []
    mark = 2
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 1:
                islands.append(dfs_mark_island(i, j, mark))
                mark += 1

    # 2. 用多源 BFS 计算最短距离
    queue = deque()
    visited = set()
    for x, y in islands[0]:
        queue.append((x, y, 0))  # (x, y, 当前路径长度)
        visited.add((x, y))

    while queue:
        x, y, dist = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if in_bounds(nx, ny):
                if grid[nx][ny] == 3:  # 到达第二个孤岛
                    return dist
                if grid[nx][ny] == 0 and (nx, ny) not in visited:
                    queue.append((nx, ny, dist + 1))
                    visited.add((nx, ny))

    return -1  # 理论上不会发生，因为总有两个孤岛

# 输入处理
def main():
    n = int(input())
    grid = [list(map(int, input().strip())) for _ in range(n)]
    print(min_bridge_steps(n, grid))

if __name__ == "__main__":
    main()
```



![image-20241209205055624](C:\Users\남궁성광\AppData\Roaming\Typora\typora-user-images\image-20241209205055624.png)

### T28776: 国王游戏



greedy, http://cs101.openjudge.cn/practice/28776

思路：首先输入大臣数量 nnn，国王左手和右手上的数字 a0,b0a0, b0a0,b0。然后输入每位大臣左手和右手的数字 a,ba, ba,b，并存储在列表 numbers 中。根据每位大臣的 a×ba \times ba×b 值对大臣排序，目的是尽量减少后续大臣因左手乘积增长而带来的影响。奖励公式为 a0//ba0 // ba0//b，即前面所有人的左手乘积除以当前大臣右手上的数字。在每次计算后，将当前大臣的左手数字 aaa 累乘到 a0a0a0。

花费时间：1h

代码：

```
##国王游戏

n = int(input())  # 大臣数量
a0, b0 = map(int, input().split())  # 国王左手和右手的数字
numbers = []
for _ in range(n):
    a, b = map(int, input().split())  # 每位大臣左手和右手的数字
    numbers.append((a, b))

# 按照大臣的左手数字 * 右手数字排序，减少后续影响
numbers.sort(key=lambda x: (x[0] * x[1]))

result = 0
for i in range(n):
    # 当前大臣获得的金币
    result = max(result, a0 // numbers[i][1])
    # 更新当前乘积（前面所有人的左手数字的乘积）
    a0 *= numbers[i][0]

print(result)  # 输出金币最多的大臣的最少金币数

```



![image-20241209210105264](C:\Users\남궁성광\AppData\Roaming\Typora\typora-user-images\image-20241209210105264.png)

## 2. 学习总结和收获



左后一次月考也结束了，对考试的负担也增加了不少。但是没办法，就得付出时间学习我不会的dp, dfs。加油！