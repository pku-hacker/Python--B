# Assignment #D: 十全十美



Updated 1254 GMT+8 Dec 17, 2024

2024 fall, Complied by 南宫圣光，2000094627

**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora [https://typoraio.cn](https://typoraio.cn/) ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。

## 1. 题目



### 02692: 假币问题



brute force, http://cs101.openjudge.cn/practice/02692

思路：1h, 根据三次称量的结果，逐步缩小假币的范围。最后确定唯一的假币，并判断其是轻还是重。

代码：

```
## 假币问题(brute force)

def find_counterfeit_coin (test_cases):
    for i in range(test_cases):
        coins = {chr(c): {'light': True, 'heavy': True} for c in range(ord('A'), ord('L') + 1)}
        
        for _ in range(3):
            left, right, result = input().split()

            if result == "even":
                for coin in left + right:
                    coins[coin]['light'] = coins[coin]['heavy'] = False
                    
            else:
                for coin in coins:
                    if coin not in left and coin not in right:
                        coins[coin]['light'] = coins[coin]['heavy'] = False
            
                if result == "up":
                    for coin in left:
                        coins[coin]['light'] = False
                    for coin in right:
                        coins[coin]['heavy'] = False
                
                if result == "down":
                    for coin in right:
                        coins[coin]['light'] = False
                    for coin in left:
                        coins[coin]['heavy'] = False
                    
        
        for coin, states in coins.items():
            if states['light']:
                print(f"{coin} is the counterfeit coin and it is light.")
                break   
            elif states['heavy']:
                print(f"{coin} is the counterfeit coin and it is heavy.")
                break


n = int(input())
find_counterfeit_coin(n)
```



![image-20241217153424122](C:\Users\남궁성광\AppData\Roaming\Typora\typora-user-images\image-20241217153424122.png)

### 01088: 滑雪



dp, dfs similar, http://cs101.openjudge.cn/practice/01088

思路：40min, 

遍历整个矩阵，从每个点作为起点开始滑行。

使用深度优先搜索（DFS）尝试向四个方向滑行，并记录当前点的最长滑行路径。

如果某个点的最长路径已经被计算过，直接返回结果，避免重复计算。

代码：

```
##滑雪(DP, Dfs)

directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

def longest_slide(R, C, height):

    memo = [[0] * C for _ in range(R)]
    
    def dfs(x, y):
        
        if memo[x][y] != 0:
            return memo[x][y]
        
        max_length = 1
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < R and 0 <= ny < C and height[nx][ny] < height[x][y]:
                max_length = max(max_length, 1 + dfs(nx, ny))
                
        memo[x][y] = max_length
        return max_length
    
    result = 0
    for i in range(R):
        for j in range(C):
            result = max(result, dfs(i, j))
        
    return result

R, C = map(int, input().split())
height = [list(map(int, input().split())) for _ in range(R)]

print(longest_slide(R, C, height))
```



![image-20241217164526544](C:\Users\남궁성광\AppData\Roaming\Typora\typora-user-images\image-20241217164526544.png)

### 25572: 螃蟹采蘑菇



bfs, dfs, http://cs101.openjudge.cn/practice/25572/

思路：1h, 从两个值为 `5` 的格子开始，计算两者的 **相对坐标差**，判断小螃蟹的初始方向

代码：

```
from collections import deque

n = int(input()) 
mat = []  # 用于存储迷宫地图
for i in range(n):
    mat.append(list(map(int, input().split())))  # 输入每一行的迷宫信息，并存入列表中

a = []  # 用于存储小螃蟹身体占据的两个格子的坐标
for i in range(n):
    for j in range(n):
        if mat[i][j] == 5:  # 找到值为 5 的位置（表示小螃蟹的身体）
            a.append([i, j])  # 将小螃蟹的坐标存入列表 a 中

# 计算小螃蟹身体两格之间的相对坐标差，判断方向
lx = a[1][0] - a[0][0]  # x 方向上的差值（用于判断是横向还是纵向）
ly = a[1][1] - a[0][1]  # y 方向上的差值

# 定义四个移动方向：上、右、下、左
dire = [[-1, 0], [0, 1], [1, 0], [0, -1]]

# 初始化一个访问数组 v，用于标记哪些位置已经访问过
v = [[0] * n for i in range(n)]


def bfs(x, y):
    """
    广度优先搜索(BFS)函数
    输入：x, y - 小螃蟹当前身体一侧的起始坐标
    输出：'yes' - 如果可以到达目标点（值为9）
          'no' - 如果无法到达目标点
    """
    v[x][y] = 1  # 标记起始位置已访问
    quene = deque([(x, y)])  # 初始化队列，并将起始位置加入队列

    while quene:  # 当队列非空时继续搜索
        x, y = quene.popleft()  # 从队列中取出当前位置

        # 检查是否到达目标点（9），身体两格中的任意一格到达即可
        if (mat[x][y] == 9 and mat[x + lx][y + ly] != 1) or \
           (mat[x][y] != 1 and mat[x + lx][y + ly] == 9):
            return 'yes'

        # 尝试向四个方向移动
        for i in range(4):
            dx = x + dire[i][0]  # 新的 x 坐标
            dy = y + dire[i][1]  # 新的 y 坐标

            # 检查移动条件：边界范围内、未访问过、两格位置都不为墙体(1)
            if 0 <= dx < n and 0 <= dy < n and 0 <= dx + lx < n \
                    and 0 <= dy + ly < n and v[dx][dy] == 0 \
                    and mat[dx][dy] != 1 and mat[dx + lx][dy + ly] != 1:
                quene.append([dx, dy])  # 将新的位置加入队列
                v[dx][dy] = 1  # 标记新位置为已访问

    return 'no'  # 如果队列为空，仍未到达目标点，则返回 'no'


# 调用 BFS 函数，从小螃蟹身体一侧的起始坐标开始搜索
print(bfs(a[0][0], a[0][1]))

```



![image-20241217192814609](C:\Users\남궁성광\AppData\Roaming\Typora\typora-user-images\image-20241217192814609.png)

### 27373: 最大整数



dp, http://cs101.openjudge.cn/practice/27373/

思路：1h, 比较字符串拼接后的大小，确保 `l[j] + l[j+1]` > `l[j+1] + l[j]`，从而使大的数字拼接优先级更高。

代码：

```
def f(string):
    # 辅助函数：如果字符串为空，返回0，否则转换为整数
    if string == '':
        return 0
    else:
        return int(string)

# 输入最大位数 m 和正整数的数量 n
m = int(input())  # 最大位数
n = int(input())  # 正整数数量

# 输入所有的数字
l = input().split()

# 冒泡排序：根据字符串拼接规则排序，确保能组成最大的数
for i in range(n):
    for j in range(n - 1 - i):
        # 比较两个字符串拼接后的大小，保证最大数优先
        if l[j] + l[j + 1] > l[j + 1] + l[j]:
            l[j], l[j + 1] = l[j + 1], l[j]

# weight数组：存储每个元素的位数
weight = []  
for num in l:
    weight.append(len(num))

# dp[i][j]：表示前i个数中选择，不超过j位时，能组成的最大可能数值
dp = [[''] * (m + 1) for _ in range(n + 1)]

# 初始化dp数组：位数为0或者前0个数时，都无法组成整数
for k in range(m + 1):
    dp[0][k] = ''  # 选择0个数，无法组成数字
for q in range(n + 1):
    dp[q][0] = ''  # 位数为0，无法组成数字

# 动态规划状态转移
for i in range(1, n + 1):  # 遍历前i个数
    for j in range(1, m + 1):  # 遍历位数限制j
        if weight[i - 1] > j:  # 当前数的位数超过限制，不能选择当前数
            dp[i][j] = dp[i - 1][j]  # 直接继承不选当前数的状态
        else:
            # 选择当前数与不选择当前数之间取最大值
            dp[i][j] = str(max(
                f(dp[i - 1][j]),  # 不选当前数
                int(l[i - 1] + dp[i - 1][j - weight[i - 1]])  # 选择当前数并拼接
            ))

# 输出最终结果
print(dp[n][m])

```

![image-20241217193214992](C:\Users\남궁성광\AppData\Roaming\Typora\typora-user-images\image-20241217193214992.png)

### 02811: 熄灯问题



brute force, http://cs101.openjudge.cn/practice/02811

思路：1h, 

通过 **暴力枚举** 所有可能的第1行按钮状态。

逐行处理按钮按下的影响，使得上一行的灯全部熄灭。

最后检查第5行是否全部熄灭。

如果满足条件，输出按钮按下的状态。

代码：

```
from copy import deepcopy  
from itertools import product  

# 定义一个映射字典，用于翻转灯的状态 (0变1，1变0)
rmap = {0: 1, 1: 0}

# 初始化矩阵，增加边界行(全0)避免边界处理复杂
# 输入5行，每行6个数字。外加上下边界的全0行，左侧和右侧的0作为边界
matrix_backup = [[0] * 8] + [[0, *map(int, input().split()), 0] for i in range(5)] \
    + [[0] * 8]

# 遍历所有可能的第1行的按钮按下状态(二进制的6位数，共 2^6 = 64 种组合)
for test in product(range(2), repeat=6):
    # 深拷贝初始矩阵，避免修改原始矩阵
    matrix = deepcopy(matrix_backup)
    # 初始化触发记录列表，记录每一行的按钮按下状态
    triggers = [list(test)]

    # 遍历从第1行到第5行
    for i in range(1, 6):  # 从第1行到第5行 (1-based)
        for j in range(1, 7):  # 遍历第1列到第6列 (1-based)
            # 如果当前按钮的触发状态为1，按下这个按钮
            if triggers[i - 1][j - 1]:
                # 翻转当前按钮及其上下左右的灯状态
                matrix[i][j] = rmap[matrix[i][j]]           # 当前灯
                matrix[i - 1][j] = rmap[matrix[i - 1][j]]   # 上方灯
                matrix[i + 1][j] = rmap[matrix[i + 1][j]]   # 下方灯
                m

```

![image-20241217193420166](C:\Users\남궁성광\AppData\Roaming\Typora\typora-user-images\image-20241217193420166.png)

### 08210: 河中跳房子



binary search, greedy, http://cs101.openjudge.cn/practice/08210/

思路：1h, 

在尝试跳跃距离 `mid` 的情况下，依次遍历岩石，记录前一个保留的岩石位置。

如果当前岩石与前一个保留岩石之间的距离小于 `mid`，那么当前岩石必须移除。

统计移除的岩石数量，并判断是否超过 `M`。

代码：

```
def can_achieve_distance(mid, rocks, L, M):
    """ 判断在跳跃距离为 mid 的情况下，移除 M 个岩石是否可行 """
    removed_count = 0  # 移除的岩石数
    prev_position = 0  # 上一个保留的岩石位置
    
    for rock in rocks:
        if rock - prev_position < mid:  # 间隔小于 mid，需要移除当前岩石
            removed_count += 1
            if removed_count > M:  # 超过允许的移除数量
                return False
        else:
            prev_position = rock  # 保留当前岩石
    
    # 最终检查终点是否满足条件
    if L - prev_position < mid:
        removed_count += 1
    
    return removed_count <= M

def max_min_jump_distance(L, N, M, rocks):
    rocks.sort()  # 对岩石位置排序
    left, right = 1, L  # 二分查找的初始范围
    result = 0
    
    while left <= right:
        mid = (left + right) // 2
        if can_achieve_distance(mid, rocks, L, M):
            result = mid  # 更新答案
            left = mid + 1  # 尝试更大的跳跃距离
        else:
            right = mid - 1  # 缩小范围
    
    return result

# 输入处理
if __name__ == "__main__":
    L, N, M = map(int, input().split())
    rocks = [int(input()) for _ in range(N)]
    print(max_min_jump_distance(L, N, M, rocks))

```

![image-20241217193627499](C:\Users\남궁성광\AppData\Roaming\Typora\typora-user-images\image-20241217193627499.png)

## 2. 学习总结和收获

担心下周考试……

如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。