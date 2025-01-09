# Assignment #A: dp & bfs



Updated 2 GMT+8 Nov 25, 2024

2024 fall, Complied by 南宫圣光-元培

**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora [https://typoraio.cn](https://typoraio.cn/) ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。

## 1. 题目



### LuoguP1255 数楼梯



dp, bfs, https://www.luogu.com.cn/problem/P1255

思路：对于剩下 nnn 阶楼梯的情况，走法总数是 f(n−1)（先走一步）和 f(n−2)（先走两步）的和，

即： **f(n) = f(n−1) + f(n−2)**

这实际上是**斐波那契数列的一个变体**

花费时间：10min

代码：

```
##数楼梯(dp)

n = int(input())

def num_stairs (n):
    if n == 1:
        return 1
    if n == 2:
        return 2
    
    dp = [0] * (n+1)
    dp[1] = 1
    dp[2] = 2
    for i in range(3, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
        
    return dp[n]

print(num_stairs(n))
```



![image-20241129143450522](C:\Users\남궁성광\AppData\Roaming\Typora\typora-user-images\image-20241129143450522.png)

### 27528: 跳台阶



dp, http://cs101.openjudge.cn/practice/27528/

思路：走到第 iii 级的方式数 **f(i)等于前i−1 级台阶所有方式数之和再加上直接一步到达的1种方式**：

**f(i)=f(1)+f(2)+⋯+f(i−1)+1**

花费时间：10min

代码：

```
## 跳台阶（DP）

N = int(input())

def find_method (N):
    
    dp = [0] * (N + 1)
    dp[1] = 1
    
    for i in range(2, N + 1):
        dp[i] = dp[i-1] * 2
    
    return dp[N]

print(find_method(N))    
```

![image-20241129145236315](C:\Users\남궁성광\AppData\Roaming\Typora\typora-user-images\image-20241129145236315.png)

### 474D. Flowers



dp, https://codeforces.com/problemset/problem/474/D

思路：实现方法解释

1. **动态规划 (`dp`) 计算：**
   - 逐步计算吃掉 xxx 朵花的方案数。
   - 对于 x≥kx \geq kx≥k，检查是否满足白色花必须成组的条件。
2. **前缀和 (`prefix`)：**
   - 通过前缀和数组，快速计算某个区间内的方案总数。
3. **查询处理：**
   - 每个查询 [a,b][a, b][a,b] 的结果可通过前缀和数组在 O(1)O(1)O(1) 时间内计算。

花费时间： 1h

代码：

```
MOD = 1000000007

def marmot_dinner(t, k, queries):
    # 找到所有查询中最大的 b 值
    max_b = max(b for _, b in queries)

    # 第1步：计算 dp 数组
    dp = [0] * (max_b + 1)
    dp[0] = 1  # 基本情况：吃 0 朵花的方案数为 1

    for x in range(1, max_b + 1):
        dp[x] = dp[x - 1]  # 吃全是红色花的情况
        if x >= k:
            dp[x] += dp[x - k]  # 加入一个白色花组
        dp[x] %= MOD  # 取模保持结果范围

    # 第2步：计算 prefix 数组
    prefix = [0] * (max_b + 1)
    for x in range(1, max_b + 1):
        prefix[x] = (prefix[x - 1] + dp[x]) % MOD

    # 第3步：计算每个查询的结果
    results = []
    for a, b in queries:
        # 区间 [a, b] 的方案数
        result = (prefix[b] - prefix[a - 1]) % MOD
        results.append(result)

    return results

# 输入处理
t, k = map(int, input().split())  # 读取查询个数和 k 值
queries = [tuple(map(int, input().split())) for _ in range(t)]  # 每个查询 [a, b]
results = marmot_dinner(t, k, queries)
print("\n".join(map(str, results)))  # 输出结果
```



![image-20241129150332648](C:\Users\남궁성광\AppData\Roaming\Typora\typora-user-images\image-20241129150332648.png)

### LeetCode5.最长回文子串



dp, two pointers, string, https://leetcode.cn/problems/longest-palindromic-substring/

思路：状态转移方程：

- 如果 s[i]==s[j]s[i] == s[j]s[i]==s[j]，且子串 s[i+1:j−1]s[i+1:j-1]s[i+1:j−1] 是回文，则 dp[i][j]=Truedp[i][j] = Truedp[i][j]=True。

- 子串长度为 1 时总是回文。

- 子串长度为 2 时，只要两个字符相同就为回文。

  

  花费时间：1h

  代码：

```
class Solution:
    def longestPalindrome(self, s: str) -> str:
        def expand_around_center(left: int, right: int) -> (int, int):
            # 以中心为基准扩展回文
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            return left + 1, right - 1  # 返回回文的起始和结束索引

        start, end = 0, 0  # 记录最长回文的起始和结束索引
        for i in range(len(s)):
            # 奇数长度回文
            left1, right1 = expand_around_center(i, i)
            # 偶数长度回文
            left2, right2 = expand_around_center(i, i + 1)

            # 选择更长的回文
            if right1 - left1 > end - start:
                start, end = left1, right1
            if right2 - left2 > end - start:
                start, end = left2, right2

        return s[start:end + 1]
```

![image-20241130193930728](C:\Users\남궁성광\AppData\Roaming\Typora\typora-user-images\image-20241130193930728.png)

### 12029: 水淹七军



bfs, dfs, http://cs101.openjudge.cn/practice/12029/

思路：自己做出来要么报错，要么runtime error 最终参照了参考答案……

花费时间：1.5h

代码：

```
from collections import deque
import sys
input = sys.stdin.read

# 判断坐标是否有效
def is_valid(x, y, m, n):
    return 0 <= x < m and 0 <= y < n

# 广度优先搜索模拟水流
def bfs(start_x, start_y, start_height, m, n, h, water_height):
    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]
    q = deque([(start_x, start_y, start_height)])
    water_height[start_x][start_y] = start_height

    while q:
        x, y, height = q.popleft()
        for i in range(4):
            nx, ny = x + dx[i], y + dy[i]
            if is_valid(nx, ny, m, n) and h[nx][ny] < height:
                if water_height[nx][ny] < height:
                    water_height[nx][ny] = height
                    q.append((nx, ny, height))

# 主函数
def main():
    data = input().split()  # 快速读取所有输入数据
    idx = 0
    k = int(data[idx])
    idx += 1
    results = []

    for _ in range(k):
        m, n = map(int, data[idx:idx + 2])
        idx += 2
        h = []
        for i in range(m):
            h.append(list(map(int, data[idx:idx + n])))
            idx += n
        water_height = [[0] * n for _ in range(m)]

        i, j = map(int, data[idx:idx + 2])
        idx += 2
        i, j = i - 1, j - 1

        p = int(data[idx])
        idx += 1

        for _ in range(p):
            x, y = map(int, data[idx:idx + 2])
            idx += 2
            x, y = x - 1, y - 1
            if h[x][y] <= h[i][j]:
                continue
            bfs(x, y, h[x][y], m, n, h, water_height)

        results.append("Yes" if water_height[i][j] > 0 else "No")

    sys.stdout.write("\n".join(results) + "\n")

if __name__ == "__main__":
    main()
```



![image-20241130194936083](C:\Users\남궁성광\AppData\Roaming\Typora\typora-user-images\image-20241130194936083.png)

### 02802: 小游戏



bfs, http://cs101.openjudge.cn/practice/02802/

思路：这道题也是……

花费时间：2h

代码：

```
from collections import deque

def bfs(start, end, grid, h, w):
    queue = deque([start])
    visited = set()
    dirs = [(0, -1), (-1, 0), (0, 1), (1, 0)]

    ans = []
    while queue:
        x, y, d_i_r, seg = queue.popleft()
        #print(x,y,end)
        if (x, y) == end:
            #return seg
            ans.append(seg)
            break

        for i, (dx, dy) in enumerate(dirs):
            nx, ny = x + dx, y + dy
 
            if 0 <= nx < h+2 and 0 <= ny < w+2 and ((nx, ny, i) not in visited):
                new_dir = i
                new_seg = seg if new_dir == d_i_r else seg + 1
                if (nx, ny) == end:
                    #return new_seg
                    ans.append(new_seg)
                    continue
                
                if grid[nx][ny] != 'X':
                    visited.add((nx, ny, i))
                    queue.append((nx, ny, new_dir, new_seg))

    if len(ans) == 0:
        return -1
    else:
        return min(ans)

board_num = 1
while True:
    w, h = map(int, input().split())
    if w == h == 0:
        break

    #grid = [[' '] * (w + 2)] + \
            #[[' '] + list(input()) + [' '] for _ in range(h)] + \
            #[[' '] * (w + 2)]
    grid = [' ' * (w + 2)] + [' ' + input() + ' ' for _ in range(h)] + [' ' * (w + 2)]
    print(f"Board #{board_num}:")
    pair_num = 1
    while True:
        y1, x1, y2, x2 = map(int, input().split())
        if x1 == y1 == x2 == y2 == 0:
            break

        start = (x1, y1, -1, 0)
        end = (x2, y2)

        seg = bfs(start, end, grid, h, w)
        if seg == -1:
            print(f"Pair {pair_num}: impossible.")
        else:
            print(f"Pair {pair_num}: {seg} segments.")
        pair_num += 1

    print()
    board_num += 1
```

![image-20241130195610939](C:\Users\남궁성광\AppData\Roaming\Typora\typora-user-images\image-20241130195610939.png)

## 2. 学习总结和收获

整体上花费的时间很多，主要是因为一开始没有思路，而且做起来题又复杂。这次感觉有点崩溃，强烈感到期末前得好好熟悉这些算法。
