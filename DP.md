# Dynamic Programming

### Dynamic Programming (DP)

**Dynamic Programming** is a method used in computer science and mathematics to solve problems by breaking them down into simpler subproblems. It is particularly useful for problems with overlapping subproblems and optimal substructure properties.

### Key Concepts of Dynamic Programming:
1. **Optimal Substructure**: 
   - A problem has an optimal substructure if an optimal solution can be constructed from optimal solutions of its subproblems.
   
2. **Overlapping Subproblems**:
   - Instead of solving the same subproblem multiple times, DP solves each subproblem once and stores the result for future reference (memoization or tabulation).

### Steps to Solve Problems Using DP:
1. **Define the Problem in Terms of States**:
   - Identify what states represent in your problem (e.g., current index, sum, etc.).
   
2. **Recursive Relation**:
   - Determine how the solution of the current state can be expressed in terms of solutions of smaller subproblems.

3. **Base Case(s)**:
   - Define the base cases which are the smallest problems whose solutions are known directly.

4. **Memoization or Tabulation**:
   - **Memoization**: Store intermediate results to avoid redundant calculations (top-down approach).
   - **Tabulation**: Use a table to iteratively solve subproblems (bottom-up approach).

---

### Examples of Dynamic Programming Problems:

1. **Fibonacci Sequence**:
   Recurrence relation:  
   \[
   F(n) = F(n-1) + F(n-2)
   \]
   Base case:  
   \[
   F(0) = 0, \quad F(1) = 1
   \]

   **Top-down (Memoization):**
   ```python
   def fib(n, memo={}):
       if n in memo:
           return memo[n]
       if n <= 1:
           return n
       memo[n] = fib(n-1, memo) + fib(n-2, memo)
       return memo[n]
   ```

   **Bottom-up (Tabulation):**
   ```python
   def fib(n):
       if n <= 1:
           return n
       dp = [0] * (n + 1)
       dp[1] = 1
       for i in range(2, n + 1):
           dp[i] = dp[i-1] + dp[i-2]
       return dp[n]
   ```

2. **Knapsack Problem**:
   - Given a set of items with weights and values, determine the maximum value of items that can be included in a knapsack of a given capacity.

3. **Longest Common Subsequence (LCS)**:
   - Find the longest subsequence common to two sequences.

---

### Tips for Using Dynamic Programming:
- Start by identifying whether the problem has overlapping subproblems and optimal substructure.
- Write down the recursive solution first, then optimize it using memoization or tabulation.
- Use a table or dictionary to store intermediate results.

Would you like to see a specific problem solved using Dynamic Programming?
