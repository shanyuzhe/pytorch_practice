"""
题目：994
链接：https://leetcode.cn/problems/rotting-oranges/

复杂度分析：
- 时间复杂度：O(N)
- 空间复杂度：O(N)

边界条件 (Edge Cases):
1. 
"""

import sys
from collections import Counter, defaultdict, deque
from typing import List
import heapq


class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        q = deque()
        fresh_count = 0
        
        # 1. 初始化：统计新鲜橘子，并将所有腐烂橘子入队
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 2:
                    q.append((i, j))
                elif grid[i][j] == 1:
                    fresh_count += 1
        
        # 如果一开始就没有新鲜橘子，直接返回 0
        if fresh_count == 0:
            return 0
        
        minutes = 0
        # 2. 开始多源 BFS
        while q and fresh_count > 0:
            minutes += 1
            # 这一层处理当前所有腐烂橘子
            for _ in range(len(q)):
                x, y = q.popleft()
                
                # 上下左右四个方向
                for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                    nx, ny = x + dx, y + dy
                    
                    # 边界检查 + 是否是新鲜橘子
                    if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] == 1:
                        grid[nx][ny] = 2  # 变腐烂
                        fresh_count -= 1  # 新鲜橘子少一个
                        q.append((nx, ny))
        
        # 3. 最后检查：如果还有新鲜橘子没烂掉，返回 -1
        return minutes if fresh_count == 0 else -1