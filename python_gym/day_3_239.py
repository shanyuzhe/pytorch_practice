"""
题目：239 滑动窗口最大值
链接：https://leetcode.cn/problems/sliding-window-maximum/description/

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

#q中维护下标 方便传递窗口尺寸信息
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        q = deque(maxlen = k)
        ans = []
        for i,v in enumerate(nums):
            while q and nums[q[-1]] <= v:
                q.pop()
            q.append(i)
            
            if q[0] == i - k:
                q.popleft()
            if i >= k - 1:
                ans.append(nums[q[0]])
        return ans
# --- 测试用例 ---
if __name__ == "__main__":
    sol = Solution()
    # Case 1
