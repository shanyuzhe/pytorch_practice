"""
题目：102. 二叉树的层序遍历
链接：https://leetcode.cn/problems/binary-tree-level-order-traversal/description/

复杂度分析：
- 时间复杂度：O(N)
- 空间复杂度：O(N)

边界条件 (Edge Cases):
1. 
"""

import sys
import math
from collections import *
from typing import List
import heapq

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    
    def solve(self, root:Optional[TreeNode]) -> List[List[int]]:
    #Optional[]修饰符意思是TreeNode有可能为空 并不是列表
        if not root:
            return []
            
            # 优化1: 队列只存 node，不存层级索引
        q = deque([root])
        ans = []
        
        while q:
            # 优化2: 记录当前层的节点数量
            # 这一步是核心！此时队列里的长度，就是这一层的所有节点数
            current_layer_size = len(q)
            current_layer_vals = []
            
            # 优化3: 一次性把这一层的节点全部取出来
            for _ in range(current_layer_size):
                node = q.popleft()
                current_layer_vals.append(node.val)
                
                # 把下一层的孩子加入队列（为下一轮 while 做准备）
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            
            # 这一层完事了，直接加到结果里
            ans.append(current_layer_vals)
            
        return ans
        

# --- 测试用例 ---
if __name__ == "__main__":
    sol = Solution()
    # Case 1
