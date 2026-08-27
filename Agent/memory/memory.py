import numpy as np
import os
import datetime

class Memory:
    def __init__(
        self,
        max_short_term_size=3,
        use_short_term_memory=True, # 默认开启
        max_long_term_size=3,
        use_long_term_memory=True,  # 默认开启
        default_unit="MPa",         # 默认单位
        optimization_mode="max"     # 模式: 'max', 'min', 'target'
    ):
        # 记录 Planner 的思考过程（工具选择、反思）
        self.memory_planning = {
            "reflection": [],
            "choice": [],
            "reason": [],
        }
        # 记录实际执行结果（材料成分、性能反馈）
        self.memory = {
            "reflection": [],
            "composition": [],      # 或者是 structure_name / idea_title
            "reason": [],
            "predicted_value": [],  # 存储提取出的数值结果
            "feedback": [],         # 存储完整的文本反馈
        }
        self.max_short_term_size = max_short_term_size
        self.max_long_term_size = max_long_term_size
        self.use_short_term_memory = use_short_term_memory
        self.use_long_term_memory = use_long_term_memory
        self.default_unit = default_unit
        self.optimization_mode = optimization_mode

    def store_plan(self, plan):
        """存储 Planner 的决策过程"""
        for key in ["reflection", "choice", "reason"]:
            if key in plan:
                self.memory_planning[key].append(plan[key])
            else:
                self.memory_planning[key].append("N/A")

    def get_situation(self, max_situation_size=5):
        """生成当前状态描述，用于 Prompt 的上下文 (Current Situation)"""
        unit = self.default_unit
        num_trials = len(self.memory_planning["choice"])

        if num_trials == 0:
            return "You have not yet attempted to propose a new polymer structure and do not have enough experience.\n"
        
        output = f"You have made {num_trials} attempts to propose new polymers so far.\n"

        # 如果还没有具体的性能数据
        if len(self.memory["predicted_value"]) == 0:
            return output
        
        # 获取最近的一次回馈
        output += f"Here are the outcomes of your latest proposals:\n"

        # [修复] 确保遍历长度不超限
        available_count = min(len(self.memory_planning["choice"]), len(self.memory["predicted_value"]))
        safe_len = min(available_count, max_situation_size)

        for i in range(safe_len):
            idx = - (i + 1)
            
            # 获取当前值
            curr_val = self.memory['predicted_value'][idx]
            choice = self.memory_planning['choice'][idx]
            
            # [修复] 尝试获取前一次值，如果索引越界则只显示当前值
            prev_idx = idx - 1
            if abs(prev_idx) <= len(self.memory['predicted_value']):
                prev_val = self.memory['predicted_value'][prev_idx]
                output += f"* Action [{choice}] -> Property changed from {prev_val:.3f} to {curr_val:.3f} {unit}\n"
            else:
                # 这是第一步，没有"前一次"
                output += f"* Action [{choice}] -> Resulting property: {curr_val:.3f} {unit}\n"
            
        return output

    def store(self, next_guess, feedback, predicted_value):
        """存储一次迭代的完整结果"""
        # 1. 存 LLM 的输出 (Idea/Composition)
        for key in ["reflection", "composition", "reason"]:
            self.memory[key].append(next_guess.get(key, "No info"))
            
        # 2. 存 环境/仿真/RAG 的反馈
        self.memory["feedback"].append(feedback)
        self.memory["predicted_value"].append(predicted_value)

    def get_memory(self, target=None):
        """根据配置返回短期或长期记忆"""
        mem_str = ""
        if self.use_short_term_memory:
            st_mem = self.get_short_term_memory()
            if st_mem:
                mem_str += "### Short-term Memory (Recent Evolution):\n"
                mem_str += st_mem + "\n"
        
        if self.use_long_term_memory:
            lt_mem = self.get_long_term_memory(target)
            if lt_mem:
                mem_str += "### Long-term Memory (Best Historical Attempts):\n"
                mem_str += lt_mem + "\n"
            
        return mem_str

    def get_short_term_memory(self):
        """获取最近的几次尝试链条"""
        unit = self.default_unit
        num_memory = len(self.memory["composition"])
        
        # [修复] 只有1条记录时无法对比，返回空或者简单描述
        if num_memory < 2:
            return ""

        count = min(num_memory - 1, self.max_short_term_size)
        output = ""

        # 倒序读取
        for i in range(count):
            idx = - (i + 1) # -1, -2...
            
            prev_idx = idx - 1
            
            # [修复] 越界检查
            if abs(prev_idx) > len(self.memory["predicted_value"]): 
                break
            
            prev_val = self.memory["predicted_value"][prev_idx]
            curr_val = self.memory["predicted_value"][idx]
            
            comp_prev = self.memory["composition"][prev_idx]
            comp_curr = self.memory["composition"][idx]
            reason = self.memory["reason"][idx]
            
            # 注意：这里的 Iteration 编号逻辑是：总次数 + 当前倒序索引 + 1
            # 例如总共5次，idx=-1(最后一次)，iteration = 5
            iter_num = num_memory + idx + 1
            
            output += f"""
- Iteration {iter_num}:
  From: {comp_prev} -> To: {comp_curr}
  Reasoning: {reason}
  Outcome: {prev_val:.3f} -> {curr_val:.3f} {unit}
"""
        return output

    def get_long_term_memory(self, target=None):
        """获取历史上表现最好的几次尝试"""
        unit = self.default_unit
        
        # 如果数据太少，没必要区分长期记忆
        if len(self.memory["predicted_value"]) < 2:
            return ""

        # MatAgent 逻辑：通常跳过初始种子(index 0)，看后续生成的
        # 这里为了稳健，如果只有2条数据，就都看
        start_idx = 1 if len(self.memory["predicted_value"]) > 2 else 0
        
        val_hist = np.array(self.memory["predicted_value"][start_idx:])
        indices_offset = start_idx 

        num_memory = min(len(val_hist), self.max_long_term_size)
        
        # [逻辑] 根据优化模式选择最佳索引
        if self.optimization_mode == "target" and target is not None:
            # 接近目标值 (绝对差最小)
            best_local_indices = np.argsort(np.abs(val_hist - target))[:num_memory]
        elif self.optimization_mode == "max":
            # 最大化 (取最大的几个)
            best_local_indices = np.argsort(val_hist)[-num_memory:][::-1]
        elif self.optimization_mode == "min":
            # 最小化 (取最小的几个)
            best_local_indices = np.argsort(val_hist)[:num_memory]
        else:
            # 默认 fallback 到 max
            best_local_indices = np.argsort(val_hist)[-num_memory:][::-1]

        output = ""
        for local_idx in best_local_indices:
            # 还原真实索引
            idx_real = local_idx + indices_offset
            
            # 获取该次尝试的结果
            curr_val = self.memory["predicted_value"][idx_real]
            
            # 尝试获取前一次的结果作为对比（Context）
            prev_val_str = "N/A"
            if idx_real - 1 >= 0:
                prev_val = self.memory["predicted_value"][idx_real-1]
                prev_val_str = f"{prev_val:.3f}"

            output += f"""
- Successful Attempt:
  From: {self.memory["composition"][idx_real-1] if idx_real-1 >=0 else "Start"}
  Proposed: {self.memory["composition"][idx_real]}
  Reasoning: {self.memory["reason"][idx_real]}
  Result: {curr_val:.3f} {unit} (Prev: {prev_val_str})
"""
        return output

    def clear(self):
        """清空所有短期和长期记忆，重置为初始状态"""
        self.memory_planning = {
            "reflection": [],
            "choice": [],
            "reason": [],
        }
        self.memory = {
            "reflection": [],
            "composition": [],
            "reason": [],
            "predicted_value": [],
            "feedback": [],
        }
        print("🧹 Memory has been wiped clean.")

    def save_memory_snapshot(self, folder_path, original_question=None, final_answer=None):
        """
        保存记忆快照，并支持归档原始问题和最终答案
        :param folder_path: 保存文件夹路径
        :param original_question: 用户最初输入的 Prompt 或问题
        :param final_answer: Agent 最终生成的总结性回答
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 固定文件名
        filename = "Agent_Memory_History_Log.txt"
        filepath = os.path.join(folder_path, filename)
        
        # 获取过程记忆内容
        memory_content = self.get_memory()
        situation_content = self.get_situation()
        
        # 追加写入模式 'a'
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(f"\n\n")
            f.write(f"##################################################\n")
            f.write(f"🧠 SNAPSHOT AT {timestamp}\n")
            f.write(f"##################################################\n")
            
            # 1. 写入原始问题 (如果有)
            if original_question:
                f.write(f"❓ [Original Question]\n{original_question}\n")
                f.write(f"-" * 50 + "\n")

            # 2. 写入当前的 Context 和 记忆池
            f.write(f"--- Situation ---\n{situation_content}\n")
            f.write(f"--- Memory Dump ---\n{memory_content}\n")
            
            # 3. 写入最终答案 (如果有)
            if final_answer:
                f.write(f"-" * 50 + "\n")
                f.write(f"✅ [Final Answer / Summary]\n{final_answer}\n")

            f.write(f"##################################################\n") # 结束分割线
            
        return filepath
        