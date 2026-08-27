# domain_router.py
import json
import os
import sys

current_file = os.path.abspath(__file__)
intent_dir = os.path.dirname(current_file)
agent_dir = os.path.dirname(intent_dir)
project_root = os.path.dirname(agent_dir)
if project_root not in sys.path:
    sys.path.append(project_root)
from Agent.Agent_Config.deepseek_client import call_deepseek_llm

# 🔥【修改点】：Prompt 内容已深度定制，与 expert JSON 的节点 ID 严格对齐
DOMAIN_PROMPT = """
You are a domain classifier for degradable polymer research.

Your task:
Given a user query, classify it into a structured ontology. 
CRITICAL: You must map user terms to the STANDARD KNOWLEDGE GRAPH NODES listed below.

### 🔴 STRICT CONSTRAINTS:
1. **ALIGNMENT**: If the user asks for "Young's modulus", output "Mechanical Strength" (the standard node). 
2. **NO HALLUCINATIONS**: If the material is not specified, output "unknown".
3. **NO ASSUMPTIONS**: Do not assume "PLGA" just because "drug delivery" is mentioned.

The ontology fields and their ALLOWED VALUES (based on Knowledge Graph Nodes):

1. material_family
   # Knowledge Graph Nodes: PLA, PGA, PCL, PLGA_50_50, Polyanhydride
   # Instruction: Map broad terms to specific nodes if context implies, otherwise use strict extraction.
   Examples: 
   - PLA (for polylactic acid, PLLA, PDLLA)
   - PGA (for polyglycolic acid)
   - PCL (for polycaprolactone)
   - PLGA_50_50 (if user mentions PLGA roughly or 50:50 ratio)
   - Polyanhydride (for polysebacic acid, PSA)
   - Hydrogel
   - unknown

2. modification_type
   Examples: copolymer, surface modification, blending, crosslinking, none, unknown

3. target_property
   # Knowledge Graph Nodes Mapping (CRITICAL):
   # - "Mechanical Strength" <--- covers: tensile strength, modulus, stiffness, toughness, strong, weak
   # - "Degradation Rate"    <--- covers: degradation time, degradation speed, mass loss, erosion rate
   # - "Hydrolysis Rate"     <--- covers: bond cleavage, hydrolysis speed
   # - "Water Diffusion"     <--- covers: water permeability, diffusion coefficient
   # - "Water Uptake"        <--- covers: absorption, hydration
   # - "Swelling"            <--- covers: swelling ratio, expansion
   # - "Chain Mobility"      <--- covers: flexibility, rigid, movement
   # - "Tg" / "Tm"           <--- covers: glass transition, melting point
   # - "Hydrophilic" / "Hydrophobic" <--- covers: wettability, contact angle
   Examples: Mechanical Strength, Degradation Rate, Hydrolysis Rate, Water Diffusion, High Tg, unknown

4. degradation_environment
   Examples: Acidic Microenvironment, PBS, in vivo, enzymatic, unknown

5. mechanism_type
   # Knowledge Graph Nodes:
   # - "Surface Erosion"
   # - "Bulk Erosion"
   # - "Autocatalysis"
   # - "Enzymatic Attack"
   Examples: Surface Erosion, Bulk Erosion, Autocatalysis, unknown

6. method_type
   Examples: tensile testing, DSC, GPC, SEM, unknown

Return JSON STRICTLY in the form:

{
  "material_family": "...",
  "modification_type": "...",
  "target_property": "...",
  "degradation_environment": "...",
  "mechanism_type": "...",
  "method_type": "...",
  "reason": "Explain mapping. E.g., 'Mapped stiffness to Mechanical Strength'."
}

User query:
\"\"\"{query}\"\"\"
"""

def classify_domain(query: str) -> dict:
    # 替换占位符
    prompt = DOMAIN_PROMPT.replace("{query}", query)

    # 调用 LLM
    raw = call_deepseek_llm(prompt, system_prompt="You produce STRICT JSON. Map terms to KG nodes.")

    # ===== 解析 JSON =====
    try:
        text = raw.strip()
        # 提取第一个 { 到最后一个 } 之间的内容
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = text[start:end+1]
            data = json.loads(json_str)

            # 字段兜底
            expected_keys = [
                "material_family", "modification_type", "target_property", 
                "degradation_environment", "mechanism_type", "method_type", "reason"
            ]
            for key in expected_keys:
                data.setdefault(key, "unknown")
            
            return data
    except Exception as e:
        print(f"[DomainRouter Debug] JSON parse failed: {e}")
        print(f"[DomainRouter Debug] Raw LLM output: {raw}")

    # 失败兜底
    return {
        "material_family": "unknown",
        "modification_type": "unknown",
        "target_property": "unknown",
        "degradation_environment": "unknown",
        "mechanism_type": "unknown",
        "method_type": "unknown",
        "reason": "Fallback rule triggered.",
    }

if __name__ == "__main__":
    print("🤖 Domain Router Initialized (Aligned with Expert KG)")
    while True:
        q = input("\n请输入一个可降解聚合物相关问题 (输入 q 退出)：\n> ").strip()
        if q.lower() == 'q': break
        dom = classify_domain(q)
        print(json.dumps(dom, ensure_ascii=False, indent=2))

REPORT_CATEGORY_DISPLAY = {
    "mechanical_design": "力学性能设计",
    "thermal_design": "热学性能设计",
    "degradation_regulation": "降解行为与结构调控",
}


def coarsen_domain_to_report_hints(domain: dict | None) -> dict:
    """将 domain_router 的 ontology 结果粗映射到 design report 模板类别，供 writer 作为辅助信号使用。"""
    scores = {key: 0 for key in REPORT_CATEGORY_DISPLAY}
    if not isinstance(domain, dict):
        return {"scores": scores, "reason": "domain missing"}

    target_property = str(domain.get("target_property", "") or "").lower()
    modification_type = str(domain.get("modification_type", "") or "").lower()
    mechanism_type = str(domain.get("mechanism_type", "") or "").lower()
    method_type = str(domain.get("method_type", "") or "").lower()

    if any(tok in target_property for tok in ["mechanical", "strength", "modulus", "stiffness"]):
        scores["mechanical_design"] += 10
    if any(tok in target_property for tok in ["tg", "tm", "thermal", "glass transition", "high tg"]):
        scores["thermal_design"] += 10
    if any(tok in target_property for tok in ["degradation", "hydrolysis", "water diffusion", "water uptake"]):
        scores["degradation_regulation"] += 10
    
    # 💡 这里的 modification_type (如 copolymer/blend) 不再指向独立分类，
    # 而是可以给力学或降解加一点基础分，因为共混/共聚通常是为了改性这两者
    if any(tok in modification_type for tok in ["copolymer", "blend", "blending", "crosslink"]):
        scores["mechanical_design"] += 3
        scores["degradation_regulation"] += 3

    if any(tok in mechanism_type for tok in ["erosion", "autocatalysis", "hydrolysis"]):
        scores["degradation_regulation"] += 8
    if any(tok in method_type for tok in ["dsc", "dma"]):
        scores["thermal_design"] += 3
    if any(tok in method_type for tok in ["tensile", "compression", "mechanical"]):
        scores["mechanical_design"] += 3

    return {
        "scores": scores,
        "reason": "Coarse report-category hints inferred from target_property / modification_type / mechanism_type / method_type.",
    }
