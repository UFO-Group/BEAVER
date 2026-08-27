import os
from openai import OpenAI


def call_deepseek_llm(prompt):
    api_key = os.getenv("LLM_API_KEY")

    if not api_key:
        raise RuntimeError("未设置环境变量 LLM_API_KEY")

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv(
            "LLM_BASE_URL",
            "https://llmapi.paratera.com",
        ),
    )

    system_prompt = """
Extract all polymer-based material systems.

Output the structure of each system using the following JSON format:

{
  "Material Name": <string or null>,  // e.g., "PLA/PCL blend with TiO₂"

  "Base Polymer(s)": {
    "Polymer A": {
      "Name": <string>,   // exact name from text
      "Type": <string>,   // one of: "Homopolymer", "Block copolymer", "Random copolymer", 
                          // "Graft copolymer", "Crosslinked polymer", etc.
      "Role": <string>    // e.g., "Matrix", "Reinforcement", "Compatibilizer", "Plasticizer", "Modifier"
    },
    "Polymer B": {
      "Name": <string>,
      "Type": <string>,
      "Role": <string>
    },
    ...
  } or null,

  "Polymer Structure Type": <"Pure polymer" | "Blend" | "Block copolymer" | "Random copolymer" | 
                             "Graft copolymer" | "Crosslinked" | "Filled composite" | null>,

  "Copolymerized/Blended/Crosslinked/Filled With": <string or list of directly bonded or mixed components, or null>,

  "Other Material(Dopants, Additives or Modifiers)": {
    "Material A": {
      "Name": <string>,   // exact name from text
      "Type": <string>,   // one of: "Inorganic nanoparticle", "Carbon nanomaterial", "Plasticizer", 
                          // "Compatibilizer", "Natural fiber", "Functional additive", 
                          // "Chain extender", "Nucleating agent", etc.
      "Role": <string>    // e.g., "Reinforcement", "Plasticizer", "Compatibilizer", "Modifier", "Dopants"
    },
    ...
  } or null,

  "Component Ratio": {
    "Type": <"wt_percent" | "mass_ratio" | "molar_ratio" | "phr" | "range" | "qualitative" | "unknown">,
    "Components": {
      "<component name>": <number or [number, number]>
    },
    "Original": <string or null>
  },
  
  "Mechanical Properties": {
    "Tensile Strength": {
      "value": <number | [number, number] | null>,
      "unit": <string | null>,
      "description": <string | null>
    },
    "Elongation at Break": {
      "value": <number | [number, number] | null>,
      "unit": <"%" | string | null>,
      "description": <string | null>
    },
    "Young's Modulus": {
      "value": <number | [number, number] | null>,
      "unit": <string | null>,
      "description": <string | null>
    },
    "Flexural Modulus": {
      "value": <number | [number, number] | null>,
      "unit": <string | null>,
      "description": <string | null>
    },
    "Impact Strength": {
      "value": <number | [number, number] | null>,
      "unit": <string | null>,
      "description": <string | null>
    },
    "Stress-Strain": {
      "value": <string | null>,
      "unit": null,
      "description": <string | null>  // e.g., "yielded at ~3% strain with necking"
    },
    "Hardness": {
      "value": <number | [number, number] | null>,
      "unit": <string | null>,
      "description": <string | null>
    }
  },
  
  "Glass Transition Temperature": {
    "value": <number | [number, number] | null>,
    "unit": <string | null>,                      // e.g., "°C"
    "description": <string | null>
  },

  "Melting Temperature": {
    "value": <number | [number, number] | null>,
    "unit": <string | null>,                      // e.g., "°C"
    "description": <string | null>
  }
}

Your task includes:

1. **Polymer Structure Type Identification**:
   - Classify based on phrasing in the text:
     - "PLA-b-PCL" → "Block copolymer"
     - "PLA-co-PBS" → "Random copolymer"
     - "PLA/PCL blend" → "Blend"
     - "PLA-g-PEG" → "Graft copolymer"
     - "crosslinked PLA" → "Crosslinked"
     - "PLA with nanoclay" → "Filled composite"
   - Place components like PCL, PEG, TiO₂ under `"Copolymerized/Blended/Crosslinked/Filled With"`.

2. **Base Polymer(s) Identification**:
   - Place all core polymers into `"Base Polymer(s)"` as `"Polymer A"`, `"Polymer B"` etc.
   - For each, extract:
     - `"Name"`: exact name (e.g., "PLA", "PCL")
     - `"Type"`: one of "Homopolymer", "Block copolymer", "Crosslinked polymer", etc.
     - `"Role"`: one of "Matrix", "Reinforcement", "Compatibilizer", "Modifier", etc.

3. **Other Materials Extraction**:
   - Non-polymeric or additive components (e.g., TiO₂, Joncryl, CNT) go under `"Other Material"`.
   - Each should include:
     - `"Name"`: literal string
     - `"Type"`: see controlled vocabulary above
     - `"Role"`: see controlled vocabulary above
   - If no such components, return `null`.

4. **Component Ratio Extraction**:
   - Extract any numeric or qualitative description of composition ratio.
   - Examples:
     - `"PLA:PCL = 70:30"` → `"mass_ratio"`
     - `"20 wt% PEG"` → `"wt_percent"`
     - `"PEG content ranged from 10% to 30%"` → `"range"`

5. **Mechanical Properties Extraction**:  
   In addition to standard metrics, extract any mention of the following properties wherever they appear in the text:
   - Young’s modulus, modulus of elasticity, elastic modulus, tensile modulus, mechanical modulus  
   - Tensile strength, compressive strength, compression strength, breaking strength, tensile properties
   - Elongation at break, breaking elongation
   - Young's Modulus, tensile modulus
   - Flexural Modulus, bending modulus, flexural stiffness
   - Impact Strength, impact toughness
   - Stress-Strain, mechanical behavior
   - Hardness, shore hardness, rockwell, durometer
    
   Map each to the closest JSON fields (e.g., Young’s modulus → `"Young's Modulus"`, tensile strength → `"Tensile Strength"`, etc.). If a property does not have a predefined field, include it under `"Other Mechanical Info"` with its name and value.

6. **Thermal Properties**:
   - Extract `"Glass Transition Temperature"` and `"Melting Temperature"` as before.

Output rules:
- If only one material system is discussed, return **a single JSON object**.
- If multiple distinct Material systems are discussed, return **a JSON list** of multiple such objects.
- If no valid material or property is found, return the string **"NONE"**.
- Do **not** add any extra text or formatting outside the JSON.
- For missing fields, use **null**.
"""

    try:
        # Send request and get response
        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "DeepSeek-R1"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.05,
            max_tokens=4960,
        )
        
        # Return API response content
        return response.choices[0].message.content

    except Exception as e:
        # Error handling
        print("🚨 Exception occurred:", str(e))
        return None