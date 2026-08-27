# Design of Aromatic-Aliphatic Copolyesters for Enhanced Glass Transition Temperature with Preserved Toughness and Controlled Degradation

## Abstract

This study proposes an aromatic-aliphatic copolyester system designed to simultaneously increase glass transition temperature (T₉), preserve toughness, and maintain clinically relevant degradation rates. The core hypothesis posits that incorporating aromatic segments ≤2 units and aliphatic segments ≥2 units achieves synergistic thermal-mechanical-degradation balance by leveraging aromatic rigidity for T₉ elevation while retaining enzymatic hydrolysis sites. Literature evidence indicates feasibility through analogous systems like PBTGA and PBST copolymers [1, 2, 3], though direct validation of the specific microstructure is limited. The experimental plan involves synthesizing terephthalate/succinate copolymers with systematic aromatic-aliphatic ratio variations (30-60 mol% aromatic), followed by comprehensive thermal, mechanical, and degradation characterization. Key risks include potential brittleness at high aromatic content and degradation rate suppression beyond 55 mol% aromatic units [4]. Conservative optimization of composition windows is recommended to navigate trade-offs.

## Graphical Repeat-Unit Representation

![Representative repeat-unit schematic](E:/MultiAgent/额外文件/Design-0506/力学+热学+降解/idea2_Aromatic-Aliphatic_Copolyester/repeat_unit_test/figures/docx_test_idea2_Aromatic-Aliphatic_Copolyester_repeat_unit.png)

**Figure.** Representative polymer-structure schematic of poly(butylene terephthalate-co-succinate) (PBTS) with variable aromatic-aliphatic composition.

Structure confidence: high. Architecture: random copolymer. Composition note: terephthalate:succinate ratio TBD. Dictionary/RDKit validation: 1/1 mapped structure(s) validated.

## 1. Introduction

### 1.1 Scientific Background

Biodegradable polymers face inherent trade-offs between thermal stability, mechanical integrity, and degradation kinetics. Aliphatic polyesters (e.g., PCL, PLA) offer tunable degradation but suffer from low T₉ (<60°C) and insufficient stiffness for many clinical applications. Incorporating aromatic units enhances thermal-mechanical properties but typically suppresses biodegradability through reduced chain mobility and crystallinity alterations [5, 6, 7].

### 1.2 Design Hypothesis

We hypothesize that an aromatic-aliphatic copolyester with constrained aromatic segment lengths (≤2 units) and sufficient aliphatic spacers (≥2 units) will simultaneously:

1) Increase T₉ via aromatic ring rigidity [6, 7]

2) Preserve toughness through aliphatic segment flexibility [2, 3]

3) Maintain enzymatic degradation pathways via accessible aliphatic ester bonds [5, 8]

This approach is scientifically significant for enabling high-temperature applications (e.g., sterilization-resistant implants) without compromising degradation or mechanical resilience.

### 1.3 Scope of This Study

Direct evidence supports individual mechanisms (T₉ elevation by aromatics [6, 7], degradation dependence on segment length [4, 5]), but no literature validates the combined performance triad. Degradation environment specificity (physiological vs. compost) and long-term mechanical retention data are currently missing.

## 2. Mechanistic Rationale and Design Hypothesis

### 2.1 Mechanistic Basis

- **T₉ elevation**: Aromatic terephthalate units restrict chain mobility, raising T₉ proportionally to aromatic content [6, 7]. PBT-based copolymers achieve T₉ >50°C at >40 mol% aromatic [1, 7].

- **Toughness preservation**: Aliphatic segments (e.g., butylene succinate) provide chain flexibility, enabling >200% elongation at break in balanced compositions [2, 3].

- **Degradation control**: Enzymatic hydrolysis occurs preferentially at aliphatic esters. Degradation rate decreases exponentially when aromatic segment length exceeds 3 units or aromatic content >55 mol% [4, 5]. Crystallinity reduction accelerates degradation by enhancing water ingress [8, 9].

### 2.2 Evidence-Supported Design Rules

| Property | Key Variables | Evidence Strength |

|-------------------|------------------------------------------------------------------------------|-------------------|

| T₉ | Aromatic content (↑T₉), aliphatic spacer length (↓T₉) | Strong [6, 7] |

| Toughness | Aliphatic content (↑elongation), aromatic block length (↓toughness) | Moderate [2, 3] |

| Degradation rate | Aliphatic segment length (↑degradation), crystallinity (↓degradation) | Strong [4, 5, 8] |

| Biocompatibility | No direct evidence; dependent on monomer selection and byproducts | Insufficient |

### 2.3 Trade-Offs and Constraints

- **T₉ vs. degradation**: Aromatic content >55 mol% risks unacceptable degradation slowdown [4]

- **Toughness vs. T₉**: High aromatic content (>60 mol%) may reduce elongation below 100% [1, 2]

- **Crystallinity effects**: Increased crystallinity from aromatic segments slows degradation but enhances stiffness [8, 9]

## 3. Materials and Methods

### 3.1 Material System and Variable Definition

- **Base system**: Poly(butylene terephthalate-co-succinate) (PBTS)

- **Primary variables**:

- Aromatic content (terephthalate): 30, 40, 50, 60 mol%

- Aliphatic segment length: Fixed at C4 (succinate) initially

- **Fixed parameters**:

- Diol: 1,4-butanediol (BDO)

- Catalyst: Titanium butoxide (0.4 mol%) [3]

- **Unknowns**:

- Optimal polymerization temperature (TBD: 240-260°C screening)

- Molecular weight target (TBD: inherent viscosity >0.8 dL/g)

### 3.2 Formulation / Sample Matrix Design

| Group | Terephthalate (mol%) | Succinate (mol%) | Primary Comparison Target |

|-------|----------------------|------------------|--------------------------------|

| A | 30 | 70 | Degradation control |

| B | 40 | 60 | Target balance |

| C | 50 | 50 | T₉ elevation |

| D | 60 | 40 | High-T₉ risk assessment |

| Control 1 | 100 (PBT) | 0 | Non-degradable reference |

| Control 2 | 0 (PBS) | 100 | Degradation reference |

### 3.3 Sample Preparation / Fabrication Procedure

## 1. **Melt polycondensation**:

- Charge terephthalic acid, succinic acid, and 1,4-butanediol (BDO/acid = 1.3:1 mol/mol [3])

- Catalyst: Titanium butoxide (0.4 mol%)

- Stage 1: 180°C for 2h under N₂

- Stage 2: 250°C for 3h under vacuum (<5 mbar)

## 2. **Processing**:

- Compression mold at 10°C above Tm (DSC-determined)

- Quench-cool to suppress crystallization for degradation testing

## 3. **Post-processing**:

- Anneal subgroup at 90°C for crystallinity control

### 3.4 Structural and Physicochemical Characterization

- **Molecular weight**: GPC (triplicate)

- **Thermal properties**: DSC (T₉, Tₘ, crystallinity), TGA (degradation onset)

- **Crystallinity**: XRD, DSC cold crystallization

- **Morphology**: SEM pre/post-degradation

*Rationale: Crystallinity quantification essential for degradation-mechanics coupling [8, 9]*

### 3.5 Mechanical Testing Protocol

- **Tensile properties** (ASTM D638):

- Dry state: Tensile strength, elongation at break, Young's modulus

- Wet state: After 7d PBS immersion (37°C)

- **Toughness assessment**: Area under stress-strain curve

- **Groups**: All compositions + annealed vs. quenched (n=5)

- **Priority metrics**: Elongation at break >150% (toughness proxy)

### 3.6 Degradation Evaluation Protocol

- **Conditions**:

- Enzymatic: Lipase/PBS (pH 7.4, 37°C) [8]

- Hydrolytic: PBS (pH 7.4, 37°C)

- **Monitoring**:

- Mass loss (%) weekly (8 weeks)

- Molecular weight drop (GPC every 2 weeks)

- pH tracking (byproduct accumulation)

- SEM surface morphology evolution

- **Mechanical coupling**: Tensile tests at 4/8 weeks

### 3.7 Biocompatibility / Biofunction Evaluation

- **Preliminary screening**:

- Cytotoxicity (ISO 10993-5): Fibroblast viability (72h extract)

- Hemolysis (ASTM F756)

- **Scope note**: Full biocompatibility assessment beyond initial screening is TBD

### 3.8 Controls, Decision Criteria, and Statistical Comparison

- **Positive controls**: PBS (rapid degradation), PCL (moderate degradation)

- **Failure thresholds**:

- Degradation: <5% mass loss at 8 weeks (enzymatic)

- Toughness: <100% elongation at break

- **Statistical design**: Two-way ANOVA (composition × annealing) with Tukey post-hoc (p<0.05)

## 4. Results and Evidence-Based Discussion

### 4.1 Directly Supported Expectations

- T₉ will increase linearly with terephthalate content (ΔT₉ ≈ 0.8°C/mol% [7])

- Degradation rates will decrease significantly at >50 mol% aromatic content [4]

- Elongation at break >200% achievable at ≤50 mol% aromatic [2, 3]

### 4.2 Mechanistically Inferred Expectations

- **Trade-off inversion point**: Optimal balance expected at 40-50 mol% terephthalate

- **Crystallinity effects**: Annealing will reduce degradation rate by 30-50% versus quenched samples [8]

- **Degradation heterogeneity**: Surface erosion dominant at low crystallinity; bulk erosion possible at high aromatic content

### 4.3 Evidence Gaps and Uncertainty

- **Critical gaps**:

- Degradation kinetics in physiological conditions (current evidence: compost/enzymatic [4, 8])

- Long-term mechanical retention during degradation

- pH drop magnitude from acidic byproducts

- **Uncertainties**:

- Impact of monomer sequence (random vs. block) on degradation

- Minimum aliphatic segment length for enzymatic recognition

### 4.4 Comparison with Prior Systems

- **PBTGA copolymers** [1]: Achieved T₉~50°C and 46 MPa strength but degradation not quantified

- **PBST systems** [2]: High toughness (196-480% elongation) but low T₉ (<0°C)

- **Key advance**: Our design explicitly constrains segment lengths (aromatic ≤2 units) for degradation preservation [5]

## 5. Risk Analysis and Optimization Path

### 5.1 Major Failure Modes

## 1. **Degradation too slow** (>55 mol% aromatic): Suppresses enzymatic cleavage [4]

## 2. **Brittleness** (high T₉): Aromatic stacking reduces chain mobility

## 3. **Late-stage failure**: Accelerated mechanical loss during degradation

### 5.2 Optimization Pathway

- **Degradation too slow**:

- Reduce aromatic content to 30-40 mol%

- Incorporate shorter aliphatic diacids (e.g., oxalate)

- Increase amorphous phase via copolymer branching

- **Toughness too low**:

- Introduce longer aliphatic spacers (C6-C8)

- Plasticizer screening (citrate esters)

- **T₉ insufficient**:

- Increase aromatic content to 50-55 mol%

- Substitute terephthalate with rigid bioaromatics (vanillate [7])

- **Wet-state failure**: Hydrophilicity modification via PEG segments

## 6. Conclusion

The aromatic-aliphatic copolyester design demonstrates strong mechanistic plausibility for balancing T₉ elevation, toughness retention, and degradation control. Evidence supports feasibility through segment-length engineering [4, 5], but experimental validation is crucial given trade-off sensitivities. The highest-risk gap remains degradation kinetics in physiological environments. Immediate next steps should prioritize synthesizing 40-50 mol% terephthalate compositions with strict segment-length control, coupled with enzymatic degradation screening. Conservative optimization around this window offers the most promising path toward clinically viable materials.

## 8. References

- [1] 101016_jeurpolymj2022111613

- [2] 101002_app54939

- [3] 101007_s10965-020-02096-3

- [4] 101002_app55915

- [5] 101016_jpolymer2025128488

- [6] 101016_jpolymer201702054

- [7] 101016_jeurpolymj2019109296

- [8] 101007_s10965-017-1318-0

- [9] 101002_app57210
