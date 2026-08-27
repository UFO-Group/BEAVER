# Design and Verification of an Aromatic-Aliphatic Copolyester for Enhanced Thermal Stability with Preserved Biodegradation Kinetics

## Abstract  
This study proposes an aromatic-aliphatic copolyester system designed to achieve improved heat resistance (targeting elevated glass transition temperature, *T*_g) without significantly suppressing degradation kinetics. The hypothesis posits that controlled incorporation of aromatic units into an aliphatic polyester backbone enhances chain rigidity while maintaining enzymatic/hydrolytic degradation pathways through limited aromatic segment length (≤2 units) [1]. Literature evidence confirms aromatic segments elevate *T*_g (e.g., +50–80°C vs. aliphatic analogs [2]), but degradation kinetics remain poorly quantified for designed microstructures. The experimental route involves synthesizing butylene glycol-based copolymers with systematic terephthalate/adipate variations, characterizing thermal transitions (DSC), and monitoring hydrolytic/enzymatic degradation. Key risks include trade-offs between aromatic content and degradation rates, phase separation, and crystallization interference. Preliminary evidence supports feasibility, but degradation validation is critical.

## Graphical Repeat-Unit Representation

![Representative repeat-unit schematic](D:/FXR/Agent/1219/2/Run_20260506_111935/Design_a_biodegradable_polyester_with_improved_hea/idea1_Aromatic-Aliphatic_Copolymer/figures/idea1_Aromatic-Aliphatic_Copolymer_repeat_unit.png)

**Figure.** Representative polymer-structure schematic of poly(butylene adipate-co-terephthalate) with variable terephthalate (T) to adipate (A) ratio.

Structure confidence: high. Architecture: random copolymer. Composition note: T:A TBD. Dictionary/RDKit validation: 1/1 mapped structure(s) validated.


## 1. Introduction  
### 1.1 Scientific Background  
Biodegradable polyesters like poly(butylene succinate) (PBS, *T*_g ≈ −36°C [3, 4]) and poly(butylene adipate-co-terephthalate) (PBAT, *T*_g = −30–0°C [5]) face limitations in heat-resistant applications due to low glass transition temperatures. Aliphatic polyesters typically exhibit faster degradation but poor thermal stability, while aromatic segments improve rigidity at the cost of biodegradability [1, 6].  

### 1.2 Design Hypothesis  
We hypothesize that an aromatic-aliphatic copolyester with constrained aromatic segment lengths (≤2 units) and dominant aliphatic segments (≥2 units) will significantly increase *T*_g without drastically impeding degradation kinetics. This leverages the rigidity of aromatic moieties for thermal stability while preserving enzymatic cleavage sites in aliphatic regions [1, 7].  

### 1.3 Scope of This Study  
Direct evidence supports *T*_g enhancement via aromatic incorporation [2, 5, 7], and theoretical biodegradability of short aromatic sequences [1]. However, degradation kinetics data for precisely engineered copolymers are absent, and optimal aromatic/aliphatic ratios remain unquantified.  

## 2. Mechanistic Rationale and Design Hypothesis  
### 2.1 Mechanistic Basis  
Aromatic units (e.g., terephthalate) restrict chain mobility through steric hindrance and π-orbital interactions, elevating *T*_g [2, 7]. For example, cyclohexanedicarboxylate units raise *T*_g by 50–80°C versus adipate analogs [2]. Degradation occurs primarily via hydrolysis/enzymatic cleavage of aliphatic ester bonds; aromatic esters resist scission but remain accessible if segment lengths are short (≤2 units) [1].  

### 2.2 Evidence-Supported Design Rules  
- **Thermal properties:** *T*_g increases linearly with aromatic content [2, 7]. Melting temperature (*T*_m) may rise but risks thermal degradation during processing.  
- **Degradation behavior:** Aliphatic segment length ≥2 units maintains degradation pathways [1]. Higher crystallinity may slow degradation.  
- **Biocompatibility:** Insufficient evidence; dependent on monomer selection and byproducts.  

### 2.3 Trade-Offs and Constraints  
A trade-off exists between *T*_g elevation and degradation rate suppression. For instance, poly(propylene terephthalate)-diol (*T*_g = 69.1°C) degrades slower than PBS-diol (*T*_g = −42.7°C) [7]. Crystallinity may increase with symmetry but could hinder degradation [3].  

## 3. Materials and Methods  
### 3.1 Material System and Variable Definition  
- **Base system:** Butylene glycol (1,4-butanediol) + adipic acid (aliphatic) + terephthalic acid (aromatic).  
- **Primary variable:** Molar ratio of terephthalate (T) to adipate (A) units (T:A = 0:100 to 60:40).  
- **Fixed:** Diol excess (20 mol%), catalyst (titanium tetrabutoxide).  
- **Unknowns:** Optimal polymerization time/temperature, molecular weight target.  

### 3.2 Formulation / Sample Matrix Design  
| Sample | T:A (mol:mol) | Target *T*_g (°C) | Degradation Screening |  
|--------|---------------|----------------------------|------------------------|  
| PBS    | 0:100         | Baseline (−36°C [3])       | Control                |  
| C20    | 20:80         | >−10°C                     | Phase 1                |  
| C40    | 40:60         | >10°C                      | Phase 1                |  
| C60    | 60:40         | >30°C                      | Phase 2                |  
| PBAT*  | ∼50:50†       | −30–0°C [5]                | Reference              |  
*Commercial PBAT reference; †exact ratio varies.  

### 3.3 Sample Preparation / Fabrication Procedure  
1. **Esterification:** React diacids (adipic + terephthalic) with excess 1,4-butanediol (1:1.2 diacid:diol) at 180–220°C under N_2 until acid value <10 mg KOH/g.  
2. **Polycondensation:** Reduce pressure to <1 mbar, raise temperature to 240–250°C, add catalyst (0.1 wt%), and react until target melt viscosity.  
3. **Processing:** Compression-mold films (0.5 mm thickness) at *T*_m + 20°C. Annealing conditions TBD.  

### 3.4 Structural and Physicochemical Characterization  
- **Molecular weight:** GPC (THF, 35°C).  
- **Composition:** ^1H-NMR (CDCl_3) to confirm T:A ratio.  
- **Thermal transitions:** DSC (N_2, 10°C/min; 2 heat/cool cycles).  
- **Crystallinity:** XRD and DSC crystallinity (%*X*_c).  
- **Morphology:** SEM/AFM for phase separation.  

### 3.5 Mechanical Testing Protocol  
- **Tensile properties:** ASTM D638 (dry, 23°C; 5 specimens/group).  
- **Priority metrics:** Young’s modulus (stiffness), elongation at break (toughness).  
- **Comparison:** PBS vs. C20–C60 at equivalent crystallinity.  

### 3.6 Degradation Evaluation Protocol  
- **Hydrolytic degradation:** Phosphate buffer (pH 7.4, 37°C); measure mass loss, M_n change (GPC), and surface erosion (SEM) at 4, 8, 12 weeks.  
- **Enzymatic degradation:** *Pseudomonas* lipase (1 mg/mL, pH 7.0); monitor mass loss at 37°C.  
- **Degradation kinetics:** Fit data to first-order model; compare rate constants (*k*).  

### 3.7 Biocompatibility / Biofunction Evaluation  
- **Cytotoxicity:** ISO 10993-5 elution assay (L929 fibroblasts).  
- **Scope:** Preliminary screening only; insufficient evidence for in-depth analysis.  

### 3.8 Controls, Decision Criteria, and Statistical Comparison  
- **Controls:** PBS (0% T), commercial PBAT.  
- **Success criteria:** *T*_g ≥ 0°C (C20–C60) with enzymatic degradation rate ≥50% of PBS.  
- **Statistics:** One-way ANOVA (α=0.05) with Tukey post-hoc; n=3 per group.  

## 4. Results and Evidence-Based Discussion  
### 4.1 Directly Supported Expectations  
- *T*_g elevation with aromatic content is well-established: PPT-diol (*T*_g = 69.1°C) vs. PBS-diol (−42.7°C) [7].  
- PBAT demonstrates feasibility of combining aromatic/aliphatic units [5].  

### 4.2 Mechanistically Inferred Expectations  
- Degradation rates should decrease with higher T content but remain measurable if aromatic blocks are short (≤2 units) [1].  
- Crystallinity may increase with symmetric terephthalate units, potentially slowing degradation [3].  

### 4.3 Evidence Gaps and Uncertainty  
- **Critical gap:** No quantitative degradation kinetics for copolymers with controlled block lengths.  
- **Uncertainty:** Phase separation risk at T:A >40:60; effect on mechanical properties unknown.  
- **Missing data:** Enzymatic degradation rates for terephthalate-containing systems.  

### 4.4 Comparison with Prior Systems  
Compared to PBS (*T*_g ≈ −36°C [3]), C40 is expected to achieve *T*_g >10°C [7]—superior to PBAT (*T*_g ≤0°C [5]). However, degradation rates for designed copolymers may exceed PBAT if aromatic sequences are shorter [1].  

## 5. Risk Analysis and Optimization Path  
### 5.1 Major Failure Modes  
1. **Degradation too slow:** Excessive T content or long aromatic blocks.  
2. **Phase separation:** Poor compatibility between aromatic/aliphatic segments.  
3. **Low *T*_g:** Inadequate aromatic incorporation or low molecular weight.  

### 5.2 Optimization Pathway  
- **Degradation too slow:** Reduce T content; increase aliphatic segment length; add hydrolytically labile comonomers (e.g., diglycolate [8]).  
- **Degradation too fast:** Increase T content or crystallinity; optimize annealing.  
- **Stiffness too low:** Raise T ratio; consider rigid diols (e.g., isosorbide [9]).  
- **Toughness too low:** Adjust aliphatic segment length; add impact modifiers.  
- **Biocompatibility issues:** Purify to remove catalysts/monomers; end-cap with biocompatible groups.  

## 6. Conclusion  
The aromatic-aliphatic copolyester design is promising for enhancing heat resistance (*T*_g >0°C) while preserving biodegradability, supported by mechanistic evidence but requiring degradation kinetics validation. Synthesis feasibility is high, yet phase separation and degradation trade-offs pose risks. Immediate experimental focus should synthesize C20–C60 series and quantify hydrolytic/enzymatic degradation. The critical next step is establishing the maximum aromatic content sustaining practical degradation rates.  

## 8. References  
- [1] 101016_jpolymer2025128488
- [2] 101002_macp201100052
- [3] 101016_jpolymdegradstab201812031
- [4] 101016_jpolymer2023125711
- [5] 101016_jpolymer2023125685
- [6] 101002_app33935
- [7] 101002_pi3000
- [8] 101002_app44186
- [9] 101002_macp201200612
