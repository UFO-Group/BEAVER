import re

# Dictionary of PLA mechanical property keywords
keyword_dict = {
    "Polymers": [
        # General polymers and copolymers
        "polymer", "copolymer", "blend", "biopolymer",
        # Common commercial or synthetic polymers
        "PLA", "polylactic acid", "PCL", "polycaprolactone",
        "PET", "polyethylene terephthalate", "PMMA", "polymethyl methacrylate",
        "PU", "polyurethane", "PA", "polyamide", "nylon",
        "PVC", "polyvinyl chloride", "PVA", "polyvinyl alcohol",
        "PAN", "polyacrylonitrile", "PVP", "polyvinylpyrrolidone",
        "PDMS", "polydimethylsiloxane", "PC", "polycarbonate",
        "PBS", "polybutylene succinate", "PGA", "poly(γ-glutamic acid)",
        "PE", "polyethylene", "PP", "polypropylene", "polyester",
        # Bio-based/natural polymers
        "GelMA", "gelatin methacrylate", "gelatin", "collagen",
        "chitosan", "alginate", "sodium alginate", "cellulose",
        "nanocellulose", "pectin", "lignin", "starch", "hyaluronic acid",
        "silk fibroin", "polyethylene glycol", "polydopamine",
        "polyacrylamide"
    ],
    "Additives or Modifiers": [
        # Basic functional additives
        "additive", "modifier", "plasticizer", "compatibilizer", "filler", "blend",
        "hybrid", "nanocomposite", "composite",
        # Inorganic nanomaterials
        "nanoparticle", "nanofiller", "nanoclay", "TiO2", "SiO2", "ZnO", "CaCO3",
        "clay", "montmorillonite", "halloysite", "bentonite",
        # Organic/polymer blends
        "PBAT", "PEG", "PHA", "PBSA", "PPC", "EVA", "PLA-g-MA",
        # Bio-based/natural materials
        "cellulose nanocrystal", "microcrystalline cellulose", "hemicellulose",
        "soy protein", "wheat bran", "rice husk",
        # Carbon-based materials
        "CNT", "carbon nanotube", "carbon black", "graphene", "graphene oxide",
        "reduced graphene oxide",
        # Fibers and reinforcements
        "fiber", "natural fiber", "glass fiber", "bamboo fiber", "hemp fiber",
        "basalt fiber", "jute fiber", "kenaf fiber",
        # Plasticizers
        "glycerol", "triacetin", "citrate", "ATBC", "TEC", "tributyl citrate",
        "polyethylene glycol",
        # Blending and compatibilization
        "blending", "blended", "copolymerized", "reactive compatibilization",
        "immiscible", "miscible",
        # Other additives
        "antioxidant", "nucleating agent", "chain extender", "crosslinker",
        "UV stabilizer", "thermal stabilizer", "fire retardant", "flame retardant"
    ],
    "Tensile Strength": ["tensile strength", "breaking strength", "tensile properties"],
    "Elongation at Break": ["elongation at break", "breaking elongation"],
    "Young's Modulus": ["young's modulus", "tensile modulus"],
    "Flexural Modulus": ["flexural modulus", "bending modulus", "flexural stiffness"],
    "Impact Strength": ["impact strength", "impact toughness"],
    "Stress-Strain": ["stress-strain", "mechanical behavior"],
    "Hardness": ["hardness", "shore hardness", "rockwell", "durometer"],
    "Glass Transition": ["glass transition", "Tg"],
    "Melting Point": ["melting point", "melting temperature", "Tm"]
}

def contains_keywords(text, keyword_dict):

    matched_categories = []
    matched_keywords = {}

    for category, synonyms in keyword_dict.items():
        found = []
        for word in synonyms:
            pattern = r'\b' + re.escape(word) + r'\b'
            if re.search(pattern, text, flags=re.IGNORECASE):
                found.append(word)
        if found:
            matched_categories.append(category)
            matched_keywords[category] = found

    return bool(matched_categories), matched_categories, matched_keywords