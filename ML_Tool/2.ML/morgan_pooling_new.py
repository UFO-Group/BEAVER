# -*- coding: utf-8 -*-

import argparse
import sys

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors, Descriptors

# ================== 聚合物基团指纹：SMARTS 定义 ==================
# 只对“聚合物块”（带 * 的 SMILES）统计这些结构片段的出现次数
POLY_FG_SMARTS = {
    # 你可以按需增删/调整
    "fg_carbonyl_C=O": "[CX3]=[OX1]",                       # 所有 C=O（酰胺/酯/酸）
    "fg_amide": "C(=O)N",                                  # 酰胺
    "fg_ester": "C(=O)O",                                  # 酯
    "fg_ether": "[OD2](-[#6])-[#6]",                       # 醚 C-O-C
    "fg_amine": "[NX3;H2,H1,H0;!$(NC=O)]",                 # 非酰胺 N
    "fg_urea_imide_like": "N-C(=O)-N",                     # 尿素/酰亚胺类 N-C(=O)-N
    "fg_sulfonyl": "S(=O)(=O)",                            # 磺酰
    "fg_aromatic_ring": "a",                               # 芳环原子
    "fg_aliphatic_ring": "[R;!a]",                         # 非芳环原子
    "fg_hetero_in_ring": "[R;!#6]",                        # 环上杂原子
}

POLY_FG_PATTERNS = {
    name: Chem.MolFromSmarts(sma) for name, sma in POLY_FG_SMARTS.items()
}

# ================== 聚合物物化特征（Structure–Aggregation） ==================
# 基础物化特征（你原来那 10 个）
POLY_PHYS_BASE_NAMES = [
    "poly_phys_MolWt",              # 分子量（repeat 单元）
    "poly_phys_MolLogP",            # 疏水性 logP
    "poly_phys_MolMR",              # 折射率相关，近似 polarizability
    "poly_phys_TPSA",               # 极性表面积
    "poly_phys_NumHBD",             # H-bond donor
    "poly_phys_NumHBA",             # H-bond acceptor
    "poly_phys_NumAromaticRings",   # 芳环数 → 刚性/芳香度
    "poly_phys_NumAliphaticRings",  # 脂环数
    "poly_phys_FractionCSP3",       # sp3 碳比例 → 柔性/三维度
    "poly_phys_LabuteASA",          # 近似可达表面积 ASA
]

# 扩展：更偏力学/热学的特征
POLY_PHYS_EXTRA_NAMES = [
    "poly_phys_NumRotatableBonds",  # 链柔性，可旋转键
    "poly_phys_HeavyAtomCount",     # 重原子数，近似体积/密度
    "poly_phys_RingCount",          # 总环数，整体刚性
]

# 全部物化特征名（基础 + 扩展）
POLY_PHYS_ALL_NAMES = POLY_PHYS_BASE_NAMES + POLY_PHYS_EXTRA_NAMES


def parse_args():
    parser = argparse.ArgumentParser(description="构建聚合物块 + 其他块的 Morgan 指纹特征")

    parser.add_argument("--in_csv", type=str, required=True, help="输入 CSV 路径")
    parser.add_argument("--out_csv", type=str, required=True, help="输出 CSV 路径")

    parser.add_argument(
        "--polymer_cols",
        nargs="+",
        required=True,
        help="包含 SMILES 的列名列表（其中含 '*' 的视为聚合物）",
    )

    parser.add_argument(
        "--target_cols",
        nargs="+",
        required=True,
        help="需要保留到输出中的目标列名（例如性质列）",
    )

    parser.add_argument("--radius", type=int, default=2, help="Morgan 指纹半径")
    parser.add_argument("--nbits", type=int, default=1024, help="Morgan 指纹长度")

    # 兼容旧参数，不使用
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="兼容旧命令参数。在当前 block 拼接模式下不使用。",
    )

    # 可选指定编码，不指定就自动尝试常见编码
    parser.add_argument(
        "--encoding",
        type=str,
        default=None,
        help="CSV 编码（如 utf-8-sig, gb18030 等）。不指定则自动尝试常见编码。",
    )

    # 特征模式开关（仅在使用 Morgan 时有效）
    #  - 2048    : 只输出 poly_fp + add_fp
    #  - 2055    : poly_fp + add_fp + 7 个 scalar
    #  - extended: 在 2055 基础上，再加 聚合物基团指纹 + 物化特征
    parser.add_argument(
        "--feature_mode",
        type=str,
        choices=["2048", "2055", "extended"],
        default="extended",
        help="特征输出模式：2048 / 2055 / extended（默认 extended）",
    )

    # 指纹降维开关：只对 poly_fp + add_fp 做降维
    parser.add_argument(
        "--dr_method",
        type=str,
        choices=["none", "pca", "umap"],
        default="none",
        help="指纹降维方法：none / pca / umap（默认 none）",
    )

    parser.add_argument(
        "--dr_dim",
        type=int,
        default=None,
        help="降维后指纹维度；未指定则默认 128 维（若大于原始维度会自动截断）。",
    )

    # ✅ 新增：不要 Morgan 指纹，只用工程化特征（scalar + poly_fg + poly_phys）
    parser.add_argument(
        "--no_morgan",
        action="store_true",
        help="不输出 Morgan 指纹（poly_fp/add_fp），只使用工程化特征。",
    )

    # ✅ 新增：开启扩展物化特征（基础 10 + 额外 3 个：NumRotatableBonds 等）
    parser.add_argument(
        "--use_extended_phys",
        action="store_true",
        help="使用扩展的聚合物物化特征（基础 10 + NumRotatableBonds 等 3 个）。",
    )

    return parser.parse_args()


def smiles_to_mol(smiles: str):
    """从 SMILES 构建 RDKit Mol，若失败返回 None。"""
    if smiles is None:
        return None
    s = str(smiles).strip()
    if s == "" or s.lower() in ("nan", "none"):
        return None
    mol = Chem.MolFromSmiles(s)
    return mol


def mol_to_morgan_fp(mol, radius: int, nbits: int) -> np.ndarray:
    """将 RDKit Mol 转为 Morgan bit 指纹（0/1 向量，float 类型方便后续平均）。"""
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nbits)
    arr = np.zeros((nbits,), dtype=float)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def detect_inorganic_and_metal(mol) -> (bool, bool):
    """
    粗略判断：
    - has_inorganic: 分子中没有碳原子（num_C == 0）
    - has_metal:     是否存在原子序号 > 20 的元素
    """
    if mol is None:
        return False, False

    atoms = list(mol.GetAtoms())
    num_C = sum(1 for a in atoms if a.GetAtomicNum() == 6)
    has_inorganic = (num_C == 0)

    # 原子序号 > 20 （从 Ca 开始）粗略视为金属
    has_metal = any(a.GetAtomicNum() > 20 for a in atoms)

    return has_inorganic, has_metal


def compute_poly_physchem(poly_mols):
    """
    对一行中的所有聚合物 mol 计算物化特征，并按 '每个聚合物单元平均值' 聚合。

    返回顺序对应 POLY_PHYS_ALL_NAMES（基础 10 + 扩展 3），
    后面可以根据 args.use_extended_phys 决定用前 10 还是全部 13。
    """
    n = len(poly_mols)
    if n == 0:
        return [0.0] * len(POLY_PHYS_ALL_NAMES)

    # 累加再平均
    sums = np.zeros(len(POLY_PHYS_ALL_NAMES), dtype=float)

    for m in poly_mols:
        try:
            mw = Descriptors.MolWt(m)
        except Exception:
            mw = 0.0
        try:
            logp = Descriptors.MolLogP(m)
        except Exception:
            logp = 0.0
        try:
            mr = Descriptors.MolMR(m)  # 近似 polarizability
        except Exception:
            mr = 0.0
        try:
            tpsa = rdMolDescriptors.CalcTPSA(m)
        except Exception:
            tpsa = 0.0
        try:
            hbd = rdMolDescriptors.CalcNumHBD(m)
        except Exception:
            hbd = 0.0
        try:
            hba = rdMolDescriptors.CalcNumHBA(m)
        except Exception:
            hba = 0.0
        try:
            nar = rdMolDescriptors.CalcNumAromaticRings(m)
        except Exception:
            nar = 0.0
        try:
            nal = rdMolDescriptors.CalcNumAliphaticRings(m)
        except Exception:
            nal = 0.0
        try:
            fcsp3 = rdMolDescriptors.CalcFractionCSP3(m)
        except Exception:
            fcsp3 = 0.0
        try:
            asa, _ = rdMolDescriptors.CalcLabuteASA(m)
        except Exception:
            asa = 0.0

        # 扩展特征
        try:
            n_rot = rdMolDescriptors.CalcNumRotatableBonds(m)
        except Exception:
            n_rot = 0.0
        try:
            heavy_atoms = Descriptors.HeavyAtomCount(m)
        except Exception:
            heavy_atoms = 0.0
        try:
            ring_count = rdMolDescriptors.CalcNumRings(m)
        except Exception:
            ring_count = 0.0

        vals = [
            mw, logp, mr, tpsa, hbd, hba, nar, nal, fcsp3, asa,
            n_rot, heavy_atoms, ring_count
        ]
        sums += np.array(vals, dtype=float)

    return (sums / float(n)).tolist()


def main():
    args = parse_args()

    if args.alpha is not None:
        print(
            f"[INFO] 提示：检测到 --alpha={args.alpha}，"
            f"但当前使用的是“聚合物块 + 其他块拼接”方案，alpha 将被忽略。",
            file=sys.stderr,
        )

    print(f"[INFO] 读取数据: {args.in_csv}")

    # 编码处理
    if args.encoding is not None:
        print(f"[INFO] 使用指定编码读取 CSV: {args.encoding}")
        df = pd.read_csv(args.in_csv, encoding=args.encoding)
    else:
        tried = []
        for enc in ["utf-8-sig", "gb18030", "latin1"]:
            try:
                df = pd.read_csv(args.in_csv, encoding=enc)
                print(f"[INFO] 成功读取文件，编码方式: {enc}")
                break
            except Exception as e:
                tried.append((enc, str(e)))
        else:
            msg = "[ERROR] 无法读取 CSV 文件，请检查文件编码！尝试过的编码：\n"
            for enc, err in tried:
                msg += f"  - {enc}: {err}\n"
            raise RuntimeError(msg)

    for col in args.polymer_cols + args.target_cols:
        if col not in df.columns:
            raise ValueError(f"列不存在: {col}")

    n_rows = len(df)
    radius = args.radius
    nbits = args.nbits

    print(f"[INFO] 原始数据行数: {n_rows}")
    print(f"[INFO] 使用 Morgan 参数: radius={radius}, nbits={nbits}")
    print(f"[INFO] 特征模式: {args.feature_mode}")
    print(f"[INFO] 降维方法: {args.dr_method}")
    print(f"[INFO] no_morgan: {args.no_morgan}")
    print(f"[INFO] use_extended_phys: {args.use_extended_phys}")

    # 如果不使用 Morgan，则无视降维设置
    if args.no_morgan and args.dr_method != "none":
        print(
            "[WARN] 设置了 --no_morgan，但 dr_method 非 none。将忽略降维，仅使用工程化特征。",
            file=sys.stderr,
        )

    # 预分配矩阵
    poly_mat = np.zeros((n_rows, nbits), dtype=float)
    add_mat = np.zeros((n_rows, nbits), dtype=float)

    scalar_names = [
        "num_poly",
        "num_nonpoly",
        "num_components",
        "has_poly",
        "has_additive",
        "has_inorganic",
        "has_metal",
    ]
    scalar_mat = np.zeros((n_rows, len(scalar_names)), dtype=float)

    # 聚合物基团指纹矩阵
    poly_fg_names = list(POLY_FG_SMARTS.keys())
    poly_fg_mat = np.zeros((n_rows, len(poly_fg_names)), dtype=float)

    # 聚合物物化特征矩阵（先按 ALL 名称初始化，后面再根据 use_extended_phys 选子集）
    poly_phys_mat_full = np.zeros((n_rows, len(POLY_PHYS_ALL_NAMES)), dtype=float)

    smiles_cols = args.polymer_cols

    for i, (_, row) in enumerate(df.iterrows()):
        poly_vecs = []
        add_vecs = []

        # 用于基团统计和物化特征的聚合物 mol 列表
        poly_mols = []

        has_inorganic_row = False
        has_metal_row = False

        num_poly = 0
        num_nonpoly = 0

        for col in smiles_cols:
            raw = row[col]
            if pd.isna(raw):
                continue

            s = str(raw).strip()
            if s == "" or s.lower() in ("nan", "none"):
                continue

            mol = smiles_to_mol(s)
            if mol is None:
                # print(f"[WARN] 无效 SMILES 在行 {i}, 列 {col}: {s}", file=sys.stderr)
                continue

            # 只有在需要 Morgan 时才真正计算指纹，否则可以跳过（为了简单，这里仍然计算，但你也可以加 if not args.no_morgan）
            fp_arr = mol_to_morgan_fp(mol, radius, nbits) if not args.no_morgan else None

            # 判断是否为聚合物：简单用 '*' 作为标志
            if "*" in s:
                if fp_arr is not None:
                    poly_vecs.append(fp_arr)
                poly_mols.append(mol)
                num_poly += 1
            else:
                if fp_arr is not None:
                    add_vecs.append(fp_arr)
                num_nonpoly += 1

            # 判断无机/金属
            has_inorg, has_met = detect_inorganic_and_metal(mol)
            if has_inorg:
                has_inorganic_row = True
            if has_met:
                has_metal_row = True

        # 聚合物块 f_poly：若有多个聚合物，则取平均；否则为 0 向量
        if not args.no_morgan:
            if len(poly_vecs) > 0:
                poly_mat[i, :] = np.mean(poly_vecs, axis=0)
            else:
                poly_mat[i, :] = 0.0

            # 非聚合物块 f_add：同理
            if len(add_vecs) > 0:
                add_mat[i, :] = np.mean(add_vecs, axis=0)
            else:
                add_mat[i, :] = 0.0

        num_components = num_poly + num_nonpoly
        has_poly = 1.0 if num_poly > 0 else 0.0
        has_additive = 1.0 if num_nonpoly > 0 else 0.0

        scalar_mat[i, :] = [
            float(num_poly),
            float(num_nonpoly),
            float(num_components),
            has_poly,
            has_additive,
            1.0 if has_inorganic_row else 0.0,
            1.0 if has_metal_row else 0.0,
        ]

        # 聚合物基团指纹：对所有聚合物 mol 的匹配次数取平均
        if len(poly_mols) > 0:
            fg_counts = []
            for fg_name in poly_fg_names:
                patt = POLY_FG_PATTERNS[fg_name]
                if patt is None:
                    fg_counts.append(0.0)
                    continue
                count = 0
                for m in poly_mols:
                    matches = m.GetSubstructMatches(patt)
                    count += len(matches)
                fg_counts.append(float(count) / float(len(poly_mols)))
            poly_fg_mat[i, :] = np.array(fg_counts, dtype=float)
        else:
            poly_fg_mat[i, :] = 0.0

        # 聚合物物化特征：同样对 poly_mols 平均（ALL）
        poly_phys_mat_full[i, :] = np.array(compute_poly_physchem(poly_mols), dtype=float)

        if (i + 1) % 500 == 0:
            print(f"[INFO] 已处理 {i + 1}/{n_rows} 行...", file=sys.stderr)

    print("[INFO] 不删除无机/含金属样本，将通过 has_inorganic / has_metal 特征让模型学习其影响。")

    # 物化特征根据 use_extended_phys 选择子集
    if args.use_extended_phys:
        phys_cols = POLY_PHYS_ALL_NAMES
        poly_phys_mat = poly_phys_mat_full
    else:
        phys_cols = POLY_PHYS_BASE_NAMES
        base_dim = len(POLY_PHYS_BASE_NAMES)
        poly_phys_mat = poly_phys_mat_full[:, :base_dim]

    # ---------- 可选：对指纹做降维（只有在使用 Morgan 时才有意义） ----------
    dr_method = args.dr_method if not args.no_morgan else "none"
    fp_dr = None
    dr_cols = []

    if (not args.no_morgan) and dr_method != "none":
        fp_full = np.concatenate([poly_mat, add_mat], axis=1)  # shape: (n_rows, 2*nbits)
        orig_dim = fp_full.shape[1]
        dr_dim = args.dr_dim if args.dr_dim is not None else 128
        dr_dim = min(dr_dim, orig_dim)

        print(f"[INFO] 对指纹做降维: method={dr_method}, dim={dr_dim}")

        if dr_method == "pca":
            from sklearn.decomposition import PCA

            pca = PCA(n_components=dr_dim)
            fp_dr = pca.fit_transform(fp_full)
        elif dr_method == "umap":
            try:
                from umap import UMAP
            except ImportError as e:
                raise RuntimeError("需要安装 umap-learn 才能使用 UMAP 降维。pip install umap-learn") from e
            umap_model = UMAP(n_components=dr_dim, random_state=42)
            fp_dr = umap_model.fit_transform(fp_full)
        else:
            raise ValueError(f"未知降维方法: {dr_method}")

        dr_cols = [f"fp_dr_{j}" for j in range(dr_dim)]
        print(f"[INFO] 指纹降维完成，输出维度: {fp_dr.shape[1]}")

    # ---------- 构建最终特征 DataFrame ----------
    fg_cols = [f"poly_fg_{name}" for name in poly_fg_names]

    if args.no_morgan:
        # ✅ 只用工程化特征：scalar + poly_fg + poly_phys
        feat_array = np.concatenate([scalar_mat, poly_fg_mat, poly_phys_mat], axis=1)
        feat_cols = scalar_names + fg_cols + phys_cols
    else:
        # 仍然使用 Morgan（支持 feature_mode + 降维）
        poly_cols = [f"poly_fp_{j}" for j in range(nbits)]
        add_cols = [f"add_fp_{j}" for j in range(nbits)]

        if dr_method == "none":
            # 使用原始 2*nbits 维指纹
            if args.feature_mode == "2048":
                feat_array = np.concatenate([poly_mat, add_mat], axis=1)
                feat_cols = poly_cols + add_cols
            elif args.feature_mode == "2055":
                feat_array = np.concatenate([poly_mat, add_mat, scalar_mat], axis=1)
                feat_cols = poly_cols + add_cols + scalar_names
            else:  # extended
                feat_array = np.concatenate(
                    [poly_mat, add_mat, scalar_mat, poly_fg_mat, poly_phys_mat],
                    axis=1,
                )
                feat_cols = poly_cols + add_cols + scalar_names + fg_cols + phys_cols
        else:
            # 使用降维后的指纹 fp_dr 代替 poly_fp + add_fp
            if args.feature_mode == "2048":
                feat_array = fp_dr
                feat_cols = dr_cols
            elif args.feature_mode == "2055":
                feat_array = np.concatenate([fp_dr, scalar_mat], axis=1)
                feat_cols = dr_cols + scalar_names
            else:  # extended
                feat_array = np.concatenate(
                    [fp_dr, scalar_mat, poly_fg_mat, poly_phys_mat],
                    axis=1,
                )
                feat_cols = dr_cols + scalar_names + fg_cols + phys_cols

    feat_df = pd.DataFrame(feat_array, columns=feat_cols)

    # 拼上 target 列
    out_df = pd.concat([df[args.target_cols].reset_index(drop=True), feat_df], axis=1)

    print(f"[INFO] 最终特征维度: {feat_df.shape[1]}")
    print(f"[INFO] 写出到: {args.out_csv}")
    out_df.to_csv(args.out_csv, index=False)
    print("[INFO] 完成。")


if __name__ == "__main__":
    main()
