#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基线训练脚本（配合 concat_features.py 的输出特征CSV使用）
- 模型：RandomForestRegressor / XGBRegressor / MLPRegressor(3层可调) / RF+MLP(残差) / SVM(SVR)【新增】
- 划分：train_test_split 或 GroupKFold（--group_col）
- 目标变换：--y_log1p（训练 log1p(y)，预测 expm1 还原）
- 输出：metrics.json、(可选) feature_importance.csv、best_model.joblib、
       (可选) valid_predictions.csv（--save_train_pred 时）
       (可选) cvK_oof.csv（--cv10 且 --save_train_pred 时，K 为折数）
       (可选) test_predictions.csv（提供 --test_csv 时）
- 测试对齐：自动按训练特征列对齐测试特征列（缺失列补 0，多余列丢弃）

【新增/修改点】
- 修复 RFPlusMLP 在被 TransformedTargetRegressor/clone 时的参数名不一致问题（完全 sklearn 兼容）
- 新增交叉验证开关：--cv10，折数由 --cv_folds 控制（默认10，可设为5等）
- 新增预测表导出：valid_predictions.csv / cvK_oof.csv / test_predictions.csv
- 新增：交叉验证训练集每折指标 cvK_train_metrics.csv
- 新增：交叉验证训练集明细 cvK_train_infold.csv
- 新增：每折单独CSV导出：fold_XX_train.csv / fold_XX_valid.csv
- 新增：每折模型导出：fold_models/fold_XX_best_model.joblib
- 新增：所有交叉验证输出文件名前缀自适应 K（cv{K}_*.csv/json）
- 新增：SVM (SVR) 模型支持：--model svm，并提供核函数/正则/epsilon/gamma 可调
"""

import os, json, argparse
from pathlib import Path
from typing import List, Union, Tuple, Dict, Optional
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GroupKFold, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR  # ? 新增
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import BaseEstimator, RegressorMixin

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MaxAbsScaler
from sklearn.impute import SimpleImputer

import joblib
import warnings

# ---- 可选：XGBoost（若没装会自动报错提示）----
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


# ============== I/O 与预处理 ==============

def read_csv_robust(path: str) -> pd.DataFrame:
    encs = ["utf-8","utf-8-sig","gbk","cp936","latin1","iso-8859-1","utf-16","utf-16le","utf-16be"]
    seps = [",","\t",";","|"]
    last_err = None
    for e in encs:
        for s in seps:
            try:
                df = pd.read_csv(path, encoding=e, sep=s, low_memory=False)
                if df.shape[1] >= 1:
                    return df
            except Exception as err:
                last_err = err
    if last_err:
        raise last_err
    raise RuntimeError(f"无法读取CSV: {path}")

def split_xy(df: pd.DataFrame, target: str, drop_cols: List[str]):
    """
    拆分特征与目标：
    - 自动排除 row_index / index / id / SMILES 等非特征列
    - 仅保留数值型列作为特征
    """
    # ========== ① 构造排除列 ==========
    exclude = set(drop_cols + [target])
    auto_exclude = {
        c for c in df.columns
        if str(c).strip().lower() in {
            "row_index", "index", "id", "idx", "sample_id", "recipeid",
            "smile", "smiles", "smile a", "smile b", "smile c"
        }
    }
    exclude |= auto_exclude

    # ========== ② 选择数值型特征列 ==========
    cols = [
        c for c in df.columns
        if c not in exclude and np.issubdtype(df[c].dtype, np.number)
    ]
    if not cols:
        raise ValueError("未检测到数值型特征列，请检查输入CSV文件。")

    # ========== ③ 输出日志（一次性打印） ==========
    print(f"[INFO] 使用 {len(cols)} 个特征列进行训练（排除列: {sorted(exclude)}）")

    # ========== ④ 返回结果 ==========
    X = df[cols].copy()
    y = df[target].values
    return X, y, cols

def eval_metrics(y_true, y_pred) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    return {"R2": r2, "RMSE": rmse, "MAE": mae}

def dump_pred_table(df_src: pd.DataFrame,
                    y_true: Optional[np.ndarray],
                    y_pred: np.ndarray,
                    id_cols: List[str],
                    out_path: Union[str, Path],
                    extra_cols: Dict[str, Union[int, float, str, np.ndarray]] = None):
    """
    统一写出预测表：
    - idx: 0..n-1
    - y_pred
    - （可选）y_true, residual = y_true - y_pred
    - （可选）ID列（存在才保留，不会报错）
    - （可选）extra_cols: 比如 fold / row_index 等
    """
    out = pd.DataFrame({"idx": np.arange(len(y_pred)), "y_pred": y_pred})
    if y_true is not None:
        out["y_true"] = y_true
        out["residual"] = out["y_true"] - out["y_pred"]

    keep = []
    if id_cols:
        for c in id_cols:
            if c in df_src.columns:
                keep.append(c)
            else:
                pass
    if keep:
        out = pd.concat([df_src.reset_index(drop=True)[keep], out], axis=1)

    if extra_cols:
        for k, v in extra_cols.items():
            if np.isscalar(v):
                out[k] = v
            else:
                out[k] = np.asarray(v)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("[saved]", out_path, " shape=", out.shape)


# ============== 工具函数 ==============

def parse_hidden(s: str) -> List[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]

def parse_batch_size(s: Union[str,int]) -> Union[str,int]:
    if isinstance(s, int):
        return s
    s = str(s).strip().lower()
    return 'auto' if s == 'auto' else int(s)


# ============== 模型配置 ==============

def parse_max_features(val, default=0.25):
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return val
    s = str(val).strip().lower()
    if s in {"sqrt", "log2"}:
        return s
    if s in {"auto", "all"}:
        warnings.warn("rf_max_features='auto' 已映射为 1.0（使用全部特征）。", RuntimeWarning)
        return 1.0
    try:
        if s.isdigit():
            return int(s)
        f = float(s)
        if f.is_integer():
            return int(f)
        return f
    except Exception as e:
        raise ValueError(f"无法解析 --rf_max_features={val!r}，请用 0<float<=1、正整数，或 'sqrt'/'log2'/'auto'。") from e

def parse_min_samples_split(val, default=2):
    if val is None:
        return default
    if isinstance(val, (int, np.integer)):
        if val >= 2:
            return int(val)
        raise ValueError("min_samples_split int 必须 >= 2")
    try:
        f = float(val)
    except Exception as e:
        raise ValueError(f"无法解析 rf_min_samples_split={val!r}") from e
    if 0.0 < f <= 1.0:
        return f
    if f.is_integer() and f >= 2.0:
        return int(f)
    raise ValueError("min_samples_split 需为 int>=2 或 float in (0,1]")

def default_rf(seed: int, args=None) -> RandomForestRegressor:
    defaults = dict(
        rf_n_estimators=1500,
        rf_max_depth=24,
        rf_max_features=0.25,
        rf_min_samples_leaf=5,
        rf_min_samples_split=2,
    )
    def get(name):
        if args is not None and hasattr(args, name) and getattr(args, name) is not None:
            return getattr(args, name)
        return defaults[name]
    mf = parse_max_features(get('rf_max_features'), default=defaults['rf_max_features'])
    mss = parse_min_samples_split(get('rf_min_samples_split'), default=defaults['rf_min_samples_split'])
    return RandomForestRegressor(
        n_estimators=get('rf_n_estimators'),
        max_depth=get('rf_max_depth'),
        max_features=mf,
        min_samples_leaf=get('rf_min_samples_leaf'),
        min_samples_split=mss,
        n_jobs=-1,
        random_state=seed,
        bootstrap=True,
        oob_score=False,
    )

def default_xgb(seed: int, args=None):
    if not _HAS_XGB:
        raise RuntimeError("未安装 xgboost，请先安装或改用 --model rf")
    defaults = dict(
        xgb_n_estimators=800,
        xgb_learning_rate=0.05,
        xgb_max_depth=4,
        xgb_subsample=0.7,
        xgb_colsample_bytree=0.3,
        xgb_min_child_weight=6.0,
        xgb_reg_lambda=8.0,
        xgb_tree_method="hist",
    )
    def get(name):
        if args is not None and hasattr(args, name) and getattr(args, name) is not None:
            return getattr(args, name)
        return defaults[name]
    return XGBRegressor(
        n_estimators=get('xgb_n_estimators'),
        learning_rate=get('xgb_learning_rate'),
        max_depth=get('xgb_max_depth'),
        min_child_weight=get('xgb_min_child_weight'),
        subsample=get('xgb_subsample'),
        colsample_bytree=get('xgb_colsample_bytree'),
        reg_lambda=get('xgb_reg_lambda'),
        tree_method=get('xgb_tree_method'),
        random_state=seed,
    )

def make_mlp_pipeline(
    hidden_layers: List[int],
    activation: str,
    alpha: float,
    lr_init: float,
    max_iter: int,
    early_stopping: bool,
    val_fraction: float,
    batch_size: Union[str,int],
    solver: str,
    seed: int
):
    batch_size = parse_batch_size(batch_size)
    if solver == "lbfgs" and early_stopping:
        print("[warn] solver=lbfgs 不支持 early_stopping，已自动关闭。")
        early_stopping = False
    mlp = MLPRegressor(
        hidden_layer_sizes=tuple(hidden_layers),
        activation=activation,
        alpha=alpha,
        learning_rate_init=lr_init,
        max_iter=max_iter,
        early_stopping=early_stopping,
        validation_fraction=val_fraction,
        batch_size=batch_size,
        solver=solver,
        random_state=seed
    )
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("var", VarianceThreshold(threshold=0.0)),
        ("scaler", MaxAbsScaler()),
        ("mlp", mlp)
    ])
    return pipe

class RFPlusMLP(BaseEstimator, RegressorMixin):
    """sklearn 风格封装：先拟合 RF，再用 MLP 拟合残差；predict = rf + mlp"""
    def __init__(self, rf=None, mlp_pipe=None):
        self.rf = rf
        self.mlp_pipe = mlp_pipe
    def get_params(self, deep=True):
        return {"rf": self.rf, "mlp_pipe": self.mlp_pipe}
    def set_params(self, **params):
        if "mlp" in params and "mlp_pipe" not in params:
            params["mlp_pipe"] = params.pop("mlp")
        for k, v in params.items():
            setattr(self, k, v)
        return self
    def fit(self, X, y):
        if self.rf is None or self.mlp_pipe is None:
            raise ValueError("RFPlusMLP 需要传入 rf 与 mlp_pipe 两个子模型")
        self.rf.fit(X, y)
        residual = y - self.rf.predict(X)
        self.mlp_pipe.fit(X, residual)
        return self
    def predict(self, X):
        return self.rf.predict(X) + self.mlp_pipe.predict(X)

def _parse_svm_gamma(val: str):
    """将 --svm_gamma 解析为 float 或 'scale'/'auto' 字符串"""
    if val is None:
        return "scale"
    s = str(val).strip().lower()
    if s in {"scale","auto"}:
        return s
    try:
        return float(s)
    except Exception:
        return "scale"

def build_base_estimator(
    model_name: str,
    seed: int,
    mlp_hidden: List[int],
    mlp_activation: str,
    mlp_alpha: float,
    mlp_lr: float,
    mlp_max_iter: int,
    mlp_early_stop: bool,
    mlp_val_fraction: float,
    mlp_batch_size: Union[str,int],
    mlp_solver: str,
    args=None,
):
    if model_name == "rf":
        base = default_rf(seed, args=args)
    elif model_name == "xgb":
        base = default_xgb(seed, args=args)
    elif model_name == "mlp":
        base = make_mlp_pipeline(
            hidden_layers=mlp_hidden,
            activation=mlp_activation,
            alpha=mlp_alpha,
            lr_init=mlp_lr,
            max_iter=mlp_max_iter,
            early_stopping=mlp_early_stop,
            val_fraction=mlp_val_fraction,
            batch_size=mlp_batch_size,
            solver=mlp_solver,
            seed=seed
        )
    elif model_name == "rf+mlp":
        rf = default_rf(seed, args=args)
        mlp = make_mlp_pipeline(
            hidden_layers=mlp_hidden,
            activation=mlp_activation,
            alpha=mlp_alpha,
            lr_init=mlp_lr,
            max_iter=mlp_max_iter,
            early_stopping=mlp_early_stop,
            val_fraction=mlp_val_fraction,
            batch_size=mlp_batch_size,
            solver=mlp_solver,
            seed=seed
        )
        base = RFPlusMLP(rf=rf, mlp_pipe=mlp)
    elif model_name == "svm":  # ? 新增 SVM
        gamma = _parse_svm_gamma(getattr(args, "svm_gamma", "scale"))
        base = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            ("svm", SVR(
                kernel=getattr(args, "svm_kernel", "rbf"),
                C=getattr(args, "svm_C", 1.0),
                epsilon=getattr(args, "svm_epsilon", 0.1),
                gamma=gamma
            ))
        ])
    else:
        raise ValueError("未知模型类型")
    return base

def wrap_y_transform_if_needed(base_estimator, use_log1p: bool):
    if not use_log1p:
        return base_estimator
    return TransformedTargetRegressor(
        regressor=base_estimator,
        func=np.log1p,
        inverse_func=np.expm1
    )


# ============== 训练/评估/保存（单次划分） ==============

def train_and_eval(
    df_feat: pd.DataFrame,
    target: str,
    valid_size: float,
    seed: int,
    model_name: str,
    drop_cols: List[str],
    save_dir: Path,
    mlp_hidden: List[int],
    mlp_activation: str,
    mlp_alpha: float,
    mlp_lr: float,
    mlp_max_iter: int,
    mlp_early_stop: bool,
    mlp_val_fraction: float,
    mlp_batch_size: Union[str,int],
    mlp_solver: str,
    y_log1p: bool,
    group_col: str,
    n_splits: int,
    fold_idx: int,
    no_perm: bool,
    save_train_pred: bool,
    id_cols: List[str],
    args=None,
):
    use_group = bool(group_col) and (group_col in df_feat.columns)
    if use_group:
        groups = df_feat[group_col].values
        gkf = GroupKFold(n_splits=int(n_splits))
        splits = list(gkf.split(df_feat, groups=groups))
        tr_idx, va_idx = splits[fold_idx % len(splits)]
        df_tr, df_va = df_feat.iloc[tr_idx], df_feat.iloc[va_idx]
    else:
        df_tr, df_va = train_test_split(df_feat, test_size=valid_size, random_state=seed)

    Xtr, ytr, feat_cols = split_xy(df_tr, target, drop_cols)
    Xva, yva, _         = split_xy(df_va, target, drop_cols)

    base = build_base_estimator(
        model_name=model_name, seed=seed,
        mlp_hidden=mlp_hidden, mlp_activation=mlp_activation, mlp_alpha=mlp_alpha,
        mlp_lr=mlp_lr, mlp_max_iter=mlp_max_iter, mlp_early_stop=mlp_early_stop,
        mlp_val_fraction=mlp_val_fraction, mlp_batch_size=mlp_batch_size, mlp_solver=mlp_solver,
        args=args
    )
    model = wrap_y_transform_if_needed(base, use_log1p=y_log1p)

    Xtr = Xtr.fillna(0.0)
    Xva = Xva.fillna(0.0)

    model.fit(Xtr, ytr)

    pred_tr = model.predict(Xtr)
    pred_va = model.predict(Xva)
    m_tr = eval_metrics(ytr, pred_tr)
    m_va = eval_metrics(yva, pred_va)

    metrics_path = save_dir / "metrics.json"
    meta_extra = {
        "use_group": use_group,
        "group_col": group_col if use_group else None,
        "n_splits": int(n_splits) if use_group else None,
        "fold_idx": int(fold_idx) if use_group else None,
        "y_log1p": bool(y_log1p),
        "model_name": model_name
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"train": m_tr, "valid": m_va, "meta": meta_extra}, f, ensure_ascii=False, indent=2)
    print("[metrics] train:", m_tr)
    print("[metrics] valid:", m_va)
    print("[saved]", metrics_path)

    if not no_perm:
        try:
            pi = permutation_importance(model, Xva, yva, n_repeats=10, random_state=seed, n_jobs=-1)
            imp = pd.DataFrame({"feature": feat_cols, "importance": pi.importances_mean})
            imp.sort_values("importance", ascending=False, inplace=True)
            imp.to_csv(save_dir/"feature_importance.csv", index=False, encoding="utf-8-sig")
            print("[saved]", save_dir/"feature_importance.csv")
        except Exception as e:
            print("[warn] permutation_importance 失败：", e)

    payload = {
        "model": model,
        "features": feat_cols,
        "target": target,
        "drop_cols": drop_cols,
        "mlp_config": {
            "hidden": mlp_hidden,
            "activation": mlp_activation,
            "alpha": mlp_alpha,
            "lr": mlp_lr,
            "max_iter": mlp_max_iter,
            "early_stopping": mlp_early_stop,
            "val_fraction": mlp_val_fraction,
            "batch_size": mlp_batch_size,
            "solver": mlp_solver
        } if model_name in ["mlp","rf+mlp"] else None,
        "y_log1p": bool(y_log1p),
        "group_col": group_col if use_group else None
    }
    joblib.dump(payload, save_dir/"best_model.joblib")
    print("[saved]", save_dir/"best_model.joblib")

    if save_train_pred:
        dump_pred_table(df_src=df_va, y_true=yva, y_pred=pred_va,
                        id_cols=id_cols, out_path=save_dir/"valid_predictions.csv")


# ============== 交叉验证（可变 K，含每折CSV与每折模型导出） ==============

def cv10_evaluate(
    df_feat: pd.DataFrame,
    target: str,
    seed: int,
    model_name: str,
    drop_cols: List[str],
    save_dir: Path,
    # MLP 参数
    mlp_hidden: List[int],
    mlp_activation: str,
    mlp_alpha: float,
    mlp_lr: float,
    mlp_max_iter: int,
    mlp_early_stop: bool,
    mlp_val_fraction: float,
    mlp_batch_size: Union[str,int],
    mlp_solver: str,
    # 目标变换
    y_log1p: bool,
    # 分组
    group_col: str,
    # 导出预测
    save_train_pred: bool,
    id_cols: List[str],
    oof_path: Optional[str] = None,
    # 传入 RF/XGB 的 args
    args=None,
    # 新增：折数
    cv_folds: int = 10,
):
    """
    交叉验证评估：
    - 验证集每折指标：cvK_metrics.csv
    - 训练集每折指标：cvK_train_metrics.csv
    - 若 save_train_pred：
        * 验证集OOF明细：cvK_oof.csv（含 fold）
        * 训练集in-fold明细：cvK_train_infold.csv（含 fold）
        * 每折单独CSV：fold_XX_valid.csv / fold_XX_train.csv
    - 每折模型：fold_models/fold_XX_best_model.joblib
    """
    Xall, yall, feat_cols = split_xy(df_feat, target, drop_cols)

    use_group = bool(group_col) and (group_col in df_feat.columns)
    if use_group:
        uniq = pd.Series(df_feat[group_col]).nunique()
        cv_k = max(2, min(int(cv_folds), int(uniq)))
        splitter = GroupKFold(n_splits=cv_k)
        splits = splitter.split(Xall, yall, groups=df_feat[group_col])
        cv_name = f"GroupKFold({cv_k})"
    else:
        cv_k = max(2, int(cv_folds))
        splitter = KFold(n_splits=cv_k, shuffle=True, random_state=seed)
        splits = splitter.split(Xall, yall)
        cv_name = f"KFold({cv_k}, shuffle=True)"

    prefix = f"cv{cv_k}"

    rows_valid, rows_train = [], []

    oof_pred,  oof_true,  oof_idx,  oof_fold  = [], [], [], []
    tr_pred,   tr_true,   tr_idx,   tr_fold   = [], [], [], []

    # 每折模型保存目录
    fold_model_dir = save_dir / "fold_models"
    fold_model_dir.mkdir(parents=True, exist_ok=True)

    fold_idx = 0
    for tr_ind, va_ind in splits:
        fold_idx += 1
        Xtr, Xva = Xall.iloc[tr_ind], Xall.iloc[va_ind]
        ytr, yva = yall[tr_ind],      yall[va_ind]

        base = build_base_estimator(
            model_name=model_name, seed=seed+fold_idx,
            mlp_hidden=mlp_hidden, mlp_activation=mlp_activation, mlp_alpha=mlp_alpha,
            mlp_lr=mlp_lr, mlp_max_iter=mlp_max_iter, mlp_early_stop=mlp_early_stop,
            mlp_val_fraction=mlp_val_fraction, mlp_batch_size=mlp_batch_size, mlp_solver=mlp_solver,
            args=args
        )
        model = wrap_y_transform_if_needed(base, use_log1p=y_log1p)

        Xtr = Xtr.fillna(0.0)
        Xva = Xva.fillna(0.0)

        model.fit(Xtr, ytr)

        # 保存本折模型
        fold_payload = {
            "model": model,
            "features": feat_cols,
            "target": target,
            "drop_cols": drop_cols,
            "mlp_config": {
                "hidden": mlp_hidden,
                "activation": mlp_activation,
                "alpha": mlp_alpha,
                "lr": mlp_lr,
                "max_iter": mlp_max_iter,
                "early_stopping": mlp_early_stop,
                "val_fraction": mlp_val_fraction,
                "batch_size": mlp_batch_size,
                "solver": mlp_solver
            } if model_name in ["mlp","rf+mlp"] else None,
            "y_log1p": bool(y_log1p),
            "group_col": group_col if use_group else None,
            "cv_fold": int(fold_idx),
            "cv_name": cv_name
        }
        fold_path = fold_model_dir / f"fold_{fold_idx:02d}_best_model.joblib"
        joblib.dump(fold_payload, fold_path)
        print("[saved]", fold_path)

        # 验证集
        yhat_va = model.predict(Xva)
        m_va = eval_metrics(yva, yhat_va)
        m_va["fold"] = fold_idx
        rows_valid.append(m_va)
        print(f"[{prefix}][valid] fold {fold_idx}: {m_va}")

        # 训练集
        yhat_tr = model.predict(Xtr)
        m_tr = eval_metrics(ytr, yhat_tr)
        m_tr["fold"] = fold_idx
        rows_train.append(m_tr)
        print(f"[{prefix}][train] fold {fold_idx}: {m_tr}")

        # 收集明细用于汇总CSV
        oof_pred.append(yhat_va);  oof_true.append(yva);  oof_idx.append(va_ind)
        oof_fold.append(np.full_like(va_ind, fill_value=fold_idx, dtype=int))
        tr_pred.append(yhat_tr);   tr_true.append(ytr);   tr_idx.append(tr_ind)
        tr_fold.append(np.full_like(tr_ind, fill_value=fold_idx, dtype=int))

        # 每折单独 CSV
        if save_train_pred:
            # 训练折内数据
            df_tr_src = df_feat.iloc[tr_ind]
            dump_pred_table(
                df_src=df_tr_src,
                y_true=ytr,
                y_pred=yhat_tr,
                id_cols=id_cols,
                out_path=save_dir / f"fold_{fold_idx:02d}_train.csv",
                extra_cols={"fold": np.full_like(tr_ind, fold_idx, dtype=int)}
            )
            # 验证折数据
            df_va_src = df_feat.iloc[va_ind]
            dump_pred_table(
                df_src=df_va_src,
                y_true=yva,
                y_pred=yhat_va,
                id_cols=id_cols,
                out_path=save_dir / f"fold_{fold_idx:02d}_valid.csv",
                extra_cols={"fold": np.full_like(va_ind, fold_idx, dtype=int)}
            )

    # 指标表
    cv_valid_df = pd.DataFrame(rows_valid)[["fold","R2","RMSE","MAE"]]
    cv_valid_df.to_csv(save_dir/f"{prefix}_metrics.csv", index=False, encoding="utf-8-sig")
    print("[saved]", save_dir/f"{prefix}_metrics.csv")

    cv_train_df = pd.DataFrame(rows_train)[["fold","R2","RMSE","MAE"]]
    cv_train_df.to_csv(save_dir/f"{prefix}_train_metrics.csv", index=False, encoding="utf-8-sig")
    print("[saved]", save_dir/f"{prefix}_train_metrics.csv")

    summary = {
        "cv": cv_name,
        "R2_mean": float(cv_valid_df["R2"].mean()),
        "R2_std":  float(cv_valid_df["R2"].std(ddof=1) if len(cv_valid_df)>1 else 0.0),
        "RMSE_mean": float(cv_valid_df["RMSE"].mean()),
        "RMSE_std":  float(cv_valid_df["RMSE"].std(ddof=1) if len(cv_valid_df)>1 else 0.0),
        "MAE_mean":  float(cv_valid_df["MAE"].mean()),
        "MAE_std":   float(cv_valid_df["MAE"].std(ddof=1) if len(cv_valid_df)>1 else 0.0),
        "n_folds":   int(len(cv_valid_df))
    }
    with open(save_dir/f"{prefix}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[saved]", save_dir/f"{prefix}_summary.json")
    print(f"[{prefix}] summary (valid only):", summary)

    # 汇总版明细（OOF/训练）导出
    if save_train_pred:
        # 验证 OOF
        idx_all = np.concatenate(oof_idx)
        order = np.argsort(idx_all)
        y_true_all = np.concatenate(oof_true)[order]
        y_pred_all = np.concatenate(oof_pred)[order]
        fold_all   = np.concatenate(oof_fold)[order]
        df_src_all = df_feat.iloc[idx_all[order]]
        path_oof = oof_path or str(save_dir / f"{prefix}_oof.csv")
        dump_pred_table(df_src=df_src_all, y_true=y_true_all, y_pred=y_pred_all,
                        id_cols=id_cols, out_path=path_oof, extra_cols={"fold": fold_all})

        # 训练 in-fold
        tr_idx_all = np.concatenate(tr_idx)
        tr_order = np.argsort(tr_idx_all)
        tr_true_all = np.concatenate(tr_true)[tr_order]
        tr_pred_all = np.concatenate(tr_pred)[tr_order]
        tr_fold_all = np.concatenate(tr_fold)[tr_order]
        df_tr_src_all = df_feat.iloc[tr_idx_all[tr_order]]
        path_tr = str(save_dir / f"{prefix}_train_infold.csv")
        dump_pred_table(df_src=df_tr_src_all, y_true=tr_true_all, y_pred=tr_pred_all,
            id_cols=id_cols, out_path=path_tr, extra_cols={"fold": tr_fold_all})


# ============== 测试集预测 ==============

def predict_on_test(
    model_path: Path,
    test_feat_df: pd.DataFrame,
    id_cols: List[str],
    save_dir: Path,
    target_col: Optional[str] = None,
    pred_path: Optional[str] = None
):
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feat_cols_ref: List[str] = bundle["features"]

    test_num = test_feat_df.select_dtypes(include=[np.number]).copy()
    for c in feat_cols_ref:
        if c not in test_num.columns:
            test_num[c] = 0.0
    Xt = test_num[feat_cols_ref].astype(float)

    y_true = None
    if target_col and (target_col in test_feat_df.columns):
        y_true = pd.to_numeric(test_feat_df[target_col], errors="coerce").to_numpy()

    y_pred = model.predict(Xt)
    out_path = pred_path or str(save_dir / "test_predictions.csv")
    dump_pred_table(df_src=test_feat_df,
                    y_true=y_true,
                    y_pred=y_pred,
                    id_cols=id_cols,
                    out_path=out_path)


# ============== 主函数与参数 ==============

def main():
    ap = argparse.ArgumentParser(description="基于 concat 特征CSV 的 RF/XGB/MLP/RF+MLP/SVM 训练脚本（含交叉验证）")
    ap.add_argument("--in_csv", required=True, help="训练用特征CSV（由 concat_features.py 生成）")
    ap.add_argument("--target", required=True, help="目标列名（必须在 in_csv 中存在）")
    ap.add_argument("--model", choices=["rf","xgb","mlp","rf+mlp","svm"], default="rf")  # ? 加入 svm
    ap.add_argument("--save_dir", required=True, help="输出目录")
    ap.add_argument("--valid_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drop_cols", default="", help="训练时剔除的列（逗号分隔），如 'ID,SampleName'")
    ap.add_argument("--test_csv", default=None, help="可选：测试集特征CSV（同样由 concat_features.py 生成）")
    ap.add_argument("--id_cols", default="", help="预测输出中需要透传的列（逗号分隔）")

    # 目标变换
    ap.add_argument("--y_log1p", action="store_true", help="对目标取 log1p 训练，预测时 expm1 还原")

    # 分组划分（防泄漏）
    ap.add_argument("--group_col", default="", help="用于 GroupKFold 的分组列名（可选）")
    ap.add_argument("--n_splits", type=int, default=5, help="GroupKFold 折数（仅用于单次划分训练）")
    ap.add_argument("--fold_idx", type=int, default=0, help="选择第几个折作为验证集（仅用于单次划分训练）")

    # 置换重要性开关
    ap.add_argument("--no_perm", action="store_true", help="关闭 permutation_importance（更快）")

    # === XGBoost hyperparams ===
    ap.add_argument('--xgb_n_estimators', type=int, default=1000)
    ap.add_argument('--xgb_learning_rate', type=float, default=0.05)
    ap.add_argument('--xgb_max_depth', type=int, default=10)
    ap.add_argument('--xgb_min_child_weight', type=float, default=10.0)
    ap.add_argument('--xgb_subsample', type=float, default=0.8)
    ap.add_argument('--xgb_colsample_bytree', type=float, default=0.4)
    ap.add_argument('--xgb_reg_lambda', type=float, default=3.0)
    ap.add_argument('--xgb_tree_method', type=str, default='hist')

    # MLP 可调参数
    ap.add_argument("--mlp_hidden", default="256,128,64", help="MLP 隐层，如 '256,128,64'")
    ap.add_argument("--mlp_activation", choices=["relu","tanh","logistic","identity"], default="relu")
    ap.add_argument("--mlp_alpha", type=float, default=1e-3, help="L2 正则系数")
    ap.add_argument("--mlp_lr", type=float, default=1e-3, help="初始学习率")
    ap.add_argument("--mlp_max_iter", type=int, default=800)
    ap.add_argument("--mlp_early_stop", action="store_true", help="启用 early stopping")
    ap.add_argument("--mlp_val_fraction", type=float, default=0.1, help="early stopping 的验证比例")
    ap.add_argument("--mlp_batch_size", default="auto", help="'auto' 或 整数，如 256")
    ap.add_argument("--mlp_solver", choices=["adam","lbfgs","sgd"], default="adam")

    # ? SVM 超参
    ap.add_argument("--svm_kernel", choices=["rbf","linear","poly","sigmoid"], default="rbf")
    ap.add_argument("--svm_C", type=float, default=1.0)
    ap.add_argument("--svm_epsilon", type=float, default=0.1)
    ap.add_argument("--svm_gamma", default="scale", help="'scale' / 'auto' / 浮点数")

    # 交叉验证（折数可配）
    ap.add_argument("--cv10", action="store_true", help="启用交叉验证评估（折数由 --cv_folds 控制，默认10）")
    ap.add_argument("--cv_folds", type=int, default=10,
                    help="交叉验证折数（配合 --cv10 使用，默认10，可设为5等）")

    # 预测表导出控制
    ap.add_argument("--save_train_pred", action="store_true",
                    help="保存训练期预测（valid_predictions.csv / cvK_oof.csv / cvK_train_infold.csv / 每折CSV）")
    ap.add_argument("--pred_path", default=None, help="自定义测试集预测输出路径（默认 <save_dir>/test_predictions.csv）")
    ap.add_argument("--oof_path", default=None, help="自定义 OOF 预测输出路径（默认 <save_dir>/cvK_oof.csv）")

    # RF 超参（可被 default_rf 读取）
    ap.add_argument('--rf_n_estimators', type=int, default=1500)
    ap.add_argument('--rf_max_depth', type=int, default=24)
    ap.add_argument(
        '--rf_max_features',
        type=str,
        default=None,
        help="可填 0.3(浮点), 128(整数), 'sqrt' 或 'log2'；'auto' 会被映射为 1.0"
    )
    ap.add_argument('--rf_min_samples_leaf', type=int, default=5)
    ap.add_argument('--rf_min_samples_split', type=float, default=2,
                    help="内部节点再划分所需的最小样本数；可为整数>=2或(0,1]的小数（占比）")

    args = ap.parse_args()

    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)

    # 读取训练特征CSV
    df = read_csv_robust(args.in_csv)
    assert args.target in df.columns, f"目标列 {args.target} 不在训练CSV中"

    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]
    id_cols   = [c.strip() for c in args.id_cols.split(",") if c.strip()]

    # 单次划分训练/评估
    train_and_eval(
        df_feat=df,
        target=args.target,
        valid_size=args.valid_size,
        seed=int(args.seed),
        model_name=args.model,
        drop_cols=drop_cols,
        save_dir=save_dir,
        # MLP
        mlp_hidden=parse_hidden(args.mlp_hidden),
        mlp_activation=args.mlp_activation,
        mlp_alpha=args.mlp_alpha,
        mlp_lr=args.mlp_lr,
        mlp_max_iter=args.mlp_max_iter,
        mlp_early_stop=args.mlp_early_stop,
        mlp_val_fraction=args.mlp_val_fraction,
        mlp_batch_size=parse_batch_size(args.mlp_batch_size),
        mlp_solver=args.mlp_solver,
        # 其他
        y_log1p=args.y_log1p,
        group_col=args.group_col,
        n_splits=int(args.n_splits),
        fold_idx=int(args.fold_idx),
        no_perm=args.no_perm,
        save_train_pred=bool(args.save_train_pred),
        id_cols=id_cols,
        args=args,
    )

    # 交叉验证评估（含每折CSV与模型导出）
    if args.cv10:
        cv10_evaluate(
            df_feat=df,
            target=args.target,
            seed=int(args.seed),
            model_name=args.model,
            drop_cols=drop_cols,
            save_dir=save_dir,
            cv_folds=int(args.cv_folds),
            mlp_hidden=parse_hidden(args.mlp_hidden),
            mlp_activation=args.mlp_activation,
            mlp_alpha=args.mlp_alpha,
            mlp_lr=args.mlp_lr,
            mlp_max_iter=args.mlp_max_iter,
            mlp_early_stop=args.mlp_early_stop,
            mlp_val_fraction=args.mlp_val_fraction,
            mlp_batch_size=parse_batch_size(args.mlp_batch_size),
            mlp_solver=args.mlp_solver,
            y_log1p=args.y_log1p,
            group_col=args.group_col,
            save_train_pred=bool(args.save_train_pred),
            id_cols=id_cols,
            oof_path=args.oof_path,
            args=args,
        )

    # 测试预测（提供 --test_csv 即输出 test_predictions.csv）
    if args.test_csv:
        dft = read_csv_robust(args.test_csv)
        predict_on_test(
            model_path=save_dir/"best_model.joblib",
            test_feat_df=dft,
            id_cols=id_cols,
            save_dir=save_dir,
            target_col=args.target if args.target in dft.columns else None,
            pred_path=args.pred_path
        )
        print("[info] 已输出测试集预测：", (args.pred_path or str(save_dir / "test_predictions.csv")))


if __name__ == "__main__":
    main()
