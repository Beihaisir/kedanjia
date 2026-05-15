# -*- coding: utf-8 -*-
"""
Streamlit 客单价分析工具

安装依赖：
    pip install streamlit pandas openpyxl xlrd

运行：
    streamlit run streamlit_customer_unit_price.py

核心口径：
1. 支持菜品明细、支付明细分别上传多个文件，适合多门店汇总。
2. 菜品明细按 POS销售单号 聚合为订单。
3. 退款剔除：单据类型=退款单 的记录不参与统计，并同时读取退款行的 POS销售单号 与 POS退款单号，剔除对应整单销售数据。
4. 主菜识别：菜品名称包含维护窗口中的任一主菜关键词，即按主菜计数。
5. 组合主菜：同一菜品名称命中多个主菜关键词时，主菜数量=命中主菜数 * 菜品数量。
   例如：(套)泡椒板筋+宫保猪肝，数量=2，则主菜数量/人数=2*2=4。
6. 订单金额：订单内所有有效菜品行的 优惠后小计价格 合计。
7. 客单价：统计周期内总金额 / 总人数。无主菜订单金额计入当天/渠道总金额，但人数为 0，并单独统计订单数。
8. 渠道：支付明细表通过 POS销售单号 关联，支付类型作为渠道；一单多支付类型会合并为“多支付：A+B”。

大数据优化：
- 多文件合并时只保留分析所需字段，减少内存占用。
- 主菜计数使用向量化 str.contains，不使用逐行 apply。
- 先聚合到订单级，再关联支付，避免一单多支付导致金额重复。
- 默认只预览前 N 行明细；默认导出汇总，不导出百万级明细。
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import streamlit as st


DEFAULT_MAIN_KEYWORDS = [
    "干锅肥肠", "干锅鸡杂", "干锅牛杂", "干锅板筋",
    "宫保牛杂", "宫保鸡胗", "宫保板筋", "宫保肥肠", "宫保猪肝",
    "宫保腰花", "宫保双脆", "宫保牛肉", "宫保鸡丁",
    "老母鸡汤", "泡椒板筋", "泡椒鸡杂", "老三鲜蹄筋面",
    "经典宫保单人餐", "八珍老母鸡汤", "特色小卤拼",
]

DISH_REQUIRED = ["POS销售单号", "菜品名称", "菜品数量", "单据类型", "POS退款单号", "优惠后小计价格"]
DISH_OPTIONAL = ["创建时间", "门店"]
PAY_REQUIRED = ["POS销售单号", "支付类型", "总金额", "POS退款单号"]
PAY_OPTIONAL = ["门店"]

REQUIRED_DISH_COLUMNS = {
    "order_id": "POS销售单号",
    "dish_name": "菜品名称",
    "quantity": "菜品数量",
    "doc_type": "单据类型",
    "refund_order_id": "POS退款单号",
    "amount": "优惠后小计价格",
}

REQUIRED_PAYMENT_COLUMNS = {
    "order_id": "POS销售单号",
    "pay_type": "支付类型",
    "pay_amount": "总金额",
    "refund_order_id": "POS退款单号",
}


@dataclass
class AnalysisResult:
    valid_rows: pd.DataFrame
    order_summary: pd.DataFrame
    daily_summary: pd.DataFrame
    store_summary: pd.DataFrame
    channel_summary: pd.DataFrame
    daily_channel_summary: pd.DataFrame
    store_channel_summary: pd.DataFrame
    no_main_orders: pd.DataFrame
    removed_refund_order_ids: pd.DataFrame
    payment_order_summary: pd.DataFrame


def _clean_col_name(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().replace("\u3000", " ")


def _normalize_order_id(value) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    if re.fullmatch(r"[0-9]+(\.[0-9]+)?e\+?[0-9]+", text, flags=re.I):
        try:
            return str(int(float(text)))
        except Exception:
            return text
    return text


def _to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce").fillna(0)


def _find_header_row(raw: pd.DataFrame, expected_columns: Iterable[str], max_scan_rows: int = 30) -> int:
    expected = set(expected_columns)
    best_idx = 0
    best_hits = -1
    for i in range(min(max_scan_rows, len(raw))):
        row_values = {_clean_col_name(x) for x in raw.iloc[i].tolist()}
        hits = len(expected & row_values)
        if hits > best_hits:
            best_idx = i
            best_hits = hits
    if best_hits <= 0:
        raise ValueError(f"未能在前 {max_scan_rows} 行找到表头。请确认文件是否为正确导出的明细表。")
    return best_idx


def _read_csv_header_row(content: bytes, expected_columns: Sequence[str]) -> tuple[int, str]:
    last_error = None
    for enc in ["utf-8-sig", "gb18030", "gbk", "utf-8"]:
        try:
            raw = pd.read_csv(io.BytesIO(content), header=None, dtype=object, encoding=enc, nrows=30)
            header_row = _find_header_row(raw, expected_columns)
            return header_row, enc
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise ValueError(f"CSV 表头识别失败：{last_error}")


def read_uploaded_table(uploaded_file, expected_columns: Iterable[str], keep_columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """兼容 csv/xls/xlsx，识别 POS 导出中真实表头不在第一行的情况。"""
    if uploaded_file is None:
        return pd.DataFrame()

    name = uploaded_file.name.lower()
    content = uploaded_file.getvalue()
    expected_columns = list(expected_columns)

    if name.endswith(".csv"):
        header_row, enc = _read_csv_header_row(content, expected_columns)
        header_cols = pd.read_csv(io.BytesIO(content), header=header_row, dtype=object, encoding=enc, nrows=0).columns
        header_cols = [_clean_col_name(c) for c in header_cols]
        usecols = None
        if keep_columns:
            keep_set = set(keep_columns)
            usecols = [i for i, c in enumerate(header_cols) if c in keep_set]
        df = pd.read_csv(io.BytesIO(content), header=header_row, dtype=object, encoding=enc, usecols=usecols)
    else:
        raw = pd.read_excel(io.BytesIO(content), header=None, dtype=object, nrows=30)
        header_row = _find_header_row(raw, expected_columns)
        header_cols = pd.read_excel(io.BytesIO(content), header=header_row, dtype=object, nrows=0).columns
        header_cols = [_clean_col_name(c) for c in header_cols]
        usecols = None
        if keep_columns:
            keep_set = set(keep_columns)
            usecols = [i for i, c in enumerate(header_cols) if c in keep_set]
        df = pd.read_excel(io.BytesIO(content), header=header_row, dtype=object, usecols=usecols)

    df.columns = [_clean_col_name(c) for c in df.columns]
    df = df.loc[:, [c for c in df.columns if c]]
    df = df.dropna(how="all")
    df["来源文件"] = uploaded_file.name
    return df


def read_many_uploaded_tables(files, expected_columns: Iterable[str], keep_columns: Sequence[str], table_name: str) -> pd.DataFrame:
    if not files:
        return pd.DataFrame()
    frames = []
    errors = []
    progress = st.progress(0, text=f"正在读取{table_name}：0/{len(files)}")
    for idx, file in enumerate(files, start=1):
        try:
            df = read_uploaded_table(file, expected_columns, keep_columns=keep_columns)
            frames.append(df)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{file.name}: {exc}")
        progress.progress(idx / len(files), text=f"正在读取{table_name}：{idx}/{len(files)}")
    progress.empty()
    if errors:
        raise ValueError(f"{table_name}读取失败：\n" + "\n".join(errors))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, copy=False)


def require_columns(df: pd.DataFrame, required: dict[str, str], table_name: str) -> None:
    missing = [v for v in required.values() if v not in df.columns]
    if missing:
        raise ValueError(f"{table_name}缺少必要字段：{missing}。当前字段：{list(df.columns)}")


def parse_keywords(text: str) -> list[str]:
    raw = []
    for part in re.split(r"[\n,，;；]+", text):
        kw = part.strip()
        if kw and kw not in raw:
            raw.append(kw)

    # 关键优化和口径保护：如果用户同时维护了“宫保猪肝”和“(套)宫保猪肝”，保留较短基础词即可，避免重复计人数。
    # 如果菜名是“泡椒板筋+宫保猪肝”，基础词仍会分别命中两个主菜。
    canonical = []
    for kw in sorted(raw, key=len):
        if not any(existing in kw for existing in canonical):
            canonical.append(kw)
    return sorted(canonical, key=len, reverse=True)


def add_main_dish_columns(valid: pd.DataFrame, keywords: list[str], keep_match_detail: bool) -> pd.DataFrame:
    if valid.empty:
        valid["主菜数量"] = []
        valid["是否主菜行"] = []
        valid["其他菜品数量"] = []
        valid["命中主菜"] = []
        return valid

    names = valid["菜品名称"].fillna("").astype(str)
    hit_count = pd.Series(0, index=valid.index, dtype="int16")
    matched = pd.Series("", index=valid.index, dtype="object") if keep_match_detail else None

    for kw in keywords:
        mask = names.str.contains(re.escape(kw), regex=True, na=False)
        hit_count = hit_count + mask.astype("int16")
        if keep_match_detail and matched is not None:
            matched.loc[mask] = np.where(matched.loc[mask].eq(""), kw, matched.loc[mask] + "+" + kw)

    valid["主菜数量"] = (hit_count.astype("float64") * valid["菜品数量"].astype("float64")).round(6)
    valid["是否主菜行"] = valid["主菜数量"] > 0
    valid["其他菜品数量"] = np.where(valid["是否主菜行"], 0, valid["菜品数量"])
    valid["命中主菜"] = matched if keep_match_detail else ""
    return valid


def summarize_payments(payment_df: pd.DataFrame, exclude_refund_payment_rows: bool = True) -> tuple[pd.DataFrame, set[str]]:
    if payment_df.empty:
        return pd.DataFrame(columns=["POS销售单号", "渠道", "支付总金额", "支付类型数"]), set()

    require_columns(payment_df, REQUIRED_PAYMENT_COLUMNS, "支付明细")
    df = payment_df.copy()
    df["POS销售单号"] = df["POS销售单号"].map(_normalize_order_id)
    df["POS退款单号"] = df["POS退款单号"].map(_normalize_order_id)
    df["总金额"] = _to_number(df["总金额"])
    df["支付类型"] = df["支付类型"].fillna("未知支付类型").astype(str).str.strip().replace("", "未知支付类型")

    has_refund_marker = df["POS退款单号"].notna()
    payment_refunds = set(df.loc[has_refund_marker, "POS销售单号"].dropna().astype(str))
    payment_refunds |= set(df.loc[has_refund_marker, "POS退款单号"].dropna().astype(str))
    if exclude_refund_payment_rows:
        df = df[~has_refund_marker].copy()

    def combine_pay_types(s: pd.Series) -> str:
        vals = sorted({str(x).strip() for x in s if str(x).strip()})
        if not vals:
            return "未知支付类型"
        if len(vals) == 1:
            return vals[0]
        return "多支付：" + "+".join(vals)

    out = (
        df.dropna(subset=["POS销售单号"])
          .groupby("POS销售单号", as_index=False, sort=False)
          .agg(
              渠道=("支付类型", combine_pay_types),
              支付总金额=("总金额", "sum"),
              支付类型数=("支付类型", lambda s: len(set(s.astype(str).str.strip()))),
          )
    )
    return out, payment_refunds


def summarize_orders(order_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if order_df.empty:
        return pd.DataFrame()
    grouped = (
        order_df.groupby(group_cols, dropna=False, as_index=False, sort=True)
        .agg(
            订单数=("POS销售单号", "nunique"),
            总金额=("订单金额", "sum"),
            主菜数量=("主菜数量", "sum"),
            人数=("人数", "sum"),
            其他菜品数量=("其他菜品数量", "sum"),
            无主菜订单数=("无主菜订单", "sum"),
        )
    )
    grouped["客单价"] = np.where(grouped["人数"] > 0, grouped["总金额"] / grouped["人数"], np.nan)
    grouped["单均销售额"] = np.where(grouped["订单数"] > 0, grouped["总金额"] / grouped["订单数"], np.nan)
    return grouped.reset_index(drop=True)


def build_analysis(
    dish_df: pd.DataFrame,
    payment_df: pd.DataFrame,
    main_keywords: list[str],
    use_payment_refunds: bool = True,
    keep_match_detail: bool = True,
) -> AnalysisResult:
    require_columns(dish_df, REQUIRED_DISH_COLUMNS, "菜品明细")

    rows = dish_df.copy()
    rows["POS销售单号"] = rows["POS销售单号"].map(_normalize_order_id)
    rows["POS退款单号"] = rows["POS退款单号"].map(_normalize_order_id)
    rows["菜品名称"] = rows["菜品名称"].astype(str).str.strip()
    rows["菜品数量"] = _to_number(rows["菜品数量"])
    rows["优惠后小计价格"] = _to_number(rows["优惠后小计价格"])
    rows["单据类型"] = rows["单据类型"].fillna("").astype(str).str.strip()

    if "创建时间" in rows.columns:
        rows["创建日期"] = pd.to_datetime(rows["创建时间"], errors="coerce").dt.date
    else:
        rows["创建日期"] = pd.NaT
    if "门店" not in rows.columns:
        rows["门店"] = "未知门店"
    rows["门店"] = rows["门店"].fillna("未知门店").astype(str).str.strip().replace("", "未知门店")

    is_refund_row = rows["单据类型"].str.contains("退款", na=False)

    # POS 实际导出存在两种情况：原销售单号可能在退款行的 POS销售单号，也可能在 POS退款单号。
    refund_order_ids = set(rows.loc[is_refund_row, "POS销售单号"].dropna().astype(str))
    refund_order_ids |= set(rows.loc[is_refund_row, "POS退款单号"].dropna().astype(str))

    payment_summary, payment_refund_order_ids = summarize_payments(payment_df) if not payment_df.empty else (
        pd.DataFrame(columns=["POS销售单号", "渠道", "支付总金额", "支付类型数"]), set()
    )
    if use_payment_refunds:
        refund_order_ids |= payment_refund_order_ids

    valid = rows[
        (~is_refund_row)
        & rows["POS销售单号"].notna()
        & (~rows["POS销售单号"].astype(str).isin(refund_order_ids))
    ].copy()

    valid = add_main_dish_columns(valid, main_keywords, keep_match_detail=keep_match_detail)

    order_summary = (
        valid.groupby("POS销售单号", as_index=False, sort=False)
        .agg(
            日期=("创建日期", "min"),
            门店=("门店", lambda s: "+".join(sorted({str(x) for x in s.dropna()}))),
            订单金额=("优惠后小计价格", "sum"),
            主菜数量=("主菜数量", "sum"),
            人数=("主菜数量", "sum"),
            其他菜品数量=("其他菜品数量", "sum"),
            明细行数=("菜品名称", "size"),
            菜品数量合计=("菜品数量", "sum"),
            主菜名称命中=("命中主菜", lambda s: "+".join(sorted({x for x in s if x}))),
        )
    )
    order_summary["无主菜订单"] = order_summary["人数"] <= 0
    order_summary["订单客单价"] = np.where(order_summary["人数"] > 0, order_summary["订单金额"] / order_summary["人数"], np.nan)

    if not payment_summary.empty:
        order_summary = order_summary.merge(payment_summary, on="POS销售单号", how="left", copy=False)
    else:
        order_summary["渠道"] = np.nan
        order_summary["支付总金额"] = np.nan
        order_summary["支付类型数"] = np.nan
    order_summary["渠道"] = order_summary["渠道"].fillna("未匹配支付类型")

    daily_summary = summarize_orders(order_summary, ["日期"])
    store_summary = summarize_orders(order_summary, ["门店"])
    channel_summary = summarize_orders(order_summary, ["渠道"])
    daily_channel_summary = summarize_orders(order_summary, ["日期", "渠道"])
    store_channel_summary = summarize_orders(order_summary, ["门店", "渠道"])

    no_main_orders = order_summary[order_summary["无主菜订单"]].copy()
    actually_removed = sorted(set(rows.loc[~is_refund_row, "POS销售单号"].dropna().astype(str)) & refund_order_ids)
    removed_refund_order_ids = pd.DataFrame({"被退款剔除的POS销售单号": actually_removed})

    return AnalysisResult(
        valid_rows=valid,
        order_summary=order_summary,
        daily_summary=daily_summary,
        store_summary=store_summary,
        channel_summary=channel_summary,
        daily_channel_summary=daily_channel_summary,
        store_channel_summary=store_channel_summary,
        no_main_orders=no_main_orders,
        removed_refund_order_ids=removed_refund_order_ids,
        payment_order_summary=payment_summary,
    )


def to_excel_bytes(result: AnalysisResult, include_detail: bool = False, max_detail_rows: int = 200000) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        result.daily_summary.to_excel(writer, sheet_name="每日汇总", index=False)
        result.store_summary.to_excel(writer, sheet_name="门店汇总", index=False)
        result.channel_summary.to_excel(writer, sheet_name="渠道汇总", index=False)
        result.daily_channel_summary.to_excel(writer, sheet_name="每日渠道汇总", index=False)
        result.store_channel_summary.to_excel(writer, sheet_name="门店渠道汇总", index=False)
        result.no_main_orders.to_excel(writer, sheet_name="无主菜订单", index=False)
        result.removed_refund_order_ids.to_excel(writer, sheet_name="退款剔除订单", index=False)
        result.payment_order_summary.to_excel(writer, sheet_name="支付聚合", index=False)

        if include_detail:
            result.order_summary.head(max_detail_rows).to_excel(writer, sheet_name="订单明细", index=False)
            result.valid_rows.head(max_detail_rows).to_excel(writer, sheet_name="有效菜品明细", index=False)

        for worksheet in writer.sheets.values():
            worksheet.freeze_panes = "A2"
            for col_cells in worksheet.columns:
                values = [str(c.value) if c.value is not None else "" for c in col_cells[:200]]
                width = min(max(max((len(v) for v in values), default=8) + 2, 10), 36)
                worksheet.column_dimensions[col_cells[0].column_letter].width = width
    return output.getvalue()


def show_preview(df: pd.DataFrame, preview_rows: int) -> None:
    st.caption(f"共 {len(df):,} 行，仅预览前 {min(preview_rows, len(df)):,} 行。")
    st.dataframe(df.head(preview_rows), use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="客单价分析", layout="wide")
    st.title("客单价分析工具")
    st.caption("支持多门店多文件上传；按 POS销售单号 聚合订单，处理退款剔除、主菜人数、无主菜订单金额分摊，并按日期、门店、支付类型统计。")

    with st.sidebar:
        st.header("1. 上传文件")
        dish_files = st.file_uploader(
            "上传菜品明细，可多选（xls / xlsx / csv）",
            type=["xls", "xlsx", "csv"],
            accept_multiple_files=True,
        )
        payment_files = st.file_uploader(
            "上传支付明细，可多选（xls / xlsx / csv）",
            type=["xls", "xlsx", "csv"],
            accept_multiple_files=True,
        )
        use_payment_refunds = st.checkbox("同时使用支付明细中的 POS退款单号 剔除销售单", value=True)

        st.header("2. 主菜维护")
        st.caption("一行一个主菜关键词。菜品名称包含多个关键词时，会按多个主菜计人数。")
        keyword_text = st.text_area("主菜关键词", value="\n".join(DEFAULT_MAIN_KEYWORDS), height=260)
        main_keywords = parse_keywords(keyword_text)
        st.write(f"当前有效主菜关键词数：{len(main_keywords)}")

        st.header("3. 大数据设置")
        preview_rows = st.number_input("页面明细预览行数", min_value=100, max_value=50000, value=1000, step=100)
        keep_match_detail = st.checkbox("保留每行命中的主菜名称（更易核对，但百万行会略慢）", value=True)
        export_detail = st.checkbox("导出订单明细和菜品明细（大数据不建议默认开启）", value=False)
        max_export_detail_rows = st.number_input("明细导出最大行数", min_value=1000, max_value=1000000, value=200000, step=10000)

    if not dish_files:
        st.info("请先上传至少一个菜品明细文件。")
        return

    try:
        dish_keep_cols = DISH_REQUIRED + DISH_OPTIONAL
        pay_keep_cols = PAY_REQUIRED + PAY_OPTIONAL
        dish_df = read_many_uploaded_tables(dish_files, DISH_REQUIRED, dish_keep_cols, "菜品明细")
        payment_df = read_many_uploaded_tables(payment_files, PAY_REQUIRED, pay_keep_cols, "支付明细") if payment_files else pd.DataFrame()
        result = build_analysis(
            dish_df,
            payment_df,
            main_keywords,
            use_payment_refunds=use_payment_refunds,
            keep_match_detail=keep_match_detail,
        )
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
        st.stop()

    st.success(
        f"读取完成：菜品明细 {len(dish_df):,} 行 / {len(dish_files):,} 个文件；"
        f"支付明细 {len(payment_df):,} 行 / {len(payment_files or []):,} 个文件。"
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("订单数", f"{int(result.order_summary['POS销售单号'].nunique()):,}")
    c2.metric("总金额", f"¥{result.order_summary['订单金额'].sum():,.2f}")
    c3.metric("人数/主菜数", f"{int(result.order_summary['人数'].sum()):,}")
    overall_people = result.order_summary["人数"].sum()
    overall_aov = result.order_summary["订单金额"].sum() / overall_people if overall_people else np.nan
    c4.metric("整体客单价", "-" if pd.isna(overall_aov) else f"¥{overall_aov:,.2f}")
    c5.metric("无主菜订单", f"{int(result.no_main_orders['POS销售单号'].nunique()):,}")
    c6.metric("剔除退款订单", f"{len(result.removed_refund_order_ids):,}")

    st.divider()
    tabs = st.tabs(["每日汇总", "门店汇总", "渠道汇总", "每日×渠道", "门店×渠道", "订单明细", "有效菜品明细", "异常/剔除"])

    with tabs[0]:
        st.subheader("每日汇总")
        st.dataframe(result.daily_summary, use_container_width=True, hide_index=True)
    with tabs[1]:
        st.subheader("门店汇总")
        st.dataframe(result.store_summary, use_container_width=True, hide_index=True)
    with tabs[2]:
        st.subheader("渠道汇总")
        st.dataframe(result.channel_summary, use_container_width=True, hide_index=True)
    with tabs[3]:
        st.subheader("每日×渠道汇总")
        st.dataframe(result.daily_channel_summary, use_container_width=True, hide_index=True)
    with tabs[4]:
        st.subheader("门店×渠道汇总")
        st.dataframe(result.store_channel_summary, use_container_width=True, hide_index=True)
    with tabs[5]:
        st.subheader("订单级明细")
        show_preview(result.order_summary, int(preview_rows))
    with tabs[6]:
        st.subheader("有效菜品明细")
        show_preview(result.valid_rows, int(preview_rows))
    with tabs[7]:
        st.subheader("无主菜订单")
        st.caption("这些订单金额已进入日期/门店/渠道总金额，但人数为 0，因此会被平均到整体客单价中。")
        show_preview(result.no_main_orders, int(preview_rows))
        st.subheader("退款剔除订单")
        show_preview(result.removed_refund_order_ids, int(preview_rows))
        st.subheader("支付聚合结果")
        show_preview(result.payment_order_summary, int(preview_rows))

    st.divider()
    excel_bytes = to_excel_bytes(result, include_detail=export_detail, max_detail_rows=int(max_export_detail_rows))
    detail_note = "含部分明细" if export_detail else "仅汇总和异常表"
    st.download_button(
        f"下载分析结果 Excel（{detail_note}）",
        data=excel_bytes,
        file_name="客单价分析结果.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    main()
