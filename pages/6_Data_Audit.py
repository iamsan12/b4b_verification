import os
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="PRISM Data Audit",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main {background-color: #0E1117; color: #FAFAFA;}
        h1, h2, h3 {color: #00E5FF !important; font-family: 'Inter', sans-serif;}
        .stMetric {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 15px;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def get_csv_files() -> list[Path]:
    app_root = Path(__file__).resolve().parents[1]
    csv_files = []
    for path in app_root.rglob("*.csv"):
        # Skip hidden/system folders if any are introduced later.
        if any(part.startswith(".") for part in path.relative_to(app_root).parts):
            continue
        csv_files.append(path)
    return sorted(csv_files, key=lambda p: str(p.relative_to(app_root)).lower())


st.title("🧾 PRISM Data Audit / Raw Data Viewer")
st.markdown(
    "Use this page to inspect every CSV currently deployed with the Streamlit app. "
    "You can check shapes, columns, missing values, preview rows, and download any dataset."
)
st.markdown("---")

csv_files = get_csv_files()

if not csv_files:
    st.error("No CSV files were found in this deployed repository.")
    st.stop()

app_root = Path(__file__).resolve().parents[1]

# Build summary table once.
summary_rows = []
loaded_data: dict[str, pd.DataFrame | None] = {}
load_errors: dict[str, str] = {}

for path in csv_files:
    rel_path = str(path.relative_to(app_root))
    try:
        df = load_csv(str(path))
        loaded_data[rel_path] = df
        load_errors[rel_path] = ""
        summary_rows.append(
            {
                "file": rel_path,
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "missing_cells": int(df.isna().sum().sum()),
                "duplicate_rows": int(df.duplicated().sum()),
            }
        )
    except Exception as exc:  # Keep the audit page alive even if one CSV fails.
        loaded_data[rel_path] = None
        load_errors[rel_path] = str(exc)
        summary_rows.append(
            {
                "file": rel_path,
                "rows": 0,
                "columns": 0,
                "missing_cells": 0,
                "duplicate_rows": 0,
            }
        )

summary_df = pd.DataFrame(summary_rows)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("CSV files found", len(csv_files))
with col2:
    st.metric("Total rows", int(summary_df["rows"].sum()))
with col3:
    st.metric("Total columns", int(summary_df["columns"].sum()))
with col4:
    st.metric("Load errors", sum(1 for err in load_errors.values() if err))

st.subheader("Dataset summary")
st.dataframe(summary_df, use_container_width=True, hide_index=True)

st.download_button(
    "Download audit summary as CSV",
    data=summary_df.to_csv(index=False),
    file_name="prism_data_audit_summary.csv",
    mime="text/csv",
)

st.markdown("---")
st.subheader("Inspect individual datasets")

selected_files = st.multiselect(
    "Choose CSV files to inspect",
    options=list(loaded_data.keys()),
    default=list(loaded_data.keys())[:5],
)

preview_rows = st.slider("Preview rows per dataset", min_value=5, max_value=200, value=20, step=5)
show_full_data = st.checkbox("Show full dataframes instead of preview only", value=False)
show_missing_details = st.checkbox("Show column-wise missing-value details", value=True)

for rel_path in selected_files:
    df = loaded_data[rel_path]
    with st.expander(f"📄 {rel_path}", expanded=False):
        if df is None:
            st.error(f"Could not load this CSV: {load_errors[rel_path]}")
            continue

        info_col1, info_col2, info_col3 = st.columns(3)
        with info_col1:
            st.metric("Rows", int(df.shape[0]))
        with info_col2:
            st.metric("Columns", int(df.shape[1]))
        with info_col3:
            st.metric("Missing cells", int(df.isna().sum().sum()))

        st.markdown("**Column names**")
        st.code(", ".join(map(str, df.columns)), language="text")

        if show_missing_details:
            missing_df = (
                df.isna()
                .sum()
                .reset_index()
                .rename(columns={"index": "column", 0: "missing_count"})
            )
            missing_df["missing_percent"] = (
                missing_df["missing_count"] / max(len(df), 1) * 100
            ).round(2)
            missing_df = missing_df[missing_df["missing_count"] > 0]
            if len(missing_df) > 0:
                st.markdown("**Columns with missing values**")
                st.dataframe(missing_df, use_container_width=True, hide_index=True)
            else:
                st.success("No missing values found in this file.")

        st.markdown("**Data preview**")
        if show_full_data:
            st.dataframe(df, use_container_width=True)
        else:
            st.dataframe(df.head(preview_rows), use_container_width=True)

        st.download_button(
            label=f"Download {Path(rel_path).name}",
            data=df.to_csv(index=False),
            file_name=Path(rel_path).name,
            mime="text/csv",
            key=f"download-{rel_path}",
        )
