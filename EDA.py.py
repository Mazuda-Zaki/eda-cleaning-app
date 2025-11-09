import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import requests
from matplotlib import cm

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(page_title="EDA & Data Cleaning Tool", page_icon="📊", layout="wide")

st.markdown("""
<style>
.big-font {font-size:20px !important; font-weight:bold; color:#1f77b4;}
.info-box {background-color:#e7f3ff; padding:20px; border-radius:10px; border-left:5px solid #1f77b4;}
</style>
""", unsafe_allow_html=True)

st.title("📊 Exploratory Data Analysis & Data Cleaning Tool")
st.markdown("### Welcome! Analyze and clean your dataset easily.")
st.markdown("---")

# ==================== SESSION STATE INITIALIZATION ====================
for key in ["df", "cleaned_df", "cleaning_applied"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "cleaning_applied" else False

# ==================== FUNCTIONS ====================
def load_data(file):
    """Auto-detect file type and delimiter for CSV or Excel files."""
    try:
        ext = file.name.split('.')[-1].lower()
        if ext == 'csv':
            for sep in [',', ';', '\t', '|']:
                try:
                    file.seek(0)
                    df = pd.read_csv(file, sep=sep)
                    if df.shape[1] > 1: break
                except Exception:
                    continue
        elif ext in ['xlsx', 'xls']:
            df = pd.read_excel(file)
        else:
            st.error("❌ Unsupported file format!")
            return None

        if df.empty:
            st.error("❌ The file appears empty!")
            return None
        return df
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        return None


def download_button(df, filename, label):
    """Reusable download button for cleaned datasets or plots."""
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(label=label, data=csv, file_name=filename, mime='text/csv', use_container_width=True)


def show_basic_info(df):
    """Display dataset info, missing and duplicate summaries."""
    st.subheader("📋 Dataset Information")
    info = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes,
        'Non-Null': df.count(),
        'Null': df.isnull().sum(),
        'Unique': [df[col].nunique() for col in df.columns]
    })
    st.dataframe(info, use_container_width=True)

    total_missing = df.isnull().sum().sum()
    dupes = df.duplicated().sum()
    col1, col2 = st.columns([1, 1])  # Equal column widths
    with col1:
        st.metric("Missing Values", total_missing,help="Number of missing values in the dataset")
    with col2:
        st.metric("Duplicate Rows", dupes,help="Number of duplicate rows in the dataset")
    if total_missing:
        st.warning(f"⚠️ There are {total_missing} missing values in the dataset.")
    if dupes:
        st.warning(f"⚠️ There are {dupes} duplicate rows in the dataset.")



    #st.metric("Missing Values", total_missing)
    #st.metric("Duplicate Rows", dupes)
    #if total_missing: st.warning(f"{total_missing} missing values found.")
    #if dupes: st.warning(f"{dupes} duplicate rows found.")


def plot_chart(df, x_col, y_col, chart_type):
    """Generate line or bar charts."""
    fig, ax = plt.subplots(figsize=(10, 6))
    if chart_type == "line":
        ax.plot(df[x_col], df[y_col], marker='o', color='royalblue')
        ax.set_title(f"Line Chart: {y_col} vs {x_col}")
    elif chart_type == "bar":
        if pd.api.types.is_numeric_dtype(df[y_col]):
            # Group data and create a bar chart with a color gradient
            grouped = df.groupby(x_col)[y_col].sum().sort_values(ascending=False)

            # Using a colormap (e.g., 'coolwarm' for a nice gradient effect)
            cmap = plt.get_cmap('coolwarm')  # You can try 'viridis', 'plasma', 'inferno', etc.
            norm = plt.Normalize(vmin=grouped.min(), vmax=grouped.max())  # Normalize data for color scaling
            bar_colors = [cmap(norm(val)) for val in grouped.values]  # Map each bar to a color

            ax.bar(grouped.index, grouped.values, color=bar_colors)
        else:
            counts = df[x_col].value_counts()
            colors = sns.color_palette("Set3", len(counts))
            ax.bar(counts.index, counts.values, color=colors)
        ax.set_title(f"Bar Chart: {y_col} by {x_col}")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    plt.xticks(rotation=45)
    st.pyplot(fig)


def clean_data(df, method):
    """Handle data cleaning options."""
    cleaned = df.copy()   # Create a copy of the original DataFrame to avoid modifying `df`
    # Check the method and apply the corresponding cleaning action
    if method == "remove_missing":
        cleaned.dropna(inplace=True) # Remove rows with missing values (NaN)
    # Fill missing values based on column type
    elif method == "fill_missing":
        for col in cleaned.select_dtypes(include='object'):
            cleaned[col].fillna('Unknown', inplace=True) # Fill NaN in object columns with 'Unknown'
        for col in cleaned.select_dtypes(include=np.number):
            cleaned[col] = cleaned[col].interpolate().fillna(0)  # Interpolate NaNs and fill remaining NaNs with 0
    elif method == "remove_duplicates": 
        cleaned.drop_duplicates(inplace=True)  # Remove duplicate rows
    # Apply both "fill_missing" and "remove_duplicates"
    elif method == "apply_all":
        cleaned = clean_data(df, "fill_missing")
        cleaned.drop_duplicates(inplace=True)
    return cleaned

# ==================== FILE UPLOAD ====================

url = "https://raw.githubusercontent.com/Mazuda-Zaki/datasets/refs/heads/main/test_sales_data.csv"

st.info("💡 Don't have a dataset? Download a sample CSV below:")
st.download_button("📥 Download Sample CSV", requests.get(url).content, "sample_sales_data.csv", "text/csv")
st.header("📁 Step 1: Upload Your Dataset")
uploaded_file = st.file_uploader("Choose CSV or Excel", type=['csv', 'xlsx', 'xls'])
if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        st.session_state.df = df
        st.session_state.cleaned_df = df.copy()
        st.success("✅ File loaded successfully!")
        st.subheader("📋 Preview of Your Data (First 10 Rows)")
        st.dataframe(df.head(10), use_container_width=True)
        show_basic_info(df)

        # ==================== VISUALIZATION ====================
        st.header("📈 Step 2: Visualization")
        x_col = st.selectbox("Select X-axis", df.columns)
        y_col = st.selectbox("Select Y-axis", df.columns)
        chart_type = st.radio("Choose Chart Type", ["line", "bar"], horizontal=True)
        if st.button("Generate Chart"):
            plot_chart(df, x_col, y_col, chart_type)

        # ==================== CLEANING ====================
        st.header("🧹 Step 3: Cleaning Options")
        method = st.radio(
            "Choose Cleaning Method",
            ["remove_missing", "fill_missing", "remove_duplicates", "apply_all"],
            format_func=lambda x: {
                "remove_missing": "Remove Missing Values",
                "fill_missing": "Fill Missing Values",
                "remove_duplicates": "Remove Duplicates",
                "apply_all": "Apply All Methods"
            }[x]
        )
        if st.button("Apply Cleaning"):
            cleaned = clean_data(df, method)
            st.session_state.cleaned_df = cleaned
            st.success("✅ Cleaning complete!")
            st.dataframe(cleaned.head(10), use_container_width=True)
            download_button(cleaned, f"{method}_cleaned.csv", "📥 Download Cleaned CSV")

else:
    st.info("👆 Please upload a dataset to start.")
    st.markdown("""
    #### 🌟 Features
    - 📊 Data summary (types, nulls, duplicates)
    - 📈 Quick visualizations (line, bar)
    - 🧹 One-click cleaning (fill, drop, deduplicate)
    - 💾 Download ready-to-use cleaned files
    """)

# ==================== FOOTER ====================
st.markdown("---")
st.caption("📊 Built with Streamlit — EDA & Cleaning Tool | © 2025")

