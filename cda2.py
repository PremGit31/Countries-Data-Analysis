import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

# ==========================================
# Configuration and Setup
# ==========================================
st.set_page_config(
    page_title="Global Socio-Economic Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Common column name for merging (based on user description of standardized data)
COUNTRY_COL = "country_or_territory"

# ==========================================
# Helper Functions
# ==========================================

@st.cache_data
def load_all_data():
    """
    Loads all specified CSV files.
    Returns a dictionary of DataFrames and a merged master DataFrame.
    """
    file_map = {
        "Density": "data/all csv/density.csv",
        "GDP": "data/all csv/GDP_per_capita.csv",
        "Happiness": "data/all csv/happiness.csv",
        "HDI": "data/all csv/HDI.csv",
        "Homicide": "data/all csv/homicideRate.csv",
        "Life Expectancy": "data/all csv/life_expectancy.csv",
        "Population": "data/all csv/populations.csv",
        "Suicide Rate": "data/all csv/Suicide_rate.csv",
        "Unemployment": "data/all csv/unemployment.csv"
    }
    
    # ...existing code...

    dfs = {}
    master_df = None
    
    # Load individual files
    for name, filename in file_map.items():
        try:
            df = pd.read_csv(filename)
            # Ensure country column is string to avoid merge issues
            if COUNTRY_COL in df.columns:
                df[COUNTRY_COL] = df[COUNTRY_COL].astype(str)
            dfs[name] = df
            
            # Merging logic
            if master_df is None:
                master_df = df
            else:
                # Outer merge to keep all countries even if data is missing in some files
                if COUNTRY_COL in master_df.columns and COUNTRY_COL in df.columns:
                    master_df = pd.merge(master_df, df, on=COUNTRY_COL, how='outer')
        except FileNotFoundError:
            st.error(f"File not found: {filename}. Please ensure it is in the directory.")
        except Exception as e:
            st.error(f"Error loading {filename}: {e}")

    return dfs, master_df

def get_numeric_columns(df):
    """Returns a list of numeric column names from a dataframe."""
    return df.select_dtypes(include=[np.number]).columns.tolist()

# ==========================================
# Main App Structure
# ==========================================

def main():
    st.title("🌍 Global Socio-Economic Analysis Dashboard")
    st.markdown("Perform EDA, PCA, and Clustering on standardized global datasets.")

    # 1. Load Data
    data_dict, master_df = load_all_data()

    if not data_dict:
        st.warning("No data loaded. Please check CSV files.")
        return

    # ==========================================
    # Sidebar Logic (3 Tabs as requested)
    # ==========================================
    with st.sidebar:
        st.header("App Controls")
        sb_tab1, sb_tab2, sb_tab3 = st.tabs(["📂 Datasets", "ℹ️ How-to", "⚙️ Settings"])

        with sb_tab1:
            st.subheader("Raw Data Preview")
            dataset_choice = st.selectbox("Select Dataset to View", list(data_dict.keys()))
            if dataset_choice:
                st.dataframe(data_dict[dataset_choice], height=200)
            
            st.subheader("Merged Master Data")
            st.write("Combined data from all CSVs:")
            if master_df is not None:
                st.dataframe(master_df.head(), height=150)
                st.info(f"Master DF Shape: {master_df.shape}")

        with sb_tab2:
            st.markdown("""
            **How to use this app:**
            1. **EDA Tab:** Analyze raw data. Select specific files or the merged dataset. Compare countries and create visualizations (Scatter, Bar, Line, Heatmap).
            2. **PCA Tab:** Reduce dimensionality. Choose columns, set the number of components, and visualize how countries group in the abstract space.
            3. **K-Means Tab:** Cluster countries based on similarities. You can cluster on raw data or PCA results.
            """)

        with sb_tab3:
            st.write("Global configurations")
            missing_val_strategy = st.radio(
                "Missing Value Strategy (for PCA/KMeans)",
                ["Drop Rows (Recommended)", "Fill with Mean", "Fill with 0"]
            )

    # ==========================================
    # Main Tabs (EDA, PCA, KMeans)
    # ==========================================
    tab_eda, tab_pca, tab_kmeans = st.tabs(["📊 Exploratory Data Analysis", "🧠 PCA", "🔗 K-Means Clustering"])

    # --------------------------------------------------------
    # TAB 1: Exploratory Data Analysis (EDA)
    # --------------------------------------------------------
    with tab_eda:
        st.header("Exploratory Data Analysis")
        
        # --- Section 1: Data Selection ---
        col1, col2 = st.columns([1, 3])
        with col1:
            data_source = st.radio("Select Data Source", ["Single File", "Master Merged Data"])
        
        with col2:
            if data_source == "Single File":
                selected_file_name = st.selectbox("Choose File", list(data_dict.keys()))
                active_df = data_dict[selected_file_name]
            else:
                active_df = master_df

        # --- Section 2: Country Filtering ---
        all_countries = active_df[COUNTRY_COL].unique().tolist() if COUNTRY_COL in active_df.columns else []
        selected_countries = st.multiselect("Filter specific countries (Leave empty for all)", all_countries)
        
        if selected_countries:
            plot_df = active_df[active_df[COUNTRY_COL].isin(selected_countries)]
        else:
            plot_df = active_df

        # --- Section 3: Plotting Controls ---
        st.subheader("Visualization")
        chart_type = st.selectbox("Select Chart Type", ["Scatter Plot", "Bar Chart", "Line Chart", "Correlation Heatmap", "Pie Chart"])
        
        numeric_cols = get_numeric_columns(plot_df)

        if chart_type == "Correlation Heatmap":
            if len(numeric_cols) > 1:
                corr = plot_df[numeric_cols].corr()
                fig = px.imshow(corr, text_auto=True, title="Correlation Matrix")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough numeric columns for heatmap.")

        elif chart_type == "Pie Chart":
            target_col = st.selectbox("Select Metric for Pie Chart", numeric_cols)
            # Pie charts can get messy with too many slices
            if len(plot_df) > 20:
                st.warning("Pie chart shows top 20 entries to maintain readability.")
                pie_data = plot_df.nlargest(20, target_col)
            else:
                pie_data = plot_df
            fig = px.pie(pie_data, values=target_col, names=COUNTRY_COL, title=f"{target_col} Distribution")
            st.plotly_chart(fig, use_container_width=True)

        else:
            # For Scatter, Bar, Line
            c1, c2, c3 = st.columns(3)
            with c1:
                x_axis = st.selectbox("X-Axis", plot_df.columns)
            with c2:
                y_axis = st.selectbox("Y-Axis", numeric_cols)
            with c3:
                color_encode = st.selectbox("Color by (Optional)", [None] + list(plot_df.columns))

            if chart_type == "Scatter Plot":
                fig = px.scatter(plot_df, x=x_axis, y=y_axis, color=color_encode, hover_name=COUNTRY_COL, title=f"{y_axis} vs {x_axis}")
            elif chart_type == "Bar Chart":
                fig = px.bar(plot_df, x=x_axis, y=y_axis, color=color_encode, hover_name=COUNTRY_COL, title=f"{y_axis} by {x_axis}")
            elif chart_type == "Line Chart":
                fig = px.line(plot_df, x=x_axis, y=y_axis, color=color_encode, hover_name=COUNTRY_COL, title=f"{y_axis} Trend")
            
            st.plotly_chart(fig, use_container_width=True)

        # --- Section 4: Deep Dive Specific Country ---
        st.markdown("---")
        st.subheader("Single Country Profile")
        target_country = st.selectbox("Select a country to analyze across all columns", all_countries)
        if target_country:
            country_data = active_df[active_df[COUNTRY_COL] == target_country].T
            country_data.columns = ["Value"]
            st.dataframe(country_data)

    # --------------------------------------------------------
    # TAB 2: Principal Component Analysis (PCA)
    # --------------------------------------------------------
    with tab_pca:
        st.header("Principal Component Analysis")
        st.markdown("Dimensionality reduction on the Merged Master Dataset.")

        # Data Prep for PCA
        if master_df is not None:
            # Filter only numeric
            pca_input_df = master_df.set_index(COUNTRY_COL)
            numeric_df = pca_input_df.select_dtypes(include=[np.number])
            
            # Handle Missing Values based on sidebar selection
            if missing_val_strategy == "Drop Rows (Recommended)":
                numeric_df_clean = numeric_df.dropna()
            elif missing_val_strategy == "Fill with Mean":
                numeric_df_clean = numeric_df.fillna(numeric_df.mean())
            else:
                numeric_df_clean = numeric_df.fillna(0)
            
            st.write(f"Data available for PCA: {numeric_df_clean.shape[0]} countries and {numeric_df_clean.shape[1]} features.")

            if numeric_df_clean.shape[0] > 0 and numeric_df_clean.shape[1] > 1:
                # User controls
                n_components = st.slider("Number of Components", 2, min(numeric_df_clean.shape[1], 10), 3)
                
                # Standardization
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(numeric_df_clean)

                # Fit PCA
                pca = PCA(n_components=n_components)
                pca_result = pca.fit_transform(scaled_data)
                
                # Create PCA Dataframe
                cols = [f"PC{i+1}" for i in range(n_components)]
                df_pca = pd.DataFrame(pca_result, columns=cols, index=numeric_df_clean.index)
                
                # Visualization: 2D Scatter
                st.subheader("PCA Visualization (PC1 vs PC2)")
                fig_pca = px.scatter(
                    df_pca, x="PC1", y="PC2", 
                    hover_name=df_pca.index,
                    text=df_pca.index,
                    title="Countries in Reduced PCA Space"
                )
                fig_pca.update_traces(textposition='top center')
                st.plotly_chart(fig_pca, use_container_width=True)

                # Explained Variance
                st.subheader("Explained Variance")
                exp_var = pca.explained_variance_ratio_
                cum_var = np.cumsum(exp_var)
                
                var_df = pd.DataFrame({
                    "Principal Component": cols,
                    "Explained Variance": exp_var,
                    "Cumulative Variance": cum_var
                })
                
                col_v1, col_v2 = st.columns(2)
                with col_v1:
                    fig_scree = px.bar(var_df, x="Principal Component", y="Explained Variance", title="Scree Plot")
                    st.plotly_chart(fig_scree, use_container_width=True)
                with col_v2:
                    fig_cum = px.line(var_df, x="Principal Component", y="Cumulative Variance", markers=True, title="Cumulative Variance")
                    st.plotly_chart(fig_cum, use_container_width=True)

                # Loadings (Feature Importance)
                st.subheader("PCA Loadings (Feature Contributions)")
                loadings = pca.components_.T
                df_loadings = pd.DataFrame(loadings, columns=cols, index=numeric_df_clean.columns)
                
                fig_loadings = px.imshow(df_loadings, text_auto=True, color_continuous_scale="RdBu_r", title="Feature Loadings Heatmap")
                st.plotly_chart(fig_loadings, use_container_width=True)
            
            else:
                st.error("Not enough data points after cleaning (NaN removal) to perform PCA.")
        else:
            st.error("Master dataframe is empty.")

    # --------------------------------------------------------
    # TAB 3: K-Means Clustering
    # --------------------------------------------------------
    with tab_kmeans:
        st.header("K-Means Clustering")
        
        # --- Input Selection ---
        cluster_source = st.radio("Select Data for Clustering", ["Original Numeric Data", "PCA Result Data"])
        
        data_for_clustering = None
        
        if cluster_source == "Original Numeric Data":
            if master_df is not None:
                # Re-using the cleaning logic
                temp_df = master_df.set_index(COUNTRY_COL).select_dtypes(include=[np.number])
                if missing_val_strategy == "Drop Rows (Recommended)":
                    data_for_clustering = temp_df.dropna()
                elif missing_val_strategy == "Fill with Mean":
                    data_for_clustering = temp_df.fillna(temp_df.mean())
                else:
                    data_for_clustering = temp_df.fillna(0)
        else:
            # Check if PCA was run (check if 'df_pca' exists in session state or re-calculate)
            # For simplicity, we assume user goes linearly, but good practice is to guard.
            # We will recalculate PCA here briefly if selected to ensure robustness
            if master_df is not None:
                temp_df = master_df.set_index(COUNTRY_COL).select_dtypes(include=[np.number]).dropna() # Default drop for PCA source
                if not temp_df.empty:
                    scaler = StandardScaler()
                    scaled_temp = scaler.fit_transform(temp_df)
                    # Use a default 3 components for the 'source' of clustering if user chose PCA
                    pca_temp = PCA(n_components=3)
                    res = pca_temp.fit_transform(scaled_temp)
                    data_for_clustering = pd.DataFrame(res, columns=["PC1", "PC2", "PC3"], index=temp_df.index)

        # --- K-Means Model ---
        if data_for_clustering is not None and not data_for_clustering.empty:
            
            st.write(f"Clustering on {data_for_clustering.shape[0]} countries.")
            
            k_value = st.slider("Select Number of Clusters (k)", 2, 10, 4)
            
            # Run KMeans
            # Note: If using original data, we should scale it first. 
            # If using PCA, it's already scaled/transformed, but KMeans relies on Euclidean distance, so scaling is generally good practice.
            scaler_clust = StandardScaler()
            data_scaled_clust = scaler_clust.fit_transform(data_for_clustering)
            
            kmeans = KMeans(n_clusters=k_value, random_state=42)
            clusters = kmeans.fit_predict(data_scaled_clust)
            
            # Create Result DF
            df_results = data_for_clustering.copy()
            df_results['Cluster'] = clusters.astype(str) # Convert to string for categorical plotting
            
            # --- Visualization ---
            st.subheader("Cluster Visualization")
            
            # X and Y selectors for the cluster plot
            c_x, c_y = st.columns(2)
            with c_x:
                x_cluster = st.selectbox("X Axis for Cluster Plot", df_results.columns[:-1], index=0)
            with c_y:
                y_cluster = st.selectbox("Y Axis for Cluster Plot", df_results.columns[:-1], index=1 if len(df_results.columns)>1 else 0)
            
            fig_clust = px.scatter(
                df_results, x=x_cluster, y=y_cluster, 
                color="Cluster", 
                hover_name=df_results.index,
                title=f"K-Means Clustering (k={k_value})",
                symbol="Cluster"
            )
            fig_clust.update_traces(marker=dict(size=10))
            st.plotly_chart(fig_clust, use_container_width=True)

            # --- Analysis of Clusters ---
            st.subheader("Cluster Insights")
            
            # Show countries per cluster
            selected_cluster_view = st.selectbox("View Countries in Cluster", sorted(df_results['Cluster'].unique()))
            cluster_countries = df_results[df_results['Cluster'] == selected_cluster_view].index.tolist()
            st.write(f"**Countries in Cluster {selected_cluster_view}:** {', '.join(cluster_countries)}")

            # Compare Clusters (Mean values)
            st.subheader("Average Values per Cluster")
            cluster_means = df_results.groupby('Cluster').mean()
            st.dataframe(cluster_means.style.background_gradient(cmap="viridis"))
            
            # Compare specific countries within the clustering context
            st.subheader("Compare Countries")
            compare_countries = st.multiselect("Select Countries to Compare Cluster Assignment", df_results.index.tolist())
            if compare_countries:
                st.dataframe(df_results.loc[compare_countries])

        else:
            st.warning("Data not available for clustering. Please check data loading/cleaning steps.")

if __name__ == "__main__":
    main()