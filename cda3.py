import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

# ==========================================
# Configuration and Setup
# ==========================================
st.set_page_config(
    page_title="Global Socio-Economic Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Common column name for merging
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

    dfs = {}
    master_df = None
    
    # Load individual files
    for name, filename in file_map.items():
        try:
            df = pd.read_csv(filename)
            
            # Clean percentage columns (remove % sign and convert to numeric)
            for col in df.columns:
                if 'percent' in col.lower() or 'growth' in col.lower():
                    df[col] = df[col].astype(str).str.replace('%', '').str.strip()
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
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
    """Returns a list of numeric column names."""
    return df.select_dtypes(include=[np.number]).columns.tolist()

def clean_chart_layout(fig):
    """
    Hides x-axis tick labels (country names) to avoid clutter,
    but keeps the hover info.
    """
    fig.update_xaxes(showticklabels=False, title_text="Countries (Hover for Name)")
    return fig

# ==========================================
# Main App Structure
# ==========================================

def main():
    st.title("🌍 Global Socio-Economic Analysis Dashboard v2.0")
    st.markdown("Advanced EDA, PCA, and Clustering with standardized global datasets.")

    # 1. Load Data
    data_dict, master_df = load_all_data()

    if not data_dict:
        st.error("No data found. Please ensure CSV files are in the directory.")
        return

    # ==========================================
    # Sidebar
    # ==========================================
    with st.sidebar:
        st.header("App Controls")
        sb_tab1, sb_tab2 = st.tabs(["📂 Data & Settings", "ℹ️ Guide"])

        with sb_tab1:
            st.subheader("Missing Value Strategy")
            st.caption("Applied to PCA and K-Means")
            missing_val_strategy = st.radio(
                "Choose Strategy",
                ["Drop Rows", "Fill with Mean", "Fill with 0"]
            )
            
            st.divider()
            st.subheader("Data Preview")
            preview_choice = st.selectbox("Select Dataset", ["Master Merged"] + list(data_dict.keys()))
            if preview_choice == "Master Merged":
                if master_df is not None:
                    st.dataframe(master_df.head())
            else:
                st.dataframe(data_dict[preview_choice].head())

        with sb_tab2:
            st.markdown("""
            **How to use this app:**
            1. **EDA Tab:** Analyze raw data. Select specific files or the merged dataset. Compare countries and create visualizations (Scatter, Bar, Line, Heatmap).
            2. **PCA Tab:** Reduce dimensionality. Choose columns, set the number of components, and visualize how countries group in the abstract space.
            3. **K-Means Tab:** Cluster countries based on similarities. You can cluster on raw data or PCA results.
            """)

    # ==========================================
    # Main Tabs
    # ==========================================
    tab_eda, tab_pca, tab_kmeans = st.tabs(["📊 EDA & Visualization", "🧠 Custom PCA", "🔗 Advanced Clustering"])

    # --------------------------------------------------------
    # TAB 1: Exploratory Data Analysis (EDA)
    # --------------------------------------------------------
    with tab_eda:
        st.header("Exploratory Data Analysis")
        
        # --- 1. Data Source Selection ---
        col_src, col_filt = st.columns([1, 3])
        with col_src:
            data_source = st.radio("Data Source", ["Single File", "Master Merged Data"])
            if data_source == "Single File":
                selected_file_name = st.selectbox("Choose File", list(data_dict.keys()))
                active_df = data_dict[selected_file_name]
            else:
                active_df = master_df

        # --- 2. Advanced Filtering (Include/Exclude) ---
        all_countries = sorted(active_df[COUNTRY_COL].unique().astype(str).tolist()) if COUNTRY_COL in active_df.columns else []
        
        with col_filt:
            with st.expander("🌍 Country Filtering (Include/Exclude)", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    # Include logic
                    include_countries = st.multiselect("Include Specific Countries (Leave empty for all)", all_countries)
                with c2:
                    # Exclude logic
                    exclude_countries = st.multiselect("Exclude Specific Countries", all_countries)
        
        # Apply Logic: Include takes priority, then Exclude removes from that set
        plot_df = active_df.copy()
        if include_countries:
            plot_df = plot_df[plot_df[COUNTRY_COL].isin(include_countries)]
        if exclude_countries:
            plot_df = plot_df[~plot_df[COUNTRY_COL].isin(exclude_countries)]

        # --- 3. Sorting Logic ---
        numeric_cols = get_numeric_columns(plot_df)
        
        st.subheader("Visualization Controls")
        
        # User selections
        c_chart, c_sort, c_ord = st.columns([2, 2, 1])
        with c_chart:
            chart_type = st.selectbox("Chart Type", ["Scatter Plot", "Bar Chart", "Line Chart", "Correlation Heatmap", "Pie Chart", "Table View"])
        with c_sort:
            sort_by_col = st.selectbox("Sort By Column (for graphing)", [COUNTRY_COL] + numeric_cols)
        with c_ord:
            sort_order = st.radio("Order", ["Ascending", "Descending"])

        # Apply Sorting
        ascending_bool = True if sort_order == "Ascending" else False
        plot_df = plot_df.sort_values(by=sort_by_col, ascending=ascending_bool)

        # --- 4. Plot Generation ---
        
        if chart_type == "Table View":
            st.dataframe(plot_df, use_container_width=True)
            
        elif chart_type == "Correlation Heatmap":
            if len(numeric_cols) > 1:
                # Allow user to pick specific columns for correlation
                corr_cols = st.multiselect("Select columns for correlation", numeric_cols, default=numeric_cols[:min(5, len(numeric_cols))])
                if len(corr_cols) > 1:
                    corr = plot_df[corr_cols].corr()
                    fig = px.imshow(corr, text_auto=True, title="Correlation Matrix")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Select at least 2 columns.")
            else:
                st.warning("Not enough numeric data.")

        elif chart_type == "Pie Chart":
            # For Pie, usually we want one metric
            target_col = st.selectbox("Metric for Pie Chart", numeric_cols)
            # Pie charts typically need a limit to be readable
            limit_n = st.slider("Limit slices (Top N)", 5, 50, 15)
            pie_data = plot_df.head(limit_n) if sort_order=="Descending" else plot_df.tail(limit_n)
            
            fig = px.pie(pie_data, values=target_col, names=COUNTRY_COL, title=f"{target_col} Distribution")
            st.plotly_chart(fig, use_container_width=True)

        else:
            # Scatter, Bar, Line (Support Multiple Y)
            c1, c2, c3 = st.columns(3)
            with c1:
                x_axis = st.selectbox("X-Axis (Category)", plot_df.columns, index=plot_df.columns.get_loc(COUNTRY_COL) if COUNTRY_COL in plot_df.columns else 0)
            with c2:
                # Multi-select for Y axis
                y_axis_list = st.multiselect("Y-Axis (Select one or more)", numeric_cols, default=[numeric_cols[0]] if numeric_cols else None)
            with c3:
                # Color is only available if 1 Y is selected, otherwise color maps to variable name
                if len(y_axis_list) == 1:
                    color_encode = st.selectbox("Color Bubble/Line by", [None] + list(plot_df.columns))
                else:
                    color_encode = None
                    st.info("Color disabled for multi-column comparison.")

            if y_axis_list:
                # Base Plotly Express arguments
                common_args = {
                    "data_frame": plot_df,
                    "x": x_axis,
                    "y": y_axis_list, # List allows multi-plotting
                    "hover_name": COUNTRY_COL,
                    "title": f"Analyzing {', '.join(y_axis_list)}"
                }
                
                if chart_type == "Scatter Plot":
                    fig = px.scatter(**common_args, color=color_encode)
                elif chart_type == "Bar Chart":
                    fig = px.bar(**common_args, color=color_encode, barmode='group')
                elif chart_type == "Line Chart":
                    fig = px.line(**common_args, color=color_encode)

                # Clean Layout (Requirement: No country names on axis, but visible on hover)
                fig = clean_chart_layout(fig)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select at least one Y-axis column.")

    # --------------------------------------------------------
    # TAB 2: Principal Component Analysis (PCA)
    # --------------------------------------------------------
    with tab_pca:
        st.header("Customizable PCA")
        
        # --- Data Prep ---
        if master_df is not None:
            df_pca_prep = master_df.set_index(COUNTRY_COL)
            numeric_df = df_pca_prep.select_dtypes(include=[np.number])
            
            # Missing Value Handling
            if missing_val_strategy == "Drop Rows":
                numeric_df_clean = numeric_df.dropna()
            elif missing_val_strategy == "Fill with Mean":
                numeric_df_clean = numeric_df.fillna(numeric_df.mean())
            else:
                numeric_df_clean = numeric_df.fillna(0)
            
            # --- PCA Execution ---
            if numeric_df_clean.shape[0] > 0 and numeric_df_clean.shape[1] > 1:
                n_cols = numeric_df_clean.shape[1]
                n_components = st.slider("Number of PCs", 2, min(n_cols, 10), 3)
                
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(numeric_df_clean)
                
                pca = PCA(n_components=n_components)
                pca_result = pca.fit_transform(scaled_data)
                
                # --- Custom Renaming of PCs ---
                st.subheader("Rename Principal Components")
                st.caption("Give meaningful names to your components (e.g., 'Development Index', 'Safety Index')")
                
                default_names = [f"PC{i+1}" for i in range(n_components)]
                custom_names = []
                
                cols_ren = st.columns(n_components)
                for i, col in enumerate(cols_ren):
                    with col:
                        user_name = st.text_input(f"Name PC{i+1}", value=f"PC{i+1}")
                        custom_names.append(user_name)
                
                # Create Result DataFrame
                df_pca = pd.DataFrame(pca_result, columns=custom_names, index=numeric_df_clean.index)
                
                # --- Analysis of Results ---
                st.divider()
                st.subheader("Analyze PCA Results")
                
                # Filter countries for PCA view
                pca_countries = st.multiselect("Select Countries to view PCA Scores", df_pca.index.tolist())
                
                df_pca_display = df_pca.loc[pca_countries] if pca_countries else df_pca
                
                # View Type
                view_type = st.radio("View Format", ["2D Scatter Chart", "Bar Comparison", "Table View"], horizontal=True)
                
                if view_type == "Table View":
                    st.dataframe(df_pca_display)
                
                elif view_type == "2D Scatter Chart":
                    x_pc = st.selectbox("X Axis Component", custom_names, index=0)
                    y_pc = st.selectbox("Y Axis Component", custom_names, index=1)
                    
                    fig_pca = px.scatter(
                        df_pca_display, x=x_pc, y=y_pc,
                        hover_name=df_pca_display.index,
                        title=f"{x_pc} vs {y_pc}",
                        color=custom_names[0] # Color by first component intensity by default
                    )
                    # Requirement: No names on axis, but hover works
                    fig_pca = clean_chart_layout(fig_pca)
                    st.plotly_chart(fig_pca, use_container_width=True)
                    
                elif view_type == "Bar Comparison":
                    # Visualize multiple countries across components
                    if not pca_countries:
                        st.info("Select specific countries above to see the bar comparison clearly.")
                        # Default top 10
                        df_pca_display = df_pca.head(10)
                        
                    # Reset index to make Country a column for plotting
                    df_reset = df_pca_display.reset_index()
                    fig_bar = px.bar(
                        df_reset, x=COUNTRY_COL, y=custom_names,
                        barmode='group',
                        hover_name=COUNTRY_COL,
                        title="Principal Component Scores by Country"
                    )
                    fig_bar = clean_chart_layout(fig_bar)
                    st.plotly_chart(fig_bar, use_container_width=True)

                # --- Explainability ---
                with st.expander("Show Loadings (What makes up these PCs?)"):
                    loadings = pca.components_.T
                    df_loadings = pd.DataFrame(loadings, columns=custom_names, index=numeric_df_clean.columns)
                    fig_load = px.imshow(df_loadings, text_auto=True, color_continuous_scale="RdBu_r")
                    st.plotly_chart(fig_load)

            else:
                st.error("Insufficient data for PCA.")

    # --------------------------------------------------------
    # TAB 3: K-Means Clustering (Advanced)
    # --------------------------------------------------------
    with tab_kmeans:
        st.header("K-Means Clustering: Optimize & Visualize")
        
        # Data Selection
        c_k1, c_k2 = st.columns(2)
        with c_k1:
            cluster_source = st.selectbox("Clustering Source Data", ["Original Numeric Data", "PCA Transformed Data"])
        
        # Prepare Data
        data_for_clustering = None
        if master_df is not None:
            # Base cleanup
            temp_df = master_df.set_index(COUNTRY_COL).select_dtypes(include=[np.number])
            if missing_val_strategy == "Drop Rows":
                temp_df = temp_df.dropna()
            elif missing_val_strategy == "Fill with Mean":
                temp_df = temp_df.fillna(temp_df.mean())
            else:
                temp_df = temp_df.fillna(0)

            if cluster_source == "Original Numeric Data":
                # Scale Original
                scaler = StandardScaler()
                data_for_clustering = pd.DataFrame(scaler.fit_transform(temp_df), index=temp_df.index, columns=temp_df.columns)
            else:
                # Do PCA first then cluster
                scaler = StandardScaler()
                scaled = scaler.fit_transform(temp_df)
                pca_k = PCA(n_components=3) # Use top 3 for clustering basis
                data_for_clustering = pd.DataFrame(pca_k.fit_transform(scaled), index=temp_df.index, columns=["PC1", "PC2", "PC3"])

        if data_for_clustering is not None and not data_for_clustering.empty:
            
            # --- Requirement: Elbow Method & Silhouette Score ---
            st.subheader("1. Determine Optimal 'k'")
            st.markdown("Use the Elbow Method and Silhouette Scores to decide how many clusters to form.")
            
            k_range = range(2, 11)
            inertias = []
            silhouettes = []
            
            # Calculate metrics
            for k in k_range:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(data_for_clustering)
                inertias.append(km.inertia_)
                silhouettes.append(silhouette_score(data_for_clustering, labels))
            
            col_met1, col_met2 = st.columns(2)
            with col_met1:
                fig_elbow = px.line(x=list(k_range), y=inertias, markers=True, 
                                    labels={'x':'k (Number of Clusters)', 'y':'Inertia'}, title="Elbow Method")
                st.plotly_chart(fig_elbow, use_container_width=True)
            with col_met2:
                fig_sil = px.line(x=list(k_range), y=silhouettes, markers=True, 
                                  labels={'x':'k (Number of Clusters)', 'y':'Silhouette Score'}, title="Silhouette Analysis")
                st.plotly_chart(fig_sil, use_container_width=True)

            # --- Clustering Execution ---
            st.divider()
            st.subheader("2. Run Clustering")
            
            k_value = st.slider("Select k (Number of Clusters)", 2, 10, 3)
            
            kmeans_final = KMeans(n_clusters=k_value, random_state=42, n_init=10)
            clusters = kmeans_final.fit_predict(data_for_clustering)
            
            # Results
            df_results = data_for_clustering.copy()
            df_results['Cluster'] = clusters.astype(str)
            
            # --- Visualization ---
            viz_cols = data_for_clustering.columns.tolist()
            c_vx, c_vy = st.columns(2)
            with c_vx:
                x_axis_clust = st.selectbox("X Axis", viz_cols, index=0)
            with c_vy:
                y_axis_clust = st.selectbox("Y Axis", viz_cols, index=1 if len(viz_cols)>1 else 0)
            
            fig_clust = px.scatter(
                df_results, x=x_axis_clust, y=y_axis_clust, 
                color="Cluster", symbol="Cluster",
                hover_name=df_results.index,
                title=f"Cluster Visualization (k={k_value})"
            )
            fig_clust = clean_chart_layout(fig_clust)
            st.plotly_chart(fig_clust, use_container_width=True)
            
            # --- Analysis ---
            st.subheader("Cluster Details")
            
            # Filter by Cluster
            cluster_select = st.multiselect("Filter by Cluster", df_results['Cluster'].unique())
            if cluster_select:
                df_view = df_results[df_results['Cluster'].isin(cluster_select)]
            else:
                df_view = df_results
            
            # Compare Countries in Clusters
            st.write("Cluster Assignments:")
            st.dataframe(df_view.sort_values("Cluster"))

if __name__ == "__main__":
    main()