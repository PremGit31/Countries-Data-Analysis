# 🌍 Global Socio-Economic Analysis: Data-Driven Insights

An end-to-end Data Science project exploring the complex relationships between a nation's wealth (GDP), development (HDI), and societal well-being (Life Expectancy & Suicide Rates).

## 🚀 Overview
Does a country's wealth strictly determine its health? This project investigates socio-economic indicators across 150+ countries. Using Exploratory Data Analysis (EDA) and Unsupervised Machine Learning, I identified hidden patterns that challenge common assumptions about national prosperity.

### 📊 Interactive Dashboard
I am currently building a **Streamlit Dashboard** to allow users to filter by region and interact with the clusters.
* **Live App:** [Coming Soon]
* **Google Colab Notebook:** [Insert Link]

---

## 🛠️ Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (PCA for Dimensionality Reduction, K-Means for Clustering)
* **Deployment:** Streamlit

---

## 📈 Project Workflow

### 1. Data Acquisition & Cleaning
* Extracted socio-economic data from Wikipedia tables.
* Resolved country naming inconsistencies (e.g., standardizing "US", "USA", and "United States").
* Handled missing values and performed data type conversion for analytical readiness.

### 2. Exploratory Data Analysis (EDA)
* **Correlation Analysis:** Investigated how HDI relates to health outcomes.
* **Insight:** Found that wealth correlates with health up to a specific threshold, after which other societal factors become more influential.

### 3. Machine Learning Pipeline
* **Feature Scaling:** Standardized data to ensure features like GDP (billions) and Suicide Rates (per 100k) were weighted equally.
* **PCA:** Reduced dimensionality to visualize complex multi-dimensional data in a 2D space.
* **K-Means Clustering:** Grouped countries into distinct clusters based on shared socio-economic profiles rather than geographic location.

---

## 🔑 Key Insights
* **The Wealth-Health Plateau:** Higher GDP does not linearly increase life expectancy beyond a certain point.
* **The "Outlier" Countries:** Identified nations that outperform their economic status in terms of mental health and longevity.
* **Clustering Results:** Successfully grouped nations into 4 distinct categories: [e.g., Developing-High-Growth, Stable-Developed, etc.]

---

## ⚙️ How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/PremGit31/Countries-Data-Analysis.git](https://github.com/PremGit31/Countries-Data-Analysis.git)
   cd Countries-Data-Analysis
