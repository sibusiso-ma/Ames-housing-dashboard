# Building an advanced Ames Housing dashboard
# Sibusiso Mathebula - Updated Version

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd   
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.title('Property Data Dashboard - Advanced ML & Data Cleaning')

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/sibusiso-ma/Ames-housing-dashboard/main/AmesHousing.csv"
    df = pd.read_csv(url)
    return df



df = load_data()




   
# Data Preview
st.subheader('Data Preview')
st.write(df.head())
st.subheader('Data Summary')
st.write(df.describe(include='all'))

    # Data Filtering
st.subheader('Filter Data')
columns = df.columns.tolist()
selected_col = st.selectbox('Select column to filter by:', columns)
unique_values = df[selected_col].unique()
selected_value = st.selectbox('Select value:', unique_values)

filtered_df = df[df[selected_col] == selected_value]
st.write(filtered_df)

# Histogram Feature Selection
st.subheader('Histogram Distribution of Feature')
numerical_cols = df.select_dtypes(include=['float64','int64']).columns.tolist()
feature = st.selectbox('Select numerical feature:', numerical_cols)

fig, ax = plt.subplots()
ax.hist(df[feature], bins=30)
ax.set_xlabel(feature)
ax.set_ylabel("Count")
ax.set_title(f"Histogram of {feature}")
st.pyplot(fig)

# Correlation Heatmap Optional
if st.checkbox("Show Correlation Heatmap"):
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    st.pyplot(fig)

# Scatter Plot with Correlation Coefficient
st.subheader('Scatter Plot & Correlation Coefficient')
x_column = st.selectbox('Select x-axis column:', numerical_cols, key="scatter_x")
y_column = st.selectbox('Select y-axis column:', numerical_cols, key="scatter_y")

if st.button('Generate Scatter Plot'):
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=x_column, y=y_column, ax=ax)
    st.pyplot(fig)
        
 # Compute and show correlation
    corr_value = df[[x_column, y_column]].corr().iloc[0, 1]
    st.success(f"Correlation coefficient between **{x_column}** and **{y_column}**: **{corr_value:.4f}**")

# Machine Learning Section
st.subheader("Machine Learning Trainer")

target_col = st.selectbox("Select target variable (y):", numerical_cols)
feature_cols = st.multiselect("Select feature columns (X):", numerical_cols)

if st.checkbox("Enable Machine Learning Model Training"):
    if len(feature_cols) > 0:
        X = df[feature_cols]
        y = df[target_col]

        # Train-Test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model Selection
        model_choice = st.selectbox("Choose ML Model:", 
                                        ["Linear Regression", "Lasso", "Ridge", "Random Forest"])

        # Pipeline with Imputer + StandardScaler
        pipeline_steps = [
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]

        if model_choice == "Linear Regression":
                pipeline_steps.append(('model', LinearRegression()))
        elif model_choice == "Lasso":
                pipeline_steps.append(('model', Lasso(alpha=0.1)))
        elif model_choice == "Ridge":
                pipeline_steps.append(('model', Ridge(alpha=1.0)))
        elif model_choice == "Random Forest":
                pipeline_steps.append(('model', RandomForestRegressor(n_estimators=200, random_state=42)))

        pipeline = Pipeline(pipeline_steps)

            # Train model
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

            # Metrics
        st.subheader("Model Performance")
        st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, preds):.4f}")
        st.write(f"Mean Squared Error: {mean_squared_error(y_test, preds):.4f}")
        st.write(f"R² Score: {r2_score(y_test, preds):.4f}")

            # ============================================
            # FEATURE IMPORTANCE SECTION (Dynamic)
            # ============================================
         st.markdown("---")
        st.subheader(f"Feature Importance / Coefficients ({model_choice})")

        model = pipeline.named_steps['model']
        feature_importance = None

        # Handle model type
        if model_choice == "Random Forest":
            feature_importance = model.feature_importances_
        elif model_choice in ["Linear Regression", "Lasso", "Ridge"]:
            feature_importance = np.abs(model.coef_)  # magnitude of coefficients

        if feature_importance is not None:
            importance_df = pd.DataFrame({
                    "Feature": feature_cols,
                    "Importance": feature_importance
                }).sort_values(by="Importance", ascending=False)

            top_n = st.slider("Show top N features:", 3, len(feature_cols), min(10, len(feature_cols)))

            top_features = importance_df.head(top_n)
            st.dataframe(top_features.style.background_gradient(cmap="Blues", subset=["Importance"]))

            # Horizontal barplot
            fig, ax = plt.subplots(figsize=(8, 0.4 * top_n + 1))
            sns.barplot(x="Importance", y="Feature", data=top_features, palette="cool")
            ax.set_title(f"Top {top_n} Important Features for {model_choice}")
            st.pyplot(fig)
    else:
                st.warning("Feature importance not available for this model type.")
 else:
            st.warning("Please select at least ONE feature column!")


