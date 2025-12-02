import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Page Config
st.set_page_config(
    page_title="Refinery Yield Optimizer",
    page_icon="🏭",
    layout="wide"
)

# Load & Train Model (with caching)
@st.cache_resource
def load_and_train_model():
    """Load data and train Random Forest model"""
    # Load data
    df = pd.read_csv('data/data.csv')
    
    # Prepare features
    X = df.drop(columns=['Butane_Yield', 'Bottom_Temp_Red'])
    y = df['Butane_Yield']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    return model, X.columns.tolist(), X_test, y_test

# Load model
model, feature_names, X_test, y_test = load_and_train_model()

# Sidebar Inputs
st.sidebar.header("🎛️ Process Conditions")
st.sidebar.markdown("Adjust the sliders to set operating parameters:")

user_inputs = {}
for feature in feature_names:
    user_inputs[feature] = st.sidebar.slider(
        f"{feature.replace('_', ' ')}",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01
    )

# Make Prediction
input_df = pd.DataFrame([user_inputs])
prediction = model.predict(input_df)[0]

# Display Prediction
st.title("🏭 Refinery Yield Optimizer")
st.markdown("### Real-time Butane Impurity Prediction")

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.metric(
        label="Predicted Butane Yield",
        value=f"{prediction:.4f}",
        delta="✅ Within Spec" if prediction < 0.3 else "⚠️ High Impurity",
        delta_color="normal" if prediction < 0.3 else "inverse"
    )

with col2:
    # Status indicator
    if prediction < 0.2:
        st.success("Excellent Quality")
    elif prediction < 0.3:
        st.info("Acceptable Range")
    else:
        st.warning("Review Settings")

with col3:
    st.metric(
        label="Model Accuracy",
        value="76.5%",
        delta="R² Score"
    )

# Tabs for Visualizations
tab1, tab2, tab3 = st.tabs(["📊 Feature Importance", "🗺️ 3D Surface", "✅ Model Validation"])

with tab1:
    st.subheader("Feature Importance Analysis")
    
    # Plot feature importance
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feature_importance_df, x='Importance', y='Feature', ax=ax, palette='viridis')
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_ylabel('Process Variable', fontsize=12)
    ax.set_title('Random Forest Feature Importance', fontsize=14, fontweight='bold')
    st.pyplot(fig)
    
    st.markdown("""
    **Engineering Insight:** `Flow_Next` (draw-off rate) is the dominant variable, 
    controlling residence time on the trays. This aligns with distillation theory where 
    liquid holdup directly impacts mass transfer efficiency.
    """)

with tab2:
    st.subheader("3D Optimization Surface")
    
    try:
        # Load the pre-generated 3D surface
        with open('images/optimization_surface_3d.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=600, scrolling=True)
        
        st.markdown("""
        **How to use:** Rotate the surface to identify the "valley" region where butane 
        impurity is minimized. The optimal operating window shows the trade-off between 
        temperature and reflux flow.
        """)
    except FileNotFoundError:
        st.error("3D surface file not found. Please run the notebook to generate it.")

with tab3:
    st.subheader("Model Performance")
    
    # Predicted vs Actual scatter
    y_pred_test = model.predict(X_test)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred_test, alpha=0.6, s=50, color='blue', 
               edgecolors='black', linewidth=0.5, label='Predictions')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Butane Yield', fontsize=12)
    ax.set_ylabel('Predicted Butane Yield', fontsize=12)
    ax.set_title('Model Validation: Predicted vs Actual', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Display R² score
    r2 = r2_score(y_test, y_pred_test)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R² Score", f"{r2:.3f}")
    with col2:
        mae = np.mean(np.abs(y_test - y_pred_test))
        st.metric("MAE", f"{mae:.4f}")
    with col3:
        st.metric("Test Samples", len(y_test))

# Engineering Context Expander
with st.expander("ℹ️ Understanding Your Prediction"):
    st.markdown(f"""
    ### What does {prediction:.4f} mean?
    
    - **Butane Yield** represents the impurity level in the bottom product
    - **Lower is better:** Values below 0.3 indicate acceptable separation
    - **Your current settings:** {'✅ Optimal' if prediction < 0.2 else '⚠️ Needs adjustment'}
    
    ### Key Control Variables:
    1. **Flow_Next (Draw-off Rate):** Controls residence time on distillation trays
    2. **Reflux_Flow:** Controls separation efficiency via internal L/V ratio
    
    ### Recommended Actions:
    {
        "✅ Maintain current settings - operating in optimal range" if prediction < 0.2 
        else "⚠️ Consider increasing Reflux_Flow or reducing Flow_Next to improve separation"
    }
    
    ### Process Context:
    This is a **Debutanizer Column** that separates butane from heavier hydrocarbons. 
    The model uses Random Forest regression trained on ~2,400 industrial process states.
    """)

# Footer
st.markdown("---")
st.markdown("""
**Project:** Refinery Yield Optimization | **Model:** Random Forest (R² = 0.765) | 
**GitHub:** [Xanderflux/Refinery-Yield-Optimization](https://github.com/Xanderflux/Refinery-Yield-Optimization)
""")
