import streamlit as st
import numpy as np
import pandas as pd
import dowhy
from dowhy import CausalModel
import matplotlib.pyplot as plt
import graphviz
import altair as alt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from econml.dml import DML
from econml.dr import DRLearner

# Data generation
@st.cache_data
def generate_churn_data(n_samples):
    np.random.seed(42)
    usage = np.random.normal(10, 3, n_samples)
    sessions = np.random.normal(20, 5, n_samples)
    
    support_tickets = np.random.poisson(np.maximum(0, -0.3 * usage + 5))
    feature_adopt = 1 / (1 + np.exp(-(usage/10 + sessions/20 - 2)))
    feature_adopt = np.random.binomial(1, feature_adopt)
    
    premium_prob = 1 / (1 + np.exp(-(usage/10 - 1)))
    premium_tier = np.random.binomial(1, premium_prob)
    
    churn_prob = 1/(1 + np.exp(-(
        -usage/10 + support_tickets/5 - 
        feature_adopt - premium_tier/2 + 
        np.random.normal(0, 0.5, n_samples)
    )))
    churn = np.random.binomial(1, churn_prob)
    
    return pd.DataFrame({
        'usage_freq': usage,
        'session_length': sessions,
        'support_tickets': support_tickets,
        'feature_adopt': feature_adopt,
        'premium_tier': premium_tier,
        'churn': churn
    })

# Set page config
st.set_page_config(page_title="Causal AI for Churn Analysis", layout="wide")

# Mobile responsiveness
st.markdown("""<style>
    @media (max-width: 600px) {
        .stSelectbox {margin-top: -15px;}
        .stMetric {padding: 10px;}
    }
    </style>""", unsafe_allow_html=True)

# Title and description
st.title("Understanding User Churn with Causal AI")
st.markdown("""
**Learn how to analyze user churn through causal relationships**  
Follow these steps to discover why users leave and how to prevent it:
""")

# Generate initial dataset
data = generate_churn_data(1000)

# Data Dictionary
with st.expander("Data Dictionary"):
    st.markdown("""
    - **usage_freq**: Weekly platform visits
    - **session_length**: Average time spent per visit
    - **support_tickets**: Number of support requests
    - **feature_adopt**: 1=Used key features, 0=Didn't
    - **premium_tier**: 1=Paid plan, 0=Free tier
    - **churn**: 1=User churned, 0=User retained
    """)

# Data Preview
st.subheader("Dataset Preview")
st.dataframe(data.head(10), use_container_width=True)

# Educational sidebar
with st.sidebar:
    st.header("Learning Objectives")
    st.markdown("""
    1. 🧠 Understand key churn drivers
    2. 📈 Build causal relationships
    3. 🔍 Estimate intervention impacts
    4. 🛡️ Validate your findings
    5. 🧪 Run simulations
    """)
    st.markdown("---")
    st.markdown("**Key Concepts**")
    st.markdown("- **Treatment**: Variable you change (e.g., feature adoption)")
    st.markdown("- **Outcome**: Result to measure (churn)")
    st.markdown("- **Confounder**: Affects both treatment and outcome")

# Analysis Assumptions
with st.expander("Important Assumptions"):
    st.markdown("""
    1. **No Unmeasured Confounders**: All variables affecting both treatment and outcome are included
    2. **Causal Graph Validity**: The specified relationships reflect true causal paths
    3. **Linear Treatment Effects**: Impact of interventions scales linearly
    4. **Stable Unit Treatment Value**: One user's treatment doesn't affect others' outcomes
    5. **Data Quality**: Measurements are accurate and representative
    """)

# Data section
st.header("📊 1. Understand Your Data")
with st.expander("Why start with data exploration?"):
    st.markdown("""
    - Identify potential relationships
    - Spot data quality issues
    - Guide causal graph construction
    - Remember: **Correlation ≠ Causation**
    """)

# Data metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Users", len(data))
col2.metric("Churn Rate", f"{data.churn.mean()*100:.1f}%")
col3.metric("Premium Users", f"{data.premium_tier.mean()*100:.1f}%")

# Data visualization
st.subheader("Feature Relationships")
with st.expander("How to read this chart?"):
    st.markdown("""
    This boxplot shows the distribution of each feature for churned (1) vs retained (0) users:
    
    📊 **Chart Elements:**
    - Box: Shows where 50% of the values fall
    - Line in box: Median value
    - Whiskers: Range of typical values
    - Dots: Outliers or unusual values
    
    💡 **Interpretation:**
    - Different box positions suggest the feature affects churn
    - Overlapping boxes suggest weaker relationship
    - More separation indicates stronger predictive power
    """)

feature = st.selectbox("Compare with churn:", data.columns[:-1])
chart = alt.Chart(data).mark_boxplot().encode(
    x='churn:N',
    y=alt.Y(feature, title=feature.replace('_', ' ').title()),
    color='churn:N'
).properties(height=300)
st.altair_chart(chart, use_container_width=True)

# Causal graph section
st.header("📈 2. Build Causal Graph")
with st.expander("What's a causal graph?"):
    st.markdown("""
    - Nodes = Variables
    - Arrows = Causal relationships
    - Helps identify:
      - Direct causes
      - Confounding factors
      - Mediation paths
    """)

default_graph = """digraph {
    usage_freq -> support_tickets
    usage_freq -> feature_adopt
    usage_freq -> premium_tier
    usage_freq -> churn
    session_length -> feature_adopt
    support_tickets -> churn
    feature_adopt -> churn
    premium_tier -> churn
}"""

graph_col, edit_col = st.columns([3,1])
with graph_col:
    st.graphviz_chart(default_graph)
with edit_col:
    if st.checkbox("Edit graph (advanced)"):
        custom_graph = st.text_area("Modify graph (DOT syntax):", default_graph)
        st.graphviz_chart(custom_graph)
    else:
        custom_graph = default_graph

# Causal analysis
st.header("🔍 3. Estimate Causal Effects")
with st.expander("How does causal estimation work?"):
    st.markdown("""
    1. **Identify** relevant factors
    2. **Adjust** for confounders
    3. **Estimate** treatment effect
    4. **Validate** with robustness checks
    """)

treatment = st.selectbox("What intervention to test?", 
                        ['feature_adopt', 'premium_tier', 'support_tickets'])
outcome = 'churn'

# Initialize session state for effect storage
if 'feature_effect' not in st.session_state:
    st.session_state.feature_effect = -0.25  # Default value
if 'support_effect' not in st.session_state:
    st.session_state.support_effect = 0.15   # Default value
if 'historical_results' not in st.session_state:
    st.session_state.historical_results = []

# Method selection
method = st.selectbox("Estimation method", [
    "backdoor.propensity_score_matching",
    "backdoor.linear_regression",
    # "backdoor.econml.dml.DML",  # Double Machine Learning method
    # "backdoor.econml.dr.DRLearner"  # Doubly Robust Learner
])

if st.button("Run Analysis"):
    try:
        with st.spinner("Calculating causal effects..."):
            model = CausalModel(
                data=data,
                graph=custom_graph,
                treatment=treatment,
                outcome=outcome
            )
            
            # Identification
            st.subheader("Identified Relationships")
            estimand = model.identify_effect()
            st.code(str(estimand), language='text')
            
            # Estimation
            st.subheader("Effect Estimation")
            
            # Set up method parameters
            if method == "backdoor.econml.dml.DML":
                method_params = {
                    "init_params": {
                        "model_y": RandomForestRegressor(n_estimators=100, max_depth=5),
                        "model_t": RandomForestClassifier(n_estimators=100, max_depth=5),
                        "model_final": RandomForestRegressor(n_estimators=100, max_depth=5)
                    }
                }
            elif method == "backdoor.econml.dr.DRLearner":
                method_params = {
                    "init_params": {
                        "model_regression": RandomForestRegressor(n_estimators=100, max_depth=5),
                        "model_propensity": RandomForestClassifier(n_estimators=100, max_depth=5),
                        "model_final": RandomForestRegressor(n_estimators=100, max_depth=5),
                        "cv": 3
                    }
                }
            else:
                method_params = {}
                
            estimate = model.estimate_effect(estimand, method_name=method, method_params=method_params)
            effect = estimate.value
            ci = estimate.get_confidence_intervals()
            
            # Store the effects
            if treatment == 'feature_adopt':
                st.session_state.feature_effect = effect
            elif treatment == 'support_tickets':
                st.session_state.support_effect = effect
            
            # Store historical results
            st.session_state.historical_results.append({
                'timestamp': pd.Timestamp.now(),
                'treatment': treatment,
                'method': method,
                'effect': effect
            })
            
            st.metric("Estimated Effect on Churn", 
                     f"{effect:.2%}",
                     delta=f"95% CI: ({ci[0]:.2%}, {ci[1]:.2%})")
            
            # Interpretation
            st.markdown("**What does this mean?**")
            if effect < 0:
                st.success(f"Increasing {treatment.replace('_', ' ')} reduces churn probability")
            else:
                st.error(f"Increasing {treatment.replace('_', ' ')} may increase churn risk")
            
            # Refutation
            st.subheader("Robustness Check")
            refute = model.refute_estimate(
                estimand, estimate,
                method_name="random_common_cause"
            )
            st.write(f"Effect after adding random noise: {refute.new_effect:.2%}")
            st.progress(abs(refute.new_effect / effect) if effect !=0 else 0)
            
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        st.stop()

# Historical comparison
if st.checkbox("Compare to previous analyses") and len(st.session_state.historical_results) > 1:
    hist_df = pd.DataFrame(st.session_state.historical_results)
    hist_chart = alt.Chart(hist_df).mark_line(point=True).encode(
        x='timestamp:T',
        y='effect:Q',
        color='treatment:N',
        shape='method:N'
    ).properties(height=300)
    st.altair_chart(hist_chart, use_container_width=True)

# What-if analysis
st.header("🧪 4. Predict Intervention Impact")
col1, col2 = st.columns(2)
with col1:
    current_adopt = data.feature_adopt.mean()
    new_adopt = st.slider("Target feature adoption rate", 
                         0.0, 1.0, current_adopt)
with col2:
    current_support = data.support_tickets.mean()
    new_support = st.slider("Target support tickets", 
                          0, 10, int(current_support))

# Use actual causal effect estimates
pred_churn = data.churn.mean() * \
             (1 + (new_adopt - current_adopt)*st.session_state.feature_effect) * \
             (1 + (new_support - current_support)*st.session_state.support_effect)

st.subheader("Predicted Outcomes")
col1, col2 = st.columns(2)
col1.metric("Current Churn Rate", f"{data.churn.mean()*100:.1f}%")
col2.metric("Predicted Churn Rate", f"{pred_churn*100:.1f}%", 
           delta=f"{(pred_churn - data.churn.mean())*100:.1f}%")

# Recommendations
st.subheader("Recommended Actions")
if new_adopt > current_adopt:
    st.success("✅ Increase feature adoption through:")
    st.markdown("- Onboarding tutorials\n- Feature highlight campaigns")
if new_support < current_support:
    st.success("✅ Reduce support tickets by:")
    st.markdown("- Improving self-service resources\n- Proactive issue detection")

# Conclusion
st.markdown("---")
st.markdown("""
**Next Steps:**
- Try different treatment variables
- Modify the causal graph
- Upload your own data
- Explore advanced methods
""")
