import streamlit as st

st.set_page_config(
    page_title="AIE10 — Ensemble Methods",
    page_icon="🌳",
    layout="wide",
)

st.title("Ensemble Methods — Animated Walkthroughs")
st.caption("Applied Artificial Intelligence (AIE10)")

st.markdown(
    """
Step through each algorithm visually. Use the sidebar to navigate between topics.

| # | Topic | What you'll see |
|---|-------|-----------------|
| 1 | **Decision Trees** | Recursive splitting on a 2D dataset, growing one node at a time |
| 2 | **Bagging** | Bootstrap samples training parallel trees; votes aggregate |
| 3 | **Boosting (AdaBoost)** | Sample weights update as weak learners are added sequentially |
| 4 | **Stacking** | Base learners feed predictions into a meta-learner |
| 5 | **XGBoost** | Additive trees fit on gradients of the residuals |

Each page includes a **play** control, a **step** slider, and a short written explanation of what is happening at the current frame.
"""
)

st.divider()
st.subheader("How to read these visualizations")
st.markdown(
    """
- The **decision boundary** is shaded by predicted class probability.
- **Training points** are colored by true label; misclassified points get a red outline.
- For boosting, **point size** reflects the current sample weight.
- For bagging/stacking, each panel shows one base learner; the final panel shows the ensemble.
"""
)
