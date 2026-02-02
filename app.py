import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# -----------------------------
# Header Section
# -----------------------------
st.markdown(
    """
    <h1 style='text-align: center; color: #2E86C1;'>🎓 Student Performance Predictor</h1>
    <p style='text-align: center; font-size:18px;'>
        Predict a student's <b>Math Score</b> based on academic and demographic factors
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Info Cards
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("📚 Model Type", "Regression")

with col2:
    st.metric("🧠 ML Pipeline", "End-to-End")

with col3:
    st.metric("🎯 Output", "Math Score")

st.markdown("### 📝 Student Information")

# -----------------------------
# Input Form
# -----------------------------
with st.form("student_form"):
    c1, c2 = st.columns(2)

    with c1:
        gender = st.selectbox("👤 Gender", ["male", "female"])
        race_ethnicity = st.selectbox(
            "🌍 Race / Ethnicity",
            ["group A", "group B", "group C", "group D", "group E"]
        )
        parental_level_of_education = st.selectbox(
            "🎓 Parental Education",
            [
                "some high school",
                "high school",
                "some college",
                "associate's degree",
                "bachelor's degree",
                "master's degree"
            ]
        )

    with c2:
        lunch = st.selectbox("🍱 Lunch Type", ["standard", "free/reduced"])
        test_preparation_course = st.selectbox(
            "📖 Test Preparation Course",
            ["none", "completed"]
        )
        reading_score = st.slider("📘 Reading Score", 0, 100, 50)
        writing_score = st.slider("✍️ Writing Score", 0, 100, 50)

    submit = st.form_submit_button("🚀 Predict Performance")

# -----------------------------
# Prediction Output
# -----------------------------
if submit:
    st.markdown("---")
    st.markdown("## 📊 Prediction Result")

    try:
        data = CustomData(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score
        )

        input_df = data.get_data_as_data_frame()

        with st.expander("🔍 View Input Data"):
            st.dataframe(input_df)

        pipeline = PredictPipeline()
        prediction = pipeline.predict(input_df)[0]

        # Highlighted Result
        st.success("🎯 Prediction Successful!")
        st.markdown(
            f"""
            <div style="
                background-color:#EBF5FB;
                padding:20px;
                border-radius:12px;
                text-align:center;
            ">
                <h2 style="color:#1B4F72;">Predicted Math Score</h2>
                <h1 style="font-size:48px; color:#117864;">{round(prediction, 2)}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error("❌ Prediction failed. Please check inputs.")
        st.exception(e)

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    """
    <hr>
    <p style='text-align:center; font-size:14px;'>
        Built with ❤️ using Machine Learning & Streamlit
    </p>
    """,
    unsafe_allow_html=True
)
