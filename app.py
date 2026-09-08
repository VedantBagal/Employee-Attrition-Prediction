import pandas as pd
import streamlit as st
import pickle
import plotly.express as px

@st.cache_data
def load_scatter_data():
    return pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

scatter_df = load_scatter_data()


st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="🧑‍💼",
    layout="wide"
)

with open("attrition_pipeline.pkl", "rb") as f:
    pipeline_data = pickle.load(f)

pipeline = pipeline_data["pipeline"]
threshold = pipeline_data["threshold"]
report = pipeline_data["classification_report"]

st.markdown(
    """
    <h1 style="text-align:center;">🧑‍💼 Employee Attrition Prediction</h1>
    <p style="text-align:center; font-size:18px; color:gray;">
        Predict whether an employee is likely to leave the company
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

i1, i2, i3, i4, i5 = st.columns(5)

with i1:
    st.image("images/count_of_employee_by_department.jpg", caption = "Count of Employees by Department")
with i2:
    st.image("images/attrition_count_by_job_role.jpg", caption = "Attrition count by Job Role")
with i3:
    st.image("images/employee_left_by_gender.jpg", caption = "Count of Employees Who Left by Gender")
with i4:
    st.image("images/monthly_income.jpg", caption = "Monthly income of Employee")
with i5:
    st.image("images/age_vs_monthly_income.jpg", caption = "Age vs Monthly Income")
st.markdown("---")


st.subheader("Input Features")

p1, p2 = st.columns(2)
with p1:
    with st.form("Input Features"):
        c1, c2, c3 = st.columns(3)
        with c1:
            with st.expander("👤 Personal Details", expanded=True):
                age = st.number_input("Age", value = 30, step = 1)
                travel = st.selectbox("Business Travel", ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'])
                distance = st.number_input("Distance from home(in Km)",value = 0, step = 1)
                education = st.selectbox("Education Level", [1, 2, 3, 4, 5])
                education_field = st.selectbox("Education Field", ['Life Sciences', 'Other', 'Medical', 'Marketing','Technical Degree', 'Human Resources'])
                gender = st.selectbox("Gender", ['Male', 'Female'])
                marital_status = st.selectbox("Marital Status", ['Married', 'Divorced', 'Single'])
                relation_satisfaction = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4, 5])
                work_life_balance = st.selectbox("Work LifeBalance", [0, 1, 2, 3, 4, 5])
        with c2:
            with st.expander("💼 Job Details"):
                department = st.selectbox("Department", ['Sales', 'Research & Development', 'Human Resources'])
                satisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4, 5])
                job_involvement = st.selectbox("Job Involvement", [1, 2, 3, 4, 5])
                job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
                job_role = st.selectbox("Job Role", ['Sales Executive', 'Research Scientist', 'Laboratory Technician',
                    'Manufacturing Director', 'Healthcare Representative', 'Manager',
                    'Sales Representative', 'Research Director', 'Human Resources'])
                job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4, 5])
                performance_rating = st.selectbox("Performance Rating", [1, 2, 3, 4, 5])
                companies_worked = st.number_input("No. of Companies Worked Before", step = 1)
                overtime = st.selectbox("Overtime", ['Yes', 'No'])
                percent_salary_hike = st.number_input("Percent Salary Hike")
                years_at_company = st.number_input("Years At Company", value = 0, step = 1)
        with c3:
            with st.expander("Compensation & Workload"):
                hourly_rate = st.number_input("Hourly Rate", step = 1)
                daily_rate = st.number_input("Daily Rate", step = 1)
                monthly_income = st.number_input("Monthly Income", step = 1)
                monthly_rate = st.number_input("Monthly Rate", step = 1)
                stock_option_level = st.selectbox("Stock Option Level", [0, 1, 2, 3, 4, 5])
                working_years = st.number_input("Total Working Years", step = 1)
                last_training_year = st.number_input("Year since last training", step = 1)
                years_current_role = st.number_input("Years Current Role", step = 1)
                last_promotion = st.number_input("Years since Last Promotion", step = 1)
                current_manager = st.number_input("Years with Current Manager", step = 1)

        submit = st.form_submit_button("🔮 Predict Attrition")

pred = None
if submit:
    input_data = {
        "Age" : age,
        "BusinessTravel": travel,
        "DailyRate" : daily_rate,
        "Department": department,
        "DistanceFromHome": distance,
        "Education": education,
        "EducationField": education_field,
        "EnvironmentSatisfaction" : satisfaction,
        "Gender": gender,
        "HourlyRate": hourly_rate,
        "JobRole": job_role,
        "JobInvolvement" : job_involvement,
        "JobLevel": job_level,
        "JobSatisfaction": job_satisfaction,
        "MaritalStatus": marital_status,
        "MonthlyIncome": monthly_income,
        "MonthlyRate": monthly_rate,
        "NumCompaniesWorked" : companies_worked,
        "OverTime": overtime,
        "PercentSalaryHike" : percent_salary_hike,
        "PerformanceRating" : performance_rating,
        "RelationshipSatisfaction" : relation_satisfaction,
        "StockOptionLevel" : stock_option_level,
        "TotalWorkingYears" : working_years,
        "TrainingTimesLastYear" : last_training_year,
        "WorkLifeBalance" : work_life_balance,
        "YearsAtCompany" : years_at_company,
        "YearsInCurrentRole" : years_current_role,
        "YearsSinceLastPromotion" : last_promotion,
        "YearsWithCurrManager" : current_manager
    }

    input_df = pd.DataFrame([input_data])

    prob = pipeline.predict_proba(input_df)[:, 1][0]
    pred = 1 if prob > threshold else 0

    st.markdown("---")

    if pred == 1:
        st.error("⚠️ Employee Likely to LEAVE")
    else:
        st.success("✅ Employee Likely to STAY")

st.markdown("---")
st.subheader("📊 Model Performance (Test Data)")

report_df = pd.DataFrame(report).transpose()
report_df[["precision", "recall", "f1-score"]] *= 100
report_df = report_df.round(2)

st.dataframe(
    report_df,
    use_container_width=True
)

with p2:
    st.markdown("### 📍 Employee Attrition Prediction")
    fig = px.scatter(
        scatter_df,
        x="Age",
        y="MonthlyIncome",
        color="Attrition",
        opacity=0.7,
        labels={
            "Age": "Age",
            "MonthlyIncome": "Monthly Income",
        },
        title="Employee Distribution (Age vs Monthly Income)"
    )

    if submit:
        fig.add_scatter(
            x=[age],
            y=[monthly_income],
            mode="markers",
            marker=dict(
                size=10,
                color="red" if pred == 1 else "green",
                symbol="star",
            ),
            name="Current Employee"
        )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        scatter_df,
        x = "Age",
        y = "PercentSalaryHike",
        color = "Attrition",
        opacity = 0.7,
        labels = {
            "Age": "Age",
            "PercentSalaryHike": "Percent Salary Hike",
        },
        title = "Employee Distribution (Age vs Percent Salary Hike)"
    )

    if submit:
        fig.add_scatter(
            x = [age],
            y = [percent_salary_hike],
            mode = "markers",
            marker = dict(
                    color = "red" if pred == 1 else "green",
                    symbol = "star",
                    size = 10
            ),
            name = "Current Employee"
        )
    st.plotly_chart(fig, use_container_width=True)




