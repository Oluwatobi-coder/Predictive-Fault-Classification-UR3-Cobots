# import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# setting page configuration
st.set_page_config(page_title="UR3 CobotOps Diagnostic Tool", layout="wide", page_icon="🤖")

# adding custon CSS styles
st.markdown("""
    <style>
    .main { background-color: #ffffff; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
    </style>
    """, unsafe_allow_html=True)

# loading the model and scaler
@st.cache_resource # to prevent reloading on every button click
def load_assets():
    model = joblib.load('./model_and_scaler/ur3_balanced_model.pkl')
    scaler = joblib.load('./model_and_scaler/scaler.pkl')
    scaler.set_output(transform="pandas")
    return model, scaler

model, scaler = load_assets()


# setting up the title and description
st.title("🤖 UR3 CobotOps: Fault Prediction Tool")
st.markdown("This tool allows operators to simulate real-time sensor inputs from a **UR3 collaborative robot** and predict potential faults using a trained ML model based on the UCI UR3 CobotOps Dataset.")


# setting up the sidebar
st.sidebar.image("./app_assets/ur3cobot.png", 
                 caption="UR3 Collaborative Robot System")

if model and scaler:
    # --- SIDEBAR CONFIG ---
    st.sidebar.header("⚙️ Sensitivity Threshold")
    st.sidebar.markdown("Lower threshold increases sensitivity to potential fault.")
    threshold = st.sidebar.slider(
        "Select Sensitivity", 
        min_value=0.1, max_value=0.9, value=0.48, step=0.05, label_visibility="collapsed")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("System Performance")
    

    # setting up the main container for results
    results_container = st.container()

    st.markdown("---")

    # setting up the analysis button
    run_scan = st.button("Run Diagnostic Scan", use_container_width=True, type="primary")
    
    # setting up the input sliders for joint currents and temperatures
    st.header("Real-Time Sensor Feeds")
    st.markdown("**Adjust** the **joint currents and temperatures** and **run the diagnostic scan**.")
    col1, col2 = st.columns(2)

    # specifying feature names
    feature_names = ['Current_J0', 'Temperature_T0', 'Current_J1', 'Temperature_J1',
       'Current_J2', 'Temperature_J2', 'Current_J3', 'Temperature_J3',
       'Current_J4', 'Temperature_J4', 'Current_J5', 'Temperature_J5']

    with col1:
        st.subheader("Joint Currents (Amps)")
        c0 = st.slider("J0 Base", -2.0, 2.0, 0.1)
        c1 = st.slider("J1 Shoulder", -2.0, 2.0, 0.1)
        c2 = st.slider("J2 Elbow", -2.0, 2.0, 0.1)
        c3 = st.slider("J3 Wrist 1", -2.0, 2.0, 0.1)
        c4 = st.slider("J4 Wrist 2", -2.0, 2.0, 0.1)
        c5 = st.slider("J5 Wrist 3", -2.0, 2.0, 0.1)

    with col2:
        st.subheader("Joint Temperatures (°C)")
        t0 = st.slider("T0 Temp", 20.0, 80.0, 35.0)
        t1 = st.slider("T1 Temp", 20.0, 80.0, 35.0)
        t2 = st.slider("T2 Temp", 20.0, 80.0, 35.0)
        t3 = st.slider("T3 Temp", 20.0, 80.0, 35.0)
        t4 = st.slider("T4 Temp", 20.0, 80.0, 35.0)
        t5 = st.slider("T5 Temp", 20.0, 80.0, 35.0)

    # setting up the input processing and prediction system
    if run_scan:
        # processing the inputs
        raw_input = np.array([[c0, t0, c1, t1, c2, t2, c3, t3, c4, t4, c5, t5]])
        
        raw_input_df = pd.DataFrame(raw_input, columns=feature_names)
        
        # processing: scaling the inputs
        scaled_input = scaler.transform(raw_input_df)
        
        scaled_input_df = pd.DataFrame(scaled_input, columns=feature_names)
        
        # model prediction and latency measurement
        t_start = time.perf_counter()
        prob_fault = model.predict_proba(scaled_input_df)[0][1]
        t_end = time.perf_counter()
        
        latency = (t_end - t_start) * 1000
        
        # displaying the results
        st.markdown("---")

        with results_container:
            st.info("### 🔍 Diagnostic Results")
        
            res_col1, res_col2 = st.columns([1, 2])

            with res_col1:
                if prob_fault >= threshold:
                    st.error("🚨 FAULT DETECTED")
                    status_text = "PROTECTIVE STOP RECOMMENDED"
                else:
                    st.success("✅ SYSTEM HEALTHY")
                    status_text = "NORMAL OPERATION"
            
                st.metric("Fault Probability", f"{prob_fault:.2%}")
                st.write(f"**Action:** {status_text}")

            with res_col2:
                st.subheader("Current Sensor Inputs")
                clean_dict = {
                    'Sensor': [str(i) for i in ['J0', 'J1', 'J2', 'J3', 'J4', 'J5']],
                    'Current': [float(x) for x in [c0, c1, c2, c3, c4, c5]]
                }
                input_df = pd.DataFrame.from_dict(clean_dict)
                
                st.bar_chart(input_df, x="Sensor", y="Current")

        # setting up the performance metrics in the sidebar
        st.sidebar.write(f"Inference Latency: `{latency:.3f} ms`")
        st.sidebar.write(f"Processing Rate: `{1000/latency:.1f} Hz`")
