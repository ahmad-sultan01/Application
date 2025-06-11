import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

# Set dark theme config
st.set_page_config(
    page_title="Coal Power Plant Audit",
    page_icon="🌍",
    layout="wide"
)

# Apply dark theme
def set_dark_theme():
    st.markdown("""
    <style>
        /* Main page styling */
        .stApp {
            background-color: #0E1117;
            color: white;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #1a1a1a;
            color: white;
        }
        
        /* Text color */
        .css-1aumxhk, .css-1v0mbdj, .css-1q8dd3e, .css-1lcbmhc, .css-1outpf7 {
            color: white !important;
        }
        
        /* Input fields */
        .stTextInput>div>div>input, 
        .stNumberInput>div>div>input,
        .stSelectbox>div>div>select {
            background-color: #1a1a1a;
            color: white;
            border-color: #333;
        }
        
        /* Buttons */
        .stButton>button {
            background-color: #2e7bcf;
            color: white;
            border: none;
        }
        
        .stButton>button:hover {
            background-color: #1a5a9a;
        }
        
        /* Dataframe styling */
        .dataframe {
            background-color: #1a1a1a !important;
            color: white !important;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab"] {
            background-color: #1a1a1a;
            color: white;
            border-color: #333;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #2e7bcf;
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Set matplotlib to dark mode
    plt.style.use('dark_background')
    sns.set_style("darkgrid")

# Utility functions (unchanged)
def calculate_metrics(coal_flow, gcv, steam_flow, h_steam, h_feed,
                      power_output, flue_temp, ambient_temp):
    """
    Calculate boiler and plant performance metrics.
    """
    # Convert inputs to numeric, handling potential None/NaN values
    coal_flow = pd.to_numeric(coal_flow, errors='coerce') if pd.notna(coal_flow) else 0
    gcv = pd.to_numeric(gcv, errors='coerce') if pd.notna(gcv) else 0
    steam_flow = pd.to_numeric(steam_flow, errors='coerce') if pd.notna(steam_flow) else 0
    h_steam = pd.to_numeric(h_steam, errors='coerce') if pd.notna(h_steam) else 0
    h_feed = pd.to_numeric(h_feed, errors='coerce') if pd.notna(h_feed) else 0
    power_output = pd.to_numeric(power_output, errors='coerce') if pd.notna(power_output) else 0
    flue_temp = pd.to_numeric(flue_temp, errors='coerce') if pd.notna(flue_temp) else 0
    ambient_temp = pd.to_numeric(ambient_temp, errors='coerce') if pd.notna(ambient_temp) else 0

    heat_input = coal_flow * gcv
    steam_energy = steam_flow * (h_steam - h_feed)

    boiler_efficiency = (steam_energy / heat_input) * 100 if heat_input else 0
    heat_rate = heat_input / power_output if power_output else 0
    sfc = coal_flow / power_output if power_output else 0

    # Constants for flue gas loss calculation
    cp_flue_gas = 0.24  # kcal/kg°C
    flue_gas_flow = 1.5 * coal_flow  # Simplified estimation
    flue_gas_loss = ((flue_temp - ambient_temp) * cp_flue_gas * flue_gas_flow) / heat_input * 100 if heat_input else 0

    # CO2 emissions factor for coal (approximate)
    co2_emissions = coal_flow * 2.32  # kg CO2 per kg coal

    return {
        "Boiler Efficiency (%)": round(boiler_efficiency, 2),
        "Plant Heat Rate (kcal/kWh)": round(heat_rate, 2),
        "Specific Fuel Consumption (kg/kWh)": round(sfc, 4),
        "Flue Gas Loss (%)": round(flue_gas_loss, 2),
        "CO2 Emissions (kg/hr)": round(co2_emissions, 2)
    }

def generate_recommendations(metrics):
    rec = []

    be = metrics.get("Boiler Efficiency (%)", 0)
    if be > 85:
        rec.append("✅ **Boiler Efficiency (Avg: {:.2f}%)**: Excellent. Maintain current operation and schedule routine maintenance.".format(be))
    elif 70 <= be <= 85:
        rec.append("⚠️ **Boiler Efficiency (Avg: {:.2f}%)**: Good, but room for improvement. Optimize excess air supply, clean heat transfer surfaces.".format(be))
    else:
        rec.append("❌ **Boiler Efficiency (Avg: {:.2f}%)**: Inefficient. Check for incomplete combustion, poor coal quality, leaks.".format(be))

    hr = metrics.get("Plant Heat Rate (kcal/kWh)", 0)
    if hr < 2500 and hr > 0:
        rec.append("✅ **Plant Heat Rate (Avg: {:.2f} kcal/kWh)**: Efficient. Maintain load management.".format(hr))
    elif 2500 <= hr <= 3000:
        rec.append("⚠️ **Plant Heat Rate (Avg: {:.2f} kcal/kWh)**: Average. Inspect turbine sealing, condenser vacuum.".format(hr))
    else:
        rec.append("❌ **Plant Heat Rate (Avg: {:.2f} kcal/kWh)**: Inefficient. Audit heat exchangers, check turbine performance.".format(hr))

    sfc = metrics.get("Specific Fuel Consumption (kg/kWh)", 0)
    if sfc < 0.6 and sfc > 0:
        rec.append("✅ **Specific Fuel Consumption (Avg: {:.2f} kg/kWh)**: Efficient. Ensure consistent coal quality.".format(sfc))
    elif 0.6 <= sfc <= 0.75:
        rec.append("⚠️ **Specific Fuel Consumption (Avg: {:.2f} kg/kWh)**: Acceptable. Verify air-fuel ratio.".format(sfc))
    else:
        rec.append("❌ **Specific Fuel Consumption (Avg: {:.2f} kg/kWh)**: High. Recommend coal quality improvement.".format(sfc))

    fl = metrics.get("Flue Gas Loss (%)", 0)
    if fl < 5 and fl > 0:
        rec.append("✅ **Flue Gas Loss (Avg: {:.2f}%)**: Optimal flue gas recovery.".format(fl))
    elif 5 <= fl <= 10:
        rec.append("⚠️ **Flue Gas Loss (Avg: {:.2f}%)**: Moderate loss. Consider preheating combustion air.".format(fl))
    else:
        rec.append("❌ **Flue Gas Loss (Avg: {:.2f}%)**: High heat loss. Suggest urgent flue gas heat recovery.".format(fl))

    co2 = metrics.get("CO2 Emissions (kg/hr)", 0)
    if co2 > 8000:
        rec.append("⚠️ **CO₂ Emissions (Avg: {:.2f} kg/hr)**: High CO₂ emissions. Explore cleaner fuels or carbon capture.".format(co2))
    elif co2 > 0:
        rec.append("✅ **CO₂ Emissions (Avg: {:.2f} kg/hr)**: Monitor CO₂ emissions regularly.".format(co2))
    
    return rec

# Main app with dark theme
def main():
    set_dark_theme()
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Single Audit Calculator", "CSV Batch Analysis", "Performance Dashboard"])

    if page == "Home":
        st.title("🌍 Coal Combustion Power Plant Energy Audit Tool")
        st.markdown("""
        Welcome to the **Coal Energy Audit App**.

        Use the navigation to:
        - 🧮 Perform manual audits  
        - 📁 Upload CSVs for batch analysis  
        - 📊 View performance & emissions  
        """)

    elif page == "Single Audit Calculator":
        st.title("🧮 Manual Audit Tool")
        
        with st.form("input_form"):
            st.subheader("Plant Input Parameters")
            col1, col2 = st.columns(2)

            with col1:
                coal_flow = st.number_input("Coal Flow (kg/hr)", min_value=0.0, value=100.0)
                gcv = st.number_input("GCV of Coal (kcal/kg)", min_value=0.0, value=5000.0)
                power_output = st.number_input("Power Output (kW)", min_value=0.0, value=200.0)
                flue_temp = st.number_input("Flue Gas Temp (°C)", min_value=0.0, value=150.0)

            with col2:
                steam_flow = st.number_input("Steam Output (kg/hr)", min_value=0.0, value=400.0)
                h_steam = st.number_input("Steam Enthalpy (kcal/kg)", min_value=0.0, value=750.0)
                h_feed = st.number_input("Feedwater Enthalpy (kcal/kg)", min_value=0.0, value=100.0)
                ambient_temp = st.number_input("Ambient Temp (°C)", min_value=0.0, value=25.0)

            submitted = st.form_submit_button("Calculate")

        if submitted:
            results = calculate_metrics(coal_flow, gcv, steam_flow, h_steam, h_feed,
                                      power_output, flue_temp, ambient_temp)

            st.subheader("✅ Calculated Results")
            results_df = pd.DataFrame([results]).T.rename(columns={0: "Value"})
            results_df.index.name = "Metric"
            st.dataframe(results_df)

            st.subheader("📊 Visualizations")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=list(results.keys()), y=list(results.values()), palette='viridis', ax=ax)
            ax.set_title("Performance Metrics Overview")
            ax.set_ylabel("Value")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("💡 Performance Recommendations")
            recommendations = generate_recommendations(results)
            for rec in recommendations:
                st.markdown(rec)

    elif page == "CSV Batch Analysis":
        st.title("📂 CSV Audit - Batch Mode")
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

        required_columns = [
            "Coal Flow", "GCV", "Steam Flow", "Steam Enthalpy",
            "Feedwater Enthalpy", "Power Output", "Flue Temp", "Ambient Temp"
        ]

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.subheader("📋 Uploaded Data")
                st.dataframe(df.head())

                missing_cols = [col for col in required_columns if col not in df.columns]
                if missing_cols:
                    st.error(f"❌ CSV must have: {', '.join(missing_cols)}")
                else:
                    with st.spinner("🔄 Processing..."):
                        for col in required_columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        df.dropna(subset=required_columns, inplace=True)

                        if df.empty:
                            st.warning("No valid data rows after cleaning.")
                        else:
                            results = []
                            for _, row in df.iterrows():
                                r = calculate_metrics(
                                    row["Coal Flow"], row["GCV"], row["Steam Flow"],
                                    row["Steam Enthalpy"], row["Feedwater Enthalpy"],
                                    row["Power Output"], row["Flue Temp"], row["Ambient Temp"]
                                )
                                results.append(r)

                            result_df = pd.DataFrame(results)
                            final_df = pd.concat([df.reset_index(drop=True), result_df.reset_index(drop=True)], axis=1)

                            st.subheader("📊 Audit Results")
                            st.dataframe(final_df)

                            st.download_button("📥 Download Results", final_df.to_csv(index=False),
                                             file_name="audit_results.csv", mime="text/csv")

                            st.subheader("📈 Correlation Heatmap")
                            fig, ax = plt.subplots(figsize=(10, 8))
                            sns.heatmap(final_df.corr(numeric_only=True), ax=ax, annot=True, cmap="coolwarm")
                            st.pyplot(fig)
                            plt.close(fig)

                            st.subheader("💡 Batch Performance Recommendations")
                            avg_metrics = final_df[result_df.columns].mean().to_dict()
                            recommendations = generate_recommendations(avg_metrics)
                            for rec in recommendations:
                                st.markdown(rec)

            except Exception as e:
                st.error(f"❌ Error: {e}")

    elif page == "Performance Dashboard":
        st.title("📊 Performance & Emissions Dashboard")
        uploaded_file = st.file_uploader("Upload Power Plant Data CSV", type=["csv"])

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.subheader("📄 Preview of Raw Data")
            st.dataframe(df.head())

            numeric_cols = [
                'Coal Flow', 'GCV', 'Steam Flow', 'Steam Enthalpy',
                'Feedwater Enthalpy', 'Power Output', 'Flue Temp', 'Ambient Temp'
            ]
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                else:
                    st.error(f"Missing column: {col}")
                    st.stop()

            df.dropna(subset=numeric_cols, inplace=True)

            if df.empty:
                st.warning("No valid data rows after cleaning.")
            else:
                calculated_metrics_df = df.apply(
                    lambda row: calculate_metrics(
                        coal_flow=row['Coal Flow'],
                        gcv=row['GCV'],
                        steam_flow=row['Steam Flow'],
                        h_steam=row['Steam Enthalpy'],
                        h_feed=row['Feedwater Enthalpy'],
                        power_output=row['Power Output'],
                        flue_temp=row['Flue Temp'],
                        ambient_temp=row['Ambient Temp']
                    ),
                    axis=1
                ).apply(pd.Series)

                df_with_metrics = pd.concat([df, calculated_metrics_df], axis=1)

                st.subheader("✨ Calculated Metrics Preview")
                st.dataframe(df_with_metrics.head())

                # Plotting section
                st.subheader("📊 Performance Visualizations")
                
                col1, col2 = st.columns(2)
                with col1:
                    if 'Boiler Efficiency (%)' in df_with_metrics.columns and 'Coal Flow' in df_with_metrics.columns:
                        st.subheader("🔥 Boiler Efficiency vs Coal Flow")
                        fig1, ax1 = plt.subplots(figsize=(8, 5))
                        sns.scatterplot(data=df_with_metrics, x="Coal Flow", y="Boiler Efficiency (%)", ax=ax1)
                        st.pyplot(fig1)
                        plt.close(fig1)

                with col2:
                    if 'CO2 Emissions (kg/hr)' in df_with_metrics.columns and 'Power Output' in df_with_metrics.columns:
                        st.subheader("🌫 CO₂ Emissions vs Power Output")
                        fig2, ax2 = plt.subplots(figsize=(8, 5))
                        sns.scatterplot(data=df_with_metrics, x="Power Output", y="CO2 Emissions (kg/hr)", ax=ax2)
                        st.pyplot(fig2)
                        plt.close(fig2)

                if 'Plant Heat Rate (kcal/kWh)' in df_with_metrics.columns:
                    st.subheader("🔁 Plant Heat Rate Distribution")
                    fig3, ax3 = plt.subplots(figsize=(10, 5))
                    sns.histplot(data=df_with_metrics, x="Plant Heat Rate (kcal/kWh)", kde=True, ax=ax3)
                    st.pyplot(fig3)
                    plt.close(fig3)

                st.subheader("💡 Performance Recommendations")
                avg_metrics = {
                    "Boiler Efficiency (%)": df_with_metrics['Boiler Efficiency (%)'].mean(),
                    "Plant Heat Rate (kcal/kWh)": df_with_metrics['Plant Heat Rate (kcal/kWh)'].mean(),
                    "Specific Fuel Consumption (kg/kWh)": df_with_metrics['Specific Fuel Consumption (kg/kWh)'].mean(),
                    "Flue Gas Loss (%)": df_with_metrics['Flue Gas Loss (%)'].mean(),
                    "CO2 Emissions (kg/hr)": df_with_metrics['CO2 Emissions (kg/hr)'].mean()
                }
                recommendations = generate_recommendations(avg_metrics)
                for rec in recommendations:
                    st.markdown(rec)

if __name__ == "__main__":
    main()