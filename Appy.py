import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

# Set dark theme config
st.set_page_config(
    page_title="Power Plant Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply dark theme
def set_dark_theme():
    st.markdown("""
    <style>
        /* Main page styling */
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        
        /* Text color */
        .css-1aumxhk, .css-1v0mbdj, .css-1q8dd3e, .css-1lcbmhc, .css-1outpf7 {
            color: #FAFAFA !important;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #1a1a1a !important;
            border-right: 1px solid #333;
        }
        
        /* Sidebar text */
        [data-testid="stSidebar"] .css-1aumxhk {
            color: #FAFAFA !important;
        }
        
        /* Input fields */
        .stTextInput>div>div>input, 
        .stNumberInput>div>div>input,
        .stSelectbox>div>div>select {
            background-color: #1a1a1a !important;
            color: white !important;
            border-color: #333 !important;
        }
        
        /* Buttons */
        .stButton>button {
            background-color: #2e7bcf !important;
            color: white !important;
            border: none !important;
        }
        
        .stButton>button:hover {
            background-color: #1a5a9a !important;
        }
        
        /* Dataframe styling */
        .dataframe {
            background-color: #1a1a1a !important;
            color: white !important;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab"] {
            background-color: #1a1a1a !important;
            color: white !important;
            border-color: #333 !important;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #2e7bcf !important;
            color: white !important;
        }
        
        /* Cards */
        .card {
            background-color: #1a1a1a !important;
            border: 1px solid #333 !important;
            color: white !important;
        }
        
        /* Metric cards */
        .metric-card {
            background-color: #1a1a1a !important;
            border-left: 5px solid !important;
            color: white !important;
        }
        
        /* Plot background */
        .stPlot {
            background-color: #1a1a1a !important;
        }
        
        /* Matplotlib dark theme */
        plt.style.use('dark_background')
    </style>
    """, unsafe_allow_html=True)
    
    # Set matplotlib to dark mode
    plt.style.use('dark_background')
    sns.set_style("darkgrid")

# Utility functions (unchanged from original)
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
    # Apply dark theme
    set_dark_theme()

    # Sidebar navigation with icons
    st.sidebar.title("⚡ Power Plant Analytics")
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Navigation Menu",
        ["🏠 Home", "🧮 Single Audit", "📂 Batch Analysis", "📊 Performance Dashboard"],
        index=0
    )
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **About this app**:  
    This dashboard helps analyze and optimize coal power plant performance, efficiency, and emissions.
    """)

    if page == "🏠 Home":
        st.title("⚡ Power Plant Performance Analytics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="card">
                <h3>🧮 Single Audit</h3>
                <p>Perform manual audits by entering plant parameters</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="card">
                <h3>📂 Batch Analysis</h3>
                <p>Upload CSV files for comprehensive batch analysis</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="card">
                <h3>📊 Performance Dashboard</h3>
                <p>Interactive visualizations of plant performance metrics</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.subheader("🔍 Key Features")
        features = [
            "✅ **Performance Metrics**: Calculate boiler efficiency, heat rate, and specific fuel consumption",
            "📈 **Data Visualization**: Interactive charts and correlation analysis",
            "🌱 **Emissions Tracking**: Monitor CO₂ emissions and environmental impact",
            "💡 **Actionable Insights**: Get customized recommendations for improvement",
            "📥 **Data Export**: Download analysis results for further reporting"
        ]
        
        for feature in features:
            st.markdown(feature)
        
        st.markdown("---")
        st.info("💡 **Tip**: Use the navigation menu on the left to access different analysis tools.")

    elif page == "🧮 Single Audit":
        st.title("🧮 Single Plant Audit Calculator")
        
        with st.expander("ℹ️ About this tool", expanded=False):
            st.info("""
            This calculator helps you analyze a single set of plant parameters. 
            Enter your operational data to calculate key performance metrics.
            """)
        
        with st.form("input_form"):
            st.subheader("🔧 Plant Input Parameters")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Fuel Parameters")
                coal_flow = st.number_input("Coal Flow (kg/hr)", min_value=0.0, value=100.0, step=10.0, format="%.2f")
                gcv = st.number_input("GCV of Coal (kcal/kg)", min_value=0.0, value=5000.0, step=100.0, format="%.2f")
                power_output = st.number_input("Power Output (kW)", min_value=0.0, value=200.0, step=10.0, format="%.2f")
                flue_temp = st.number_input("Flue Gas Temp (°C)", min_value=0.0, value=150.0, step=5.0, format="%.2f")

            with col2:
                st.markdown("#### Steam Parameters")
                steam_flow = st.number_input("Steam Output (kg/hr)", min_value=0.0, value=400.0, step=10.0, format="%.2f")
                h_steam = st.number_input("Steam Enthalpy (kcal/kg)", min_value=0.0, value=750.0, step=10.0, format="%.2f")
                h_feed = st.number_input("Feedwater Enthalpy (kcal/kg)", min_value=0.0, value=100.0, step=10.0, format="%.2f")
                ambient_temp = st.number_input("Ambient Temp (°C)", min_value=0.0, value=25.0, step=1.0, format="%.2f")

            submitted = st.form_submit_button("🚀 Calculate Metrics")

        if submitted:
            with st.spinner('Calculating metrics...'):
                results = calculate_metrics(coal_flow, gcv, steam_flow, h_steam, h_feed,
                                          power_output, flue_temp, ambient_temp)

            st.success("✅ Calculation Complete!")
            
            st.subheader("📊 Key Performance Indicators")
            
            cols = st.columns(5)
            metrics = [
                ("Boiler Efficiency (%)", "#4CAF50", "🏭"),
                ("Plant Heat Rate (kcal/kWh)", "#FF9800", "🔥"),
                ("Specific Fuel Consumption (kg/kWh)", "#2196F3", "⛽"),
                ("Flue Gas Loss (%)", "#9C27B0", "💨"),
                ("CO2 Emissions (kg/hr)", "#F44336", "🌍")
            ]
            
            for i, (metric, color, icon) in enumerate(metrics):
                with cols[i]:
                    st.markdown(f"""
                    <div class="metric-card" style="border-left-color: {color}; background-color: {color}20;">
                        <h4 style="margin: 0; color: {color}; font-size: 14px;">{icon} {metric}</h4>
                        <h2 style="margin: 0; color: {color};">{results[metric]:.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.subheader("📈 Performance Visualizations")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            palette = sns.color_palette("husl", len(results))
            bars = ax.bar(results.keys(), results.values(), color=palette)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom')
            
            ax.set_title("Performance Metrics Overview", fontweight='bold')
            ax.set_ylabel("Value")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            st.subheader("💡 Optimization Recommendations")
            recommendations = generate_recommendations(results)
            
            for rec in recommendations:
                if "✅" in rec:
                    st.success(rec)
                elif "⚠️" in rec:
                    st.warning(rec)
                elif "❌" in rec:
                    st.error(rec)

    elif page == "📂 Batch Analysis":
        st.title("📂 Batch Data Analysis")
        st.markdown("Upload a CSV file containing plant operational data for comprehensive analysis.")
        
        with st.expander("📋 Required CSV Format", expanded=False):
            st.markdown("""
            Your CSV file must contain these columns:
            - `Coal Flow` (kg/hr)
            - `GCV` (kcal/kg)
            - `Steam Flow` (kg/hr)
            - `Steam Enthalpy` (kcal/kg)
            - `Feedwater Enthalpy` (kcal/kg)
            - `Power Output` (kW)
            - `Flue Temp` (°C)
            - `Ambient Temp` (°C)
            """)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        
        required_columns = [
            "Coal Flow", "GCV", "Steam Flow", "Steam Enthalpy",
            "Feedwater Enthalpy", "Power Output", "Flue Temp", "Ambient Temp"
        ]

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                
                st.subheader("📋 Data Preview")
                st.dataframe(df.head())
                
                missing_cols = [col for col in required_columns if col not in df.columns]
                if missing_cols:
                    st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                    st.stop()
                
                with st.spinner("🔄 Processing data..."):
                    for col in required_columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    df.dropna(subset=required_columns, inplace=True)
                    
                    if df.empty:
                        st.warning("⚠️ No valid data rows remaining after cleaning.")
                        st.stop()
                    
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
                
                st.success(f"✅ Successfully processed {len(final_df)} records")
                
                st.subheader("📊 Summary Statistics")
                st.dataframe(result_df.describe())
                
                st.download_button(
                    label="📥 Download Full Results",
                    data=final_df.to_csv(index=False),
                    file_name="power_plant_analysis.csv",
                    mime="text/csv"
                )
                
                st.subheader("📈 Data Visualizations")
                
                tab1, tab2 = st.tabs(["Correlation Analysis", "Metrics Distribution"])
                
                with tab1:
                    st.markdown("#### Correlation Heatmap")
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(final_df.corr(numeric_only=True), ax=ax, annot=True, cmap="coolwarm")
                    st.pyplot(fig)
                    plt.close(fig)
                
                with tab2:
                    st.markdown("#### Metrics Distribution")
                    metric_to_plot = st.selectbox("Select metric to visualize", result_df.columns)
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.histplot(data=final_df, x=metric_to_plot, kde=True, ax=ax)
                    st.pyplot(fig)
                    plt.close(fig)
                
                st.subheader("💡 Performance Recommendations")
                avg_metrics = final_df[result_df.columns].mean().to_dict()
                recommendations = generate_recommendations(avg_metrics)
                
                for rec in recommendations:
                    if "✅" in rec:
                        st.success(rec)
                    elif "⚠️" in rec:
                        st.warning(rec)
                    elif "❌" in rec:
                        st.error(rec)

            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")

    elif page == "📊 Performance Dashboard":
        st.title("📊 Plant Performance Dashboard")
        st.markdown("Interactive dashboard for visualizing plant performance metrics and trends over time.")
        
        uploaded_file = st.file_uploader("Upload Plant Data CSV", type=["csv"])
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                
                numeric_cols = [
                    'Coal Flow', 'GCV', 'Steam Flow', 'Steam Enthalpy',
                    'Feedwater Enthalpy', 'Power Output', 'Flue Temp', 'Ambient Temp'
                ]
                
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    else:
                        st.error(f"Missing required column: {col}")
                        st.stop()
                
                df.dropna(subset=numeric_cols, inplace=True)
                
                if df.empty:
                    st.warning("⚠️ No valid data rows remaining after cleaning.")
                    st.stop()
                
                with st.spinner("🔄 Calculating performance metrics..."):
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
                
                st.success(f"✅ Successfully processed {len(df_with_metrics)} records")
                
                st.subheader("🔍 Data Overview")
                st.dataframe(df_with_metrics.head())
                
                time_col_options = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])] + ["Index"]
                time_col = st.selectbox("Select time column for trends", time_col_options)
                
                st.subheader("📈 Interactive Performance Charts")
                
                metric_options = [col for col in calculated_metrics_df.columns if col not in numeric_cols]
                selected_metrics = st.multiselect("Select metrics to visualize", 
                                                metric_options,
                                                default=metric_options[:3])
                
                if selected_metrics:
                    tab1, tab2, tab3 = st.tabs(["Line Charts", "Scatter Plots", "Histograms"])
                    
                    with tab1:
                        st.markdown("#### Trend Analysis")
                        fig, ax = plt.subplots(figsize=(12, 6))
                        for metric in selected_metrics:
                            if time_col != "Index":
                                ax.plot(df_with_metrics[time_col], df_with_metrics[metric], label=metric)
                            else:
                                ax.plot(df_with_metrics.index, df_with_metrics[metric], label=metric)
                        ax.set_title("Performance Metrics Over Time", fontweight='bold')
                        ax.set_ylabel("Value")
                        ax.legend()
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    with tab2:
                        st.markdown("#### Correlation Analysis")
                        x_axis = st.selectbox("X-axis", selected_metrics)
                        y_axis = st.selectbox("Y-axis", [m for m in selected_metrics if m != x_axis])
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.scatterplot(data=df_with_metrics, x=x_axis, y=y_axis, 
                                       hue=df_with_metrics[time_col] if time_col != "Index" else None,
                                       palette="viridis", ax=ax)
                        ax.set_title(f"{y_axis} vs {x_axis}", fontweight='bold')
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    with tab3:
                        st.markdown("#### Distribution Analysis")
                        cols = st.columns(2)
                        for i, metric in enumerate(selected_metrics):
                            with cols[i % 2]:
                                fig, ax = plt.subplots(figsize=(8, 4))
                                sns.histplot(data=df_with_metrics, x=metric, kde=True, ax=ax)
                                ax.set_title(f"{metric} Distribution", fontweight='bold')
                                st.pyplot(fig)
                                plt.close(fig)
                
                st.subheader("📊 Key Performance Indicators")
                avg_metrics = df_with_metrics[metric_options].mean().to_dict()
                
                cols = st.columns(5)
                metric_colors = {
                    "Boiler Efficiency (%)": "#4CAF50",
                    "Plant Heat Rate (kcal/kWh)": "#FF9800",
                    "Specific Fuel Consumption (kg/kWh)": "#2196F3",
                    "Flue Gas Loss (%)": "#9C27B0",
                    "CO2 Emissions (kg/hr)": "#F44336"
                }
                
                for i, (metric, value) in enumerate(avg_metrics.items()):
                    with cols[i % 5]:
                        st.markdown(f"""
                        <div class="metric-card" style="border-left-color: {metric_colors.get(metric, '#FFFFFF')}; 
                                    background-color: {metric_colors.get(metric, '#FFFFFF')}20;">
                            <h4 style="margin: 0; color: {metric_colors.get(metric, '#FFFFFF')}; font-size: 14px;">{metric}</h4>
                            <h2 style="margin: 0; color: {metric_colors.get(metric, '#FFFFFF')};">{value:.2f}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.subheader("💡 Performance Recommendations")
                recommendations = generate_recommendations(avg_metrics)
                
                for rec in recommendations:
                    if "✅" in rec:
                        st.success(rec)
                    elif "⚠️" in rec:
                        st.warning(rec)
                    elif "❌" in rec:
                        st.error(rec)

            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")

if __name__ == "__main__":
    main()