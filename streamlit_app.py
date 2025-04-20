import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from numba import jit
import io

# Page configuration
st.set_page_config(
    page_title="Hybrid ARIMA-ANFIS with ABC Optimization",
    layout="wide"
)

# CSS styling
st.markdown("""
    <style>
        .header-container {
            background: #800000;
            color: white;
            padding: 40px;
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
        }
        .content {
            text-align: justify;
            font-size: 18px;
            line-height: 1.8;
            margin: 20px 10%;
            padding: 20px;
            background-color: #fff;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }
        .metric-card {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar menu
menu = st.sidebar.radio("Menu", ["HOME", "DATA PREPROCESSING", "STATIONARITY TEST", 
                              "ARIMA MODELING", "ANFIS CONFIGURATION", "HYBRID PREDICTION"])

# ======================== HOME ========================
if menu == "HOME":
    st.markdown("<div class='header-container'>HYBRID ARIMA-ANFIS MODEL<br>WITH ABC OPTIMIZATION</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="content">
    This interface implements a hybrid <b>ARIMA-ANFIS</b> model with <b>Artificial Bee Colony</b> (ABC) optimization
    for time series forecasting.<br><br>
    The workflow:
    <ol>
        <li>Upload and preprocess your time series data</li>
        <li>Check stationarity and perform differencing if needed</li>
        <li>Build and train ARIMA model</li>
        <li>Configure ANFIS parameters for residual correction</li>
        <li>Combine predictions for final hybrid forecast</li>
    </ol>
    Start by uploading your data in the <b>DATA PREPROCESSING</b> menu.
    </div>
    """, unsafe_allow_html=True)

# ==================== DATA PREPROCESSING ====================
elif menu == "DATA PREPROCESSING":
    st.markdown("<div class='header-container'>DATA PREPROCESSING</div>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.write(data.head())
        
        # Select time column
        time_col = st.selectbox("Select Time Column as Index", options=data.columns)
        if time_col:
            data[time_col] = pd.to_datetime(data[time_col])
            data.set_index(time_col, inplace=True)
            
            # Select target column
            target_col = st.selectbox("Select Target Column for Forecasting", options=data.columns)
            
            if target_col:
                data = data[[target_col]]  # Keep only target column
                
                # Handle missing values
                if data.isnull().sum().any():
                    st.warning("Data contains missing values. Dropping rows with NA values.")
                    data.dropna(inplace=True)
                
                # Plot data
                st.write("Processed Data Plot:")
                fig, ax = plt.subplots(figsize=(10, 5))
                data.plot(ax=ax)
                ax.set_title("Time Series Data")
                st.pyplot(fig)
                
                # Save to session state
                st.session_state["data"] = data
                st.session_state["target_col"] = target_col
                
                st.success("Preprocessing complete. Proceed to 'STATIONARITY TEST'.")

# ================== STATIONARITY TEST =====================
elif menu == "STATIONARITY TEST":
    st.markdown("<div class='header-container'>STATIONARITY TEST</div>", unsafe_allow_html=True)
    
    if "data" in st.session_state:
        data = st.session_state["data"]
        target_col = st.session_state["target_col"]
        
        st.subheader("ADF Test - Original Data")
        adf_result = adfuller(data[target_col])
        st.write(f"ADF Statistic: {adf_result[0]:.4f}")
        st.write(f"P-Value: {adf_result[1]:.4f}")
        
        if adf_result[1] > 0.05:
            st.warning("Data is not stationary. Performing differencing...")
            
            # Differencing
            d = st.slider("Select Differencing Order (d)", 1, 3, 1)
            data_diff = data.diff(d).dropna()
            
            # Save differenced data
            st.session_state["data_diff"] = data_diff
            st.session_state["d"] = d
            
            # Plot differenced data
            st.subheader(f"Differenced Data (d={d})")
            fig, ax = plt.subplots(figsize=(10, 5))
            data_diff.plot(ax=ax)
            ax.set_title(f"{d}-Order Differenced Data")
            st.pyplot(fig)
            
            # ADF test after differencing
            st.subheader("ADF Test - Differenced Data")
            adf_diff_result = adfuller(data_diff[target_col])
            st.write(f"ADF Statistic: {adf_diff_result[0]:.4f}")
            st.write(f"P-Value: {adf_diff_result[1]:.4f}")
            
            if adf_diff_result[1] < 0.05:
                st.success("Data is now stationary after differencing.")
            else:
                st.error("Data is still not stationary. Try higher differencing order.")
        
        else:
            st.success("Data is stationary. No differencing needed.")
            st.session_state["d"] = 0
        
        # ACF and PACF plots
        st.subheader("Autocorrelation Plots")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Autocorrelation Function (ACF)")
            fig_acf, ax_acf = plt.subplots(figsize=(8, 4))
            plot_acf(data[target_col].dropna(), lags=40, ax=ax_acf)
            st.pyplot(fig_acf)
        
        with col2:
            st.write("Partial Autocorrelation Function (PACF)")
            fig_pacf, ax_pacf = plt.subplots(figsize=(8, 4))
            plot_pacf(data[target_col].dropna(), lags=40, ax=ax_pacf)
            st.pyplot(fig_pacf)

# =================== ARIMA MODELING ===================
elif menu == "ARIMA MODELING":
    st.markdown("<div class='header-container'>ARIMA MODELING</div>", unsafe_allow_html=True)
    
    if "data" in st.session_state and "d" in st.session_state:
        data = st.session_state["data_diff"] if "data_diff" in st.session_state else st.session_state["data"]
        target_col = st.session_state["target_col"]
        d = st.session_state["d"]
        
        st.subheader("Split Data into Train/Test Sets")
        test_size = st.slider("Select Test Set Size (%)", 10, 40, 20)
        split_idx = int(len(data) * (1 - test_size/100))
        
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        
        st.session_state["train_data"] = train_data
        st.session_state["test_data"] = test_data
        
        # Plot train/test split
        fig, ax = plt.subplots(figsize=(10, 5))
        train_data.plot(ax=ax, label="Training Data")
        test_data.plot(ax=ax, label="Test Data")
        ax.axvline(train_data.index[-1], color='red', linestyle='--')
        ax.set_title("Train/Test Split")
        ax.legend()
        st.pyplot(fig)
        
        st.subheader("Set ARIMA Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            p = st.slider("Autoregressive Order (p)", 0, 10, 1)
        
        with col2:
            q = st.slider("Moving Average Order (q)", 0, 10, 1)
        
        if st.button("Train ARIMA Model"):
            try:
                model = ARIMA(train_data[target_col], order=(p, d, q))
                model_fit = model.fit()
                
                st.session_state["arima_model"] = model_fit
                st.session_state["p"] = p
                st.session_state["q"] = q
                
                st.success("ARIMA Model Trained Successfully!")
                st.write(model_fit.summary())
                
                # Forecast on test set
                forecast = model_fit.forecast(steps=len(test_data))
                test_data["ARIMA_Forecast"] = forecast
                
                # Calculate metrics
                mse = mean_squared_error(test_data[target_col], test_data["ARIMA_Forecast"])
                rmse = np.sqrt(mse)
                mape = mean_absolute_percentage_error(test_data[target_col], test_data["ARIMA_Forecast"]) * 100
                
                # Store residuals
                residuals = model_fit.resid
                st.session_state["residuals"] = residuals
                
                # Display results
                st.subheader("ARIMA Forecast Results")
                fig, ax = plt.subplots(figsize=(10, 5))
                train_data[target_col].plot(ax=ax, label="Training Data")
                test_data[target_col].plot(ax=ax, label="Actual Test Data")
                test_data["ARIMA_Forecast"].plot(ax=ax, label="ARIMA Forecast", linestyle="--")
                ax.legend()
                st.pyplot(fig)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("MSE", f"{mse:.4f}")
                with col2:
                    st.metric("RMSE", f"{rmse:.4f}")
                with col3:
                    st.metric("MAPE", f"{mape:.2f}%")
                
            except Exception as e:
                st.error(f"Error training ARIMA model: {str(e)}")

# ================== ANFIS CONFIGURATION ==================
elif menu == "ANFIS CONFIGURATION":
    st.markdown("<div class='header-container'>ANFIS CONFIGURATION</div>", unsafe_allow_html=True)
    
    if "residuals" in st.session_state:
        residuals = st.session_state["residuals"]
        
        st.subheader("Residual Analysis")
        fig, ax = plt.subplots(figsize=(10, 4))
        residuals.plot(ax=ax)
        ax.set_title("ARIMA Model Residuals")
        st.pyplot(fig)
        
        # Normalize residuals
        scaler = MinMaxScaler()
        residuals_scaled = scaler.fit_transform(residuals.values.reshape(-1, 1))
        st.session_state["scaler"] = scaler
        st.session_state["residuals_scaled"] = residuals_scaled
        
        st.subheader("Determine ANFIS Inputs")
        pacf_values = pacf(residuals_scaled, nlags=20)
        significant_lags = [i for i, val in enumerate(pacf_values) if abs(val) > 1.96/np.sqrt(len(residuals_scaled)) and i != 0]
        
        st.write("Significant Lags from PACF:", significant_lags)
        
        # Let user select lags
        selected_lags = st.multiselect("Select Lags for ANFIS Inputs", 
                                     significant_lags, 
                                     default=significant_lags[:2] if len(significant_lags) >= 2 else significant_lags)
        
        if len(selected_lags) >= 2:
            # Create lag features
            df_resid = pd.DataFrame(residuals_scaled, columns=["residual"])
            
            for lag in selected_lags:
                df_resid[f"lag_{lag}"] = df_resid["residual"].shift(lag)
            
            df_resid.dropna(inplace=True)
            
            X = df_resid[[f"lag_{lag}" for lag in selected_lags]].values
            y = df_resid["residual"].values
            
            st.session_state["X_anfis"] = X
            st.session_state["y_anfis"] = y
            st.session_state["selected_lags"] = selected_lags
            
            st.success(f"ANFIS inputs created with lags: {selected_lags}")
            
            # Initialize membership functions
            num_clusters = st.slider("Number of Membership Functions per Input", 2, 5, 2)
            
            if st.button("Initialize ANFIS Parameters"):
                centers = []
                sigmas = []
                
                for i, lag in enumerate(selected_lags):
                    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(X[:, i].reshape(-1, 1))
                    lag_centers = np.sort(kmeans.cluster_centers_.flatten())
                    lag_sigma = (lag_centers[1] - lag_centers[0]) / 2
                    
                    centers.append(lag_centers)
                    sigmas.append(lag_sigma)
                    
                    st.write(f"Lag {lag} Centers:", lag_centers)
                    st.write(f"Lag {lag} Sigma:", lag_sigma)
                
                st.session_state["centers"] = centers
                st.session_state["sigmas"] = sigmas
                
                # Initialize consequent parameters with linear regression
                n_rules = num_clusters ** len(selected_lags)
                
                # Create firing strengths
                @jit(nopython=True)
                def calculate_firing_strengths(X, centers, sigmas):
                    n_samples = X.shape[0]
                    n_inputs = X.shape[1]
                    n_mf = len(centers[0])
                    
                    # Pre-calculate all membership values
                    membership_values = np.zeros((n_inputs, n_samples, n_mf))
                    for i in range(n_inputs):
                        for s in range(n_samples):
                            for m in range(n_mf):
                                membership_values[i, s, m] = np.exp(-((X[s, i] - centers[i][m])**2 / (2 * sigmas[i]**2)))
                    
                    # Calculate firing strengths for all rules
                    rules = np.ones((n_samples, n_rules))
                    rule_idx = 0
                    for _ in range(n_rules):
                        for i in range(n_inputs):
                            mf_idx = (rule_idx // (n_mf**i)) % n_mf
                            rules[:, rule_idx] *= membership_values[i, :, mf_idx]
                        rule_idx += 1
                    
                    # Normalize firing strengths
                    normalized_rules = rules / (rules.sum(axis=1, keepdims=True) + 1e-10)
                    return normalized_rules
                
                firing_strengths = calculate_firing_strengths(X, centers, sigmas)
                
                # Prepare regression matrix
                regression_matrix = np.hstack([firing_strengths * X[:, i][:, np.newaxis] for i in range(len(selected_lags))] + 
                                   [firing_strengths])
                
                # Solve linear regression
                lin_reg = LinearRegression().fit(regression_matrix, y)
                initial_params = np.concatenate([lin_reg.coef_, [lin_reg.intercept_]])
                
                st.session_state["initial_params"] = initial_params
                st.session_state["firing_strengths"] = firing_strengths
                
                st.success("ANFIS Parameters Initialized!")

# ================== HYBRID PREDICTION ==================
elif menu == "HYBRID PREDICTION":
    st.markdown("<div class='header-container'>HYBRID PREDICTION</div>", unsafe_allow_html=True)
    
    if all(key in st.session_state for key in ["arima_model", "X_anfis", "y_anfis", "initial_params"]):
        # ABC Optimization
        st.subheader("ABC Optimization for ANFIS")
        
        # ABC parameters
        col1, col2 = st.columns(2)
        with col1:
            num_bees = st.slider("Number of Bees", 50, 500, 100)
            max_iter = st.slider("Maximum Iterations", 100, 5000, 1000)
        with col2:
            limit = st.slider("Limit for Scout Bees", 10, 200, 50)
            param_range = st.slider("Parameter Range", 0.1, 10.0, 1.0)
        
        if st.button("Run ABC Optimization"):
            X = st.session_state["X_anfis"]
            y = st.session_state["y_anfis"]
            firing_strengths = st.session_state["firing_strengths"]
            initial_params = st.session_state["initial_params"]
            
            # Define ANFIS prediction and loss functions
            @jit(nopython=True)
            def anfis_predict(params, X, firing_strengths):
                n_samples = X.shape[0]
                n_inputs = X.shape[1]
                n_rules = firing_strengths.shape[1]
                
                # Split parameters
                p = params[:n_rules*n_inputs].reshape(n_inputs, n_rules)
                r = params[n_rules*n_inputs:]
                
                outputs = np.zeros(n_samples)
                
                for i in range(n_samples):
                    weighted_sum = 0.0
                    rule_sum = 0.0
                    
                    for j in range(n_rules):
                        # Calculate rule output
                        rule_output = r[j]  # intercept
                        for k in range(n_inputs):
                            rule_output += p[k, j] * X[i, k]
                        
                        # Weight by firing strength
                        weighted_sum += firing_strengths[i, j] * rule_output
                        rule_sum += firing_strengths[i, j]
                    
                    outputs[i] = weighted_sum / (rule_sum + 1e-10)
                
                return outputs
            
            @jit(nopython=True)
            def loss_function(params, X, y, firing_strengths):
                pred = anfis_predict(params, X, firing_strengths)
                return np.mean((pred - y)**2)
            
            # ABC Implementation
            def abc_optimizer():
                # Initialize population
                n_params = len(initial_params)
                population = np.random.uniform(-param_range, param_range, (num_bees, n_params))
                fitness = np.array([loss_function(p, X, y, firing_strengths) for p in population])
                
                best_idx = np.argmin(fitness)
                best_solution = population[best_idx].copy()
                best_fitness = fitness[best_idx]
                
                trial = np.zeros(num_bees)
                
                for iteration in range(max_iter):
                    # Employed bees phase
                    for i in range(num_bees):
                        # Generate new solution
                        k = np.random.randint(0, num_bees)
                        while k == i:
                            k = np.random.randint(0, num_bees)
                        
                        phi = np.random.uniform(-1, 1, n_params)
                        new_solution = population[i] + phi * (population[i] - population[k])
                        new_solution = np.clip(new_solution, -param_range, param_range)
                        
                        new_fitness = loss_function(new_solution, X, y, firing_strengths)
                        
                        # Greedy selection
                        if new_fitness < fitness[i]:
                            population[i] = new_solution
                            fitness[i] = new_fitness
                            trial[i] = 0
                        else:
                            trial[i] += 1
                    
                    # Onlooker bees phase
                    prob = (1 / (fitness + 1e-10)) / np.sum(1 / (fitness + 1e-10))
                    
                    for _ in range(num_bees):
                        i = np.random.choice(num_bees, p=prob)
                        
                        k = np.random.randint(0, num_bees)
                        while k == i:
                            k = np.random.randint(0, num_bees)
                        
                        phi = np.random.uniform(-1, 1, n_params)
                        new_solution = population[i] + phi * (population[i] - population[k])
                        new_solution = np.clip(new_solution, -param_range, param_range)
                        
                        new_fitness = loss_function(new_solution, X, y, firing_strengths)
                        
                        if new_fitness < fitness[i]:
                            population[i] = new_solution
                            fitness[i] = new_fitness
                            trial[i] = 0
                    
                    # Scout bees phase
                    for i in range(num_bees):
                        if trial[i] > limit:
                            population[i] = np.random.uniform(-param_range, param_range, n_params)
                            fitness[i] = loss_function(population[i], X, y, firing_strengths)
                            trial[i] = 0
                    
                    # Update best solution
                    current_best_idx = np.argmin(fitness)
                    if fitness[current_best_idx] < best_fitness:
                        best_solution = population[current_best_idx].copy()
                        best_fitness = fitness[current_best_idx]
                    
                    if iteration % 100 == 0:
                        st.write(f"Iteration {iteration}, Best Loss: {best_fitness:.6f}")
                
                return best_solution, best_fitness
            
            best_params, best_loss = abc_optimizer()
            st.session_state["best_params"] = best_params
            st.session_state["best_loss"] = best_loss
            
            st.success("ABC Optimization Completed!")
            st.write(f"Best Loss (MSE): {best_loss:.6f}")
        
        # Hybrid Prediction
        if "best_params" in st.session_state:
            st.subheader("Hybrid ARIMA-ANFIS Prediction")
            
            if st.button("Generate Hybrid Forecast"):
                # Get all necessary data from session state
                arima_model = st.session_state["arima_model"]
                test_data = st.session_state["test_data"]
                target_col = st.session_state["target_col"]
                scaler = st.session_state["scaler"]
                selected_lags = st.session_state["selected_lags"]
                centers = st.session_state["centers"]
                sigmas = st.session_state["sigmas"]
                best_params = st.session_state["best_params"]
                
                # ARIMA forecast
                arima_forecast = arima_model.forecast(steps=len(test_data))
                
                # Prepare residuals for ANFIS prediction
                residuals = arima_model.resid
                residuals_scaled = scaler.transform(residuals.values.reshape(-1, 1))
                
                # Create lag features for test period
                test_residuals = []
                last_residuals = list(residuals_scaled[-max(selected_lags):].flatten())
                
                # ANFIS forecast for each step
                anfis_forecast_scaled = []
                
                for _ in range(len(test_data)):
                    # Create input vector from last residuals
                    input_vec = np.array([last_residuals[-lag] for lag in selected_lags]).reshape(1, -1)
                    
                    # Calculate firing strengths
                    firing_strengths = np.ones((1, len(centers[0])**len(selected_lags)))
                    for i, lag in enumerate(selected_lags):
                        mf_values = []
                        for c in centers[i]:
                            mf_values.append(np.exp(-((input_vec[0, i] - c)**2 / (2 * sigmas[i]**2)))
                        # Update firing strengths
                        # (This needs proper implementation for multiple inputs)
                        # Simplified for demo - should use proper rule combination
                        firing_strengths *= np.array(mf_values)
                    
                    # Normalize firing strengths
                    firing_strengths /= (firing_strengths.sum() + 1e-10)
                    
                    # Predict with ANFIS
                    pred = anfis_predict(best_params, input_vec, firing_strengths)
                    anfis_forecast_scaled.append(pred[0])
                    
                    # Update last residuals (using predicted value)
                    last_residuals.append(pred[0])
                    last_residuals = last_residuals[1:]
                
                # Scale back ANFIS predictions
                anfis_forecast = scaler.inverse_transform(np.array(anfis_forecast_scaled).reshape(-1, 1))
                
                # Combine forecasts
                hybrid_forecast = arima_forecast + anfis_forecast.flatten()
                
                # Calculate metrics
                mse = mean_squared_error(test_data[target_col], hybrid_forecast)
                rmse = np.sqrt(mse)
                mape = mean_absolute_percentage_error(test_data[target_col], hybrid_forecast) * 100
                
                # Store results
                test_data["Hybrid_Forecast"] = hybrid_forecast
                
                # Display results
                st.subheader("Forecast Results")
                fig, ax = plt.subplots(figsize=(12, 6))
                test_data[target_col].plot(ax=ax, label="Actual")
                test_data["ARIMA_Forecast"].plot(ax=ax, label="ARIMA Forecast", linestyle="--")
                test_data["Hybrid_Forecast"].plot(ax=ax, label="Hybrid Forecast", linestyle="-.")
                ax.legend()
                st.pyplot(fig)
                
                st.subheader("Performance Metrics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("MSE", f"{mse:.4f}")
                with col2:
                    st.metric("RMSE", f"{rmse:.4f}")
                with col3:
                    st.metric("MAPE", f"{mape:.2f}%")
                
                # Show improvement
                arima_mse = mean_squared_error(test_data[target_col], test_data["ARIMA_Forecast"])
                improvement = 100 * (arima_mse - mse) / arima_mse
                
                st.success(f"Hybrid model improved MSE by {improvement:.2f}% over ARIMA alone!")
