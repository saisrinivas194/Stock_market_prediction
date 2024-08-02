import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

# Function to preprocess data and handle NaN values
def preprocess_data(df):
    if df.isnull().values.any():
        st.warning("Warning: Dataset contains NaN values. Please preprocess your data accordingly.")
    df.dropna(inplace=True)
    
    # Convert Date column to datetime type if not already
    df['Date'] = pd.to_datetime(df['Date'])

    # Convert Close column to float64 if not already
    df['Close'] = df['Close'].astype('float64')

    return df

# Function to create dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# Function to create and train Random Forest model
def create_model(df, num_days=30):
    scaler = MinMaxScaler(feature_range=(0, 1))
    closedf = scaler.fit_transform(np.array(df['Close']).reshape(-1, 1))

    training_size = int(len(closedf) * 0.75)
    test_size = len(closedf) - training_size
    train_data, test_data = closedf[0:training_size, :], closedf[training_size:len(closedf), :1]

    # Reshape into X=t,t+1,t+2,...,t+time_step and Y=t+time_step
    time_step = 15
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    # Initialize Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Prediction for num_days into the future
    future_data = closedf[-time_step:, :]
    future_predictions = []
    for _ in range(num_days):
        # Predict the next day
        prediction = rf_model.predict(future_data[-time_step:].reshape(1, -1))
        future_predictions.append(prediction[0])
        # Update future_data to include the prediction for the next iteration
        future_data = np.append(future_data, prediction.reshape(1, -1), axis=0)

    # Inverse transform predictions to get actual values
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

    # Transform back to original form for visualization
    train_predict = rf_model.predict(X_train)
    test_predict = rf_model.predict(X_test)

    # Inverse transform and remove None values
    train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1)).flatten()
    test_predict = scaler.inverse_transform(test_predict.reshape(-1, 1)).flatten()
    y_train = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Ensure all arrays have the same length
    min_length = min(len(train_predict), len(test_predict), len(y_train), len(y_test))
    train_predict = train_predict[-min_length:]
    test_predict = test_predict[-min_length:]
    y_train = y_train[-min_length:]
    y_test = y_test[-min_length:]

    return train_predict, test_predict, y_train, y_test, future_predictions

# Main Streamlit application code
def main():
    st.sidebar.markdown("# Stock Market Prediction")
    st.sidebar.markdown("### Upload CSV File")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success('File successfully uploaded and loaded.')

            st.sidebar.markdown("### Choose Date Range for Analysis")
            START = st.sidebar.date_input("From", datetime.date(2015, 1, 1), key="start_date")
            END = st.sidebar.date_input("To", datetime.date(2023, 2, 28), key="end_date")
            
            st.sidebar.markdown("### Choose Number of Days for Prediction")
            num_days = st.sidebar.slider("Select the number of days", 20, 200, step=10, value=30, key="num_days")

            st.sidebar.markdown("### Select Model")
            model_choice = st.sidebar.selectbox("Select model for prediction", ["Random Forest"])

            bt = st.sidebar.button('Submit')

            if bt:
                # Preprocess the data
                if df.isnull().values.any():
                    st.warning("Warning: Dataset contains NaN values. Please preprocess your data accordingly.")
                df = preprocess_data(df)

                # Filter data based on date range
                mask = (df['Date'] >= pd.to_datetime(START)) & (df['Date'] <= pd.to_datetime(END))
                df = df.loc[mask]

                # Check if any rows are left after dropping NaN values
                if df.empty:
                    st.warning("No data available after preprocessing. Please upload a different CSV file or adjust date range.")
                    return

                # Perform EDA and Visualization
                st.title('Exploratory Data Analysis (EDA)')
                st.write(df)

                st.title('Visualizations')
                # Plot actual historical data
                fig, ax = plt.subplots()
                sns.lineplot(data=df, x='Date', y='Close', ax=ax, label='Actual Prices')
                ax.set_title('Actual Close Price')
                st.pyplot(fig)

                # Choose model and predict
                if model_choice == "Random Forest":
                    train_predict, test_predict, y_train, y_test, future_predictions = create_model(df, num_days)

                    # Display results in a table
                    st.title('Prediction Results')
                    results_df = pd.DataFrame({
                        'Date': df['Date'].tolist()[-len(train_predict):],
                        'Actual Close': df['Close'].tolist()[-len(train_predict):],
                        'Train Predicted Close': train_predict,
                        'Test Predicted Close': test_predict
                    })
                    st.write(results_df)

                    # Plot predicted data
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.lineplot(data=results_df, x='Date', y='Actual Close', ax=ax, label='Actual Prices')
                    sns.lineplot(data=results_df, x='Date', y='Train Predicted Close', ax=ax, label='Train Predicted Prices (RF)', color='blue')
                    sns.lineplot(data=results_df, x='Date', y='Test Predicted Close', ax=ax, label='Test Predicted Prices (RF)', color='orange')
                    ax.axvline(x=df['Date'].iloc[-1], color='r', linestyle='--', label='Last Historical Data')
                    ax.axvline(x=df['Date'].iloc[-1] + pd.DateOffset(days=1), color='g', linestyle='--', label='Start of Predictions')
                    ax.set_title('Random Forest Prediction Results')
                    ax.legend()
                    st.pyplot(fig)

                    # Display future predictions
                    st.title(f'Future Predictions for {num_days} Days')
                    future_dates = pd.date_range(start=df['Date'].max() + pd.DateOffset(days=1), periods=num_days, freq='D')
                    future_df = pd.DataFrame({
                        'Date': future_dates,
                        'Future Predicted Close': future_predictions
                    })
                    st.write(future_df)
                    
                    # Plotting train and test predictions
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

                    # Plot training predictions
                    train_dates = df['Date'].iloc[-len(train_predict):]
                    ax1.plot(train_dates, y_train, label='Actual Train Data', color='blue')
                    ax1.plot(train_dates, train_predict, label='Predicted Train Data', color='orange')
                    ax1.set_title('Training Data vs Predicted Training Data')
                    ax1.legend()

                    # Plot test predictions
                    test_dates = df['Date'].iloc[-len(test_predict):]
                    ax2.plot(test_dates, y_test, label='Actual Test Data', color='blue')
                    ax2.plot(test_dates, test_predict, label='Predicted Test Data', color='orange')
                    ax2.set_title('Test Data vs Predicted Test Data')
                    ax2.legend()

                    plt.tight_layout()
                    st.pyplot(fig)

                else:
                    st.warning("Please select a valid model.")

            else:
                st.warning('Please submit to see the prediction results.')

        except Exception as e:
            st.error(f"Error: {e}")

    else:
        st.warning('Please upload a CSV file to continue.')

if __name__ == '__main__':
    main()
