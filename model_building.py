import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

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
def create_model(df):
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

    # Prediction
    train_predict = rf_model.predict(X_train)
    test_predict = rf_model.predict(X_test)

    # Transform back to original form
    train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1)).flatten()
    test_predict = scaler.inverse_transform(test_predict.reshape(-1, 1)).flatten()
    y_train = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Shift train predictions for plotting
    look_back = time_step
    trainPredictPlot = np.empty_like(closedf)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict.reshape(-1, 1)

    # Shift test predictions for plotting
    testPredictPlot = np.empty_like(closedf)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(closedf) - 1, :] = test_predict.reshape(-1, 1)

    plotdf = pd.DataFrame({'Date': df.index, 'original_close': df['Close'], 'train_predicted_close': trainPredictPlot.flatten(),
                           'test_predicted_close': testPredictPlot.flatten()})

    return plotdf

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
                    plotdf = create_model(df)

                    # Plot predicted data
                    st.title('Random Forest Prediction Results')
                    st.write(plotdf)

                    fig, ax = plt.subplots(figsize=(20, 10))
                    sns.lineplot(data=df, x='Date', y='Close', ax=ax, label='Actual Prices')
                    sns.lineplot(data=plotdf, x='Date', y='train_predicted_close', ax=ax, label='Train Predicted Prices (RF)', color='blue')
                    sns.lineplot(data=plotdf, x='Date', y='test_predicted_close', ax=ax, label='Test Predicted Prices (RF)', color='orange')
                    ax.legend()
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
