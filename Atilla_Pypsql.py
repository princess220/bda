import pandas as pd
import numpy as np
import psycopg2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Database Configuration
DB_CONFIG = {
    "host": "localhost",
    "dbname": "PSQL_FINALLAB",
    "user": "postgres",
    "password": "1234",
}
CSV_FILE_PATH = "powerconsumption.csv"

#query for creating table
def create_table(cursor):
    """Create the power_consumption table if it doesn't exist."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS public.power_consumption (
            "datetime" TIMESTAMP NOT NULL PRIMARY KEY,
            "temperature" NUMERIC,
            "humidity" NUMERIC,
            "windspeed" NUMERIC,
            "generaldiffuseflows" NUMERIC,
            "diffuseflows" NUMERIC,
            "powerConsumption_Zone1" NUMERIC,
            "powerConsumption_Zone2" NUMERIC,
            "powerConsumption_Zone3" NUMERIC
        );
    """)

# function to load csv into postgre
def load_data_to_postgres():
    """Load data from CSV to PostgreSQL."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Create table
        create_table(cursor)

        # insert csv data into the power_consumption table
        df = pd.read_csv(CSV_FILE_PATH)
        for _, row in df.iterrows():
            cursor.execute("""
                INSERT INTO power_consumption (
                    datetime, temperature, humidity, windspeed, 
                    generaldiffuseflows, diffuseflows, 
                    powerConsumption_Zone1, powerConsumption_Zone2, powerConsumption_Zone3
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (datetime) DO NOTHING
            """, tuple(row))

        conn.commit()
        print("Data successfully loaded to PostgreSQL!")
    except Exception as e:
        print(f"Error loading data: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()

# fetch data from postgreSQL
def fetch_data():
    """Fetch data from PostgreSQL into a pandas DataFrame."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        query = "SELECT * FROM power_consumption"
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None
    finally:
        if conn:
            conn.close()

# descriptive analysis
def descriptive_analysis(df):
    """Perform descriptive analysis and visualization on the data."""
    print("Descriptive Statistics:")
    print(df.describe())

    print("Columns:", df.columns)

    zones = ['powerconsumption_zone1', 'powerconsumption_zone2', 'powerconsumption_zone3']
    if all(col in df.columns for col in zones):
        avg_consumptions = df[zones].mean()
    else:
        print("Column names do not match expected zones.")
        return

    overall_avg = avg_consumptions.mean()

    print("\nAverage Power Consumption:")
    for zone, avg in avg_consumptions.items():
        print(f"{zone}: {avg:.2f}")
    print(f"\nOverall Average Power Consumption: {overall_avg:.2f}")

    # Visualization
    avg_consumptions.plot(kind='bar', color=['blue', 'orange', 'green'], title="Average Power Consumption per Zone")
    plt.ylabel("Power Consumption")
    plt.show()


#predictive analysis
def predictive_analysis_rf(df):
    """Perform predictive analysis using Random Forest Regression."""
    print("\nPredictive Analysis - Random Forest Regression:")

    # Feature Engineering
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['time_of_day'] = df['datetime'].dt.hour

    # Prepare data
    X = df[['time_of_day']]
    y = df['powerconsumption_zone1']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate model
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")

    # Visualization
    plt.scatter(X_test, y_test, color='blue', label='Actual', alpha=0.6)
    plt.scatter(X_test, y_pred, color='red', label='Predicted', alpha=0.6)
    plt.title("Power Consumption Prediction (Zone 1) - Random Forest")
    plt.xlabel("Time of Day (Hour)")
    plt.ylabel("Power Consumption")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    load_data_to_postgres()
    df = fetch_data()

    if df is not None and not df.empty:
        descriptive_analysis(df)
        predictive_analysis_rf(df)
    else:
        print("No data available!")