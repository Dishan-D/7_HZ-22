import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MultiLabelBinarizer
import gspread
from google.oauth2.service_account import Credentials
import random

# Google Sheets Setup
SERVICE_ACCOUNT_FILE = "fiery-surf-449613-a3-d178e5b0aaed.json"
SHEET_ID = "1qle3vNbnunMeZKKWbK9Puek0wMrSbgYV8CDCFUkz3E8"

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=scope)
client = gspread.authorize(creds)
sheet = client.open_by_key(SHEET_ID).sheet1

# Fetch the last 10 values from Google Sheets
def get_last_10_values():
    data = sheet.get_all_values()
    last_10_rows = data[-10:]  # Get last 10 rows
    timestamps = [row[0] for row in last_10_rows]  # Assuming time is in the first column
    TDS_values = [float(row[1]) for row in last_10_rows]
    pH_values = [float(row[2]) for row in last_10_rows]
    temp_values = [float(row[3]) for row in last_10_rows]
    return timestamps, TDS_values, pH_values, temp_values

# Fetch the latest values from Google Sheets
def get_google_sheets_data():
    data = sheet.get_all_values()
    last_row = data[-1]  # Get the last row with TDS, pH, and temp (ignoring time)
    TDS = float(last_row[1])
    pH = float(last_row[2])
    temp = float(last_row[3])
    return TDS, pH, temp

# Load and prepare Water Potability dataset
data = pd.read_csv("./water_potability 2.csv")
X_potability = data.drop(columns='Potability')
y_potability = data.Potability
scaler_potability = MinMaxScaler()
X_potability_scaled = scaler_potability.fit_transform(X_potability)
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_potability_scaled, y_potability, test_size=0.2, random_state=0)
model_potability = RandomForestClassifier(n_estimators=1000, max_features='sqrt', n_jobs=-1)
model_potability.fit(X_train_p, y_train_p)

# Load and prepare Electrocoagulation dataset
df = pd.read_csv("electrocoagulation_dataset.csv")
X_current = df[['temp', 'vol', 'TDS', 'voltage', 'pH']]
y_current = df['current']
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_current, y_current, test_size=0.2, random_state=42)
model_current = RandomForestRegressor(n_estimators=100, random_state=42)
model_current.fit(X_train_c, y_train_c)

# Load and preprocess Disease Risk dataset
def train_model():
    file_path = "final_disease_dataset.csv"
    df = pd.read_csv(file_path)

    # Handle missing values
    columns_with_missing = ["ph", "Sulfate", "Trihalomethanes"]
    imputer = SimpleImputer(strategy="median")
    df[columns_with_missing] = imputer.fit_transform(df[columns_with_missing])

    # Encode 'Disease' column
    df["Disease"] = df["Disease"].apply(lambda x: x.split(", "))
    mlb = MultiLabelBinarizer()
    disease_encoded = pd.DataFrame(mlb.fit_transform(df["Disease"]), columns=mlb.classes_)

    # Merge encoded diseases with dataset
    df_processed = pd.concat([df.drop(columns=["Disease", "Potability"]), disease_encoded], axis=1)

    # Define features (X) and labels (y)
    X = df_processed.drop(columns=mlb.classes_)  # Features
    y = df_processed[mlb.classes_]  # Multi-label target (diseases)

    # Train model
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(X, y)

    return model, mlb, X

# Train model
model_disease, mlb, X_disease = train_model()

# GUI setup
root = tk.Tk()
root.title("Water Analysis & Prediction System")
root.geometry("1000x800")

notebook = ttk.Notebook(root)
frame_potability = ttk.Frame(notebook)
frame_current = ttk.Frame(notebook)
frame_visualization = ttk.Frame(notebook)
frame_disease = ttk.Frame(notebook)
notebook.add(frame_potability, text="Water Potability")
notebook.add(frame_current, text="Current Prediction")
notebook.add(frame_visualization, text="Sensor Visualization")
notebook.add(frame_disease, text="Disease Risk Prediction")
notebook.pack(expand=True, fill="both")

# Frame for Plot
plot_frame = tk.Frame(frame_visualization)
plot_frame.pack()

# Create Matplotlib Figures for Three Side-by-Side Graphs
fig, axes = plt.subplots(1, 3, figsize=(10, 3))  # Smaller size, horizontal layout
ax_TDS, ax_pH, ax_temp = axes

canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.get_tk_widget().pack()

def update_plot():
    timestamps, TDS_values, pH_values, temp_values = get_last_10_values()
    
    # Clear previous plots
    ax_TDS.clear()
    ax_pH.clear()
    ax_temp.clear()

    # Plot TDS
    ax_TDS.plot(timestamps, TDS_values, marker='o', linestyle='-', color='blue', label="TDS")
    ax_TDS.set_title("TDS", fontsize=8)
    ax_TDS.set_xticks([])  # Remove x-ticks to save space
    ax_TDS.set_ylabel("ppm", fontsize=7)
    ax_TDS.grid(True)

    # Plot pH
    ax_pH.plot(timestamps, pH_values, marker='s', linestyle='-', color='green', label="pH")
    ax_pH.set_title("pH", fontsize=8)
    ax_pH.set_xticks([])
    ax_pH.set_ylabel("", fontsize=7)  # Remove redundant labels
    ax_pH.grid(True)

    # Plot Temperature
    ax_temp.plot(timestamps, temp_values, marker='^', linestyle='-', color='red', label="Temp")
    ax_temp.set_title("Temperature", fontsize=8)
    ax_temp.set_xticks([])
    ax_temp.set_ylabel("°C", fontsize=7)
    ax_temp.grid(True)

    # Draw updated plots
    canvas.draw()

# Refresh Button for Visualization
tk.Button(frame_visualization, text="Update Plot", command=update_plot).pack(pady=5)

# Water Potability Tab
labels_p = ["pH", "Hardness", "Solids", "Chloramines", "Sulfate", "Conductivity", "Organic Carbon", "Trihalomethanes", "Turbidity"]
slider_ranges_p = {
    "pH": (0, 14),
    "Hardness": (0, 500),
    "Solids": (0, 50000),
    "Chloramines": (0, 10),
    "Sulfate": (0, 500),
    "Conductivity": (0, 2000),
    "Organic Carbon": (0, 30),
    "Trihalomethanes": (0, 150),
    "Turbidity": (0, 10)
}

sliders_p = []
tk.Label(frame_potability, text="Enter Water Quality Parameters", font=("Arial", 14)).pack(pady=10)
for label in labels_p:
    frame = tk.Frame(frame_potability)
    frame.pack(pady=5)
    tk.Label(frame, text=label).pack(side=tk.LEFT)
    min_val, max_val = slider_ranges_p[label]
    slider = tk.Scale(frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL, length=200)
    slider.pack(side=tk.RIGHT)
    sliders_p.append(slider)

def randomize_potability_values():
    for label, slider in zip(labels_p, sliders_p):
        min_val, max_val = slider_ranges_p[label]
        random_value = random.uniform(min_val, max_val)
        slider.set(random_value)

def predict_potability():
    try:
        values = [slider.get() for slider in sliders_p]
        input_scaled = scaler_potability.transform(np.array(values).reshape(1, -1))
        prediction = model_potability.predict(input_scaled)
        result = "Potable" if prediction[0] == 1 else "Not Potable"
        result_label_potability.config(text=f"Prediction: {result}", font=("Arial", 20, "bold"))

    except ValueError:
        result_label_potability.config(text="Prediction: Invalid Input",font=("Arial", 20, "bold"))

tk.Button(frame_potability, text="Regenerate", command=randomize_potability_values).pack(pady=10)
tk.Button(frame_potability, text="Predict Potability", command=predict_potability).pack(pady=20)

# Add label for displaying potability result
result_label_potability = tk.Label(frame_potability, text="Prediction: ", font=("Arial", 20,"bold"))
result_label_potability.pack(pady=10)

# Current Prediction Tab
tk.Label(frame_current, text="Enter Sensor Values", font=("Arial", 14)).pack(pady=10)
labels_c = ["Temperature (°C)", "Voltage (V)", "TDS (ppm)", "Volume (L)", "pH"]
slider_ranges_c = {
    "Temperature (°C)": (0, 100),  # Replace with actual min-max from dataset
    "Voltage (V)": (0, 10),        # Replace with actual min-max from dataset
    "TDS (ppm)": (0, 5000),        # Replace with actual min-max from dataset
    "Volume (L)": (0, 100),        # Replace with actual min-max from dataset
    "pH": (0, 14)                  # Replace with actual min-max from dataset
}

sliders_c = []
for label in labels_c:
    frame = tk.Frame(frame_current)
    frame.pack(pady=5)
    tk.Label(frame, text=label).pack(side=tk.LEFT)
    min_val, max_val = slider_ranges_c[label]
    slider = tk.Scale(frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL, length=200)
    slider.pack(side=tk.RIGHT)
    sliders_c.append(slider)

def randomize_current_values():
    voltage = random.uniform(0, 10)
    volume = random.uniform(0, 100)
    sliders_c[1].set(voltage)  # Voltage slider
    sliders_c[3].set(volume)   # Volume slider
    for i, label in enumerate(labels_c[:3]):
        min_val, max_val = slider_ranges_c[label]
        random_value = random.uniform(min_val, max_val)
        sliders_c[i].set(random_value)

def predict_current():
    try:
        values = [slider.get() for slider in sliders_c]
        predicted_current = model_current.predict(np.array([values]))[0]
        result_label_current.config(text=f"Prediction: {predicted_current:.4f} A",font=("Arial", 20, "bold"))
    except ValueError:
        result_label_current.config(text="Prediction: Invalid Input",font=("Arial", 20, "bold"))

tk.Button(frame_current, text="Regenerate", command=randomize_current_values).pack(pady=10)
tk.Button(frame_current, text="Predict Current", command=predict_current).pack(pady=20)

# Add label for displaying current result
result_label_current = tk.Label(frame_current, text="Prediction: ", font=("Arial", 20))
result_label_current.pack(pady=10)

# Disease Risk Prediction Tab
parameters = [
    ("pH", 4, 10),
    ("Hardness", 0, 500),
    ("Solids", 0, 100000),
    ("Chloramines", 0, 10),
    ("Sulfate", 0, 500),
    ("Conductivity", 0, 1500),
    ("Organic Carbon", 0, 30),
    ("Trihalomethanes", 0, 150),
    ("Turbidity", 0, 10)
]

sliders_d = {}
for i, (label, min_val, max_val) in enumerate(parameters):
    ttk.Label(frame_disease, text=label).grid(row=i, column=0, padx=10, pady=5)
    slider = ttk.Scale(frame_disease, from_=min_val, to=max_val, orient="horizontal", length=200)
    slider.grid(row=i, column=1, padx=10, pady=5)
    slider.set((min_val + max_val) / 2)
    sliders_d[label.lower().replace(" ", "_")] = slider

# Assign sliders to variables
ph_slider = sliders_d["ph"]
hardness_slider = sliders_d["hardness"]
solids_slider = sliders_d["solids"]
chloramines_slider = sliders_d["chloramines"]
sulfate_slider = sliders_d["sulfate"]
conductivity_slider = sliders_d["conductivity"]
organic_carbon_slider = sliders_d["organic_carbon"]
trihalomethanes_slider = sliders_d["trihalomethanes"]
turbidity_slider = sliders_d["turbidity"]

def predict_disease():
    input_data = {
        "ph": float(ph_slider.get()),
        "Hardness": float(hardness_slider.get()),
        "Solids": float(solids_slider.get()),
        "Chloramines": float(chloramines_slider.get()),
        "Sulfate": float(sulfate_slider.get()),
        "Conductivity": float(conductivity_slider.get()),
        "Organic_carbon": float(organic_carbon_slider.get()),
        "Trihalomethanes": float(trihalomethanes_slider.get()),
        "Turbidity": float(turbidity_slider.get())
    }

    # Ensure input_df aligns with X columns
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=X_disease.columns, fill_value=0)  # Align features

    # Predict
    disease_prediction = model_disease.predict(input_df)[0]

    # Get the disease with the highest probability
    max_index = np.argmax(disease_prediction)
    max_value = disease_prediction[max_index]

    # Display the most probable disease (if its probability is > 0.5)
    predicted_disease = mlb.classes_[max_index] if max_value > 0.5 else "No Risk"
    result_label_disease.config(text="Predicted Disease: " + predicted_disease)

def randomize_disease_values():
    sulfate_slider.set(random.uniform(0, 500))
    turbidity_slider.set(random.uniform(0, 10))
    ph_slider.set(random.uniform(4, 10))
    hardness_slider.set(random.uniform(0, 500))
    solids_slider.set(random.uniform(0, 100000))
    conductivity_slider.set(random.uniform(0, 1500))
    trihalomethanes_slider.set(random.uniform(0, 150))
    organic_carbon_slider.set(random.uniform(0, 30))
    chloramines_slider.set(random.uniform(0, 10))

# Buttons
predict_button = ttk.Button(frame_disease, text="Predict Disease", command=predict_disease)
predict_button.grid(row=len(parameters), column=0, columnspan=2, pady=10)

randomize_button = ttk.Button(frame_disease, text="Randomize Values", command=randomize_disease_values)
randomize_button.grid(row=len(parameters) + 1, column=0, columnspan=2, pady=10)

# Result Label
result_label_disease = ttk.Label(frame_disease, text="Predicted Disease Risk: ", font=("Arial", 12))
result_label_disease.grid(row=len(parameters) + 2, column=0, columnspan=2, pady=10)

# Function to fetch new data and make predictions
def refresh_data():
    TDS, pH, temp = get_google_sheets_data()  # Get latest data from Google Sheets
    print(f"Fetched data from Google Sheets: TDS={TDS}, pH={pH}, temp={temp}")
    
    sliders_c[2].set(TDS)  # TDS slider
    sliders_c[4].set(pH)   # pH slider
    sliders_c[0].set(temp) # Temperature slider

    predict_current()
    predict_potability()
    update_plot()

# Refresh Button
tk.Button(root, text="Refresh Data", command=refresh_data).pack(pady=10)

root.mainloop()