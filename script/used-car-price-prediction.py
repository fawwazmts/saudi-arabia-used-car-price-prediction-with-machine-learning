# Deploy Used Car Price Prediction

# ======================================================
import pandas as pd
import numpy as np

from xgboost import XGBRegressor
import streamlit as st
import pickle
# ======================================================

# Main Title
st.write('''
# Saudi Arabia Used Car Price Prediction
''')

# ======================================================

# Add a sidebar
st.sidebar.header("Please input car's features")

# ======================================================

# Add features

types_car = (
    "Range Rover", "Optima", "CX3", "Cayenne S", "Sonata", "Avalon", "C300", 
    "Land Cruiser", "LS", "FJ", "Tucson", "Pajero", "Azera", "Focus", "Spark", 
    "Accent", "ML", "Corolla", "Tahoe", "Altima", "Expedition", "Santa Fe", 
    "Liberty", "Land Cruiser Pickup", "VTC", "Malibu", "Patrol", "Grand Cherokee", 
    "SL", "Previa", "MKZ", "Datsun", "Hilux", "Yukon", "GLC", "Edge", "Innova", 
    "Navara", "G80", "Carnival", "Suburban", "Camaro", "Accord", "Sunny", 
    "Taurus", "Camry", "Elantra", "Flex", "Cerato", "Land Cruiser 70", "Charger", 
    "H6", "Hiace", "Fusion", "Aveo", "CX9", "Yaris", "Sierra", "Durango", "ES", 
    "Navigator", "Opirus", "Creta", "CS35", "GLE", "Sedona", "Victoria", "Prestige", 
    "CLA", "Vanquish", "Safrane", "Cadenza", "Silverado", "Rio", "Maxima", 
    "X-Trail", "Cruze", "Prado", "Caprice", "Grand Marquis", "LX", "Impala", 
    "QX", "Blazer", "H1", "Rav4", "Genesis", "Pathfinder", "Traverse", "SEL", 
    "Civic", "Echo Sport", "Challenger", "Wrangler", "A6", "CX5", "Mohave", 
    "Rush", "Cherokee", "Veloster", "IS", "Fluence", "Vego", "Marquis", "Kona", 
    "Explorer", "UX", "Beetle", "F150", "Lancer", "Mustang", "DB9", "Sorento", 
    "APV", "Viano", "Safari", "RX", "Platinum", "Avanza", "D-MAX", "Coupe S", 
    "Odyssey", "Panamera", "Juke", "Sportage", "C200", "GS", "X-Terra", 
    "Picanto", "CT5", "KICKS", "Gran Max", "Cayman", "A8", "Levante", "G", 
    "Montero", "A3", "Touareg", "Passat", "Delta", "Acadia", "H3", "GS3", 
    "Coupe", "Cayenne Turbo", "Colorado", "Vitara", "Kaptiva", "CLS", "LF X60", 
    "Aurion", "Koleos", "Abeka", "Flying Spur", "Pilot", "Ranger", "Escalade", 
    "A7", "Quattroporte", "Compass", "Bus Urvan", "Macan", "Azkarra", "GL", 
    "City", "Symbol", "Ertiga", "RX5", "Envoy", "CT6", "Fleetwood", "Tiggo", 
    "Q5", "A4", "XJ", "H2", "HS", "Seltos", "RX8", "301", "EC8", "3008", 
    "Suvana", "Prius", "Eado", "Royal", "NX", "Copper", "CS75", "F-Pace", 
    "Coolray", "CS85", "Jimny", "GC7", "A5", "S300", "Superb", "Ram", "Terrain", 
    "Cressida", "500", "Armada", "5008", "Tiguan", "Golf", "CS95", "S5", "911", 
    "Camargue", "Defender", "Daily", "Nitro", "Mini Van", "Pegas", "Cores", 
    "Grand Vitara", "FX", "L300", "Coaster", "Discovery", "Montero2", "Z370", 
    "Bus County", "Stinger", "SRT", "K5", "CT4", "F Type", "CC", "ASX", "Carens", 
    "XT5", "Tuscani", "4Runner", "ATS", "CRV", "The 4", "HRV", "X7", "GX", 
    "X40", "Q7", "ZS", "G70", "Megane", "Power", "B50", "Town Car", "Van", "2", 
    "i40", "XF", "RC", "Doblo", "MKX", "Jetta", "Soul", "Dzire", "Avante", 
    "CX7", "Countryman", "GTB 599 Fiorano", "Prestige Plus", "MKS", "Milan", 
    "Savana", "S8", "Others"
)

regions_car = (
    "Riyadh", "Hafar Al-Batin", "Abha", "Makkah", "Dammam", "Jeddah", 
    "Khobar", "Al-Baha", "Jazan", "Aseer", "Al-Medina", "Al-Namas", 
    "Qassim", "Taef", "Al-Ahsa", "Sabya", "Al-Jouf", "Yanbu", "Najran", 
    "Hail", "Tabouk", "Jubail", "Wadi Dawasir", "Arar", "Besha", "Qurayyat"
)

make_car = (
    "Land Rover", "Kia", "Mazda", "Porsche", "Hyundai", "Toyota", "Chrysler", 
    "Lexus", "Mitsubishi", "Ford", "MG", "Chevrolet", "Mercedes", "Nissan", 
    "Jeep", "BMW", "Lincoln", "GMC", "Genesis", "Honda", "Zhengzhou", "Dodge", 
    "HAVAL", "Cadillac", "Changan", "Aston Martin", "Renault", "Mercury", 
    "INFINITI", "Audi", "Rolls-Royce", "BYD", "Volkswagen", "Victory Auto", 
    "Suzuki", "Geely", "Isuzu", "Daihatsu", "Maserati", "Hummer", "GAC", 
    "Lifan", "Bentley", "Chery", "Jaguar", "Peugeot", "Foton", "MINI", 
    "Škoda", "Fiat", "Iveco", "FAW", "Great Wall", "Ferrari"
)

gear_types_car = ("Automatic", "Manual")

origins_car = ("Gulf Arabic", "Saudi", "Other", "Unknown")

options_car = ("Full", "Semi Full", "Standard")

def user_input_feature():
    # numerical feature --> number_input
    inputYear = st.sidebar.number_input(label="Year", min_value=1978, max_value=2021, value=2020)
    inputEngineSize = st.sidebar.number_input(label="EngineSize", min_value=1.0, max_value=7.5, value=6.0, step=0.1)
    inputMileage = st.sidebar.number_input(label="Mileage", min_value=100, max_value=749000, value=6000)

    # categorical feature --> select_box
    inputType = st.sidebar.selectbox(label='Type', options=types_car)
    inputRegion = st.sidebar.selectbox(label='Region', options=regions_car)
    inputMake = st.sidebar.selectbox(label='Make', options=make_car)
    inputGearType = st.sidebar.selectbox(label='GearType', options=gear_types_car)
    inputOrigin = st.sidebar.selectbox(label='Origin', options=origins_car)
    inputOptions = st.sidebar.selectbox(label='Options', options=options_car)

    df = pd.DataFrame()
    df["Year"] = [inputYear]
    df["Engine_Size"] = [inputEngineSize]
    df["Mileage"] = [inputMileage]
    df["Type"] = [inputType]
    df["Region"] = [inputRegion]
    df["Make"] = [inputMake]
    df["Gear_Type"] = [inputGearType]
    df["Origin"] = [inputOrigin]
    df["Options"] = [inputOptions]

    return df

df_car = user_input_feature()

# predict customer
model_loaded = pickle.load(open('../model/Model_SaudiArabiaUsedCars_XGB.sav', 'rb'))

price = model_loaded.predict(df_car)

# Create 2 containers left and right

col1, col2 = st.columns(2)

# col1 --> first container
# displaying the dataframe from user input
with col1:
    st.subheader("Car Features")
    st.write(df_car.T.rename(columns={0: 'Value'}))

# col2 --> second container
with col2:

    # display the prediction result
    st.subheader('Prediction')

    st.write(f'Predicted used car price: {price[0]:,} SAR')
