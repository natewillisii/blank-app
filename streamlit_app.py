# Data Integration Setup
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import asyncio
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import json

# API Keys setup
TOMORROW_API_KEY = st.secrets["TOMORROW_API_KEY"]
WEATHERBIT_API_KEY = st.secrets["WEATHERBIT_API_KEY"]

# Weather API Integration Setup
class WeatherAPI:
    def __init__(self):
        self.tomorrow_base_url = "https://api.tomorrow.io/v4/weather"
        self.weatherbit_base_url = "https://api.weatherbit.io/v2.0"
    
    @staticmethod
    @st.cache_data(ttl=300)
    def get_weather_data(location):
        """Get weather data for a location"""
        try:
            base_url = "https://api.tomorrow.io/v4/weather"
            params = {
                "location": location,
                "apikey": st.secrets["TOMORROW_API_KEY"],
                "fields": ["temperature", "precipitation", "windSpeed", "humidity"]
            }
            response = requests.get(f"{base_url}/realtime", params=params)
            response.raise_for_status()  # Raise an exception for bad status codes
            return response.json()
        except Exception as e:
            st.error(f"Error fetching weather data: {str(e)}")
            return {"data": {"values": {}}}
    
    @staticmethod
    @st.cache_data(ttl=600)
    def get_weather_alerts(lat, lon):
        """Get weather alerts for coordinates"""
        try:
            base_url = "https://api.weatherbit.io/v2.0"
            params = {
                "lat": lat,
                "lon": lon,
                "key": st.secrets["WEATHERBIT_API_KEY"]
            }
            response = requests.get(f"{base_url}/alerts", params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error fetching weather alerts: {str(e)}")
            return {}

# Disaster Prediction Model Setup
class DisasterPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)
        self.scaler = StandardScaler()
        # Add some initial data to fit the scaler
        self.scaler.fit(np.array([[0, 0, 0, 0]]))  # Fit with dummy data initially
        # Fit the model with dummy data and labels
        X_dummy = np.array([[0, 0, 0, 0], [1, 1, 1, 1]])
        y_dummy = np.array([0, 1])
        self.model.fit(X_dummy, y_dummy)
        
    def prepare_features(self, weather_data):
        features = np.array([[
            weather_data.get('data', {}).get('values', {}).get('temperature', 0),
            weather_data.get('data', {}).get('values', {}).get('precipitation', 0),
            weather_data.get('data', {}).get('values', {}).get('windSpeed', 0),
            weather_data.get('data', {}).get('values', {}).get('humidity', 0)
        ]])
        # First fit, then transform
        self.scaler.fit(features)  # Fit with actual data
        return self.scaler.transform(features)
    
    def predict_risk(self, weather_data):
        features = self.prepare_features(weather_data)
        try:
            risk_prob = self.model.predict_proba(features)[0]
            return {
                'risk_level': 'High' if risk_prob[1] > 0.7 else 'Medium' if risk_prob[1] > 0.3 else 'Low',
                'probability': risk_prob[1]
            }
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return {
                'risk_level': 'Unknown',
                'probability': 0.0
            }

# Real-time Updates Implementation
class FEMADisasterAPI:
    @staticmethod
    @st.cache_data(ttl=300)
    def get_disaster_declarations(state=None):
        """Get recent disaster declarations from FEMA"""
        try:
            base_url = "https://www.fema.gov/api/open/v2"
            endpoint = f"{base_url}/DisasterDeclarationsSummaries"
            params = {
                "$format": "json",
                "$orderby": "declarationDate desc",
                "$top": 100
            }
            if state:
                params["$filter"] = f"state eq '{state}'"
                
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error fetching FEMA declarations: {str(e)}")
            return {}
    
    @staticmethod
    @st.cache_data(ttl=300)
    def get_disaster_summaries():
        """Get disaster summaries from FEMA Web"""
        try:
            base_url = "https://www.fema.gov/api/open/v2"
            endpoint = f"{base_url}/FemaWebDisasterSummaries"
            params = {
                "$format": "json",
                "$orderby": "declarationDate desc",
                "$top": 50
            }
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error fetching FEMA summaries: {str(e)}")
            return {}

# Main Application 
def main():
    st.set_page_config(page_title="Disaster Monitor & Predict", layout="wide")
    
    with st.spinner("Loading disaster monitoring system..."):
        # Initialize components
        weather_api = WeatherAPI()
        disaster_predictor = DisasterPredictor()
    
    # Create placeholder for real-time updates
    placeholder = st.empty()
    
    async def update_dashboard():
        while True:
            try:
                with placeholder.container():
                    # Get real-time weather data using static methods
                    weather_data = WeatherAPI.get_weather_data("Boston")
                    alerts = WeatherAPI.get_weather_alerts(42.3601, -71.0589)
                    
                    # Safely get weather values
                    values = weather_data.get('data', {}).get('values', {})
                    
                    # Predict disaster risk
                    risk_assessment = disaster_predictor.predict_risk(weather_data)
                    
                    # Display metrics using placeholders
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Temperature", f"{values.get('temperature', 'N/A')}°C")
                    with col2:
                        st.metric("Wind Speed", f"{values.get('windSpeed', 'N/A')} km/h")
                    with col3:
                        st.metric("Risk Level", risk_assessment['risk_level'])
                    
                    # Display alerts
                    if alerts:
                        st.warning("Active Weather Alerts")
                        for alert in alerts.get('alerts', []):
                            st.write(f"⚠️ {alert.get('title', 'Unknown Alert')}")
                    
                    # Display FEMA disaster data
                    disaster_data = FEMADisasterAPI.get_disaster_declarations("MA")
                    if disaster_data:
                        st.subheader("Recent Disaster Declarations")
                        for declaration in disaster_data.get('DisasterDeclarationsSummaries', [])[:5]:
                            st.info(
                                f"🚨 {declaration.get('declarationTitle', 'Unknown Disaster')} - {declaration.get('declarationDate', 'Date unknown')}"
                            )
                    
                    # Create risk gauge chart with unique key using timestamp
                    current_time = datetime.now().strftime("%Y%m%d%H%M%S%f")
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=risk_assessment['probability'] * 100,
                        title={'text': "Disaster Risk Probability"},
                        gauge={'axis': {'range': [0, 100]}}
                    ))
                    st.plotly_chart(fig, key=f"risk_gauge_chart_{current_time}", use_container_width=True)
                
                await asyncio.sleep(30)  # Update every 30 seconds
            except Exception as e:
                st.error(f"Dashboard update error: {str(e)}")
                await asyncio.sleep(5)  # Shorter retry interval on error

    # Run the async update loop
    asyncio.run(update_dashboard())

if __name__ == "__main__":
    main()
