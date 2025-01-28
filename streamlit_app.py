#Data Integration Setup
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



#Weather API Integration Setup
class WeatherAPI:
    def __init__(self):
        self.tomorrow_base_url = "https://api.tomorrow.io/v4/weather"
        self.weatherbit_base_url = "https://api.weatherbit.io/v2.0"
    
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_weather_data(self, location):
        params = {
            "location": location,
            "apikey": TOMORROW_API_KEY,
            "fields": ["temperature", "precipitation", "windSpeed", "humidity"]
        }
        response = requests.get(f"{self.tomorrow_base_url}/realtime", params=params)
        return response.json()
    
    @st.cache_data(ttl=600)  # Cache for 10 minutes
    def get_weather_alerts(self, lat, lon):
        params = {
            "lat": lat,
            "lon": lon,
            "key": WEATHERBIT_API_KEY
        }
        response = requests.get(f"{self.weatherbit_base_url}/alerts", params=params)
        return response.json()

#Disaster Prediction Model Setup
class DisasterPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)
        self.scaler = StandardScaler()
        
    def prepare_features(self, weather_data):
        features = np.array([[
            weather_data['temperature'],
            weather_data['precipitation'],
            weather_data['windSpeed'],
            weather_data['humidity']
        ]])
        return self.scaler.transform(features)
    
    def predict_risk(self, weather_data):
        features = self.prepare_features(weather_data)
        risk_prob = self.model.predict_proba(features)[0]
        return {
            'risk_level': 'High' if risk_prob[1] > 0.7 else 'Medium' if risk_prob[1] > 0.3 else 'Low',
            'probability': risk_prob[1]
        }

#Real-time Updates Implementation
class FEMADisasterAPI:
    def __init__(self):
        self.base_url = "https://www.fema.gov/api/open/v2"
    
    @st.cache_data(ttl=300)
    def get_disaster_declarations(self, state=None):
        """Get recent disaster declarations from FEMA"""
        endpoint = f"{self.base_url}/DisasterDeclarationsSummaries"
        params = {
            "$format": "json",
            "$orderby": "declarationDate desc",
            "$top": 100  # Limit to most recent 100 declarations
        }
        if state:
            params["$filter"] = f"state eq '{state}'"
            
        response = requests.get(endpoint, params=params)
        return response.json()
    
    @st.cache_data(ttl=300)
    def get_disaster_summaries(self):
        """Get disaster summaries from FEMA Web"""
        endpoint = f"{self.base_url}/FemaWebDisasterSummaries"
        params = {
            "$format": "json",
            "$orderby": "declarationDate desc",
            "$top": 50
        }
        response = requests.get(endpoint, params=params)
        return response.json()

    
    async def receive_updates(self):
        while self.connected:
            try:
                message = await self.websocket.recv()
                data = json.loads(message)
                st.session_state.update(data)
            except Exception as e:
                st.error(f"Error receiving updates: {e}")
                break

#Main Application 
def main():
    st.set_page_config(page_title="Disaster Monitor & Predict", layout="wide")
    
    # Initialize components
    weather_api = WeatherAPI()
    disaster_predictor = DisasterPredictor()
    fema_api = FEMADisasterAPI() 
    
    # Create placeholder for real-time updates
    placeholder = st.empty()
    
    async def update_dashboard():
        while True:
            with placeholder.container():
                # Get real-time weather data
                weather_data = weather_api.get_weather_data("Boston")
                alerts = weather_api.get_weather_alerts(42.3601, -71.0589)
                
                # Predict disaster risk
                risk_assessment = disaster_predictor.predict_risk(weather_data)
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Temperature", f"{weather_data['temperature']}°C")
                with col2:
                    st.metric("Wind Speed", f"{weather_data['windSpeed']} km/h")
                with col3:
                    st.metric("Risk Level", risk_assessment['risk_level'])
                
                # Display alerts
                if alerts:
                    st.warning("Active Weather Alerts")
                    for alert in alerts:
                        st.write(f"⚠️ {alert['title']}")
                
                # Create risk gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_assessment['probability'] * 100,
                    title={'text': "Disaster Risk Probability"},
                    gauge={'axis': {'range': [0, 100]}}
                ))
                st.plotly_chart(fig)
            
            await asyncio.sleep(30)  # Update every 30 seconds

    # Run the async update loop
    asyncio.run(update_dashboard())

if __name__ == "__main__":
    main()
