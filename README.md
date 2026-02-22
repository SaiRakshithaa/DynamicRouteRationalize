# DynamicRouteRationalize

DynamicRouteRationalize is a **Django-based web application** that predicts **accurate travel times** for routes by combining real-time data from **Google Maps API** with external datasets such as **weather, holidays, and historical traffic patterns**. The system uses a **machine learning model (XGBoost)** to adjust ETAs, providing smarter route recommendations.

---

## Features

- Fetches routes and ETAs from **Google Maps API**
- Adjusts ETA using **weather, holiday, and traffic datasets**
- Predicts delays using a **trained XGBoost model**
- Visualizes routes on an interactive **Folium map**
- Highlights the **best route** based on predicted travel time

---

## Demo

- Input: Source, Destination, and Departure Time  
- Output: Recommended route with **adjusted ETA**, **extra time due to traffic**, and a **visual route map**

---

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/SaiRakshithaa/DynamicRouteRationalize.git
cd DynamicRouteRationalize
```
2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. **Set up Google Maps API key**
In settings.py, add:
```bash
GOOGLE_MAPS_API_KEY = "YOUR_GOOGLE_MAPS_API_KEY"
```
4. **Run Django server**
```bash
python manage.py runserver
```
---

## Conclusion & Future Work

DynamicRouteRationalize demonstrates how **real-time route prediction** can be enhanced by combining **Google Maps API** data with **environmental and contextual datasets** such as weather, holidays, and traffic patterns. The use of an **XGBoost model** ensures that predicted delays are accurate, making travel planning more reliable.

**Future improvements** could include:

- Incorporating **real-time traffic feeds** for even more precise predictions  
- Adding **public transport options** alongside driving routes  
- Using **more advanced ML models** or **deep learning** for better accuracy  
- Deploying the app as a **production-ready web service** with user authentication  

This project highlights the power of combining **machine learning** with **geospatial APIs** to create smarter, data-driven solutions.
