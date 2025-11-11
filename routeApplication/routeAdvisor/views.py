from django.shortcuts import render
from django.conf import settings
import googlemaps
from datetime import datetime, timedelta
import folium
import os
import requests
import json
import pandas as pd
import joblib
from .datacleaning import Api_Data
from .predict_module import predict_from_api




def get_routes_from_new_api(api_key, origin, destination, departure_time_obj):
    url = "https://routes.googleapis.com/directions/v2:computeRoutes"
    payload = {
        "origin": {"address": origin},
        "destination": {"address": destination},
        "travelMode": "DRIVE",
        "computeAlternativeRoutes": True,
        "routingPreference": "TRAFFIC_AWARE",
        "departure_time": departure_time_obj.isoformat() + "Z",
    }
    headers = {
        'Content-Type': 'application/json',
        'X-Goog-Api-Key': api_key,
        'X-Goog-FieldMask': 'routes.duration,routes.distanceMeters,routes.polyline.encodedPolyline,routes.description'
    }
    resp = requests.post(url, data=json.dumps(payload), headers=headers)
    resp.raise_for_status()
    return resp.json()


def decode_polyline(polyline_str):
    index, lat, lng = 0, 0, 0
    coordinates = []
    changes = {'latitude': 0, 'longitude': 0}
    while index < len(polyline_str):
        for unit in ['latitude', 'longitude']:
            shift, result = 0, 0
            while True:
                byte = ord(polyline_str[index]) - 63
                index += 1
                result |= (byte & 0x1f) << shift
                shift += 5
                if not byte >= 0x20:
                    break
            if result & 1:
                changes[unit] = ~(result >> 1)
            else:
                changes[unit] = (result >> 1)
        lat += changes['latitude']
        lng += changes['longitude']
        coordinates.append((lat / 100000.0, lng / 100000.0))
    return coordinates


def parse_duration_string(duration_str):
    if not isinstance(duration_str, str) or not duration_str.endswith('s'):
        return 0
    try:
        return int(float(duration_str[:-1]))
    except ValueError:
        return 0


def get_valid_coords(gmaps_client, address):
    geocode_result = gmaps_client.geocode(address)
    if geocode_result:
        location = geocode_result[0]['geometry']['location']
        return [float(location['lat']), float(location['lng'])]
    return None




def home(request):
    context = {}
    if request.method == 'POST':
        try:
            source = request.POST.get('source')
            destination = request.POST.get('destination')
            datetime_str = request.POST.get('datetime')
            departure_datetime_obj = datetime.strptime(datetime_str, '%Y-%m-%dT%H:%M')

            # Step 1: Fetch routes from Google
            api_response = get_routes_from_new_api(
                settings.GOOGLE_MAPS_API_KEY, source, destination, departure_datetime_obj
            )
            if not api_response or 'routes' not in api_response:
                raise ValueError("No routes found. Please check locations and try again.")

            processed_routes = []
            gmaps_client = googlemaps.Client(key=settings.GOOGLE_MAPS_API_KEY)
            coords = get_valid_coords(gmaps_client, source)

            # Step 2: Fetch weather data once
            api_obj = Api_Data(coords)
            weather_data = api_obj.get_forecast_data(datetime_str.replace("T", " ") + ":00")

            # Step 3: Predict traffic using your saved ML model
            predicted_traffic = None
            if weather_data:
                predicted_traffic = predict_from_api(weather_data)  # ✅ Direct call to your ML prediction

            # Step 4: Process route results
            for i, route in enumerate(api_response['routes']):
                duration_seconds = parse_duration_string(route.get('duration'))
                duration_text = str(timedelta(seconds=duration_seconds))

                adjusted_text = None
                extra_time_text = None

                if predicted_traffic:
                    # Optional: Adjust ETA using a scaling factor
                    V_avg = 40000
                    alpha = 0.5
                    adjusted_time = duration_seconds * (1 + alpha * (predicted_traffic / V_avg))
                    adjusted_text = str(timedelta(seconds=round(adjusted_time)))
                    extra_seconds = round(adjusted_time - duration_seconds)
                    extra_time_text = str(timedelta(seconds=extra_seconds))

                processed_routes.append({
                    'id': i,
                    'summary': route.get('description', 'N/A'),
                    'duration_seconds': duration_seconds,
                    'duration_text': duration_text,
                    'adjusted_duration_text': adjusted_text,
                    'predicted_traffic': predicted_traffic,
                    'extra_time_text': extra_time_text,
                    'polyline': route.get('polyline', {}).get('encodedPolyline')
                })

            # Step 5: Select best route
            best_route = min(
                processed_routes,
                key=lambda x: timedelta(seconds=int(x['adjusted_duration_text'].split(':')[0])*3600
                                        + int(x['adjusted_duration_text'].split(':')[1])*60
                                        + float(x['adjusted_duration_text'].split(':')[2]))
                if x['adjusted_duration_text'] else x['duration_seconds']
            )

            # Step 6: Build message
            extra_time_msg = f"Extra time due to predicted traffic: {best_route['extra_time_text']}" if best_route.get('extra_time_text') else ''
            context['recommendation'] = (
                f"The best route is via {best_route['summary']}. "
                f"Google ETA: {best_route['duration_text']}. "
                f"{'(Adjusted ETA: ' + best_route['adjusted_duration_text'] + ')' if best_route.get('adjusted_duration_text') else ''} "
                f"{extra_time_msg}"
            )

            # Step 7: Draw map
            start_coords = coords
            m = folium.Map(location=start_coords, zoom_start=13, tiles="cartodbpositron")

            for route in processed_routes:
                if route['polyline']:
                    points = decode_polyline(route['polyline'])
                    is_best = route['id'] == best_route['id']
                    folium.PolyLine(
                        points,
                        color='blue' if is_best else 'gray',
                        weight=6 if is_best else 4,
                        opacity=0.9 if is_best else 0.7,
                        tooltip=f"{route['summary']}<br>Predicted traffic: {route['predicted_traffic']}"
                    ).add_to(m)

            end_coords = get_valid_coords(gmaps_client, destination)
            if end_coords:
                folium.Marker(location=end_coords, popup=f"<b>End:</b><br>{destination}",
                              icon=folium.Icon(color='red', icon='stop')).add_to(m)

            folium.Marker(location=start_coords, popup=f"<b>Start:</b><br>{source}",
                          icon=folium.Icon(color='green', icon='play')).add_to(m)

            context['map_html'] = m._repr_html_()

        except Exception as e:
            context['error'] = f"An error occurred: {e}"

    return render(request, 'home.html', context)
