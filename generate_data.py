"""
generate_data.py  ·  LoRRI Shipment Data Generator
Run: python generate_data.py
Outputs: shipments.csv
"""
import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

DEPOT_LAT, DEPOT_LON = 19.0760, 72.8777

INDIAN_CITIES = [
    ("Delhi",              28.6139,  77.2090),
    ("Bengaluru",          12.9716,  77.5946),
    ("Chennai",            13.0827,  80.2707),
    ("Hyderabad",          17.3850,  78.4867),
    ("Kolkata",            22.5726,  88.3639),
    ("Pune",               18.5204,  73.8567),
    ("Ahmedabad",          23.0225,  72.5714),
    ("Jaipur",             26.9124,  75.7873),
    ("Surat",              21.1702,  72.8311),
    ("Lucknow",            26.8467,  80.9462),
    ("Nagpur",             21.1458,  79.0882),
    ("Indore",             22.7196,  75.8577),
    ("Bhopal",             23.2599,  77.4126),
    ("Visakhapatnam",      17.6868,  83.2185),
    ("Patna",              25.5941,  85.1376),
    ("Vadodara",           22.3072,  73.1812),
    ("Ludhiana",           30.9010,  75.8573),
    ("Agra",               27.1767,  78.0081),
    ("Nashik",             19.9975,  73.7898),
    ("Faridabad",          28.4089,  77.3178),
    ("Meerut",             28.9845,  77.7064),
    ("Rajkot",             22.3039,  70.8022),
    ("Varanasi",           25.3176,  82.9739),
    ("Srinagar",           34.0837,  74.7973),
    ("Aurangabad",         19.8762,  75.3433),
    ("Dhanbad",            23.7957,  86.4304),
    ("Amritsar",           31.6340,  74.8723),
    ("Allahabad",          25.4358,  81.8463),
    ("Ranchi",             23.3441,  85.3096),
    ("Coimbatore",         11.0168,  76.9558),
    ("Jabalpur",           23.1815,  79.9864),
    ("Gwalior",            26.2183,  78.1828),
    ("Vijayawada",         16.5062,  80.6480),
    ("Jodhpur",            26.2389,  73.0243),
    ("Madurai",             9.9252,  78.1198),
    ("Raipur",             21.2514,  81.6296),
    ("Kota",               25.2138,  75.8648),
    ("Chandigarh",         30.7333,  76.7794),
    ("Guwahati",           26.1445,  91.7362),
    ("Thiruvananthapuram",  8.5241,  76.9366),
    ("Solapur",            17.6599,  75.9064),
    ("Hubli",              15.3647,  75.1240),
    ("Tiruchirappalli",    10.7905,  78.7047),
    ("Bareilly",           28.3670,  79.4304),
    ("Mysuru",             12.2958,  76.6394),
    ("Bhubaneswar",        20.2961,  85.8245),
    ("Dehradun",           30.3165,  78.0322),
    ("Jammu",              32.7266,  74.8570),
    ("Mangalore",          12.9141,  74.8560),
    ("Udaipur",            24.5854,  73.7125),
]

n = len(INDIAN_CITIES)
lat_jitter = np.random.uniform(-0.05, 0.05, n)
lon_jitter = np.random.uniform(-0.05, 0.05, n)

toll_base      = {c[0]: np.random.uniform(800,  3500) for c in INDIAN_CITIES}
traffic_mult   = {c[0]: round(np.random.uniform(1.0,  1.8),  2) for c in INDIAN_CITIES}
emission_factor= {c[0]: round(np.random.uniform(0.18, 0.32), 3) for c in INDIAN_CITIES}
sla_window     = {"HIGH": 24, "MEDIUM": 48, "LOW": 72}
priorities     = np.random.choice(["HIGH", "MEDIUM", "LOW"], n, p=[0.2, 0.5, 0.3])

data = {
    "id":              [f"SHIP_{i:03d}" for i in range(1, n + 1)],
    "city":            [c[0] for c in INDIAN_CITIES],
    "latitude":        np.round(np.array([c[1] for c in INDIAN_CITIES]) + lat_jitter, 6),
    "longitude":       np.round(np.array([c[2] for c in INDIAN_CITIES]) + lon_jitter, 6),
    "weight":          np.round(np.random.uniform(10, 500, n), 2),
    "priority":        priorities,
    "sla_hours":       [sla_window[p] for p in priorities],
    "toll_cost_inr":   [round(toll_base[c[0]], 2) for c in INDIAN_CITIES],
    "traffic_mult":    [traffic_mult[c[0]] for c in INDIAN_CITIES],
    "emission_factor": [emission_factor[c[0]] for c in INDIAN_CITIES],
}

df = pd.DataFrame(data)
df.to_csv("shipments.csv", index=False)

print("✅ shipments.csv created — 50 Indian cities")
print(f"   Depot: Mumbai ({DEPOT_LAT}°N, {DEPOT_LON}°E)")
print(f"\n   Shipments : {n}")
print(f"   Priorities: {dict(df['priority'].value_counts())}")
print(f"   Avg toll  : ₹{df['toll_cost_inr'].mean():.0f}")
print(f"   Avg traffic: {df['traffic_mult'].mean():.2f}x")
