"""
generate_data.py — India-Realistic Synthetic Data Generator
============================================================
Generates shipment, vehicle, and lane data modelled on real Indian
logistics conditions: LoRRI-style truck categories, real city coordinates,
INR-based freight rates, actual toll plazas, and stochastic demand patterns.

Run:
    python generate_data.py

Outputs:
    shipments.csv       — Individual shipment records
    vehicles.csv        — Fleet / vehicle master data
    lanes.csv           — Lane-level metadata (distance, tolls, road type)
    demand_forecast.csv — Stochastic demand variations per lane
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# ─────────────────────────────────────────────
# SEED FOR REPRODUCIBILITY
# ─────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────
# 1. INDIAN CITY / HUB MASTER DATA
#    lat/lon sourced from real coordinates
# ─────────────────────────────────────────────
CITIES = {
    # (city_name): (lat, lon, state, tier)
    "Mumbai":       (19.0760,  72.8777, "Maharashtra",    1),
    "Delhi":        (28.6139,  77.2090, "Delhi",          1),
    "Bengaluru":    (12.9716,  77.5946, "Karnataka",      1),
    "Chennai":      (13.0827,  80.2707, "Tamil Nadu",     1),
    "Hyderabad":    (17.3850,  78.4867, "Telangana",      1),
    "Pune":         (18.5204,  73.8567, "Maharashtra",    1),
    "Ahmedabad":    (23.0225,  72.5714, "Gujarat",        1),
    "Kolkata":      (22.5726,  88.3639, "West Bengal",    1),
    "Surat":        (21.1702,  72.8311, "Gujarat",        2),
    "Jaipur":       (26.9124,  75.7873, "Rajasthan",      2),
    "Lucknow":      (26.8467,  80.9462, "Uttar Pradesh",  2),
    "Nagpur":       (21.1458,  79.0882, "Maharashtra",    2),
    "Indore":       (22.7196,  75.8577, "Madhya Pradesh", 2),
    "Bhopal":       (23.2599,  77.4126, "Madhya Pradesh", 2),
    "Nashik":       (19.9975,  73.7898, "Maharashtra",    2),
    "Vadodara":     (22.3072,  73.1812, "Gujarat",        2),
    "Coimbatore":   (11.0168,  76.9558, "Tamil Nadu",     2),
    "Visakhapatnam":(17.6868,  83.2185, "Andhra Pradesh", 2),
    "Patna":        (25.5941,  85.1376, "Bihar",          2),
    "Chandigarh":   (30.7333,  76.7794, "Punjab",         2),
    "Kochi":        ( 9.9312,  76.2673, "Kerala",         2),
    "Aurangabad":   (19.8762,  75.3433, "Maharashtra",    3),
    "Rajkot":       (22.3039,  70.8022, "Gujarat",        3),
    "Ludhiana":     (30.9010,  75.8573, "Punjab",         3),
    "Agra":         (27.1767,  78.0081, "Uttar Pradesh",  3),
    "Varanasi":     (25.3176,  82.9739, "Uttar Pradesh",  3),
    "Amritsar":     (31.6340,  74.8723, "Punjab",         3),
    "Jodhpur":      (26.2389,  73.0243, "Rajasthan",      3),
    "Guwahati":     (26.1445,  91.7362, "Assam",          3),
    "Raipur":       (21.2514,  81.6296, "Chhattisgarh",   3),
}

# LoRRI-aligned truck types with real Indian specs
TRUCK_TYPES = {
    "Tata Ace (Mini)": {
        "capacity_ton": 0.75,
        "capacity_cbm": 4,
        "cost_per_km_inr": 12,
        "fixed_cost_inr": 800,
        "fuel_efficiency_kmpl": 18,
        "co2_per_km_kg": 0.18,
        "toll_multiplier": 0.5,
        "category": "LCV",
    },
    "Tata 407 (SCV)": {
        "capacity_ton": 2.5,
        "capacity_cbm": 16,
        "cost_per_km_inr": 18,
        "fixed_cost_inr": 1200,
        "fuel_efficiency_kmpl": 14,
        "co2_per_km_kg": 0.28,
        "toll_multiplier": 0.75,
        "category": "SCV",
    },
    "Eicher 10T (ICV)": {
        "capacity_ton": 10,
        "capacity_cbm": 40,
        "cost_per_km_inr": 28,
        "fixed_cost_inr": 2500,
        "fuel_efficiency_kmpl": 10,
        "co2_per_km_kg": 0.55,
        "toll_multiplier": 1.0,
        "category": "ICV",
    },
    "Ashok Leyland 14T (MCV)": {
        "capacity_ton": 14,
        "capacity_cbm": 55,
        "cost_per_km_inr": 35,
        "fixed_cost_inr": 3200,
        "fuel_efficiency_kmpl": 8,
        "co2_per_km_kg": 0.72,
        "toll_multiplier": 1.25,
        "category": "MCV",
    },
    "Tata 1109 20T (HCV)": {
        "capacity_ton": 20,
        "capacity_cbm": 72,
        "cost_per_km_inr": 45,
        "fixed_cost_inr": 4500,
        "fuel_efficiency_kmpl": 6,
        "co2_per_km_kg": 0.95,
        "toll_multiplier": 1.5,
        "category": "HCV",
    },
    "Volvo 32T (HXL)": {
        "capacity_ton": 32,
        "capacity_cbm": 105,
        "cost_per_km_inr": 65,
        "fixed_cost_inr": 7000,
        "fuel_efficiency_kmpl": 4.5,
        "co2_per_km_kg": 1.40,
        "toll_multiplier": 2.0,
        "category": "HXL",
    },
    "Trailer 40T (MXL)": {
        "capacity_ton": 40,
        "capacity_cbm": 130,
        "cost_per_km_inr": 80,
        "fixed_cost_inr": 9000,
        "fuel_efficiency_kmpl": 3.5,
        "co2_per_km_kg": 1.80,
        "toll_multiplier": 2.5,
        "category": "MXL",
    },
}

# Real Indian national highway corridors with tolls (INR per trip, approx)
HIGHWAY_CORRIDORS = {
    ("Mumbai",    "Pune"):        {"highway": "NH48",  "toll_inr": 320,  "road_quality": "Expressway"},
    ("Mumbai",    "Nashik"):      {"highway": "NH3",   "toll_inr": 280,  "road_quality": "NH"},
    ("Mumbai",    "Ahmedabad"):   {"highway": "NH48",  "toll_inr": 850,  "road_quality": "Expressway"},
    ("Mumbai",    "Surat"):       {"highway": "NH48",  "toll_inr": 620,  "road_quality": "NH"},
    ("Delhi",     "Jaipur"):      {"highway": "NH48",  "toll_inr": 430,  "road_quality": "Expressway"},
    ("Delhi",     "Agra"):        {"highway": "NH19",  "toll_inr": 380,  "road_quality": "Expressway"},
    ("Delhi",     "Chandigarh"):  {"highway": "NH44",  "toll_inr": 290,  "road_quality": "NH"},
    ("Delhi",     "Ludhiana"):    {"highway": "NH44",  "toll_inr": 520,  "road_quality": "NH"},
    ("Delhi",     "Lucknow"):     {"highway": "NH19",  "toll_inr": 680,  "road_quality": "Expressway"},
    ("Bengaluru", "Chennai"):     {"highway": "NH48",  "toll_inr": 560,  "road_quality": "NH"},
    ("Bengaluru", "Hyderabad"):   {"highway": "NH44",  "toll_inr": 720,  "road_quality": "NH"},
    ("Bengaluru", "Coimbatore"):  {"highway": "NH544", "toll_inr": 380,  "road_quality": "NH"},
    ("Chennai",   "Hyderabad"):   {"highway": "NH65",  "toll_inr": 640,  "road_quality": "NH"},
    ("Pune",      "Hyderabad"):   {"highway": "NH65",  "toll_inr": 820,  "road_quality": "NH"},
    ("Pune",      "Nagpur"):      {"highway": "NH65",  "toll_inr": 740,  "road_quality": "NH"},
    ("Ahmedabad", "Indore"):      {"highway": "NH47",  "toll_inr": 460,  "road_quality": "NH"},
    ("Ahmedabad", "Surat"):       {"highway": "NH48",  "toll_inr": 220,  "road_quality": "Expressway"},
    ("Nagpur",    "Hyderabad"):   {"highway": "NH44",  "toll_inr": 580,  "road_quality": "NH"},
    ("Nagpur",    "Raipur"):      {"highway": "NH53",  "toll_inr": 390,  "road_quality": "NH"},
    ("Kolkata",   "Patna"):       {"highway": "NH19",  "toll_inr": 520,  "road_quality": "NH"},
    ("Lucknow",   "Varanasi"):    {"highway": "NH19",  "toll_inr": 310,  "road_quality": "NH"},
    ("Jaipur",    "Jodhpur"):     {"highway": "NH62",  "toll_inr": 290,  "road_quality": "NH"},
}

# Cargo categories relevant to Indian freight market
CARGO_TYPES = [
    {"type": "FMCG",             "density_ton_per_cbm": 0.4,  "fragile": False, "hazmat": False, "premium_pct": 0},
    {"type": "Automotive Parts", "density_ton_per_cbm": 0.7,  "fragile": False, "hazmat": False, "premium_pct": 5},
    {"type": "Electronics",      "density_ton_per_cbm": 0.3,  "fragile": True,  "hazmat": False, "premium_pct": 15},
    {"type": "Textiles",         "density_ton_per_cbm": 0.25, "fragile": False, "hazmat": False, "premium_pct": 0},
    {"type": "Pharmaceuticals",  "density_ton_per_cbm": 0.35, "fragile": True,  "hazmat": False, "premium_pct": 20},
    {"type": "Steel/Metal",      "density_ton_per_cbm": 3.5,  "fragile": False, "hazmat": False, "premium_pct": 0},
    {"type": "Chemicals",        "density_ton_per_cbm": 1.2,  "fragile": False, "hazmat": True,  "premium_pct": 25},
    {"type": "Food & Agri",      "density_ton_per_cbm": 0.6,  "fragile": False, "hazmat": False, "premium_pct": 0},
    {"type": "E-commerce",       "density_ton_per_cbm": 0.2,  "fragile": True,  "hazmat": False, "premium_pct": 10},
    {"type": "Construction",     "density_ton_per_cbm": 2.0,  "fragile": False, "hazmat": False, "premium_pct": 0},
]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def haversine_km(city1: str, city2: str) -> float:
    """Great-circle distance between two cities in km."""
    lat1, lon1 = np.radians(CITIES[city1][:2])
    lat2, lon2 = np.radians(CITIES[city2][:2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def road_distance_km(city1: str, city2: str) -> float:
    """
    Approximate road distance = haversine * road_factor.
    Expressways ~1.15x, highways ~1.25x, others ~1.40x.
    """
    key = (city1, city2) if (city1, city2) in HIGHWAY_CORRIDORS else (city2, city1)
    if key in HIGHWAY_CORRIDORS:
        rq = HIGHWAY_CORRIDORS[key]["road_quality"]
        factor = 1.15 if rq == "Expressway" else 1.25
    else:
        factor = 1.40
    return round(haversine_km(city1, city2) * factor, 1)


def get_toll(city1: str, city2: str, truck_category: str) -> float:
    key = (city1, city2) if (city1, city2) in HIGHWAY_CORRIDORS else (city2, city1)
    if key not in HIGHWAY_CORRIDORS:
        # Estimate toll for unmapped lanes
        dist = road_distance_km(city1, city2)
        base = dist * 1.8  # ~₹1.8/km average toll
    else:
        base = HIGHWAY_CORRIDORS[key]["toll_inr"]
    multipliers = {
        "LCV": 0.5, "SCV": 0.75, "ICV": 1.0,
        "MCV": 1.25, "HCV": 1.5, "HXL": 2.0, "MXL": 2.5
    }
    return round(base * multipliers.get(truck_category, 1.0), 0)


def random_time_window(date: datetime, is_priority: bool):
    """Returns (earliest_pickup, latest_delivery) datetime strings."""
    if is_priority:
        start_hour = random.randint(6, 10)
        window_hours = random.randint(6, 12)
    else:
        start_hour = random.randint(6, 14)
        window_hours = random.randint(12, 36)
    pickup = date.replace(hour=start_hour, minute=0, second=0)
    delivery = pickup + timedelta(hours=window_hours)
    return pickup.strftime("%Y-%m-%d %H:%M"), delivery.strftime("%Y-%m-%d %H:%M")


def delay_probability(city1: str, city2: str, cargo_type: str) -> float:
    """
    Stochastic delay probability based on road quality, distance, cargo type.
    """
    dist = road_distance_km(city1, city2)
    key = (city1, city2) if (city1, city2) in HIGHWAY_CORRIDORS else (city2, city1)
    if key in HIGHWAY_CORRIDORS:
        rq = HIGHWAY_CORRIDORS[key]["road_quality"]
        base_prob = 0.08 if rq == "Expressway" else 0.15
    else:
        base_prob = 0.25

    # Longer routes are riskier
    if dist > 1000:
        base_prob += 0.10
    elif dist > 500:
        base_prob += 0.05

    # Hazmat or fragile cargo adds risk
    cargo = next((c for c in CARGO_TYPES if c["type"] == cargo_type), None)
    if cargo and (cargo["fragile"] or cargo["hazmat"]):
        base_prob += 0.05

    return round(min(base_prob, 0.60), 2)


# ─────────────────────────────────────────────
# 2. GENERATE SHIPMENTS
# ─────────────────────────────────────────────

def generate_shipments(n: int = 200) -> pd.DataFrame:
    city_names = list(CITIES.keys())
    records = []
    base_date = datetime(2025, 1, 1)

    for i in range(n):
        origin = random.choice(city_names)
        destination = random.choice([c for c in city_names if c != origin])

        cargo = random.choice(CARGO_TYPES)
        weight_ton = round(random.uniform(0.5, 35), 2)
        volume_cbm = round(weight_ton / cargo["density_ton_per_cbm"] * random.uniform(0.8, 1.2), 1)

        dist_km = road_distance_km(origin, destination)
        is_priority = random.random() < 0.25
        shipment_date = base_date + timedelta(days=random.randint(0, 89))
        earliest_pickup, latest_delivery = random_time_window(shipment_date, is_priority)

        # Determine ideal truck type based on weight
        if weight_ton <= 0.75:
            truck_type = "Tata Ace (Mini)"
        elif weight_ton <= 2.5:
            truck_type = "Tata 407 (SCV)"
        elif weight_ton <= 10:
            truck_type = "Eicher 10T (ICV)"
        elif weight_ton <= 14:
            truck_type = "Ashok Leyland 14T (MCV)"
        elif weight_ton <= 20:
            truck_type = "Tata 1109 20T (HCV)"
        elif weight_ton <= 32:
            truck_type = "Volvo 32T (HXL)"
        else:
            truck_type = "Trailer 40T (MXL)"

        truck = TRUCK_TYPES[truck_type]
        toll = get_toll(origin, destination, truck["category"])
        freight_cost_inr = round(
            truck["fixed_cost_inr"]
            + dist_km * truck["cost_per_km_inr"]
            + toll
            + (cargo["premium_pct"] / 100 * dist_km * truck["cost_per_km_inr"]),
            0
        )
        co2_kg = round(dist_km * truck["co2_per_km_kg"], 1)
        delay_prob = delay_probability(origin, destination, cargo["type"])

        key = (origin, destination) if (origin, destination) in HIGHWAY_CORRIDORS \
              else ((destination, origin) if (destination, origin) in HIGHWAY_CORRIDORS else None)
        highway = HIGHWAY_CORRIDORS[key]["highway"] if key else "State Road"
        road_quality = HIGHWAY_CORRIDORS[key]["road_quality"] if key else "State Road"

        records.append({
            "shipment_id":       f"SHP{str(i+1).zfill(4)}",
            "origin_city":       origin,
            "origin_state":      CITIES[origin][2],
            "origin_lat":        CITIES[origin][0],
            "origin_lon":        CITIES[origin][1],
            "destination_city":  destination,
            "destination_state": CITIES[destination][2],
            "destination_lat":   CITIES[destination][0],
            "destination_lon":   CITIES[destination][1],
            "cargo_type":        cargo["type"],
            "weight_ton":        weight_ton,
            "volume_cbm":        volume_cbm,
            "fragile":           cargo["fragile"],
            "hazmat":            cargo["hazmat"],
            "priority":          is_priority,
            "recommended_truck": truck_type,
            "truck_category":    truck["category"],
            "road_distance_km":  dist_km,
            "highway":           highway,
            "road_quality":      road_quality,
            "toll_inr":          toll,
            "freight_cost_inr":  freight_cost_inr,
            "co2_emission_kg":   co2_kg,
            "delay_probability": delay_prob,
            "earliest_pickup":   earliest_pickup,
            "latest_delivery":   latest_delivery,
            "shipment_date":     shipment_date.strftime("%Y-%m-%d"),
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# 3. GENERATE VEHICLES / FLEET
# ─────────────────────────────────────────────

def generate_vehicles(n_per_type: int = 5) -> pd.DataFrame:
    records = []
    hubs = [c for c in CITIES if CITIES[c][3] == 1]  # Tier-1 cities as depots
    vid = 1

    for truck_name, specs in TRUCK_TYPES.items():
        for _ in range(n_per_type):
            home_hub = random.choice(hubs)
            records.append({
                "vehicle_id":           f"VH{str(vid).zfill(3)}",
                "truck_type":           truck_name,
                "category":             specs["category"],
                "capacity_ton":         specs["capacity_ton"],
                "capacity_cbm":         specs["capacity_cbm"],
                "cost_per_km_inr":      specs["cost_per_km_inr"],
                "fixed_cost_per_day_inr": specs["fixed_cost_inr"],
                "fuel_efficiency_kmpl": specs["fuel_efficiency_kmpl"],
                "co2_per_km_kg":        specs["co2_per_km_kg"],
                "home_hub":             home_hub,
                "home_hub_lat":         CITIES[home_hub][0],
                "home_hub_lon":         CITIES[home_hub][1],
                "available":            True,
                "current_utilization_pct": round(random.uniform(40, 95), 1),
            })
            vid += 1

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# 4. GENERATE LANE METADATA
# ─────────────────────────────────────────────

def generate_lanes() -> pd.DataFrame:
    records = []
    all_cities = list(CITIES.keys())

    # All mapped corridors first
    for (orig, dest), info in HIGHWAY_CORRIDORS.items():
        dist = road_distance_km(orig, dest)
        records.append({
            "lane_id":          f"{orig[:3].upper()}-{dest[:3].upper()}",
            "origin":           orig,
            "destination":      dest,
            "highway":          info["highway"],
            "road_quality":     info["road_quality"],
            "distance_km":      dist,
            "base_toll_inr":    info["toll_inr"],
            "avg_transit_hrs":  round(dist / 55, 1),   # ~55 km/h average speed India
            "peak_delay_hrs":   round(dist / 55 * 0.20, 1),
            "lane_demand_index": round(random.uniform(0.5, 1.0), 2),
        })

    # Add some unmapped city pairs
    for _ in range(20):
        o = random.choice(all_cities)
        d = random.choice([c for c in all_cities if c != o])
        dist = road_distance_km(o, d)
        records.append({
            "lane_id":          f"{o[:3].upper()}-{d[:3].upper()}",
            "origin":           o,
            "destination":      d,
            "highway":          "State Road",
            "road_quality":     "State Road",
            "distance_km":      dist,
            "base_toll_inr":    round(dist * 1.8, 0),
            "avg_transit_hrs":  round(dist / 40, 1),   # Slower on state roads
            "peak_delay_hrs":   round(dist / 40 * 0.30, 1),
            "lane_demand_index": round(random.uniform(0.2, 0.7), 2),
        })

    df = pd.DataFrame(records).drop_duplicates(subset=["origin", "destination"])
    return df


# ─────────────────────────────────────────────
# 5. GENERATE STOCHASTIC DEMAND FORECAST
# ─────────────────────────────────────────────

def generate_demand_forecast(lanes_df: pd.DataFrame, days: int = 90) -> pd.DataFrame:
    records = []
    base_date = datetime(2025, 1, 1)

    for _, lane in lanes_df.iterrows():
        for day_offset in range(0, days, 7):  # Weekly granularity
            date = base_date + timedelta(days=day_offset)
            week_num = date.isocalendar()[1]

            # Seasonality: Q1 (Jan-Mar) moderate, festive season spike Oct-Nov
            seasonal_factor = 1.0
            if date.month in [10, 11]:   # Diwali/festive
                seasonal_factor = 1.45
            elif date.month in [3, 4]:   # Financial year end
                seasonal_factor = 1.25
            elif date.month in [6, 7]:   # Monsoon slowdown
                seasonal_factor = 0.80

            base_demand = lane["lane_demand_index"] * 20  # Base shipments/week
            demand = max(1, int(np.random.poisson(base_demand * seasonal_factor)))
            delay_risk = min(0.9, lane["base_toll_inr"] / 5000 + random.uniform(0, 0.2))

            records.append({
                "lane_id":         lane["lane_id"],
                "origin":          lane["origin"],
                "destination":     lane["destination"],
                "week_start_date": date.strftime("%Y-%m-%d"),
                "week_number":     week_num,
                "predicted_shipments": demand,
                "seasonal_factor": round(seasonal_factor, 2),
                "delay_risk_score": round(delay_risk, 2),
                "recommended_fleet_size": max(1, demand // 5),
            })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# MAIN — GENERATE AND SAVE ALL FILES
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("🚚 Generating India-realistic logistics data...\n")

    print("  → Generating 200 shipments...")
    shipments_df = generate_shipments(n=200)
    shipments_df.to_csv("shipments.csv", index=False)
    print(f"     ✅ shipments.csv saved ({len(shipments_df)} rows)")

    print("  → Generating fleet / vehicle data...")
    vehicles_df = generate_vehicles(n_per_type=5)
    vehicles_df.to_csv("vehicles.csv", index=False)
    print(f"     ✅ vehicles.csv saved ({len(vehicles_df)} rows)")

    print("  → Generating lane metadata...")
    lanes_df = generate_lanes()
    lanes_df.to_csv("lanes.csv", index=False)
    print(f"     ✅ lanes.csv saved ({len(lanes_df)} rows)")

    print("  → Generating stochastic demand forecast...")
    demand_df = generate_demand_forecast(lanes_df, days=90)
    demand_df.to_csv("demand_forecast.csv", index=False)
    print(f"     ✅ demand_forecast.csv saved ({len(demand_df)} rows)")

    print("\n📊 Data Summary:")
    print(f"   Cities covered:      {len(CITIES)} (across {len(set(v[2] for v in CITIES.values()))} states)")
    print(f"   Truck types:         {len(TRUCK_TYPES)}")
    print(f"   Highway corridors:   {len(HIGHWAY_CORRIDORS)}")
    print(f"   Shipments:           {len(shipments_df)}")
    print(f"   Vehicles in fleet:   {len(vehicles_df)}")
    print(f"   Lanes mapped:        {len(lanes_df)}")
    print(f"   Demand records:      {len(demand_df)}")

    print("\n💰 Cost Stats (INR):")
    print(f"   Avg freight cost:    ₹{shipments_df['freight_cost_inr'].mean():,.0f}")
    print(f"   Total freight value: ₹{shipments_df['freight_cost_inr'].sum():,.0f}")
    print(f"   Avg toll per trip:   ₹{shipments_df['toll_inr'].mean():,.0f}")

    print("\n🌿 Carbon Stats:")
    print(f"   Total CO2 (kg):      {shipments_df['co2_emission_kg'].sum():,.0f} kg")
    print(f"   Avg CO2 per trip:    {shipments_df['co2_emission_kg'].mean():,.1f} kg")

    print("\n✅ All files generated successfully. Run route_solver.py next.")