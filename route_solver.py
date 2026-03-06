"""
route_solver.py  ·  LoRRI Multi-Objective CVRP Solver
Run: python route_solver.py
Outputs: routes.csv, vehicle_summary.csv, metrics.csv
"""
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt

# ── Constants ─────────────────────────────────────────────────────────────────
DEPOT              = {"id": "DEPOT", "latitude": 19.0760, "longitude": 72.8777}
VEHICLE_CAP        = 800
NUM_VEHICLES       = 5
AVG_SPEED_KMPH     = 55
FUEL_COST_PER_KM   = 12
DRIVER_COST_PER_HR = 180
SLA_PENALTY_PER_HR = 500

W_TIME   = 0.30
W_COST   = 0.35
W_CARBON = 0.20
W_SLA    = 0.15


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
    return 2 * R * asin(sqrt(a))


def leg_costs(dist_km, travel_time_hr, toll_inr, emission_factor,
              weight_kg, priority, sla_hours, cumulative_time_hr):
    fuel_cost   = dist_km * FUEL_COST_PER_KM
    driver_cost = travel_time_hr * DRIVER_COST_PER_HR
    carbon_kg   = dist_km * emission_factor
    arrival_hr  = cumulative_time_hr + travel_time_hr
    sla_breach  = max(0, arrival_hr - sla_hours)
    sla_penalty = sla_breach * SLA_PENALTY_PER_HR
    total_cost  = fuel_cost + driver_cost + toll_inr + sla_penalty
    return {
        "travel_time_hr": travel_time_hr,
        "fuel_cost":      round(fuel_cost,   2),
        "driver_cost":    round(driver_cost, 2),
        "toll_cost":      round(toll_inr,    2),
        "sla_penalty":    round(sla_penalty, 2),
        "total_cost":     round(total_cost,  2),
        "carbon_kg":      round(carbon_kg,   3),
        "sla_breach_hr":  round(sla_breach,  2),
    }


def mo_score(travel_time_hr, cost, carbon_kg, sla_breach_hr,
             norm_time, norm_cost, norm_carbon, norm_sla):
    s_time   = travel_time_hr / norm_time   if norm_time   else 0
    s_cost   = cost           / norm_cost   if norm_cost   else 0
    s_carbon = carbon_kg      / norm_carbon if norm_carbon else 0
    s_sla    = sla_breach_hr  / norm_sla    if norm_sla    else 0
    return W_TIME*s_time + W_COST*s_cost + W_CARBON*s_carbon + W_SLA*s_sla


def mo_vrp(locations, demands, df):
    n       = len(locations)
    visited = [False] * n
    routes  = []

    all_dists = [haversine(locations[i]["latitude"], locations[i]["longitude"],
                           locations[j]["latitude"], locations[j]["longitude"])
                 for i in range(n) for j in range(n) if i != j]
    norm_dist   = np.percentile(all_dists, 75)
    norm_time   = norm_dist / AVG_SPEED_KMPH
    norm_cost   = norm_dist * FUEL_COST_PER_KM + norm_time * DRIVER_COST_PER_HR + 2000
    norm_carbon = norm_dist * 0.25
    norm_sla    = 12

    for v in range(NUM_VEHICLES):
        route, load, current, cum_time, route_data = [], 0.0, 0, 0.0, []

        while True:
            best_idx, best_score, best_costs = -1, float("inf"), None
            for j in range(1, n):
                if visited[j]:
                    continue
                if load + demands[j] > VEHICLE_CAP:
                    continue
                dist = haversine(locations[current]["latitude"], locations[current]["longitude"],
                                 locations[j]["latitude"],       locations[j]["longitude"])
                rec  = df.iloc[j - 1]
                t_hr = dist / (AVG_SPEED_KMPH / rec["traffic_mult"])
                costs = leg_costs(dist, t_hr, rec["toll_cost_inr"], rec["emission_factor"],
                                  demands[j], rec["priority"], rec["sla_hours"], cum_time)
                score = mo_score(costs["travel_time_hr"], costs["total_cost"],
                                 costs["carbon_kg"], costs["sla_breach_hr"],
                                 norm_time, norm_cost, norm_carbon, norm_sla)
                if score < best_score:
                    best_score, best_idx, best_costs = score, j, costs

            if best_idx == -1:
                break
            visited[best_idx] = True
            load     += demands[best_idx]
            cum_time += best_costs["travel_time_hr"]
            route.append(best_idx)
            route_data.append({**best_costs, "stop_idx": best_idx, "mo_score": round(best_score, 4)})
            current = best_idx

        if route:
            ret_dist = haversine(locations[current]["latitude"], locations[current]["longitude"],
                                 locations[0]["latitude"],       locations[0]["longitude"])
            ret_time = ret_dist / AVG_SPEED_KMPH
            total_dist = sum(
                haversine(locations[route[i-1] if i > 0 else 0]["latitude"],
                          locations[route[i-1] if i > 0 else 0]["longitude"],
                          locations[s]["latitude"], locations[s]["longitude"])
                for i, s in enumerate(route)
            ) + ret_dist
            routes.append({
                "vehicle":       v + 1,
                "stops":         route,
                "stop_data":     route_data,
                "load_kg":       round(load, 2),
                "total_dist":    round(total_dist, 2),
                "total_time_hr": round(cum_time + ret_time, 2),
                "total_fuel":    round(sum(d["fuel_cost"]   for d in route_data), 2),
                "total_toll":    round(sum(d["toll_cost"]   for d in route_data), 2),
                "total_driver":  round(sum(d["driver_cost"] for d in route_data), 2),
                "total_sla_pen": round(sum(d["sla_penalty"] for d in route_data), 2),
                "total_cost":    round(sum(d["total_cost"]  for d in route_data), 2),
                "total_carbon":  round(sum(d["carbon_kg"]   for d in route_data), 3),
                "sla_breaches":  sum(1 for d in route_data if d["sla_breach_hr"] > 0),
            })

        if all(visited[1:]):
            break

    return routes


def baseline_metrics(locations, df):
    prev = locations[0]
    tot  = dict(dist=0.0, time=0.0, fuel=0.0, toll=0.0, drv=0.0, co2=0.0, sla=0)
    cum_time = 0.0
    for i, loc in enumerate(locations[1:]):
        d   = haversine(prev["latitude"], prev["longitude"], loc["latitude"], loc["longitude"])
        rec = df.iloc[i]
        t   = d / AVG_SPEED_KMPH
        c   = leg_costs(d, t, rec["toll_cost_inr"], rec["emission_factor"],
                        df.iloc[i]["weight"], rec["priority"], rec["sla_hours"], cum_time)
        tot["dist"] += d;          tot["time"] += c["travel_time_hr"]
        tot["fuel"] += c["fuel_cost"]; tot["toll"] += c["toll_cost"]
        tot["drv"]  += c["driver_cost"]; tot["co2"] += c["carbon_kg"]
        if c["sla_breach_hr"] > 0: tot["sla"] += 1
        cum_time += c["travel_time_hr"]; prev = loc
    ret = haversine(prev["latitude"], prev["longitude"],
                    locations[0]["latitude"], locations[0]["longitude"])
    tot["dist"] += ret; tot["time"] += ret / AVG_SPEED_KMPH
    return {
        "distance_km":        round(tot["dist"], 2),
        "time_hr":            round(tot["time"], 2),
        "fuel_cost":          round(tot["fuel"], 2),
        "toll_cost":          round(tot["toll"], 2),
        "driver_cost":        round(tot["drv"],  2),
        "total_cost":         round(tot["fuel"] + tot["toll"] + tot["drv"], 2),
        "carbon_kg":          round(tot["co2"],  3),
        "sla_breaches":       tot["sla"],
        "sla_adherence_pct":  round((len(df) - tot["sla"]) / len(df) * 100, 1),
    }


def solve(csv_path="shipments.csv"):
    df        = pd.read_csv(csv_path)
    locations = [DEPOT] + df.to_dict("records")
    demands   = [0] + df["weight"].tolist()

    print(f"\n📦 {len(df)} shipments | {NUM_VEHICLES} vehicles | {VEHICLE_CAP}kg cap")
    print(f"   Weights: Time={W_TIME} Cost={W_COST} Carbon={W_CARBON} SLA={W_SLA}\n")

    base   = baseline_metrics(locations, df)
    routes = mo_vrp(locations, demands, df)

    opt = {
        "distance_km":      round(sum(r["total_dist"]    for r in routes), 2),
        "time_hr":          round(sum(r["total_time_hr"] for r in routes), 2),
        "fuel_cost":        round(sum(r["total_fuel"]    for r in routes), 2),
        "toll_cost":        round(sum(r["total_toll"]    for r in routes), 2),
        "driver_cost":      round(sum(r["total_driver"]  for r in routes), 2),
        "total_cost":       round(sum(r["total_cost"]    for r in routes), 2),
        "carbon_kg":        round(sum(r["total_carbon"]  for r in routes), 3),
        "sla_breaches":     sum(r["sla_breaches"]        for r in routes),
    }
    opt["sla_adherence_pct"] = round((len(df) - opt["sla_breaches"]) / len(df) * 100, 1)

    # Print summary
    print("=" * 62)
    print(f"  {'METRIC':<28} {'BASELINE':>10} {'OPTIMIZED':>10} {'SAVING':>8}")
    print("=" * 62)
    for label, bk, ok in [
        ("Distance (km)",        base["distance_km"],       opt["distance_km"]),
        ("Travel Time (hr)",     base["time_hr"],            opt["time_hr"]),
        ("Fuel Cost (₹)",        base["fuel_cost"],          opt["fuel_cost"]),
        ("Toll Cost (₹)",        base["toll_cost"],          opt["toll_cost"]),
        ("Driver Cost (₹)",      base["driver_cost"],        opt["driver_cost"]),
        ("Total Cost (₹)",       base["total_cost"],         opt["total_cost"]),
        ("Carbon (kg)",          base["carbon_kg"],          opt["carbon_kg"]),
        ("SLA Adherence (%)",    base["sla_adherence_pct"],  opt["sla_adherence_pct"]),
    ]:
        print(f"  {label:<28} {bk:>10.1f} {ok:>10.1f} {ok-bk:>+8.1f}")
    print("=" * 62)

    # ── routes.csv ────────────────────────────────────────────────────────────
    rows = []
    for r in routes:
        for order, (stop_idx, sd) in enumerate(zip(r["stops"], r["stop_data"]), 1):
            rec = df.iloc[stop_idx - 1]
            rows.append({
                "vehicle":           r["vehicle"],
                "stop_order":        order,
                "shipment_id":       rec["id"],
                "city":              rec["city"],
                "latitude":          locations[stop_idx]["latitude"],
                "longitude":         locations[stop_idx]["longitude"],
                "weight":            demands[stop_idx],
                "priority":          rec["priority"],
                "sla_hours":         rec["sla_hours"],
                "travel_time_hr":    round(sd["travel_time_hr"], 2),
                "fuel_cost":         sd["fuel_cost"],
                "toll_cost":         sd["toll_cost"],
                "driver_cost":       sd["driver_cost"],
                "sla_penalty":       sd["sla_penalty"],
                "total_cost":        sd["total_cost"],
                "carbon_kg":         sd["carbon_kg"],
                "sla_breach_hr":     sd["sla_breach_hr"],
                "mo_score":          sd["mo_score"],
                "route_distance_km": r["total_dist"],
            })
    pd.DataFrame(rows).to_csv("routes.csv", index=False)

    # ── metrics.csv ───────────────────────────────────────────────────────────
    row = {"num_shipments": len(df), "num_vehicles": len(routes)}
    for k, v in base.items(): row[f"baseline_{k}"] = v
    for k, v in opt.items():  row[f"opt_{k}"]      = v
    pd.DataFrame([row]).to_csv("metrics.csv", index=False)

    # ── vehicle_summary.csv ───────────────────────────────────────────────────
    pd.DataFrame([{
        "vehicle":        r["vehicle"],
        "stops":          len(r["stops"]),
        "load_kg":        r["load_kg"],
        "distance_km":    r["total_dist"],
        "time_hr":        r["total_time_hr"],
        "fuel_cost":      r["total_fuel"],
        "toll_cost":      r["total_toll"],
        "driver_cost":    r["total_driver"],
        "sla_penalty":    r["total_sla_pen"],
        "total_cost":     r["total_cost"],
        "carbon_kg":      r["total_carbon"],
        "sla_breaches":   r["sla_breaches"],
        "utilization_pct":round(r["load_kg"] / VEHICLE_CAP * 100, 1),
    } for r in routes]).to_csv("vehicle_summary.csv", index=False)

    print("\n✅  routes.csv · metrics.csv · vehicle_summary.csv saved.")
    return routes, base, opt


if __name__ == "__main__":
    solve()
