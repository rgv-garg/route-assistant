# ─────────────────────────────────────────────
# 4. PREDICTION FUNCTION - PNR CONFIRMATION
# ─────────────────────────────────────────────

# Table references (Unity Catalog)
SILVER_TABLE = "workspace.default.railways_silver"
GOLD_TABLE = "workspace.default.railways_gold"

def predict_confirmation_probability(
    train_no:        int,
    booking_status:  str,
    waitlist_number: int,
    travel_class:    str,
    from_station:    str,
    to_station:      str,
    travel_date:     str,
) -> dict:
    """
    Predict the probability that a booking will be confirmed.
 
    Parameters
    ----------
    train_no        : int   — Train number, e.g. 12951
    booking_status  : str   — Current status: "Confirmed" | "Waitlisted" | "RAC"
    waitlist_number : int   — Waitlist position (0 if not on waitlist)
    travel_class    : str   — Coach class: "1AC" | "2AC" | "3AC" | "Sleeper"
    from_station    : str   — Source station code, e.g. "NDLS"
    to_station      : str   — Destination station code, e.g. "CSMT"
    travel_date     : str   — Date of journey in "DD-Mon" format, e.g. "15-Jun"
 
    Returns
    -------
    dict with keys:
        confirmed_probability  : float  (0.0 – 1.0)
        not_confirmed_probability : float
        prediction             : str    ("Confirmed" | "Not Confirmed")
        confidence             : str    ("High" | "Medium" | "Low")
        input_summary          : dict   (echo of enriched inputs used)
    """
 
    # ── 1. Derive engineered fields from raw inputs ──────────────
    is_waitlisted        = 1 if waitlist_number > 0 else 0
    waitlist_rank        = waitlist_number
    is_confirmed_booking = 1 if booking_status == "Confirmed" else 0
 
    # Quota inference from booking_status
    if booking_status == "Waitlisted" and waitlist_number > 0:
        quota = "General"
    elif booking_status == "Confirmed":
        quota = "General"
    else:
        quota = "General"
 
    # ── 2. Look up Gold-layer route stats for enrichment ─────────
    gold_df = spark.table(GOLD_TABLE)
 
    # Try exact match on route + class first
    route_stats = gold_df.filter(
        (F.col("Source_Station") == from_station) &
        (F.col("Destination_Station") == to_station) &
        (F.col("Class_of_Travel") == travel_class)
    ).agg(
        F.avg("route_confirmation_rate").alias("route_confirmation_rate"),
        F.avg("avg_distance").alias("avg_distance"),
        F.avg("bookings_count").alias("bookings_count"),
    ).collect()
 
    if route_stats and route_stats[0]["route_confirmation_rate"] is not None:
        r                    = route_stats[0]
        route_conf_rate      = float(r["route_confirmation_rate"])
        avg_distance         = float(r["avg_distance"])
        bookings_count       = float(r["bookings_count"])
        avg_distance_num     = avg_distance
    else:
        # No historical data for this route — use dataset-wide averages
        global_stats = spark.table(SILVER_TABLE).agg(
            F.avg("label").alias("cr"),
            F.avg("Travel_Distance").alias("ad"),
            F.count("PNR_Number").alias("bc"),
        ).collect()[0]
        route_conf_rate   = float(global_stats["cr"])
        avg_distance      = float(global_stats["ad"])
        bookings_count    = float(global_stats["bc"])
        avg_distance_num  = avg_distance
 
    # Estimate fare (simplified - using average distance)
    avg_fare_on_route = avg_distance_num * 0.5  # Rough estimate: ₹0.50 per km
    fare_per_km = 0.5
 
    # ── 3. Build a single-row Spark DataFrame ────────────────────
    input_row = spark.createDataFrame([{
        # Raw schema fields
        "PNR_Number":            "QUERY",
        "Train_Number":          train_no,
        "Booking_Date":          travel_date,
        "Class_of_Travel":       travel_class,
        "Quota":                 quota,
        "Source_Station":        from_station,
        "Destination_Station":   to_station,
        "Date_of_Journey":       travel_date,
        "Current_Status":        booking_status,
        "Number_of_Passengers":  1,
        "Age_of_Passengers":     "Adult",
        "Booking_Channel":       "IRCTC Website",
        "Travel_Distance":       int(avg_distance_num),
        "Train_Type":            "Express",
        "Seat_Availability":     100,
        "Special_Considerations": "None",
        "Holiday_or_Peak_Season": "No",
        "Waitlist_Position":     f"WL{waitlist_number:03d}" if waitlist_number > 0 else None,
        "Confirmation_Status":   "Unknown",
        # Engineered features
        "IsWaitlisted":           is_waitlisted,
        "WaitlistRank":           waitlist_rank,
        "IsHoliday":              0,
        "HasConcession":          0,
        "IsConfirmed_AtBooking":  is_confirmed_booking,
        "label":                  0.0,
        # Gold-enriched route stats
        "route_confirmation_rate": route_conf_rate,
        "avg_distance":            avg_distance,
        "bookings_count":          bookings_count,
    }])
 
    # ── 4. Run the trained best_model pipeline ───────────────────
    prediction_row = best_model.transform(input_row).collect()[0]
 
    prob_vector           = prediction_row["probability"]
    confirmed_prob        = float(prob_vector[1])       # index 1 = Confirmed class
    not_confirmed_prob    = float(prob_vector[0])
    predicted_label       = int(prediction_row["prediction"])
    prediction_label_str  = "Confirmed" if predicted_label == 1 else "Not Confirmed"
 
    # Confidence band
    if confirmed_prob >= 0.75 or confirmed_prob <= 0.25:
        confidence = "High"
    elif confirmed_prob >= 0.60 or confirmed_prob <= 0.40:
        confidence = "Medium"
    else:
        confidence = "Low"
 
    result = {
        "confirmed_probability":     round(confirmed_prob,     4),
        "not_confirmed_probability": round(not_confirmed_prob, 4),
        "prediction":                prediction_label_str,
        "confidence":                confidence,
        "input_summary": {
            "train_no":        train_no,
            "booking_status":  booking_status,
            "waitlist_number": waitlist_number,
            "class":           travel_class,
            "from_station":    from_station,
            "to_station":      to_station,
            "travel_date":     travel_date,
            "route_conf_rate": round(route_conf_rate, 4),
            "is_waitlisted":   bool(is_waitlisted),
        },
    }
 
    # ── 5. Pretty-print ──────────────────────────────────────────
    print("\n" + "═" * 52)
    print("  🚆  PNR CONFIRMATION PROBABILITY REPORT")
    print("═" * 52)
    print(f"  Train No       : {train_no}")
    print(f"  Route          : {from_station} → {to_station}")
    print(f"  Class          : {travel_class}")
    print(f"  Travel Date    : {travel_date}")
    print(f"  Booking Status : {booking_status}"
          + (f"  (WL#{waitlist_number})" if waitlist_number > 0 else ""))
    print("─" * 52)
    print(f"  ✅ Confirmed Probability    : {confirmed_prob * 100:.1f}%")
    print(f"  ❌ Not Confirmed Probability: {not_confirmed_prob * 100:.1f}%")
    print(f"  🏷  Prediction              : {prediction_label_str}")
    print(f"  📊 Confidence              : {confidence}")
    print(f"  📈 Historical Route Conf.  : {route_conf_rate * 100:.1f}%")
    print("═" * 52 + "\n")
 
    return result

print("✅ Prediction function defined. Use predict_confirmation_probability() to make predictions.")
print("\nNote: Run Cell 1 first to train the model and create best_model variable.")