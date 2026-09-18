# ============================================================================
#  BusyBee Crew Scheduling — FULL pandas implementation (original + rolling)
#  ---------------------------------------------------------------------------
#  This is the complete model in pandas idiom (pd.read_csv, set_index().to_dict(),
#  LpVariable.dicts), keeping the original Pieces 1-5 structure and check-prints,
#  and adding this session's modifications. Each change is tagged [NEW]/[CHANGED].
#
#  SUMMARY OF CHANGES FROM THE ORIGINAL:
#    [NEW]     Routes carry a `day`; hours accumulate across the WEEK.
#    [NEW]     Solved DAY-BY-DAY (rolling horizon): solve a day, commit it,
#              carry each worker's hours forward as a constant (keeps it linear).
#    [CHANGED] Reliability score replaced by three HARD capability flags
#              (Boston / large-events / elite-clients).
#    [CHANGED] MU repurposed: now weights a soft TRAVEL term = a distance-weighted
#              mean of crew quality (driver 3x, helper 1x). Higher MU pushes the
#              best drivers OUT to the farthest routes.
#    Coverage is HARD (kept from the original): every route must be fully staffed.
#
#  KEPT FROM THE ORIGINAL: pandas data handling, availability, driver-only-drives,
#    one-role-per-route, the 40h regular / 1.5x overtime split, the cost objective,
#    the Pieces 1-5 layout, and its check-prints.
# ============================================================================

import pandas as pd
import pulp
import sys


# =========================================================
# PIECE 1 — Load the three data files (pandas)
# =========================================================

routes_df = pd.read_csv("data/routes.csv")
roster_df = pd.read_csv("data/roster.csv")
avail_df  = pd.read_csv("data/availability.csv")

# Confirm pandas sees everything (counts + a sample row of each)
print("--- PIECE 1: DATA LOADED ---")
print("Routes:", len(routes_df), "| sample:", routes_df.iloc[0].to_dict())
print("Roster:", len(roster_df), "| sample:", roster_df.iloc[0].to_dict())
print("Availability:", len(avail_df), "| sample:", avail_df.iloc[0].to_dict())


# =========================================================
# PIECE 2 — Organize the data into lookups (set_index().to_dict())
# =========================================================

# Simple lists of who / what / when
employees = roster_df["emp_id"].tolist()
route_ids = routes_df["route_id"].tolist()
roles     = ["driver", "helper"]
days      = ["Tue", "Wed", "Thu", "Fri", "Sat", "Sun", "Mon"]     # [NEW] the pay period

# --- Per-worker lookups ---
base_wage    = roster_df.set_index("emp_id")["base_wage"].to_dict()
can_drive    = (roster_df.set_index("emp_id")["driver_eligible"] == 1).to_dict()
is_available = (avail_df.set_index("emp_id")["available"] == 1).to_dict()

# [CHANGED] the old reliability score is replaced by three HARD capability flags
can_boston = (roster_df.set_index("emp_id")["can_boston"] == 1).to_dict()
can_events = (roster_df.set_index("emp_id")["can_events"] == 1).to_dict()
can_elite  = (roster_df.set_index("emp_id")["can_elite"]  == 1).to_dict()

# [NEW] correction factor (1 = best score-quartile .. 4 = worst) for the travel knob
corr = roster_df.set_index("emp_id")["corr_factor"].to_dict()

# --- Per-route lookups ---
# derive total hours and the distance bracket right on the DataFrame
routes_df["hours"]   = routes_df["drive_hours"] + routes_df["work_hours"]
routes_df["bracket"] = routes_df["distance_mi"] // 15             # [NEW] 1 near .. 4 far

route_day   = routes_df.set_index("route_id")["day"].to_dict()    # [NEW] day of each route
route_hours = routes_df.set_index("route_id")["hours"].to_dict()
crew_size   = routes_df.set_index("route_id")["crew_size"].to_dict()
bracket     = routes_df.set_index("route_id")["bracket"].to_dict()
need_boston = (routes_df.set_index("route_id")["need_boston"] == 1).to_dict()   # [CHANGED]
need_events = (routes_df.set_index("route_id")["need_events"] == 1).to_dict()
need_elite  = (routes_df.set_index("route_id")["need_elite"]  == 1).to_dict()

# --- Settings ---
OT_THRESHOLD = 40        # first 40 weekly hours regular; the rest are overtime at 1.5x
# [CHANGED] MU: was the reliability penalty weight; now the soft travel/quality weight.

# Ask the user how strongly to prioritize sending better crews to farther routes.
# 0 = ignore travel, just minimize cost.  Higher = push the best drivers out to the far routes.
user_input = input("Travel priority (0 = pure cost, higher = better crews on far routes) [default 20]: ")
travel_priority = float(user_input) if user_input.strip() != "" else 20.0
MU = travel_priority   # keep MU as the internal name so the rest of the code is unchanged

# --- PIECE 2 CHECK ---
print("\n--- PIECE 2 CHECK ---")
print("Employees:", len(employees), "| Routes:", len(route_ids))
print("Capability drivers -> boston:", sum(can_boston.values()),
      "| events:", sum(can_events.values()), "| elite:", sum(can_elite.values()))
print("Sample route hours:", {r: route_hours[r] for r in route_ids[:4]})
print("Sample base wages:", {p: base_wage[p] for p in employees[:4]})


# =========================================================
# PIECE 3 — Build and solve the model for ONE day
# ---------------------------------------------------------
# The original Pieces 3-4 (variables, constraints, objective, solve) live here so
# the rolling loop can call them once per day. `hours_used` carries each worker's
# already-committed hours into this day's solve.
# =========================================================

def solve_one_day(day, hours_used):
    # today's routes come straight from a DataFrame filter
    todays_routes = routes_df[routes_df["day"] == day]["route_id"].tolist()

    # --- Decision variables ---
    # x[p][r][d] = 1 if worker p works route r in role d  (LpVariable.dicts, original style)
    x   = pulp.LpVariable.dicts("x", (employees, todays_routes, roles), cat=pulp.LpBinary)
    reg = pulp.LpVariable.dicts("reg", employees, lowBound=0)   # regular hours (<= 40)
    ot  = pulp.LpVariable.dicts("ot",  employees, lowBound=0)   # overtime hours (1.5x)

    model = pulp.LpProblem("BusyBee_day", pulp.LpMinimize)

    # --- Constraint 1: Crew coverage (HARD) ---
    # Each route MUST have exactly 1 driver and (crew_size - 1) helpers.
    for r in todays_routes:
        model += pulp.lpSum(x[p][r]["driver"] for p in employees) == 1, f"driver_{r}"
        model += pulp.lpSum(x[p][r]["helper"] for p in employees) == crew_size[r] - 1, f"helpers_{r}"

    # --- Constraint 2: Only eligible drivers drive; drivers don't help (kept) ---
    for p in employees:
        for r in todays_routes:
            if not can_drive[p]:
                model += x[p][r]["driver"] == 0, f"cant_drive_{p}_{r}"
            else:
                model += x[p][r]["helper"] == 0, f"driver_not_helper_{p}_{r}"

            # --- Constraint 3 [CHANGED]: HARD capabilities ---
            # A route needing a capability can only be driven by someone who holds it.
            if need_boston[r] and not can_boston[p]:
                model += x[p][r]["driver"] == 0, f"need_boston_{p}_{r}"
            if need_events[r] and not can_events[p]:
                model += x[p][r]["driver"] == 0, f"need_events_{p}_{r}"
            if need_elite[r] and not can_elite[p]:
                model += x[p][r]["driver"] == 0, f"need_elite_{p}_{r}"

    # --- Constraint 4: Availability — unavailable people take nothing (kept) ---
    for p in employees:
        if not is_available[p]:
            for r in todays_routes:
                for d in roles:
                    model += x[p][r][d] == 0, f"unavailable_{p}_{r}_{d}"

    # --- Constraint 5: one role per route + [NEW] one route per person per day ---
    for p in employees:
        for r in todays_routes:
            model += x[p][r]["driver"] + x[p][r]["helper"] <= 1, f"one_role_{p}_{r}"
        model += pulp.lpSum(x[p][r][d] for r in todays_routes for d in roles) <= 1, f"one_route_{p}"

    # --- Hours split WITH the weekly carry-forward [NEW] ---
    # total hours today = regular + overtime; regular hours still available this
    # week = (40 - hours already worked), passed in as a CONSTANT (keeps it linear).
    for p in employees:
        day_hours = pulp.lpSum(route_hours[r] * x[p][r][d]
                               for r in todays_routes for d in roles)
        model += reg[p] + ot[p] == day_hours, f"hours_split_{p}"
        model += reg[p] <= max(0.0, OT_THRESHOLD - hours_used[p]), f"reg_cap_{p}"

    # --- [NEW] Soft travel term: distance-weighted MEAN of crew quality ---
    # crew quality per route = (3*driver_corr + sum helper_corr)/(crew_size+2)
    # weighted by the route's distance bracket; dividing by total weight makes it a
    # per-route mean, so a weak crew is penalized most on FAR routes -> higher MU
    # pushes the best (low-corr) drivers OUT to the farthest routes.
    travel_terms = []
    total_weight = 0
    for r in todays_routes:
        divisor     = crew_size[r] + 2
        driver_part = 3 * pulp.lpSum(corr[p] * x[p][r]["driver"] for p in employees)
        helper_part =     pulp.lpSum(corr[p] * x[p][r]["helper"] for p in employees)
        crew_quality = (driver_part + helper_part) / divisor
        travel_terms.append(bracket[r] * crew_quality)
        total_weight += bracket[r]

    # --- The objective ---
    # regular at base wage + overtime at 1.5x, plus MU * (weighted-mean travel term)
    model += (
        pulp.lpSum(base_wage[p] * reg[p] + 1.5 * base_wage[p] * ot[p] for p in employees)
        + MU * (pulp.lpSum(travel_terms) / total_weight)
    ), "Total_Payroll_plus_Travel"

    # --- Solve this day ---
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    return model, x, reg, ot, todays_routes


# =========================================================
# PIECE 4 — Rolling loop: solve each day, commit it, carry hours forward
# =========================================================

hours_used = {p: 0.0 for p in employees}    # everyone starts the week at 0 hours
schedule   = []                             # (day, route, crew) tuples build up here

print("\n--- PIECE 4: SOLVING DAY BY DAY ---")
first_day = True
for today in days:
    model, x, reg, ot, todays_routes = solve_one_day(today, hours_used)
    status = pulp.LpStatus[model.status]

    # On the first day, print a PIECE 3-style check (variable / constraint counts)
    if first_day:
        print("\n--- PIECE 3 CHECK (first day) ---")
        n_vars = sum(1 for p in employees for r in todays_routes for d in roles)
        print("Decision variables this day (x):", n_vars)
        print("Constraints this day:", len(model.constraints))
        first_day = False

    # HARD coverage means an unstaffable day comes back Infeasible (not under-filled)
    if status != "Optimal":
        print(f"  {today}: {status} — this day could not be fully staffed with qualified people")
        break

    # Commit ONLY today's assignments and lock the hours into the running total
    for r in todays_routes:
        crew = [(p, d) for p in employees for d in roles if x[p][r][d].value() == 1]
        for p, d in crew:
            hours_used[p] += route_hours[r]
        schedule.append((today, r, crew))
    print(f"  {today}: {len(todays_routes)} routes | status {status}")


# =========================================================
# PIECE 5 — Print the schedule and the pay
# =========================================================

print("\n=================== WEEKLY SCHEDULE ===================")
for day, r, crew in schedule:
    crew_str = ",  ".join(f"{p} ({d})" for p, d in crew)
    print(f"  {day} {r}:  {crew_str}")

print("\n-------------------- HOURS & PAY --------------------")
grand_total = 0.0
for p in employees:
    total = hours_used[p]
    if total == 0:
        continue
    r_hours = min(total, OT_THRESHOLD)
    o_hours = max(0.0, total - OT_THRESHOLD)
    pay = base_wage[p] * r_hours + 1.5 * base_wage[p] * o_hours
    grand_total += pay
    print(f"  {p}:  {r_hours:.1f} reg + {o_hours:.1f} OT hrs   =  ${pay:,.2f}")

working = sum(1 for p in employees if hours_used[p] > 0)
print(f"\n  People working: {working}/{len(employees)}")
print(f"  TOTAL WEEKLY PAYROLL:  ${grand_total:,.2f}")
print("======================================================")
