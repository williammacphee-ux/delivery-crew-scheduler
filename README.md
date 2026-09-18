# delivery-crew-scheduler
Mixed-integer program that schedules an 87-person delivery crew across a 140-route week to minimize payroll, solved day-by-day on a rolling horizon. Python, PuLP.


Delivery Crew Scheduling Optimizer

An operations-research model that builds a full week's delivery-crew schedule — assigning drivers and helpers to routes to minimize payroll — for a regional party-rental and equipment-delivery company with a ~90-person workforce. Built in Python with mixed-integer programming, solved day-by-day on a rolling horizon.

Two-person course project (Introduction to Python for Business Data Analysis).

About the problem — real operations, simulated data

This project models the actual operations of a real regional delivery company: its workforce structure, weekly route patterns, driver-capability requirements, overtime rules, and the real staffing tradeoffs its managers make every week are all drawn from how the business genuinely runs. The company is not named here, and because its real operational data is proprietary and was not shared, the input files are simulated data built to match that real structure rather than taken from the company's systems. In other words: the decisions and processes being optimized are real; the numbers are a faithful stand-in.

Overview

The company runs dozens of equipment deliveries a day across Massachusetts, each route needing a qualified driver and one or more helpers. Staffing a week by hand means juggling driver eligibility, route-specific skill requirements (Boston driving, large events, high-value "elite" clients), overtime limits, and who's available — all while keeping payroll down. This model does it as an optimization instead of by gut feel.

Workforce: 39 drivers, 48 helpers (87 employees)
Routes: 140 across a 7-day week (Tue–Mon)
Objective: minimize weekly payroll, with a tunable preference for putting stronger crews on the farther routes
Approach

A mixed-integer linear program (PuLP + CBC), solved one day at a time on a rolling horizon so that weekly overtime is handled correctly without making the model non-linear.

Decision variables (per day)

x[p][r][role] — binary: worker p works route r as driver or helper
reg[p], ot[p] — regular vs. overtime hours for worker p

Objective

sum( base_wage[p]*reg[p] + 1.5*base_wage[p]*ot[p] ) + MU * (distance-weighted mean crew-quality term)

Constraints

Coverage (hard): each route gets exactly 1 driver and crew_size − 1 helpers
Driver eligibility: only licensed drivers drive, and drivers don't double as helpers
Capability floors (hard): Boston-driving / large-event / elite-client routes can only be driven by someone who holds that qualification
Availability: unavailable workers take nothing that day
One route per person per day
Overtime split: first 40 weekly hours at base rate, the rest at 1.5×

Rolling horizon: solve a day, commit its assignments, add each worker's committed hours to a running total, and pass that total into the next day as a constant. This is what keeps each day's problem linear while still enforcing a weekly 40-hour overtime threshold — the main modeling idea in the project.

Data

Simulated data modeling the real operation (see the note above): an 87-person roster with wages, driver flags, three capability flags, and a crew-quality "correction factor"; 140 routes with day, drive/work hours, crew size, distance, and capability requirements; and an availability table. Files are in data/.

Results

Running the full week at the default travel priority (20):

Every one of the 7 days solves to Optimal
140 routes fully staffed; 75 of 87 workers used
Total weekly payroll: $45,870.28
The first day alone builds 2,610 decision variables and 2,901 constraints — a sizable MILP, not a toy instance
The cost-vs-quality lever

The travel-priority knob (MU) is the interesting part. Raising it pushes higher-rated crews toward the farthest routes, trading payroll for service quality — and the model lets you price that trade exactly:

Travel priority	Weekly payroll
0 (pure cost)	$45,856.56
20	$45,870.28
60	$45,878.78

The effect is deliberately modest — it's a soft preference layered on top of a hard cost objective — but it's real and directional: about $22/week buys a systematically better-crewed set of long-haul routes. That quantifies the cost of a service-quality policy managers could actually choose to adopt or not — the kind of real operational decision this project is built around.

Repository contents
delivery_scheduler.py      # the optimization model (rolling-horizon weekly solve)
data/roster.csv            # 87 workers: wages, driver flag, capability flags, quality factor
data/routes.csv            # 140 routes: day, hours, crew size, distance, capability needs
data/availability.csv      # per-worker availability
README.md
How to run
bash
pip install pandas pulp
python delivery_scheduler.py

The script prompts for a travel priority — enter 20 for the default behavior, or 0 for pure cost minimization. It prints the day-by-day solve status, the full weekly schedule, and per-worker hours and pay.

What this project demonstrates
Turning a real, multi-constraint workforce-scheduling problem into a mixed-integer program
A rolling-horizon decomposition that keeps a weekly-overtime model linear
Encoding nuanced business rules — capability floors, driver eligibility, availability, overtime — as constraints
A tunable cost-vs-quality objective, and quantifying the tradeoff in real dollars
Building a faithful simulated dataset when the real operational data is confidential
