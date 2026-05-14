# Forecasting Facilities Demand using Time Series Analysis for Return-to-Office Policies Using Badge Data for Facility Management

### Forecasting Facilities Demand using Time Series Analysis for Return-to-Office Policies
#### Using Badge Data for Facility Management
As companies shift to hybrid work, return-to-office policies demand data-driven decision-making. Badge-in data from employees provides real-time insights into office utilization, helping facility managers make informed decisions. We can use daily attendance patterns to optimize cafeteria planning for example --- so we can ensure the right amount of food is prepared without waste. Office space allocation benefits from accurate forecasting, preventing unnecessary workspace provisioning while ensuring enough seating for those who show up. Building automation systems use occupancy data to adjust HVAC controls dynamically, fine-tuning heating, cooling, and ventilation based on actual usage rather than static schedules.

Facilities teams rely on time series analysis to forecast office space needs. By analyzing badge-in patterns over time, organizations can anticipate fluctuations in attendance, identify trends, and adapt office policies accordingly.

### Understanding Badge-In Data for RTO Monitoring
Badge-in data captures employee movement in and out of the office. Attendance is sporadic --- employees do not come in every day, and their arrival and departure times vary. Weekly seasonality is a key feature, with higher attendance midweek (Tuesday through Thursday) and lower turnout on Mondays and Fridays. External events also impact attendance, including holidays and company-wide meetings that may cause dips or surges in office presence.

### Simulating Badge-In Data for 1,000 Employees Over 300 Days
To model return-to-office trends, a simulated dataset represents badge-ins and badge-outs for 1,000 employees over 300 days. The simulation assumes that, on any given day, about 60% of employees come to the office. Badge-in times follow a normal distribution centered around 9 AM, with slight variations based on individual preferences. Similarly, badge-out times generally occur eight to nine hours later, clustering around 5--6 PM.



<figcaption>Simulated Data</figcaption>


The simulated time series data reveals daily fluctuations in office attendance, typically ranging between 550 and 650 employees. Weekly patterns emerge, with midweek peaks and Friday dips. Some days show lower-than-expected attendance, likely due to holidays or work-from-home flexibility.

### How Facilities Managers Use This Data
Forecasting office attendance ensures smooth operations. Short-term forecasts, such as predicting attendance for the upcoming week, help cafeteria managers determine food quantities, reducing waste while ensuring enough meals are available. Long-term projections support broader space planning and lease decisions, guiding companies on whether to expand, consolidate, or reconfigure their offices.

Optimizing space allocation becomes more efficient when companies identify consistent attendance trends. If data shows that Fridays have low badge-in rates, organizations can introduce hot-desking policies to reduce unused workstations. Instead of maintaining a fully open office, facilities teams can consolidate workspaces, lower maintenance costs, and reduce energy consumption.

Building automation integrates with time series forecasting to enhance efficiency. HVAC systems adjust dynamically based on expected occupancy, preventing unnecessary heating or cooling of unoccupied areas. Lighting systems synchronize with badge-in data to illuminate workspaces only when employees are present. Predictive maintenance for elevators, restrooms, and other shared facilities can be planned based on expected office usage, preventing overcrowding and optimizing operations.

### Building a Forecasting Model
With badge-in data in hand, the next step is to develop a forecasting model that anticipates future attendance trends. A well-trained time series model detects patterns, identifies anomalies, and provides actionable insights.



<figcaption>Simulated data</figcaption>


This model detects long-term trends and short-term fluctuations, allowing facilities managers to proactively adjust operations based on data-driven forecasts. By continuously refining the model with new badge-in data, organizations can adapt their return-to-office policies in real time, ensuring an optimal balance between workspace efficiency and employee flexibility.

### Coffee Badgers: Employees Who Badge In, Grab a Coffee, and Leave
Not everyone who badges into the office stays for a full workday. Some employees show up, grab a coffee, chat with a few coworkers, and leave within the hour. These are coffee badgers, a distinct group that facilities managers need to account for when analyzing badge-in data. Their behavior skews attendance numbers, making it seem like more employees are working in the office than actually are.

Tracking coffee badgers helps companies refine their return-to-office policies, ensuring that reported attendance reflects real workspace usage. If a company bases space planning or cafeteria demand on badge-in counts alone, it risks overestimating office utilization. A company might assume it needs to prepare lunch for 1,000 people when, in reality, 10% of those employees have already left for the day.

This model assumes that about 10% of the workforce exhibits coffee badging behavior. These employees follow the same badge-in patterns as their colleagues, typically arriving around 9 AM, but instead of staying for a full workday, they leave within 20 to 60 minutes. This short visit duration makes them easy to classify. By analyzing badge-out times, we can separate normal office workers from coffee badgers and adjust forecasting models accordingly.

The time series visualization of badge-ins reveals an important insight: coffee badgers appear consistently over time, fluctuating slightly but maintaining their share of total badge-ins. While overall attendance patterns show weekly seasonality --- higher midweek and lower on Fridays --- coffee badging remains a persistent behavior. Whether employees are testing the waters of office life, attending a quick in-person meeting, or simply preferring their home work setup, coffee badgers represent a stable sub-pattern within the larger return-to-office trend.

To detect this behavior, we build a classification model that distinguishes between coffee badgers and full-day employees. Using badge-in time and duration in the office as key features, a Random Forest classifier learns to predict whether an employee is likely to be a coffee badger. After training on 80% of the dataset and testing on the remaining 20%, the model achieves near-perfect accuracy, correctly identifying those who stay for only a short time.



<figcaption>Simulated data</figcaption>


Understanding coffee badging is more than just an interesting anomaly --- it's a necessary adjustment for accurate facility planning. By incorporating this classification model into attendance forecasting, companies can make better decisions about workspace allocation, energy efficiency, and cafeteria planning. Instead of assuming that everyone who badges in needs a desk, an office, or a meal, facilities managers can refine their predictions, ensuring that real office usage --- not just badge-ins --- drives workplace policy.
