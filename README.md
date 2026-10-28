# Smart Energy Microgrid \& Demand Forecasting Engine (HackSphere 3.0)

**1. #Overview**

The app forecasts hourly demand for the next 24 hours with a QuickML regression model, then decides Main Grid or Renewable Storage mode and raises overload alerts.



**2.#Architecture**

\##Browser dashboard (Flask-served HTML with Chart.js)

\##Flask backend on Catalyst AppSail

\##Cleaning and feature engineering (features.py)

\##Calls to the published QuickML endpoint (quickml\_client.py)

\##Switching rules and battery simulation (controller.py)

\##Results back to the dashboard, with optional saving to Data Store



**3. #Setup**

\##Python 3.9 or newer

\##pip install -r requirements.txt

\##Run python prepare\_training\_data.py to create train\_features.csv

\##Upload that file to QuickML, set the target to target\_demand\_kw, exclude timestamp, choose Regression, and publish



**4. #Environment variables**

\##QUICKML\_ENDPOINT: published endpoint URL

\##QUICKML\_ENDPOINT\_KEY: endpoint key from the endpoint page

\##ORG\_ID (or CATALYST\_ORG\_ID): Catalyst organization ID

\##ZOHO\_CLIENT\_ID, ZOHO\_CLIENT\_SECRET, ZOHO\_REFRESH\_TOKEN: OAuth credentials, created with get\_refresh\_token.py

\##ENVIRONMENT: Development or Production

\##.env is never committed, and that .env.example shows the format.



**5. #Switching rules**

\##Renewable Storage Mode: demand above 160 kW, or hour 18:00 to 22:00 inclusive

\##Main Grid Mode: otherwise

\##Critical Overload alert: demand 200 kW or more, in any mode



**6. #Battery assumptions**

Say clearly these are assumptions, since the brief didn't define them: 1500 kWh capacity, 80% starting charge, Storage mode covers 50% of demand, Main Grid mode charges at 200 kW plus a solar bonus, clamped between 0 and 100%.



**7. #Model**

\##Features used: weather columns plus hour, weekday, month, sin and cos of hour and weekday, cooling and heating degrees, temperature times humidity, and an evening flag. No lag or past-demand features.

\##Validation: RMSE 9.15 kW, MAE 7.34 kW, R² 0.9665, on a time-ordered split.



**8. #Testing**

\##py test\_eval.py checks the predict.py contract

\##py tests\\test\_controller.py checks the switching edge cases

\##py check\_setup.py checks settings and the endpoint


**9.#Repository**

Github:https://github.com/kruthikay247/smart-energy-grid

