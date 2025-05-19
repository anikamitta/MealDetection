Meal Detection 

After a meal, glucose levels tend to spike rapidly. By applying peak detection algorithms to glucose trends, the script aims to identify these spikes, enabling researchers or 
clinicians to estimate when a meal was likely consumed—without requiring manual logging. Anomalies such as sensor malfunctions, sudden glucose spikes, or drops due to insulin 
overdosing or hypoglycemia can be critical in managing diabetes. This code uses feature engineering to calculate the rate of change in glucose levels and uses Isolation Forest to 
detect unusual patterns.
