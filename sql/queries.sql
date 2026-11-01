-- SYNTHETIC churn EDA
SELECT contract_type, AVG(churn) AS churn_rate, COUNT(*) AS n FROM customers GROUP BY contract_type;
SELECT CASE WHEN tenure_months<12 THEN '0-11' WHEN tenure_months<36 THEN '12-35' ELSE '36+' END AS tenure_band, AVG(churn) AS churn_rate FROM customers GROUP BY 1;
