-- ShipCast Task A: daily order volume per customer state, with forecasting features.
-- One row per (state, day), zero-filled via a state x date spine.
-- Window: 2017-01-01 .. 2018-08-21. 2016 is too sparse (Nov 2016 has no orders) and
-- the feed collapses after 2018-08-22 (snapshot artefact, not real demand).
-- Features: lags 1/7/14/28, 7- and 28-day rolling mean and std, day of week, month,
-- national-holiday and Black Friday flags. Rolling windows include the current day,
-- so every feature on row d uses only data up to and including d.

CREATE OR REPLACE TABLE daily_volume AS
WITH orders AS (
    SELECT o.order_id,
           c.customer_state AS state,
           CAST(o.order_purchase_timestamp AS DATE) AS d
    FROM read_csv_auto('data/raw/olist_orders_dataset.csv') o
    JOIN read_csv_auto('data/raw/olist_customers_dataset.csv') c USING (customer_id)
    WHERE o.order_status <> 'canceled'
      AND CAST(o.order_purchase_timestamp AS DATE) BETWEEN DATE '2017-01-01' AND DATE '2018-08-21'
),
spine AS (
    SELECT CAST(d AS DATE) AS d
    FROM range(DATE '2017-01-01', DATE '2018-08-22', INTERVAL 1 DAY) t(d)
),
states AS (
    SELECT DISTINCT state FROM orders
),
counts AS (
    SELECT state, d, COUNT(DISTINCT order_id) AS y
    FROM orders
    GROUP BY state, d
),
holidays(d, is_black_friday) AS (
    -- Brazilian national holidays 2017-2018 (incl. Carnival, Good Friday, Corpus Christi)
    VALUES
    (DATE '2017-01-01', 0), (DATE '2017-02-27', 0), (DATE '2017-02-28', 0), (DATE '2017-04-14', 0),
    (DATE '2017-04-21', 0), (DATE '2017-05-01', 0), (DATE '2017-06-15', 0), (DATE '2017-09-07', 0),
    (DATE '2017-10-12', 0), (DATE '2017-11-02', 0), (DATE '2017-11-15', 0), (DATE '2017-11-24', 1),
    (DATE '2017-12-25', 0), (DATE '2018-01-01', 0), (DATE '2018-02-12', 0), (DATE '2018-02-13', 0),
    (DATE '2018-03-30', 0), (DATE '2018-04-21', 0), (DATE '2018-05-01', 0), (DATE '2018-05-31', 0)
),
panel AS (
    SELECT s.state, sp.d, COALESCE(c.y, 0) AS y
    FROM states s
    CROSS JOIN spine sp
    LEFT JOIN counts c ON c.state = s.state AND c.d = sp.d
)
SELECT
    p.state,
    p.d,
    p.y,
    LAG(p.y, 1)  OVER w AS lag1,
    LAG(p.y, 7)  OVER w AS lag7,
    LAG(p.y, 14) OVER w AS lag14,
    LAG(p.y, 28) OVER w AS lag28,
    AVG(p.y)         OVER (w ROWS BETWEEN 6  PRECEDING AND CURRENT ROW) AS mean7,
    AVG(p.y)         OVER (w ROWS BETWEEN 27 PRECEDING AND CURRENT ROW) AS mean28,
    STDDEV_SAMP(p.y) OVER (w ROWS BETWEEN 6  PRECEDING AND CURRENT ROW) AS std7,
    STDDEV_SAMP(p.y) OVER (w ROWS BETWEEN 27 PRECEDING AND CURRENT ROW) AS std28,
    ISODOW(p.d) AS dow,                                  -- 1 = Monday .. 7 = Sunday
    MONTH(p.d) AS month,
    CASE WHEN h.d IS NOT NULL AND h.is_black_friday = 0 THEN 1 ELSE 0 END AS is_holiday,
    COALESCE(h.is_black_friday, 0) AS is_black_friday
FROM panel p
LEFT JOIN holidays h ON h.d = p.d
WINDOW w AS (PARTITION BY p.state ORDER BY p.d)
ORDER BY p.state, p.d;
