-- ShipCast Task B: one row per delivered, single-seller order with purchase-time features.
-- Features are only what is known when the customer clicks "buy": customer/seller state,
-- seller->customer haversine distance, freight, weight, item count, purchase day/hour.
-- No carrier timestamps (approval, handover to carrier) are used - those would be leakage.
-- Olist's estimated delivery date is kept only as the business baseline, never as a feature.

CREATE OR REPLACE TABLE orders_b AS
WITH geo AS (
    -- ~1M geolocation rows -> one centroid per zip prefix (avoids join fan-out)
    SELECT geolocation_zip_code_prefix AS zip,
           AVG(geolocation_lat) AS lat,
           AVG(geolocation_lng) AS lng
    FROM read_csv_auto('data/raw/olist_geolocation_dataset.csv')
    GROUP BY 1
),
items AS (
    SELECT oi.order_id,
           COUNT(*)                                   AS n_items,
           COUNT(DISTINCT oi.seller_id)               AS n_sellers,
           ANY_VALUE(oi.seller_id)                    AS seller_id,
           SUM(oi.price)                              AS price,
           SUM(oi.freight_value)                      AS freight,
           SUM(p.product_weight_g)                    AS weight_g,
           SUM(p.product_length_cm * p.product_height_cm * p.product_width_cm) AS volume_cm3,
           MODE(p.product_category_name)              AS category
    FROM read_csv_auto('data/raw/olist_order_items_dataset.csv') oi
    LEFT JOIN read_csv_auto('data/raw/olist_products_dataset.csv') p USING (product_id)
    GROUP BY oi.order_id
),
base AS (
    SELECT o.order_id,
           CAST(o.order_purchase_timestamp AS TIMESTAMP) AS purchase_ts,
           c.customer_state,
           s.seller_state,
           2 * 6371 * ASIN(SQRT(
               POW(SIN(RADIANS(gs.lat - gc.lat) / 2), 2) +
               COS(RADIANS(gc.lat)) * COS(RADIANS(gs.lat)) *
               POW(SIN(RADIANS(gs.lng - gc.lng) / 2), 2)
           )) AS distance_km,
           i.n_items, i.weight_g, i.volume_cm3, i.price, i.freight, i.category,
           DATE_DIFF('day', CAST(o.order_purchase_timestamp AS DATE),
                            CAST(o.order_delivered_customer_date AS DATE)) AS actual_days,
           DATE_DIFF('day', CAST(o.order_purchase_timestamp AS DATE),
                            CAST(o.order_estimated_delivery_date AS DATE)) AS olist_promise_days
    FROM read_csv_auto('data/raw/olist_orders_dataset.csv') o
    JOIN read_csv_auto('data/raw/olist_customers_dataset.csv') c USING (customer_id)
    JOIN items i USING (order_id)
    JOIN read_csv_auto('data/raw/olist_sellers_dataset.csv') s ON s.seller_id = i.seller_id
    LEFT JOIN geo gc ON gc.zip = c.customer_zip_code_prefix
    LEFT JOIN geo gs ON gs.zip = s.seller_zip_code_prefix
    WHERE o.order_status = 'delivered'
      AND o.order_delivered_customer_date IS NOT NULL
      AND i.n_sellers = 1
      AND CAST(o.order_purchase_timestamp AS DATE) BETWEEN DATE '2017-01-01' AND DATE '2018-08-21'
)
SELECT *,
       ISODOW(purchase_ts)             AS purchase_dow,
       HOUR(purchase_ts)               AS purchase_hour,
       actual_days > olist_promise_days AS late
FROM base
WHERE distance_km IS NOT NULL
  AND actual_days >= 0
ORDER BY purchase_ts, order_id;
