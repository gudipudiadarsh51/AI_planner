WITH philly_reviews AS (
    SELECT r.user_id, r.business_id, r.date AS review_ts
    FROM `sigr-ag-13719.yelp.stg_reviews` r
    JOIN `sigr-ag-13719.yelp.stg_businesses` b ON r.business_id = b.business_id
    WHERE b.city='Philadelphia' AND b.categories LIKE '%Restaurant%'
),
pairs AS (
    SELECT
        a.business_id,
        a.user_id AS ua, b.user_id AS ub,
        -- bucket the co-visit to a representative day for grouping
        DATE(a.review_ts) AS event_day
    FROM philly_reviews a
    JOIN philly_reviews b
      ON a.business_id=b.business_id AND a.user_id<b.user_id
      AND ABS(TIMESTAMP_DIFF(a.review_ts, b.review_ts, DAY)) <= 2
    JOIN `sigr-ag-13719.yelp.int_social_edges` e
      ON a.user_id=e.user_id AND b.user_id=e.friend_id
)
SELECT
    COUNT(*) total_pairs,
    COUNT(DISTINCT CONCAT(business_id, CAST(event_day AS STRING))) events,
    ROUND(COUNT(*)/COUNT(DISTINCT CONCAT(business_id, CAST(event_day AS STRING))),2) avg_pairs_per_event
FROM pairs