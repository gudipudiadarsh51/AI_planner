WITH base AS (
    SELECT
        business_id,
        TIMESTAMP(TRIM(checkin_dt)) AS checkin_dt
    FROM {{ ref('stg_checkins') }},
    UNNEST(SPLIT(date, ',')) AS checkin_dt WHERE category = 'Checkin'
)

SELECT COUNT(*) AS matching_reviews
FROM base
JOIN {{ ref('stg_reviews') }} r
    ON base.business_id = r.business_id
    AND DATE(r.date) = DATE(base.checkin_dt)