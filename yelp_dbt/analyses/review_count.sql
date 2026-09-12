WITH totals AS (

    SELECT COUNT(*) AS total_reviews
    FROM {{ ref('stg_reviews') }}

),

matches AS (

    SELECT matching_reviews
    FROM {{ ref('int_group_formation') }}

)

SELECT
    total_reviews,
    matching_reviews,
    SAFE_DIVIDE(matching_reviews, total_reviews) AS review_match_ratio,
    SAFE_DIVIDE(matching_reviews, total_reviews) * 100 AS review_match_percentage

FROM totals
CROSS JOIN matches