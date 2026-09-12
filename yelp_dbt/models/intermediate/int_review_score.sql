-- models/intermediate/int_review_score.sql
{{ config(materialized='table') }}

with signals as (

    select * from {{ ref('int_review_signals') }}

),

flagged as (

    select
        review_id,
        user_id,
        business_id,
        stars,
        review_dt,
        text_length,
        useful,
        funny,
        cool,
        mins_since_prev_review,
        reviews_same_day,
        same_text_count,
        users_same_text,

        coalesce(mins_since_prev_review between 0 and 10 
        and business_id != prev_business_id,
            false)                                  as flag_rapid_review,
        (reviews_same_day > 5)                       as flag_same_day_burst,
        (reviews_same_day > 15)                      as flag_extreme_burst,
        (same_text_count > 1)                        as flag_repeated_text,
        (users_same_text > 3)                        as flag_coordinated_text,
        (text_length < 20)                           as flag_short_text

    from signals

),

scored as (

    select
        *,

        greatest(0.0,
            1.0
            - (case when flag_rapid_review     then 0.10 else 0.0 end)
            - (case when flag_same_day_burst   then 0.20 else 0.0 end)
            - (case when flag_extreme_burst    then 0.40 else 0.0 end)
            - (case when flag_repeated_text    then 0.50 else 0.0 end)
            - (case when flag_coordinated_text then 0.40 else 0.0 end)
            - (case when flag_short_text       then 0.10 else 0.0 end)
        ) as review_trust_score,

        (flag_repeated_text
            or flag_extreme_burst
            or flag_coordinated_text)             as is_suspicious

    from flagged

)

select * from scored