{{config(
    materialized='table',
    tags=['intermediate'],
    description="This model represents the social edges between users in the Yelp dataset. It captures the relationships between users and their friends, ensuring that each (user_id, friend_id) pair is unique and valid."
)}}
with base as (
    select
        review_id,
        user_id,
        business_id,
        stars,
        date         as review_dt,
        text         as review_text,
        length(text) as text_length,
        useful,
        funny,
        cool
    from {{ ref('stg_reviews') }} 
),
signals as (
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

        timestamp_diff(review_dt, lag(review_dt) over (partition by user_id 
        order by review_dt, review_id), minute) as mins_since_prev_review,

        lag(business_id) over (partition by user_id order by review_dt, review_id) as prev_business_id,

        count(*) over (partition by user_id, DATE(review_dt)) as reviews_same_day,

        count(*) over( partition by user_id, review_text ) as same_text_count,

        count(distinct user_id) over (partition by review_text ) as users_same_text
        
        from base
        )
select * from signals
