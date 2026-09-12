with source as (
    select * from {{ source('yelp', 'raw_reviews') }}
),
cleaned as (
    select
        review_id,
        user_id,
        business_id,
        stars,
        cast(date as timestamp)     as date,
        trim(text)                  as text,
        useful,
        funny,
        cool,
        category
    from source
    where review_id is not null
)
select * from cleaned