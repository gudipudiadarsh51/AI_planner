with source as (
    select * from {{ source('yelp', 'raw_tips') }}
),
cleaned as (
    select
        user_id,
        business_id,
        cast(date as timestamp)     as date,
        trim(text)                  as text,
        compliment_count
    from source
    where user_id is not null
      and business_id is not null
)
select * from cleaned