with source as (
    select * from {{ source('yelp', 'raw_checkins') }}
),
cleaned as (
    select
        business_id,
        -- date: comma-separated string of checkin timestamps in source; kept raw.
        -- UNNEST to one-row-per-checkin in intermediate if you need that grain.
        date
    from source
    where business_id is not null
)
select * from cleaned