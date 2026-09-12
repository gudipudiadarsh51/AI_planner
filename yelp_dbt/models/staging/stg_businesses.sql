-- models/staging/stg_businesses.sql

with source as (

    select * from {{ source('yelp', 'raw_businesses') }}

),

cleaned as (

    select
        -- identifier
        business_id,

        -- descriptive text (trimmed)
        trim(name)      as name,
        trim(address)   as address,
        trim(city)      as city,
        trim(state)     as state,
        postal_code,

        -- geo (already FLOAT64 in source)
        latitude,
        longitude,

        -- metrics (already correctly typed)
        stars,
        review_count,
        is_open,

        -- categories: comma-separated string, kept as-is
        categories,

        -- selected flat attributes, normalized from Yelp's "True"/"False" strings
        safe_cast(json_value(attributes, '$.RestaurantsPriceRange2') as int64) as price_range,

        case
            when json_value(attributes, '$.RestaurantsTakeOut') = 'True'  then true
            when json_value(attributes, '$.RestaurantsTakeOut') = 'False' then false
        end as takeout,

        case
            when json_value(attributes, '$.RestaurantsDelivery') = 'True'  then true
            when json_value(attributes, '$.RestaurantsDelivery') = 'False' then false
        end as delivery,

        -- keep full blobs for downstream extraction
        attributes as attributes_json,
        hours       as hours_json

    from source
    where business_id is not null

)

select * from cleaned