with source as (
    select * from {{ source('yelp', 'raw_users') }}
),
cleaned as (
    select
        user_id,
        trim(name)                          as name,
        review_count,
        cast(yelping_since as timestamp)    as yelping_since,
        useful,
        funny,
        cool,
        fans,
        average_stars,
        -- friends: comma-separated string in source, kept raw for UNNEST in intermediate
        friends,
        -- elite: comma-separated years string, kept raw
        elite,
        -- compliment_* counts
        compliment_hot,
        compliment_more,
        compliment_profile,
        compliment_cute,
        compliment_list,
        compliment_note,
        compliment_plain,
        compliment_cool,
        compliment_funny,
        compliment_writer,
        compliment_photos
    from source
    where user_id is not null
)
select * from cleaned