with philly_restaurants as (
    select business_id
    from `sigr-ag-13719.yelp.stg_businesses`
    where city = 'Philadelphia' and categories like '%Restaurant%'
),
checkin_events as (
    select extract(hour from timestamp(trim(ci))) as hr
    from `sigr-ag-13719.yelp.stg_checkins` c
    join philly_restaurants p on c.business_id = p.business_id,
    unnest(split(c.date, ',')) as ci
    where trim(ci) != ''
),
review_events as (
    select extract(hour from r.date) as hr
    from `sigr-ag-13719.yelp.stg_reviews` r
    join philly_restaurants p on r.business_id = p.business_id
)
select
    hr as hour_of_day,
    (select count(*) from checkin_events c where c.hr = h.hr) as checkins,
    (select count(*) from review_events rv where rv.hr = h.hr) as reviews
from unnest(generate_array(0,23)) as hr
cross join (select 1) h   -- placeholder
order by hour_of_day