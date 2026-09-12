-- models/intermediate/int_covisit_friend_pairs.sql
{{ config(materialized='table') }}

with philly_reviews as (
    select distinct
        r.user_id,
        r.business_id,
        date(r.date) as visit_day
    from {{ ref('stg_reviews') }} r
    join {{ ref('stg_businesses') }} b
        on r.business_id = b.business_id
    where b.city = 'Philadelphia'
      and b.categories like '%Restaurant%'
),

covisits as (
    select
        a.business_id,
        a.visit_day,
        a.user_id as user_a,
        b.user_id as user_b
    from philly_reviews a
    join philly_reviews b
        on a.business_id = b.business_id
        and a.visit_day = b.visit_day
        and a.user_id < b.user_id
)

select
    c.business_id,
    c.visit_day,
    c.user_a,
    c.user_b
from covisits c
join {{ ref('int_social_edges') }} e
    on c.user_a = e.user_id
    and c.user_b = e.friend_id