{{config(materialized='table')}}

with users as (
    select user_id, friends from {{ ref('stg_users') }}
    where friends is not null and friends != 'None' and trim(friends) != ''
),
exploded as (
    select user_id, trim(friend_id) as friend_id from users, unnest(split(friends,',')) as friend_id
    where trim(friend_id)!= ''
)
select distinct user_id, friend_id from exploded where friend_id!= user_id