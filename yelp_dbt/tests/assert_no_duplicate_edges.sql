--Each (user_id, friend_id) pair should be unique. This is a directed graph, so (user_id, friend_id) is different from (friend_id, user_id)
-- return rows only if the duplicates exists 
select user_id, friend_id, count(*) as duplicate_count
from {{ref('int_social_edges')}}
group by user_id, friend_id
having count(*) > 1