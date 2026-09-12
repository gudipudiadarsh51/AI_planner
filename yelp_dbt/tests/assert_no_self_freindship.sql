-- no user should be their own friend
--returns rows where only self loop exists
select user_id from {{ref('int_social_edges')}} where user_id = friend_id