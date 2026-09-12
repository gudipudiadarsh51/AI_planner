-- Trust score must be between 0 and 1. Returns rows if violated.
select review_id, review_trust_score
from {{ ref('int_review_score') }}
where review_trust_score < 0 or review_trust_score > 1