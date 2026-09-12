-- is_suspicious must be true exactly when one of the hard fraud flags is true.
-- Returns rows only if the logic is inconsistent.
select review_id, is_suspicious, flag_repeated_text, flag_extreme_burst, flag_coordinated_text
from {{ ref('int_review_score') }}
where is_suspicious != (flag_repeated_text or flag_extreme_burst or flag_coordinated_text)