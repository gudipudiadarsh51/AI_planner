-- analyses/null_check_all.sql

{% set tables = ['stg_users', 'stg_businesses', 'stg_reviews', 'stg_tips', 'stg_checkins'] %}

{% for table in tables %}
select '{{ table }}' as source_table, column_name, null_count
from ({{ null_check(table) }})
{% if not loop.last %}union all{% endif %}
{% endfor %}
order by source_table, null_count desc