{% macro null_check(table_name)%}
{%set relation = ref(table_name)%}
{% set columns = adapter.get_columns_in_relation(relation) %} 

with null_counts as (
    {%for column in columns%}
    select '{{column.name}}' as column_name, count(*) - count({{ column.name }}) 
    as null_count
    from {{ relation }}
    {%if not loop.last %}union all{% endif %}
    {%endfor%}
)

select column_name, null_count from null_counts
where null_count > 0 order by null_count desc
{% endmacro %}
