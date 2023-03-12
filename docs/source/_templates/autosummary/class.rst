{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
    {% block attributes %}
    {% if attributes %}
    .. rubric:: {{ _('Attributes') }}

    .. autosummary::
    {% for item in attributes %}
        {{ objname }}.{{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}


    {% block methods %}
    {% if methods %}
    .. rubric:: {{ _('Methods') }}

    .. autosummary::
        :nosignatures:
    {% for item in methods %}
        {{ objname }}.{{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}


{% block attributes_detail %}
{% if attributes %}

Attributes
----------

{% for item in attributes %}
    .. autoattribute:: {{ module }}.{{ objname }}.{{ item }}
{%- endfor %}
{% endif %}
{% endblock %}


{% block methods_detail %}
{% if methods %}

Methods
-------

{% for item in methods %}
    .. automethod:: {{ module }}.{{ objname }}.{{ item }}
{%- endfor %}
{% endif %}
{% endblock %}
