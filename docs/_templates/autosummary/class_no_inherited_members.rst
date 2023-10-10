{# This is identical to class.rst, except for the filtering in `set wanted_methods`. -#}

{% if referencefile %}
.. include:: {{ referencefile }}
{% endif %}

{{ objname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :no-members:
   :no-inherited-members:
   :no-special-members:
   :show-inheritance:

{% block attributes_summary %}
  {% if attributes %}
   .. rubric:: Attributes
    {% for item in attributes %}
   .. autoattribute:: {{ name }}.{{ item }}
    {%- endfor %}
  {% endif %}
{% endblock -%}

{% block methods_summary %}
  {% set wanted_methods = (methods | reject('in', inherited_members) | reject('==', '__init__') | list) %}
  {% if wanted_methods %}
   .. rubric:: Methods
    {% for item in wanted_methods %}
   .. automethod:: {{ name }}.{{ item }}
    {%- endfor %}
  {% endif %}
{% endblock %}
