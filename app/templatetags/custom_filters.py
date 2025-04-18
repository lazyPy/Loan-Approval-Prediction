from decimal import Decimal
from django import template
from datetime import datetime
from dateutil.relativedelta import relativedelta

register = template.Library()


@register.filter
def div(value, arg):
    """Divide the value by the argument"""
    try:
        return Decimal(str(value)) / Decimal(str(arg))
    except (ValueError, ZeroDivisionError):
        return 0


@register.filter
def mul(value, arg):
    """Multiply the value by the argument"""
    try:
        return Decimal(str(value)) * Decimal(str(arg))
    except ValueError:
        return 0


@register.filter
def sub(value, arg):
    """Subtract the argument from the value"""
    try:
        return Decimal(str(value)) - Decimal(str(arg))
    except ValueError:
        return 0


@register.filter
def add_months(value, months):
    try:
        if isinstance(value, str):
            date = datetime.strptime(value, '%Y-%m-%d')
        else:
            date = value
        new_date = date + relativedelta(months=int(months))
        return new_date.strftime('%b %d, %Y')
    except (ValueError, TypeError):
        return value


@register.filter
def make_range(value):
    """Create a range of numbers from 1 to the given value"""
    try:
        value = int(value)
        return range(1, value + 1)
    except (ValueError, TypeError):
        return range(0)
