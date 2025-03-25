from decimal import Decimal
from django import template

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
