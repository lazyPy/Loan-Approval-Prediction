# Generated by Django 4.2.18 on 2025-03-21 05:13

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0022_areamanager_dailyloandisbursement_forecastresult_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='areamanager',
            name='user',
        ),
        migrations.DeleteModel(
            name='DailyLoanDisbursement',
        ),
        migrations.DeleteModel(
            name='ForecastResult',
        ),
        migrations.RemoveField(
            model_name='loandisbursement',
            name='area_manager',
        ),
        migrations.RemoveField(
            model_name='loandisbursement',
            name='sales_representative',
        ),
        migrations.DeleteModel(
            name='MonthlyLoanDisbursement',
        ),
        migrations.RemoveField(
            model_name='salesrepresentative',
            name='area_manager',
        ),
        migrations.RemoveField(
            model_name='salesrepresentative',
            name='user',
        ),
        migrations.DeleteModel(
            name='WeeklyLoanDisbursement',
        ),
        migrations.DeleteModel(
            name='YearlyLoanDisbursement',
        ),
        migrations.DeleteModel(
            name='AreaManager',
        ),
        migrations.DeleteModel(
            name='LoanDisbursement',
        ),
        migrations.DeleteModel(
            name='SalesRepresentative',
        ),
    ]
