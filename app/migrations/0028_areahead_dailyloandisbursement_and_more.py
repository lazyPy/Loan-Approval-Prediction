# Generated by Django 4.2.18 on 2025-03-21 08:06

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0027_delete_dailyloandisbursement_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='AreaHead',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('area', models.CharField(max_length=100)),
                ('monthly_quota', models.DecimalField(decimal_places=2, max_digits=12)),
            ],
        ),
        migrations.CreateModel(
            name='DailyLoanDisbursement',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date', models.DateField(unique=True)),
                ('total_amount', models.DecimalField(decimal_places=2, default=0, max_digits=15)),
                ('disbursement_percentage', models.DecimalField(decimal_places=2, default=0, max_digits=5)),
            ],
            options={
                'ordering': ['-date'],
            },
        ),
        migrations.CreateModel(
            name='YearlyLoanDisbursement',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('year', models.IntegerField(unique=True)),
                ('total_amount', models.DecimalField(decimal_places=2, default=0, max_digits=15)),
                ('disbursement_percentage', models.DecimalField(decimal_places=2, default=0, max_digits=5)),
                ('yearly_quota', models.DecimalField(decimal_places=2, default=0, max_digits=15)),
                ('quota_met', models.BooleanField(default=False)),
            ],
            options={
                'ordering': ['-year'],
            },
        ),
        migrations.CreateModel(
            name='WeeklyLoanDisbursement',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('year', models.IntegerField()),
                ('month', models.IntegerField()),
                ('week', models.IntegerField(choices=[(1, 'Week 1'), (2, 'Week 2'), (3, 'Week 3'), (4, 'Week 4'), (5, 'Week 5')])),
                ('start_date', models.DateField()),
                ('end_date', models.DateField()),
                ('total_amount', models.DecimalField(decimal_places=2, default=0, max_digits=15)),
                ('disbursement_percentage', models.DecimalField(decimal_places=2, default=0, max_digits=5)),
            ],
            options={
                'ordering': ['-year', '-month', '-week'],
                'unique_together': {('year', 'month', 'week')},
            },
        ),
        migrations.CreateModel(
            name='SalesRepresentative',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('area_head', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='sales_representatives', to='app.areahead')),
            ],
        ),
        migrations.CreateModel(
            name='MonthlyLoanDisbursement',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('year', models.IntegerField()),
                ('month', models.IntegerField()),
                ('total_amount', models.DecimalField(decimal_places=2, default=0, max_digits=15)),
                ('disbursement_percentage', models.DecimalField(decimal_places=2, default=0, max_digits=5)),
                ('monthly_quota', models.DecimalField(decimal_places=2, default=0, max_digits=15)),
                ('quota_met', models.BooleanField(default=False)),
            ],
            options={
                'ordering': ['-year', '-month'],
                'unique_together': {('year', 'month')},
            },
        ),
    ]
