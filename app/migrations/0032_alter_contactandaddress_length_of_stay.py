# Generated by Django 4.2.18 on 2025-03-24 03:25

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0031_quotasettings'),
    ]

    operations = [
        migrations.AlterField(
            model_name='contactandaddress',
            name='length_of_stay',
            field=models.CharField(choices=[('less_than_6_months', 'Less than 6 months'), ('6_months_to_1_year', '6 months – 1 year'), ('1_to_3_years', '1 – 3 years'), ('3_to_5_years', '3 – 5 years'), ('5_to_10_years', '5 – 10 years'), ('more_than_10_years', 'More than 10 years'), ('since_birth', 'Since birth')], max_length=50),
        ),
    ]
