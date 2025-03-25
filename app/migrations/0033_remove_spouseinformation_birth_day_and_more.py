# Generated by Django 4.2.18 on 2025-03-24 05:09

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0032_alter_contactandaddress_length_of_stay'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='spouseinformation',
            name='birth_day',
        ),
        migrations.RemoveField(
            model_name='spouseinformation',
            name='birth_month',
        ),
        migrations.RemoveField(
            model_name='spouseinformation',
            name='birth_year',
        ),
        migrations.AddField(
            model_name='spouseinformation',
            name='date_of_birth',
            field=models.DateField(default=django.utils.timezone.now),
            preserve_default=False,
        ),
    ]
