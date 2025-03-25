# Generated by Django 4.2.18 on 2025-03-21 05:34

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0024_areamanager_loandisbursementdata'),
    ]

    operations = [
        migrations.AlterUniqueTogether(
            name='loandisbursementdata',
            unique_together=None,
        ),
        migrations.RemoveField(
            model_name='loandisbursementdata',
            name='area_head',
        ),
        migrations.RemoveField(
            model_name='loandisbursementdata',
            name='loan',
        ),
        migrations.DeleteModel(
            name='AreaManager',
        ),
        migrations.DeleteModel(
            name='LoanDisbursementData',
        ),
    ]
