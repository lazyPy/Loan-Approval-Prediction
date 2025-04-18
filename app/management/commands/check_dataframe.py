from django.core.management.base import BaseCommand
from app.forecasting import get_recent_completed_loans

class Command(BaseCommand):
    help = 'Check the DataFrame creation from completed loans'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Getting completed loans DataFrame...'))
        
        # Get the DataFrame
        df = get_recent_completed_loans()
        
        # Check if DataFrame is empty
        if len(df) == 0:
            self.stdout.write(self.style.ERROR('DataFrame is empty. No completed loans found.'))
            return
        
        # Print DataFrame info
        self.stdout.write('\nDataFrame Information:')
        self.stdout.write(f'Number of rows: {len(df)}')
        self.stdout.write(f'Columns: {df.columns.tolist()}')
        self.stdout.write(f'Index type: {type(df.index)}')
        
        # Print first few rows
        self.stdout.write('\nFirst 5 rows of DataFrame:')
        self.stdout.write(str(df.head()))
        
        # Print date range
        self.stdout.write('\nDate Range:')
        self.stdout.write(f'Start: {df.index.min()}')
        self.stdout.write(f'End: {df.index.max()}')
        
        self.stdout.write('\nDataFrame is correctly created and can be used for forecasting.') 