from django.core.management.base import BaseCommand
# Configure matplotlib to use non-interactive backend
import matplotlib
matplotlib.use('Agg')
from app.forecasting import train_and_save_models, generate_initial_images

class Command(BaseCommand):
    help = 'Train forecasting models and generate visualizations'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Starting forecasting model training...'))
        
        try:
            # First, generate the initial image visualizations
            self.stdout.write('Generating initial visualizations...')
            visualization_success = generate_initial_images()
            
            if visualization_success:
                self.stdout.write(self.style.SUCCESS('Successfully generated visualizations.'))
            else:
                self.stdout.write(self.style.WARNING('Failed to generate some visualizations.'))
            
            # Then train and save models
            self.stdout.write('Training models...')
            results = train_and_save_models()
            
            if results:
                self.stdout.write(self.style.SUCCESS('Successfully trained and saved all models.'))
                
                # Show information about the trained models
                for freq, data in results.items():
                    if 'forecast' in data and len(data['forecast']) > 0:
                        self.stdout.write(f"\n{freq} Frequency Model:")
                        self.stdout.write(f"  Future data points: {len(data['forecast'])}")
                        self.stdout.write(f"  Forecast range: {data['forecast'].index.min()} to {data['forecast'].index.max()}")
            else:
                self.stdout.write(self.style.WARNING('Failed to train some models.'))
                
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error during training: {str(e)}'))
            import traceback
            traceback.print_exc() 