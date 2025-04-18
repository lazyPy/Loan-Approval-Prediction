from django.core.management.base import BaseCommand
from app.forecasting import generate_initial_images

class Command(BaseCommand):
    help = 'Generate forecast images for historical data and model performance'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Starting image generation...'))
        
        # Run the function to generate the images
        success = generate_initial_images()
        
        if success:
            self.stdout.write(self.style.SUCCESS('Successfully generated forecast images'))
        else:
            self.stdout.write(self.style.ERROR('Failed to generate forecast images')) 