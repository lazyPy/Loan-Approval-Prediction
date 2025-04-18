from django.core.management.base import BaseCommand
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tkinter issues
import matplotlib.pyplot as plt
import numpy as np
import os

class Command(BaseCommand):
    help = 'Create placeholder images for model performance visualizations'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Creating placeholder images for model performance...'))
        
        # Create directory if it doesn't exist
        os.makedirs('static/img', exist_ok=True)
        
        # Create a "no_image" placeholder if it doesn't exist
        no_image_path = 'static/img/no_image.png'
        if not os.path.exists(no_image_path):
            # Create a simple "No Image Available" image
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'No Image Available', 
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=20, color='gray')
            plt.axis('off')
            plt.savefig(no_image_path)
            plt.close()
            self.stdout.write(f"Created {no_image_path}")
        
        # Create model performance placeholders for each frequency
        frequencies = ['Weekly', 'Monthly', 'Quarterly']
        
        for freq in frequencies:
            # Create mock data for model performance
            x = np.arange(10)
            y_actual = np.sin(x) * 10000 + 50000 + np.random.normal(0, 2000, 10)
            y_pred = y_actual + np.random.normal(0, 3000, 10)
            
            # Generate model performance image
            plt.figure(figsize=(12, 6))
            plt.plot(x, y_actual, 'o-', label='Actual')
            plt.plot(x, y_pred, 'x-', label='Predicted')
            plt.title(f'{freq} Frequency - Model Performance')
            
            # Set appropriate x-axis label based on frequency
            if freq == 'Weekly':
                x_label = 'Week'
                # Create week labels like "Week 1", "Week 2", etc.
                x_ticks = x
                x_tick_labels = [f"Week {i+1}" for i in x]
            elif freq == 'Monthly':
                x_label = 'Month'
                # Create month labels
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                x_ticks = x
                x_tick_labels = [months[i % 12] for i in x]
            elif freq == 'Quarterly':
                x_label = 'Quarter'
                # Create quarter labels
                x_ticks = x
                x_tick_labels = [f"Q{i+1}" for i in x]
            else:
                x_label = 'Date'
                x_ticks = x
                x_tick_labels = [f"Date {i+1}" for i in x]
            
            # Set the custom tick positions and labels
            plt.xticks(x_ticks, x_tick_labels)
                
            plt.xlabel(x_label)
            plt.ylabel('Loan Amount')
            plt.legend()
            plt.grid(True)
            
            # Save image
            image_path = f'static/img/{freq}_Frequency_-_Model_Performance.png'
            plt.savefig(image_path)
            plt.close()
            self.stdout.write(f"Created {image_path}")
        
        self.stdout.write(self.style.SUCCESS('Successfully created all placeholder images')) 