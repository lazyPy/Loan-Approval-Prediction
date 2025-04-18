import csv
import os
import django
from decimal import Decimal
from datetime import datetime

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'main.settings')

# Force using production database (Render PostgreSQL)
os.environ['RENDER'] = 'true'  # This will trigger the production database settings

django.setup()

from app.models import (
    Borrower, PersonalInformation, ContactAndAddress, Education,
    Employment, Expense, Vehicle, LoanDetails, Marketing, LoanStatus
)

def import_loans_from_csv(csv_file, update_existing=False):
    success_count = 0
    update_count = 0
    error_count = 0
    skip_count = 0
    
    with open(csv_file, 'r') as file:
        csv_reader = csv.DictReader(file)
        
        for row in csv_reader:
            reference_number = row['reference_number']
            
            # Check if borrower with this reference number already exists
            existing_borrower = Borrower.objects.filter(reference_number=reference_number).first()
            
            if existing_borrower:
                if update_existing:
                    try:
                        # Update existing borrower's information
                        if hasattr(existing_borrower, 'personal_info'):
                            personal_info = existing_borrower.personal_info
                            personal_info.first_name = row['first_name']
                            personal_info.middle_name = row['middle_name']
                            personal_info.last_name = row['last_name']
                            personal_info.gender = row['gender']
                            personal_info.date_of_birth = datetime.strptime(row['date_of_birth'], '%Y-%m-%d').date()
                            personal_info.civil_status = row['civil_status']
                            personal_info.religion = row['religion']
                            personal_info.sss_number = row['sss_number']
                            personal_info.tin_number = row['tin_number']
                            personal_info.property_area = row['property_area']
                            personal_info.residency_and_citizenship = row['residency_and_citizenship']
                            personal_info.save()
                        
                        # Similar updates for other related models...
                        # (Code omitted for brevity but would follow the same pattern)
                        
                        print(f"Updated existing loan application: {reference_number}")
                        update_count += 1
                        
                    except Exception as e:
                        print(f"Error updating loan {reference_number}: {str(e)}")
                        error_count += 1
                else:
                    print(f"Skipping loan {reference_number}: Reference number already exists")
                    skip_count += 1
                continue
            
            # Create new borrower if not exists
            try:
                # Create Borrower
                borrower = Borrower.objects.create(
                    reference_number=reference_number
                )
                
                # Create PersonalInformation
                PersonalInformation.objects.create(
                    loan=borrower,
                    first_name=row['first_name'],
                    middle_name=row['middle_name'],
                    last_name=row['last_name'],
                    gender=row['gender'],
                    date_of_birth=datetime.strptime(row['date_of_birth'], '%Y-%m-%d').date(),
                    civil_status=row['civil_status'],
                    religion=row['religion'],
                    sss_number=row['sss_number'],
                    tin_number=row['tin_number'],
                    property_area=row['property_area'],
                    residency_and_citizenship=row['residency_and_citizenship']
                )
                
                # Create ContactAndAddress
                ContactAndAddress.objects.create(
                    loan=borrower,
                    contact_number=row['contact_number'],
                    email_address=row['email_address'],
                    facebook_account=row['facebook_account'],
                    no_and_street=row['no_and_street'],
                    barangay=row['barangay'],
                    municipality=row['municipality'],
                    province=row['province'],
                    resident_status=row['resident_status'],
                    length_of_stay=row['length_of_stay'],
                    permanent_address=row['permanent_address'],
                    provincial_address=row['provincial_address'],
                    parents_address=row['parents_address']
                )
                
                # Create Education
                Education.objects.create(
                    loan=borrower,
                    education=row['education'],
                    course=row['course'],
                    school_last_attended=row['school_last_attended'],
                    year_graduated=int(row['year_graduated'])
                )
                
                # Create Employment
                Employment.objects.create(
                    loan=borrower,
                    source_of_funds=row['source_of_funds'],
                    employer_business_name=row['employer_business_name'],
                    employer_contact_number=row['employer_contact_number'],
                    position=row['position'],
                    employment_status=row['employment_status'],
                    monthly_net_income=Decimal(row['monthly_net_income']),
                    employer_business_address=row['employer_business_address'],
                    other_income=row['other_income'],
                    other_monthly_income=Decimal(row['other_monthly_income'])
                )
                
                # Create Expense
                Expense.objects.create(
                    loan=borrower,
                    food_and_groceries=Decimal(row['food_and_groceries']),
                    electric_and_water=Decimal(row['electric_and_water']),
                    education_and_misc=Decimal(row['education_and_misc']),
                    other_expense=Decimal(row['other_expense'])
                )
                
                # Create Vehicle
                Vehicle.objects.create(
                    loan=borrower,
                    make_brand=row['make_brand'],
                    series=row['series'],
                    year_model=int(row['year_model']),
                    variant=row['variant'],
                    color=row['color'],
                    plate_no=row['plate_no'],
                    engine_no=row['engine_no'],
                    chassis_no=row['chassis_no'],
                    transmission=row['transmission'],
                    fuel=row['fuel'],
                    dealer_name=row['dealer_name'],
                    dealer_address=row['dealer_address'],
                    dealer_contact_number=row['dealer_contact_number']
                )
                
                # Create LoanDetails
                LoanDetails.objects.create(
                    loan=borrower,
                    loan_type=row['loan_type'],
                    estimated_vehicle_value=Decimal(row['estimated_vehicle_value']),
                    down_payment_percentage=Decimal(row['down_payment_percentage']),
                    loan_amount_term=int(row['loan_amount_term']),
                    loan_purpose=row['loan_purpose']
                )
                
                # Create Marketing
                Marketing.objects.create(
                    loan=borrower,
                    marketing_source=row['marketing_source'],
                    sales_representative=row['sales_representative']
                )
                
                # Create initial LoanStatus
                LoanStatus.objects.create(
                    loan=borrower
                )
                
                print(f"Successfully imported loan application: {reference_number}")
                success_count += 1
                
            except Exception as e:
                print(f"Error importing loan {reference_number}: {str(e)}")
                error_count += 1
                continue
    
    # Print summary
    print("\nImport Summary:")
    print(f"Total processed: {success_count + update_count + skip_count + error_count}")
    print(f"Successfully imported: {success_count}")
    if update_existing:
        print(f"Updated existing: {update_count}")
    else:
        print(f"Skipped existing: {skip_count}")
    print(f"Errors: {error_count}")

if __name__ == '__main__':
    csv_file = 'loan_template.csv'  # Make sure this file is in the same directory
    
    # Set to True if you want to update existing records instead of skipping them
    update_existing = False
    
    import_loans_from_csv(csv_file, update_existing) 