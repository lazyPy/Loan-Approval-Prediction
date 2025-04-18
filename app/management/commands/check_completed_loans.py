from django.core.management.base import BaseCommand
from app.models import Borrower, LoanDisbursementOfficerRemarks

class Command(BaseCommand):
    help = 'Check for completed loans in the database'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Checking for completed loans...'))
        
        # Query for completed loans
        completed_loans = Borrower.objects.filter(
            loan_disbursement_officer_remarks__status='COMPLETED'
        ).select_related(
            'loan_details',
            'loan_disbursement_officer_remarks'
        )
        
        # Count total loans
        total_loans = Borrower.objects.count()
        
        self.stdout.write(f"Total loans in database: {total_loans}")
        self.stdout.write(f"Completed loans: {completed_loans.count()}")
        
        # Check loan details
        if completed_loans.exists():
            self.stdout.write("\nDetails of completed loans:")
            
            for idx, loan in enumerate(completed_loans[:5], 1):  # Show first 5 loans
                self.stdout.write(f"\nLoan #{idx}:")
                self.stdout.write(f"  Loan ID: {loan.loan_id}")
                self.stdout.write(f"  Reference Number: {loan.reference_number}")
                
                if hasattr(loan, 'loan_details'):
                    self.stdout.write(f"  Loan Amount: {loan.loan_details.loan_amount_applied}")
                else:
                    self.stdout.write("  No loan_details associated")
                
                if hasattr(loan, 'loan_disbursement_officer_remarks'):
                    self.stdout.write(f"  Disbursement Date: {loan.loan_disbursement_officer_remarks.disbursement_date}")
                    self.stdout.write(f"  Status: {loan.loan_disbursement_officer_remarks.status}")
                else:
                    self.stdout.write("  No loan_disbursement_officer_remarks associated")
            
            if completed_loans.count() > 5:
                self.stdout.write(f"\n... and {completed_loans.count() - 5} more completed loans")
        else:
            self.stdout.write(self.style.WARNING("\nNo completed loans found in the database"))
            self.stdout.write(self.style.WARNING("This is why forecast images cannot be generated"))
            
            # Suggest solution
            self.stdout.write("\nSuggested solution:")
            self.stdout.write("1. Make sure there are loans with status 'COMPLETED' in the database")
            self.stdout.write("2. Each completed loan should have associated loan_details and loan_disbursement_officer_remarks")
            self.stdout.write("3. Manually create some completed loans for testing if needed") 