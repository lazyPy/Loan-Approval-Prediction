from django.db import models
import uuid
from datetime import date, datetime
from decimal import Decimal, InvalidOperation, DivisionByZero
import time

class Borrower(models.Model):
    LOAN_STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('PROCEED_CI', 'Proceed to Credit Investigator'),
        ('PROCEED_LAO', 'Proceed to Loan Approval Officer'),
        ('PROCEED_LDO', 'Proceed to Loan Disbursement Officer'),
        ('COMPLETED', 'Completed'),
        ('HOLD', 'Hold'),
        ('CANCELLED', 'Cancelled'),
        ('DECLINED', 'Declined'),
    ]
    
    # Basic Information
    loan_id = models.AutoField(primary_key=True)
    reference_number = models.CharField(max_length=12, unique=True, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        # Generate reference number if not set
        if not self.reference_number:
            # Format: YYMMDDXXXXXX where XXXXXX is a random number
            import random
            date_part = datetime.now().strftime('%y%m%d')
            random_part = str(random.randint(0, 999999)).zfill(6)
            self.reference_number = f"{date_part}{random_part}"
        super().save(*args, **kwargs)

    def __str__(self):
        if hasattr(self, 'personal_info'):
            return f"{self.personal_info.last_name}, {self.personal_info.first_name}"
        return f"Borrower {self.loan_id}"

    class Meta:
        ordering = ['-created_at']

class PersonalInformation(models.Model):
    GENDER_CHOICES = [('M', 'Male'), ('F', 'Female')]
    CIVIL_STATUS_CHOICES = [('Y', 'Married'), ('N', 'Single')]
    PROPERTY_AREA_CHOICES = [
        ('Urban', 'Urban'),
        ('Semi-Urban', 'Semi-Urban'),
        ('Rural', 'Rural')
    ]
    RESIDENCY_CHOICES = [
        ('Resident Filipino Citizen', 'Resident Filipino Citizen'),
        ('Resident Alien', 'Resident Alien'),
        ('Non-Resident Citizen/Alien', 'Non-Resident Citizen/Alien')
    ]
    
    loan = models.OneToOneField(Borrower, on_delete=models.CASCADE, related_name='personal_info')
    first_name = models.CharField(max_length=100)
    middle_name = models.CharField(max_length=100, blank=True)
    last_name = models.CharField(max_length=100)
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES)
    date_of_birth = models.DateField()
    age = models.IntegerField()
    civil_status = models.CharField(max_length=1, choices=CIVIL_STATUS_CHOICES)
    religion = models.CharField(max_length=50)
    sss_number = models.CharField(max_length=20)
    tin_number = models.CharField(max_length=20)
    property_area = models.CharField(max_length=20, choices=PROPERTY_AREA_CHOICES)
    residency_and_citizenship = models.CharField(max_length=50, choices=RESIDENCY_CHOICES)

    def save(self, *args, **kwargs):
        # Calculate age before saving
        if self.date_of_birth:
            today = date.today()
            self.age = today.year - self.date_of_birth.year - ((today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day))
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.last_name}, {self.first_name}"

class ContactAndAddress(models.Model):
    RESIDENT_STATUS_CHOICES = [
        ('Residence', 'Residence'),
        ('Renting', 'Renting')
    ]
    
    LENGTH_OF_STAY_CHOICES = [
        ('less_than_6_months', 'Less than 6 months'),
        ('6_months_to_1_year', '6 months – 1 year'),
        ('1_to_3_years', '1 – 3 years'),
        ('3_to_5_years', '3 – 5 years'),
        ('5_to_10_years', '5 – 10 years'),
        ('more_than_10_years', 'More than 10 years'),
        ('since_birth', 'Since birth')
    ]
    
    loan = models.OneToOneField(Borrower, on_delete=models.CASCADE, related_name='contact_info')
    contact_number = models.CharField(max_length=20)
    email_address = models.EmailField()
    facebook_account = models.CharField(max_length=100, blank=True)
    no_and_street = models.CharField(max_length=200)
    barangay = models.CharField(max_length=100)
    municipality = models.CharField(max_length=100)
    province = models.CharField(max_length=100)
    resident_status = models.CharField(max_length=20, choices=RESIDENT_STATUS_CHOICES)
    length_of_stay = models.CharField(max_length=50, choices=LENGTH_OF_STAY_CHOICES)
    permanent_address = models.TextField()
    provincial_address = models.TextField(blank=True)
    parents_address = models.TextField()

class Education(models.Model):

    EDUCATION_CHOICES = [
        ('Graduate', 'Graduate'),
        ('Under Graduate', 'Under Graduate')
    ]

    loan = models.OneToOneField(Borrower, on_delete=models.CASCADE, related_name='education')
    education = models.CharField(max_length=20, choices=EDUCATION_CHOICES)
    course = models.CharField(max_length=100)
    school_last_attended = models.CharField(max_length=200)
    year_graduated = models.IntegerField()

class Dependent(models.Model):
    loan = models.ForeignKey(Borrower, on_delete=models.CASCADE, related_name='dependents')
    name = models.CharField(max_length=100)
    age = models.IntegerField()
    school = models.CharField(max_length=200)
    relation = models.CharField(max_length=50)
    self_employed = models.CharField(max_length=1, choices=[('Y', 'Yes'), ('N', 'No')])

    def __str__(self):
        return f"{self.name} ({self.relation})"
    
    @classmethod
    def get_dependent_count(cls, loan):
        """Get the count of dependents for a loan"""
        return cls.objects.filter(loan=loan).count()

class SpouseInformation(models.Model):

    CIVIL_STATUS_CHOICES = [('Y', 'Married'), ('N', 'Single')]
    EDUCATION_CHOICES = [
        ('Graduate', 'Graduate'),
        ('Under Graduate', 'Under Graduate')
    ]

    loan = models.OneToOneField(Borrower, on_delete=models.CASCADE, related_name='spouse')
    first_name = models.CharField(max_length=100)
    middle_name = models.CharField(max_length=100, blank=True)
    last_name = models.CharField(max_length=100)
    relation_to_borrower = models.CharField(max_length=50)
    civil_status = models.CharField(max_length=1, choices=CIVIL_STATUS_CHOICES)
    date_of_birth = models.DateField()
    education = models.CharField(max_length=20, choices=EDUCATION_CHOICES)
    net_income = models.DecimalField(max_digits=12, decimal_places=2)
    employer_business_name = models.CharField(max_length=200)
    employer_contact_number = models.CharField(max_length=20)
    other_income = models.CharField(max_length=200, blank=True)
    other_monthly_income = models.DecimalField(max_digits=12, decimal_places=2, default=0)

class Employment(models.Model):
    EMPLOYMENT_STATUS_CHOICES = [
        ('Employee', 'Employee'),
        ('Registered Business', 'Registered Business'),
        ('Retired', 'Retired'),
        ('Gaming', 'Gaming and gambling entities'),
        ('Unemployed', 'Unemployed (Dependent on Family)'),
        ('Unemployed_IND', 'Unemployed AND NOT dependent on Family member'),
        ('Student', 'Student'),
        ('Lawyer', 'Lawyer'),
        ('Accountant', 'Accountant'),
    ]
    
    SOURCE_OF_FUNDS_CHOICES = [
        ('Salary', 'Salary'),
        ('Income from Business', 'Income from Business'),
        ('Support From Relative', 'Support From Relative'),
        ('Commissions', 'Commissions'),
        ('Remittance', 'Remittance'),
    ]
    
    loan = models.OneToOneField(Borrower, on_delete=models.CASCADE, related_name='employment')
    source_of_funds = models.CharField(max_length=50, choices=SOURCE_OF_FUNDS_CHOICES)
    employer_business_name = models.CharField(max_length=200)
    employer_contact_number = models.CharField(max_length=20)
    position = models.CharField(max_length=100)
    employment_status = models.CharField(max_length=50, choices=EMPLOYMENT_STATUS_CHOICES)
    monthly_net_income = models.DecimalField(max_digits=12, decimal_places=2)
    employer_business_address = models.TextField()
    other_income = models.CharField(max_length=200, blank=True)
    other_monthly_income = models.DecimalField(max_digits=12, decimal_places=2, default=0)

class Expense(models.Model):
    loan = models.OneToOneField(Borrower, on_delete=models.CASCADE, related_name='expenses')
    food_and_groceries = models.DecimalField(max_digits=12, decimal_places=2)
    electric_and_water = models.DecimalField(max_digits=12, decimal_places=2)
    education_and_misc = models.DecimalField(max_digits=12, decimal_places=2)
    other_expense = models.DecimalField(max_digits=12, decimal_places=2)

class Vehicle(models.Model):
    TRANSMISSION_CHOICES = [('Automatic', 'Automatic'), ('Manual', 'Manual')]
    FUEL_CHOICES = [('Gas', 'Gas'), ('Diesel', 'Diesel')]
    
    loan = models.OneToOneField(Borrower, on_delete=models.CASCADE, related_name='vehicle')
    make_brand = models.CharField(max_length=100)
    series = models.CharField(max_length=100)
    year_model = models.IntegerField()
    variant = models.CharField(max_length=50)
    color = models.CharField(max_length=50)
    plate_no = models.CharField(max_length=20, blank=True)
    engine_no = models.CharField(max_length=50)
    chassis_no = models.CharField(max_length=50)
    transmission = models.CharField(max_length=20, choices=TRANSMISSION_CHOICES)
    fuel = models.CharField(max_length=20, choices=FUEL_CHOICES)
    dealer_name = models.CharField(max_length=200)
    dealer_address = models.TextField()
    dealer_contact_number = models.CharField(max_length=20)

class LoanDetails(models.Model):
    LOAN_TYPE_CHOICES = [('New', 'New'), ('Renewal', 'Renewal')]
    LOAN_PURPOSE_CHOICES = [
        ('Personal', 'For Personal Use'),
        ('Business', 'For Business Use')
    ]
    DOWN_PAYMENT_CHOICES = [
        (Decimal('0.20'), '20%'),
        (Decimal('0.30'), '30%'),
        (Decimal('0.40'), '40%')
    ]
    LOAN_TERM_CHOICES = [
        (12, '12 months'),
        (18, '18 months'),
        (24, '24 months'),
        (30, '30 months'),
        (36, '36 months'),
    ]
    
    loan = models.OneToOneField(Borrower, on_delete=models.CASCADE, related_name='loan_details')
    loan_type = models.CharField(max_length=10, choices=LOAN_TYPE_CHOICES)
    estimated_vehicle_value = models.DecimalField(max_digits=12, decimal_places=2)
    down_payment_percentage = models.DecimalField(max_digits=4, decimal_places=2, choices=DOWN_PAYMENT_CHOICES)
    loan_amount_term = models.IntegerField(choices=LOAN_TERM_CHOICES)
    loan_purpose = models.CharField(max_length=20, choices=LOAN_PURPOSE_CHOICES)
    loan_amount_applied = models.DecimalField(max_digits=12, decimal_places=2)
    monthly_amortization = models.DecimalField(max_digits=12, decimal_places=2)

    def save(self, *args, **kwargs):
        try:
            # Ensure all values are valid Decimals
            estimated_value = Decimal(str(self.estimated_vehicle_value))
            down_payment_pct = Decimal(str(self.down_payment_percentage))
            loan_term = Decimal(str(self.loan_amount_term))
            
            # Get the current active interest rate
            from .models import InterestRate
            interest_rate = InterestRate.get_active_rate()
            
            # Calculate loan amount applied (Principal)
            down_payment_amount = estimated_value * down_payment_pct
            self.loan_amount_applied = estimated_value - down_payment_amount
            
            # Calculate monthly amortization using the formula:
            # (((Loan Amount × Interest Rate) × Loan Term) + Loan Amount) / Loan Term
            annual_interest_rate = Decimal(str(interest_rate)) / Decimal('100')
            total_interest = (self.loan_amount_applied * annual_interest_rate * loan_term)
            total_amount = total_interest + self.loan_amount_applied
            self.monthly_amortization = total_amount / loan_term
            
            # Round to 2 decimal places
            self.monthly_amortization = self.monthly_amortization.quantize(Decimal('0.01'), rounding='ROUND_HALF_UP')
            
        except (InvalidOperation, DivisionByZero, ValueError) as e:
            # Handle any decimal errors
            self.loan_amount_applied = Decimal('0')
            self.monthly_amortization = Decimal('0')
            
        super().save(*args, **kwargs)
        
    def __str__(self):
        return f"Loan {self.loan.loan_id} - {self.loan_type} - {self.loan_amount_applied}"

class RequiredDocument(models.Model):
    loan = models.OneToOneField(Borrower, on_delete=models.CASCADE, related_name='documents')
    valid_id = models.FileField(upload_to='documents/valid_ids/')
    proof_of_income = models.FileField(upload_to='documents/proof_of_income/')
    utility_bill = models.FileField(upload_to='documents/utility_bills/')
    
    def __str__(self):
        return f"Documents for Loan {self.loan.loan_id}"

class Marketing(models.Model):
    MARKETING_SOURCE_CHOICES = [
        ('Walk-In', 'Walk-In'),
        ('Website', 'Website'),
        ('Social Media', 'Social Media'),
        ('Thru Agent', 'Through Agent'),
    ]
    
    loan = models.OneToOneField(Borrower, on_delete=models.CASCADE, related_name='marketing')
    marketing_source = models.CharField(max_length=20, choices=MARKETING_SOURCE_CHOICES)
    sales_representative = models.CharField(max_length=100, blank=True)
    
    def __str__(self):
        return f"{self.loan.loan_id} - {self.marketing_source}"

class MonthlyCashFlow(models.Model):
    loan = models.OneToOneField(Borrower, on_delete=models.CASCADE, related_name='cash_flow')
    applicant_total_income = models.DecimalField(max_digits=12, decimal_places=2)
    spouse_total_income = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    total_income = models.DecimalField(max_digits=12, decimal_places=2)
    total_expenses = models.DecimalField(max_digits=12, decimal_places=2)
    net_disposal = models.DecimalField(max_digits=12, decimal_places=2)

    def save(self, *args, **kwargs):
        try:
            # Get employment data if available
            if hasattr(self.loan, 'employment'):
                self.applicant_total_income = self.loan.employment.monthly_net_income + self.loan.employment.other_monthly_income
            else:
                self.applicant_total_income = Decimal('0')
            
            # Get spouse data if available
            if hasattr(self.loan, 'spouse'):
                self.spouse_total_income = self.loan.spouse.net_income + self.loan.spouse.other_monthly_income
            else:
                self.spouse_total_income = Decimal('0')
            
            # Calculate total income
            self.total_income = self.applicant_total_income + self.spouse_total_income
            
            # Get expense data if available
            if hasattr(self.loan, 'expenses'):
                expenses = self.loan.expenses
                self.total_expenses = (
                    expenses.food_and_groceries + 
                    expenses.electric_and_water + 
                    expenses.education_and_misc + 
                    expenses.other_expense
                )
            else:
                self.total_expenses = Decimal('0')
            
            # Calculate net disposal (70% of disposable income)
            disposable_income = self.total_income - self.total_expenses
            self.net_disposal = disposable_income * Decimal('0.7')
            
            # Round to 2 decimal places
            self.applicant_total_income = self.applicant_total_income.quantize(Decimal('0.01'), rounding='ROUND_HALF_UP')
            self.spouse_total_income = self.spouse_total_income.quantize(Decimal('0.01'), rounding='ROUND_HALF_UP')
            self.total_income = self.total_income.quantize(Decimal('0.01'), rounding='ROUND_HALF_UP')
            self.total_expenses = self.total_expenses.quantize(Decimal('0.01'), rounding='ROUND_HALF_UP')
            self.net_disposal = self.net_disposal.quantize(Decimal('0.01'), rounding='ROUND_HALF_UP')
            
        except (InvalidOperation, DivisionByZero, ValueError, AttributeError) as e:
            # Handle any errors
            print(f"Error calculating cash flow: {e}")
        
        super().save(*args, **kwargs)
    
    def __str__(self):
        return f"Cash Flow for Loan {self.loan.loan_id}"

class LoanStatus(models.Model):
    loan = models.OneToOneField(Borrower, on_delete=models.CASCADE, related_name='status')
    remarks = models.TextField(blank=True)
    status = models.CharField(max_length=50, choices=Borrower.LOAN_STATUS_CHOICES, default='PENDING')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        if hasattr(self.loan, 'personal_info'):
            return f"{self.loan.personal_info.last_name}, {self.loan.personal_info.first_name} - {self.status}"
        return f"Loan {self.loan.loan_id} - {self.status}"

    class Meta:
        ordering = ['-updated_at']
        verbose_name_plural = 'Loan Statuses'

class MarketingOfficerRemarks(models.Model):
    DOCUMENT_COMPLETE_CHOICES = [
        ('YES', 'Yes'),
        ('NO', 'No')
    ]
    
    loan = models.OneToOneField(Borrower, on_delete=models.CASCADE, related_name='marketing_officer_remarks')
    marketing_officer_name = models.CharField(max_length=100)
    complete_documents = models.CharField(max_length=3, choices=DOCUMENT_COMPLETE_CHOICES, default='NO')
    remarks = models.TextField()
    
    def __str__(self):
        return f"Marketing Officer Remarks for Loan {self.loan.loan_id}"
    
    def save(self, *args, **kwargs):
        # When saving, also update the loan status if needed
        super().save(*args, **kwargs)
        
        # If documents are not complete, set status to HOLD
        if self.complete_documents == 'NO' and hasattr(self.loan, 'status'):
            status = self.loan.status
            if status.status != 'HOLD':
                status.status = 'HOLD'
                status.remarks = f"Documents incomplete. {self.remarks}"
                status.save()
        
        # If documents are complete, allow proceeding to Credit Investigator
        elif self.complete_documents == 'YES' and hasattr(self.loan, 'status'):
            status = self.loan.status
            if status.status == 'PENDING':
                status.status = 'PROCEED_CI'
                status.remarks = f"Documents complete. {self.remarks}"
                status.save()

class CreditInvestigatorRemarks(models.Model):
    VERIFICATION_CHOICES = [
        ('YES', 'Yes'),
        ('NO', 'No')
    ]
    
    SUSPICIOUS_INDICATOR_CHOICES = [
        (0, 'None'),
        (1, 'Suspicious Transaction')
    ]
    
    CREDIT_RISK_CHOICES = [
        ('LOW', 'Low Risk'),
        ('MEDIUM', 'Medium Risk'),
        ('HIGH', 'High Risk')
    ]
    
    STATUS_CHOICES = [
        ('PROCEED_LAO', 'Proceed to Loan Approval Officer'),
        ('HOLD', 'Hold'),
        ('DECLINED', 'Declined')
    ]
    
    loan = models.OneToOneField(Borrower, on_delete=models.CASCADE, related_name='credit_investigator_remarks')
    credit_investigator_name = models.CharField(max_length=100)
    verified = models.CharField(max_length=3, choices=VERIFICATION_CHOICES, default='NO')
    suspicious_indicator = models.IntegerField(choices=SUSPICIOUS_INDICATOR_CHOICES, default=0)
    credit_risk_assessment = models.CharField(max_length=6, choices=CREDIT_RISK_CHOICES)
    remarks = models.TextField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='HOLD')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Credit Investigator Remarks for Loan {self.loan.loan_id}"
    
    def save(self, *args, **kwargs):
        # Automatically determine credit risk based on various factors
        self._calculate_credit_risk()
        
        # When saving, also update the loan status if needed
        super().save(*args, **kwargs)
        
        # Update the loan status based on the credit investigator's assessment
        if hasattr(self.loan, 'status'):
            status = self.loan.status
            status.status = self.status
            
            if self.status == 'PROCEED_LAO':
                status.remarks = f"Verified by {self.credit_investigator_name}. {self.remarks}"
            elif self.status == 'HOLD':
                status.remarks = f"On hold by Credit Investigator. {self.remarks}"
            elif self.status == 'DECLINED':
                status.remarks = f"Declined by Credit Investigator. {self.remarks}"
                
            status.save()
    
    def _calculate_credit_risk(self):
        """Calculate credit risk based on various factors using a point system"""
        total_points = 0
        
        # 1. Points for Residency and Citizenship (0-10 points)
        if hasattr(self.loan, 'personal_info'):
            residency = self.loan.personal_info.residency_and_citizenship
            if residency == 'RFC':  # Resident Filipino Citizen
                total_points += 10
            elif residency == 'RA':  # Resident Alien
                total_points += 5
            elif residency == 'NRCA':  # Non-Resident Citizen/Alien
                total_points += 0
        
        # 2. Points for Source of Fund (0-10 points)
        if hasattr(self.loan, 'employment'):
            source_of_funds = self.loan.employment.source_of_funds
            if source_of_funds in ['Salary', 'Income from Business']:
                total_points += 10
            elif source_of_funds == 'Support From Relative':
                total_points += 5
            elif source_of_funds in ['Commissions', 'Remittance']:
                total_points += 0
        
        # 3. Points for Employment Status (0-10 points)
        if hasattr(self.loan, 'employment'):
            employment_status = self.loan.employment.employment_status
            if employment_status in ['Employee', 'Registered Business', 'Retired']:
                total_points += 10
            elif employment_status in ['Unemployed', 'Student']:
                total_points += 5
            elif employment_status in ['Lawyer', 'Accountant', 'Gaming', 'Unemployed_IND']:
                total_points += 0
        
        # 4. Points for Suspicious Indicator (0-10 points)
        if self.suspicious_indicator == 0:  # None
            total_points += 10
        else:  # Suspicious Transaction
            total_points += 0
        
        # Determine credit risk assessment based on total points (0-40)
        if 0 <= total_points <= 13:
            self.credit_risk_assessment = 'HIGH'
        elif 14 <= total_points <= 26:
            self.credit_risk_assessment = 'MEDIUM'
        elif 27 <= total_points <= 40:
            self.credit_risk_assessment = 'LOW'
        else:
            # Fallback in case of unexpected point total
            self.credit_risk_assessment = 'HIGH'
        
        return self.credit_risk_assessment

class LoanApprovalOfficerRemarks(models.Model):
    APPROVAL_CHOICES = [
        ('APPROVED', 'Approved'),
        ('DECLINED', 'Declined')
    ]
    
    loan = models.OneToOneField(Borrower, on_delete=models.CASCADE, related_name='loan_approval_officer_remarks')
    loan_approval_officer_name = models.CharField(max_length=100)
    approval_status = models.CharField(max_length=10, choices=APPROVAL_CHOICES, blank=True, null=True)
    remarks = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.loan.loan_id} - {self.get_approval_status_display()}"
    
    def save(self, *args, **kwargs):
        # Update the loan status based on approval decision
        super().save(*args, **kwargs)
        
        # Update the loan status
        try:
            loan_status = LoanStatus.objects.get(loan=self.loan)
            if self.approval_status == 'APPROVED':
                loan_status.status = 'PROCEED_LDO'
            elif self.approval_status == 'DECLINED':
                loan_status.status = 'DECLINED'
            else:
                # If approval_status is blank/null, keep the loan in LAO stage
                loan_status.status = 'PROCEED_LAO'
            loan_status.remarks = f"Loan Approval Officer: {self.remarks}"
            loan_status.save()
        except LoanStatus.DoesNotExist:
            if self.approval_status == 'APPROVED':
                status = 'PROCEED_LDO'
            elif self.approval_status == 'DECLINED':
                status = 'DECLINED'
            else:
                status = 'PROCEED_LAO'
            LoanStatus.objects.create(
                loan=self.loan,
                status=status,
                remarks=f"Loan Approval Officer: {self.remarks}"
            )

class LoanDisbursementOfficerRemarks(models.Model):
    STATUS_CHOICES = [
        ('COMPLETED', 'Completed'),
        ('HOLD', 'Hold'),
        ('CANCELLED', 'Cancelled')
    ]
    
    loan = models.OneToOneField(Borrower, on_delete=models.CASCADE, related_name='loan_disbursement_officer_remarks')
    loan_disbursement_officer_name = models.CharField(max_length=100)
    disbursement_date = models.DateField()
    loan_due_date = models.DateField()
    remarks = models.TextField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='HOLD')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.loan.loan_id} - {self.get_status_display()}"
    
    def save(self, *args, **kwargs):
        # Update the loan status based on disbursement decision
        super().save(*args, **kwargs)
        
        # Update the loan status
        try:
            loan_status = LoanStatus.objects.get(loan=self.loan)
            if self.status == 'COMPLETED':
                loan_status.status = 'COMPLETED'
            elif self.status == 'CANCELLED':
                loan_status.status = 'CANCELLED'
            else:  # HOLD
                loan_status.status = 'PROCEED_LDO'
            loan_status.remarks = f"Loan Disbursement Officer: {self.remarks}"
            loan_status.save()
        except LoanStatus.DoesNotExist:
            LoanStatus.objects.create(
                loan=self.loan,
                status='PROCEED_LDO' if self.status == 'HOLD' else self.status,
                remarks=f"Loan Disbursement Officer: {self.remarks}"
            )

class DailyLoanDisbursement(models.Model):
    date = models.DateField(unique=True)
    total_amount = models.DecimalField(max_digits=15, decimal_places=2, default=0)
    disbursement_percentage = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    
    def __str__(self):
        return f"Daily Disbursement for {self.date}"
    
    class Meta:
        ordering = ['-date']

class WeeklyLoanDisbursement(models.Model):
    WEEK_CHOICES = [
        (1, 'Week 1'),
        (2, 'Week 2'),
        (3, 'Week 3'),
        (4, 'Week 4'),
        (5, 'Week 5'),
    ]
    
    year = models.IntegerField()
    month = models.IntegerField()
    week = models.IntegerField(choices=WEEK_CHOICES)
    start_date = models.DateField()
    end_date = models.DateField()
    total_amount = models.DecimalField(max_digits=15, decimal_places=2, default=0)
    disbursement_percentage = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    
    def __str__(self):
        return f"Week {self.week} ({self.start_date} to {self.end_date})"
    
    class Meta:
        ordering = ['-year', '-month', '-week']
        unique_together = ['year', 'month', 'week']

class MonthlyLoanDisbursement(models.Model):
    year = models.IntegerField()
    month = models.IntegerField()
    total_amount = models.DecimalField(max_digits=15, decimal_places=2, default=0)
    disbursement_percentage = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    monthly_quota = models.DecimalField(max_digits=15, decimal_places=2, default=0)
    quota_met = models.BooleanField(default=False)
    
    def __str__(self):
        return f"{self.get_month_name()} {self.year}"
    
    def get_month_name(self):
        import calendar
        return calendar.month_name[self.month]
    
    class Meta:
        ordering = ['-year', '-month']
        unique_together = ['year', 'month']

class YearlyLoanDisbursement(models.Model):
    year = models.IntegerField(unique=True)
    total_amount = models.DecimalField(max_digits=15, decimal_places=2, default=0)
    disbursement_percentage = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    yearly_quota = models.DecimalField(max_digits=15, decimal_places=2, default=0)
    quota_met = models.BooleanField(default=False)
    
    def __str__(self):
        return f"Year {self.year}"
    
    class Meta:
        ordering = ['-year']

# System Administrator related models
class InterestRate(models.Model):
    rate = models.DecimalField(max_digits=5, decimal_places=2, default=Decimal('10.00'), help_text="Interest rate percentage (e.g., 10.00 for 10%)")
    effective_date = models.DateField(auto_now_add=True)
    is_active = models.BooleanField(default=True)
    created_by = models.ForeignKey('auth.User', on_delete=models.SET_NULL, null=True, related_name='interest_rates')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.rate}% (from {self.effective_date})"
    
    class Meta:
        ordering = ['-effective_date']
        
    def save(self, *args, **kwargs):
        # Delete all other instances if they exist
        if not self.pk:
            InterestRate.objects.all().delete()
        super().save(*args, **kwargs)
    
    @classmethod
    def get_active_rate(cls):
        """Get the current active interest rate"""
        active_rate = cls.objects.first()
        if active_rate:
            return active_rate.rate
        return Decimal('10.00')  # Default interest rate if none is set

class QuotaSettings(models.Model):
    """
    Stores the monthly loan quota settings.
    This model only ever has one instance.
    """
    monthly_quota = models.DecimalField(max_digits=12, decimal_places=2)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Monthly Quota: {self.monthly_quota}"
    
    class Meta:
        verbose_name = "Quota Setting"
        verbose_name_plural = "Quota Settings"
        
    def save(self, *args, **kwargs):
        # Ensure only one instance exists
        if not self.pk and QuotaSettings.objects.exists():
            # Update existing instance instead of creating a new one
            existing = QuotaSettings.objects.first()
            existing.monthly_quota = self.monthly_quota
            existing.save()
            return existing
        return super().save(*args, **kwargs)
    
    @classmethod
    def get_monthly_quota(cls):
        """Get the current monthly quota"""
        quota = cls.objects.first()
        if quota:
            return quota.monthly_quota
        # Create default if none exists
        quota = cls.objects.create(monthly_quota=Decimal('10000000.00'))
        return quota.monthly_quota
        
class UserAccount(models.Model):
    POSITION_CHOICES = [
        ('MARKETING', 'Marketing Officer'),
        ('CREDIT', 'Credit Investigator'),
        ('APPROVAL', 'Loan Approval Officer'),
        ('DISBURSEMENT', 'Loan Disbursement Officer'),
        ('AREA', 'Area Manager'),
        ('ADMIN', 'System Administrator'),
    ]
    
    user = models.OneToOneField('auth.User', on_delete=models.CASCADE, related_name='user_account')
    address = models.TextField()
    contact_number = models.CharField(max_length=20)
    position = models.CharField(max_length=20, choices=POSITION_CHOICES)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.user.username} - {self.get_position_display()}"
