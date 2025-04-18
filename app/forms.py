from django import forms
from .models import *
from datetime import date
from django.contrib.auth.models import User

class PersonalInformationForm(forms.ModelForm):
    class Meta:
        model = PersonalInformation
        exclude = ['loan', 'age']
        widgets = {
            'first_name': forms.TextInput(attrs={'class': 'form-control'}),
            'middle_name': forms.TextInput(attrs={'class': 'form-control'}),
            'last_name': forms.TextInput(attrs={'class': 'form-control'}),
            'gender': forms.Select(attrs={'class': 'form-control'}),
            'date_of_birth': forms.DateInput(attrs={
                'class': 'form-control',
                'type': 'date',
                'max': date.today().isoformat()
            }),
            'civil_status': forms.Select(attrs={'class': 'form-control'}),
            'religion': forms.TextInput(attrs={'class': 'form-control'}),
            'sss_number': forms.TextInput(attrs={'class': 'form-control'}),
            'tin_number': forms.TextInput(attrs={'class': 'form-control'}),
            'property_area': forms.Select(attrs={'class': 'form-control'}),
            'residency_and_citizenship': forms.Select(attrs={'class': 'form-control'}),
        }

class ContactAddressForm(forms.ModelForm):
    class Meta:
        model = ContactAndAddress
        exclude = ['loan']
        widgets = {
            'contact_number': forms.TextInput(attrs={'class': 'form-control'}),
            'email_address': forms.EmailInput(attrs={'class': 'form-control'}),
            'facebook_account': forms.TextInput(attrs={'class': 'form-control'}),
            'no_and_street': forms.TextInput(attrs={'class': 'form-control'}),
            'barangay': forms.TextInput(attrs={'class': 'form-control'}),
            'municipality': forms.TextInput(attrs={'class': 'form-control'}),
            'province': forms.TextInput(attrs={'class': 'form-control'}),
            'resident_status': forms.Select(attrs={'class': 'form-control'}),
            'length_of_stay': forms.Select(attrs={'class': 'form-control'}),
            'permanent_address': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'provincial_address': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'parents_address': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
        }

class EducationForm(forms.ModelForm):
    class Meta:
        model = Education
        exclude = ['loan']
        widgets = {
            'education': forms.Select(attrs={'class': 'form-control'}),
            'course': forms.TextInput(attrs={'class': 'form-control'}),
            'school_last_attended': forms.TextInput(attrs={'class': 'form-control'}),
            'year_graduated': forms.NumberInput(attrs={'class': 'form-control'}),
        }

class DependentForm(forms.ModelForm):
    class Meta:
        model = Dependent
        exclude = ['loan']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'age': forms.NumberInput(attrs={'class': 'form-control', 'min': '0'}),
            'school': forms.TextInput(attrs={'class': 'form-control'}),
            'relation': forms.TextInput(attrs={'class': 'form-control'}),
            'self_employed': forms.Select(attrs={'class': 'form-control'}),
        }

class SpouseInformationForm(forms.ModelForm):
    class Meta:
        model = SpouseInformation
        exclude = ['loan']
        widgets = {
            'first_name': forms.TextInput(attrs={'class': 'form-control'}),
            'middle_name': forms.TextInput(attrs={'class': 'form-control'}),
            'last_name': forms.TextInput(attrs={'class': 'form-control'}),
            'relation_to_borrower': forms.TextInput(attrs={'class': 'form-control'}),
            'civil_status': forms.Select(attrs={'class': 'form-control'}),
            'date_of_birth': forms.DateInput(attrs={
                'class': 'form-control',
                'type': 'date',
                'max': date.today().isoformat()
            }),
            'education': forms.Select(attrs={'class': 'form-control'}),
            'net_income': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
            'employer_business_name': forms.TextInput(attrs={'class': 'form-control'}),
            'employer_contact_number': forms.TextInput(attrs={'class': 'form-control'}),
            'other_income': forms.TextInput(attrs={'class': 'form-control'}),
            'other_monthly_income': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
        }

class EmploymentForm(forms.ModelForm):
    class Meta:
        model = Employment
        exclude = ['loan']
        widgets = {
            'source_of_funds': forms.Select(attrs={'class': 'form-control'}),
            'employer_business_name': forms.TextInput(attrs={'class': 'form-control'}),
            'employer_contact_number': forms.TextInput(attrs={'class': 'form-control'}),
            'position': forms.TextInput(attrs={'class': 'form-control'}),
            'employment_status': forms.Select(attrs={
                'class': 'form-control',
                'onchange': 'handleEmploymentStatusChange(this)'
            }),
            'monthly_net_income': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
            'employer_business_address': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'other_income': forms.TextInput(attrs={'class': 'form-control'}),
            'other_monthly_income': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
        }

class ExpenseForm(forms.ModelForm):
    class Meta:
        model = Expense
        exclude = ['loan']
        widgets = {
            'food_and_groceries': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
            'electric_and_water': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
            'education_and_misc': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
            'other_expense': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
        }

class VehicleForm(forms.ModelForm):
    class Meta:
        model = Vehicle
        exclude = ['loan']
        widgets = {
            'make_brand': forms.TextInput(attrs={'class': 'form-control'}),
            'series': forms.TextInput(attrs={'class': 'form-control'}),
            'year_model': forms.NumberInput(attrs={'class': 'form-control'}),
            'variant': forms.TextInput(attrs={'class': 'form-control'}),
            'color': forms.TextInput(attrs={'class': 'form-control'}),
            'plate_no': forms.TextInput(attrs={'class': 'form-control'}),
            'engine_no': forms.TextInput(attrs={'class': 'form-control'}),
            'chassis_no': forms.TextInput(attrs={'class': 'form-control'}),
            'transmission': forms.Select(attrs={'class': 'form-control'}),
            'fuel': forms.Select(attrs={'class': 'form-control'}),
            'dealer_name': forms.TextInput(attrs={'class': 'form-control'}),
            'dealer_address': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'dealer_contact_number': forms.TextInput(attrs={'class': 'form-control'}),
        }

class LoanDetailsForm(forms.ModelForm):
    class Meta:
        model = LoanDetails
        exclude = ['loan', 'loan_amount_applied', 'monthly_amortization']
        widgets = {
            'loan_type': forms.Select(attrs={'class': 'form-control'}),
            'estimated_vehicle_value': forms.NumberInput(attrs={
                'class': 'form-control',
                'step': '1',
                'min': '1'
            }),
            'down_payment_percentage': forms.Select(attrs={'class': 'form-control'}),
            'loan_amount_term': forms.Select(attrs={'class': 'form-control'}),
            'loan_purpose': forms.Select(attrs={'class': 'form-control'})
        }

    def clean_estimated_vehicle_value(self):
        value = self.cleaned_data['estimated_vehicle_value']
        if value <= 0:
            raise forms.ValidationError("Vehicle value must be greater than 0")
        return value

class RequiredDocumentForm(forms.ModelForm):
    class Meta:
        model = RequiredDocument
        exclude = ['loan']
        widgets = {
            'valid_id': forms.FileInput(attrs={'class': 'form-control'}),
            'proof_of_income': forms.FileInput(attrs={'class': 'form-control'}),
            'utility_bill': forms.FileInput(attrs={'class': 'form-control'}),
        }

class MarketingForm(forms.ModelForm):
    class Meta:
        model = Marketing
        exclude = ['loan']
        widgets = {
            'marketing_source': forms.Select(attrs={'class': 'form-control'}),
            'sales_representative': forms.TextInput(attrs={'class': 'form-control'}),
        }

class MarketingOfficerRemarksForm(forms.ModelForm):
    class Meta:
        model = MarketingOfficerRemarks
        exclude = ['loan', 'marketing_officer_name']
        widgets = {
            'complete_documents': forms.Select(attrs={'class': 'form-control'}),
            'remarks': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
        }

class CreditInvestigatorRemarksForm(forms.ModelForm):
    class Meta:
        model = CreditInvestigatorRemarks
        exclude = ['loan', 'credit_investigator_name', 'credit_risk_assessment', 'created_at', 'updated_at']
        widgets = {
            'verified': forms.Select(attrs={'class': 'form-control'}),
            'suspicious_indicator': forms.Select(attrs={'class': 'form-control'}),
            'remarks': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'status': forms.Select(attrs={'class': 'form-control'}),
        }

class LoanApprovalOfficerRemarksForm(forms.ModelForm):
    class Meta:
        model = LoanApprovalOfficerRemarks
        exclude = ['loan', 'loan_approval_officer_name', 'created_at', 'updated_at']
        widgets = {
            'approval_status': forms.Select(attrs={'class': 'form-control'}),
            'remarks': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
        }

class LoanDisbursementOfficerRemarksForm(forms.ModelForm):
    class Meta:
        model = LoanDisbursementOfficerRemarks
        exclude = ['loan', 'loan_disbursement_officer_name', 'created_at', 'updated_at']
        widgets = {
            'disbursement_date': forms.DateInput(attrs={'class': 'form-control', 'type': 'date'}),
            'maturity_date': forms.DateInput(attrs={'class': 'form-control', 'type': 'date'}),
            'remarks': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'status': forms.Select(attrs={'class': 'form-control'}),
        }

# System Administrator Forms
class InterestRateForm(forms.ModelForm):
    class Meta:
        model = InterestRate
        fields = ['rate']
        widgets = {
            'rate': forms.NumberInput(attrs={
                'class': 'form-control', 
                'step': '0.01', 
                'min': '0.01', 
                'max': '100.00',
                'placeholder': 'e.g. 10.00 for 10%'
            }),
        }
        labels = {
            'rate': 'Interest Rate (%)',
        }
        help_texts = {
            'rate': 'Enter the interest rate percentage (e.g., 10.00 for 10%).',
        }

class MonthlyQuotaForm(forms.Form):
    monthly_quota = forms.DecimalField(
        max_digits=12, 
        decimal_places=2, 
        min_value=1,
        widget=forms.NumberInput(attrs={
            'class': 'form-control', 
            'step': '0.01', 
            'min': '1000.00',
            'placeholder': 'e.g. 30000.00'
        })
    )
    help_text = forms.CharField(
        widget=forms.HiddenInput(),
        required=False,
        initial="Enter the monthly quota amount for loan disbursements."
    )

class UserAccountCreationForm(forms.ModelForm):
    username = forms.CharField(max_length=150, widget=forms.TextInput(attrs={'class': 'form-control'}))
    first_name = forms.CharField(max_length=150, widget=forms.TextInput(attrs={'class': 'form-control'}))
    last_name = forms.CharField(max_length=150, widget=forms.TextInput(attrs={'class': 'form-control'}))
    email = forms.EmailField(widget=forms.EmailInput(attrs={'class': 'form-control'}))
    password1 = forms.CharField(label='Password', widget=forms.PasswordInput(attrs={'class': 'form-control'}))
    password2 = forms.CharField(label='Confirm Password', widget=forms.PasswordInput(attrs={'class': 'form-control'}))
    
    class Meta:
        model = UserAccount
        fields = ['position', 'address', 'contact_number']
        widgets = {
            'position': forms.Select(attrs={'class': 'form-control'}),
            'address': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'contact_number': forms.TextInput(attrs={'class': 'form-control'}),
        }
    
    def clean_username(self):
        username = self.cleaned_data['username']
        if User.objects.filter(username=username).exists():
            raise forms.ValidationError('Username already exists')
        return username
    
    def clean_email(self):
        email = self.cleaned_data['email']
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError('Email already exists')
        return email
    
    def clean(self):
        cleaned_data = super().clean()
        password1 = cleaned_data.get('password1')
        password2 = cleaned_data.get('password2')
        
        if password1 and password2 and password1 != password2:
            self.add_error('password2', 'Passwords do not match')
        
        return cleaned_data

class UserAccountUpdateForm(forms.ModelForm):
    first_name = forms.CharField(max_length=150, widget=forms.TextInput(attrs={'class': 'form-control'}))
    last_name = forms.CharField(max_length=150, widget=forms.TextInput(attrs={'class': 'form-control'}))
    email = forms.EmailField(widget=forms.EmailInput(attrs={'class': 'form-control'}))
    
    class Meta:
        model = UserAccount
        fields = ['position', 'address', 'contact_number']
        widgets = {
            'position': forms.Select(attrs={'class': 'form-control'}),
            'address': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'contact_number': forms.TextInput(attrs={'class': 'form-control'}),
        } 