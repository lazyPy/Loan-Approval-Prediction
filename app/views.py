from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import JsonResponse
from django.db import transaction
from django.db.models import Q, Sum
from django.utils import timezone
from datetime import timedelta
import json
import os
import re
from .forms import *
from .models import *
from django.contrib.auth.models import User
import pandas as pd
import numpy as np
# Add matplotlib Agg backend configuration
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend to avoid GUI errors
import joblib
from django.db import models
from django.contrib.auth.decorators import login_required
from django.core.cache import cache
from django.conf import settings
from django.contrib.auth import authenticate, login, logout
from functools import wraps
from .forecasting import get_forecast_data

def role_required(allowed_roles):
    def decorator(view_func):
        @wraps(view_func)
        def _wrapped_view(request, *args, **kwargs):
            if not request.user.is_authenticated:
                messages.error(request, "Please login to access this page.")
                return redirect('login')
            
            # Superuser can access everything
            if request.user.is_superuser:
                return view_func(request, *args, **kwargs)
            
            # Check if user has a user_account
            if not hasattr(request.user, 'user_account'):
                messages.error(request, "You don't have permission to access this page.")
                return redirect('home')
            
            # Check if user's role is in allowed roles
            if request.user.user_account.position not in allowed_roles:
                messages.error(request, "You don't have permission to access this page.")
                return redirect('home')
            
            return view_func(request, *args, **kwargs)
        return _wrapped_view
    return decorator

def loan_application_step(request, step=1):
    
    FORMS = {
        1: {'form': PersonalInformationForm, 'template': 'personal_info.html', 'title': 'Personal Information'},
        2: {'form': ContactAddressForm, 'template': 'contact_address.html', 'title': 'Contact & Address Information'},
        3: {'form': EducationForm, 'template': 'education.html', 'title': 'Educational Background'},
        4: {'form': DependentForm, 'template': 'dependents.html', 'title': 'Dependents Information'},
        5: {'form': EmploymentForm, 'template': 'employment.html', 'title': 'Source of Income'},
        6: {'form': SpouseInformationForm, 'template': 'spouse.html', 'title': 'Spouse/Co-Borrower Information'},
        7: {'form': ExpenseForm, 'template': 'expenses.html', 'title': 'Monthly Expenses'},
        8: {'form': VehicleForm, 'template': 'vehicle.html', 'title': 'Vehicle Information'},
        9: {'form': LoanDetailsForm, 'template': 'loan_details.html', 'title': 'Loan Details'},
        10: {'form': RequiredDocumentForm, 'template': 'documents.html', 'title': 'Required Documents'},
        11: {'form': MarketingForm, 'template': 'marketing.html', 'title': 'Marketing Information'},
    }
    
    # Set a session flag to indicate privacy notice acknowledgment
    request.session['privacy_notice_acknowledged'] = True
    
    # Get loan application identifier from session, but don't create anything yet
    loan_id = request.session.get('loan_id')
    
    form_class = FORMS[step]['form']
    form = form_class()
    
    # Check if this is an AJAX request to save form data
    if request.method == 'POST' and request.headers.get('Content-Type') == 'application/json':
        form_data = json.loads(request.body)
        # Store form data in session for navigation purposes
        session_forms = request.session.get('loan_form_data', {})
        session_forms[str(step)] = form_data
        request.session['loan_form_data'] = session_forms
        return JsonResponse({'status': 'success'})
    
    if request.method == 'POST':
        form = form_class(request.POST, request.FILES)
        
        # Special handling for spouse step (step 6) - allow it to be empty
        if step == 6:
            try:
                # Check if all fields are empty (user is skipping spouse info)
                all_empty = True
                for field_name, field_value in request.POST.items():
                    # Skip CSRF token and other non-form fields
                    if field_name.startswith('csrf') or field_name in ['csrfmiddlewaretoken', 'next']:
                        continue
                    
                    # If any field has a value, not all fields are empty
                    if field_value:
                        all_empty = False
                        break
                
                # If all fields are empty, consider the form valid but store it as empty
                if all_empty:
                    # Store empty form data in session
                    form_data = request.POST.dict()
                    session_forms = request.session.get('loan_form_data', {})
                    session_forms[str(step)] = form_data
                    request.session['loan_form_data'] = session_forms
                    
                    # Redirect to next step
                    if step < len(FORMS):
                        return redirect('loan_application_step', step=step + 1)
            except Exception as e:
                # If there's any error, log it and continue with regular form validation
                print(f"Error processing spouse form: {e}")
                # Continue with regular form validation
        
        # Special handling for dependents step (step 4) - allow it to be empty or with zero dependents
        if step == 4:
            # Check for empty dependents_data or zero dependents
            dependents_data = request.POST.get('dependents_data', '[]')
            try:
                dependents_list = json.loads(dependents_data)
                if len(dependents_list) == 0:
                    # Store form data in session
                    form_data = request.POST.dict()
                    session_forms = request.session.get('loan_form_data', {})
                    session_forms[str(step)] = form_data
                    request.session['loan_form_data'] = session_forms
                    
                    # Redirect to next step
                    if step < len(FORMS):
                        return redirect('loan_application_step', step=step + 1)
            except json.JSONDecodeError:
                # If there's an error parsing, just continue with regular form validation
                print(f"Error parsing dependents data: Invalid JSON format")
                # Continue with regular form validation
        
        if form.is_valid():
            # Store form data in session
            form_data = request.POST.dict()
            
            # Handle file uploads for documents step
            if step == 10 and request.FILES:
                # Create a unique temporary directory for this session if it doesn't exist
                session_id = request.session.session_key
                if not session_id:
                    request.session.create()
                    session_id = request.session.session_key
                
                temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp', session_id)
                os.makedirs(temp_dir, exist_ok=True)
                
                # Store file paths in session
                files_data = {}
                for field_name, uploaded_file in request.FILES.items():
                    # Create a safe filename
                    filename = f"{field_name}_{uploaded_file.name}"
                    filepath = os.path.join(temp_dir, filename)
                    
                    # Save the file
                    with open(filepath, 'wb+') as destination:
                        for chunk in uploaded_file.chunks():
                            destination.write(chunk)
                    
                    # Store the relative path
                    files_data[field_name] = os.path.join('temp', session_id, filename)
                
                # Store file paths in session
                request.session[f'step_{step}_files'] = files_data
            
            # Create or update the form data dictionary in session
            session_forms = request.session.get('loan_form_data', {})
            session_forms[str(step)] = form_data
            request.session['loan_form_data'] = session_forms
            
            # If this is the final step, validate that all previous steps have been completed
            if step == 11:
                # Check if all previous steps have data
                missing_steps = []
                for step_num in range(1, 11):
                    if str(step_num) not in session_forms:
                        missing_steps.append(step_num)
                
                if missing_steps:
                    # Some steps are missing, redirect to the first missing step
                    messages.error(request, f"Please complete all previous steps before submitting your application. Missing steps: {', '.join(map(str, missing_steps))}")
                    return redirect('loan_application_step', step=missing_steps[0])
                
                # All steps are complete, proceed with processing the application
                with transaction.atomic():
                    # Create new loan application and personal information
                    loan = Borrower.objects.create()
                    request.session['loan_id'] = loan.loan_id
                    # Create initial loan status
                    LoanStatus.objects.create(loan=loan)
                    
                    # Process all form data from session
                    for step_num, step_data in session_forms.items():
                        step_num = int(step_num)
                        form_class = FORMS[step_num]['form']
                        
                        # Special handling for documents step
                        if step_num == 10:
                            files_data = request.session.get(f'step_{step_num}_files', {})
                            if files_data:
                                document_instance = RequiredDocument(loan=loan)
                                for field_name, file_path in files_data.items():
                                    # Get the full path of the temporary file
                                    temp_file_path = os.path.join(settings.MEDIA_ROOT, file_path)
                                    if os.path.exists(temp_file_path):
                                        # Open the temporary file
                                        with open(temp_file_path, 'rb') as f:
                                            # Get just the filename without the temp path
                                            filename = os.path.basename(file_path)
                                            # Save to the proper location using the FileField
                                            getattr(document_instance, field_name).save(filename, f)
                                document_instance.save()
                                
                                # Clean up temporary files
                                temp_dir = os.path.dirname(temp_file_path)
                                if os.path.exists(temp_dir):
                                    import shutil
                                    shutil.rmtree(temp_dir)
                        
                        # Special handling for dependents
                        elif step_num == 4:
                            # Process dependents data from JSON
                            dependents_data_str = step_data.get('dependents_data', '[]')
                            try:
                                dependents_data = json.loads(dependents_data_str)
                                # Only create dependents if there's actual data
                                if dependents_data and len(dependents_data) > 0:
                                    # Create dependents
                                    for data in dependents_data:
                                        # Skip empty or incomplete data
                                        if not all(key in data and data[key] for key in ['name']):
                                            continue
                                            
                                        Dependent.objects.create(
                                            loan=loan,
                                            name=data.get('name', ''),
                                            age=int(data.get('age', 0)) if data.get('age') else None,
                                            school=data.get('school', ''),
                                            relation=data.get('relation', ''),
                                            self_employed=data.get('self_employed', 'N')
                                        )
                            except json.JSONDecodeError:
                                messages.error(request, "Error processing dependents data.")
                        
                        # Special handling for spouse information (step 6)
                        elif step_num == 6:
                            # Check if form contains any actual data
                            has_spouse_data = False
                            for key, value in step_data.items():
                                if key not in ['csrfmiddlewaretoken', 'next'] and value.strip():
                                    has_spouse_data = True
                                    break
                            
                            # Only create spouse record if there's actual data
                            if has_spouse_data:
                                form = form_class(step_data)
                                if form.is_valid():
                                    instance = form.save(commit=False)
                                    instance.loan = loan
                                    instance.save()
                        
                        else:
                            # Standard form handling
                            form = form_class(step_data)
                            if form.is_valid():
                                instance = form.save(commit=False)
                                instance.loan = loan
                                instance.save()
                    
                    # Create MonthlyCashFlow record after all other records are created
                    try:
                        MonthlyCashFlow.objects.create(
                            loan=loan
                        )
                    except Exception as e:
                        print(f"Error creating monthly cash flow: {e}")
                    
                    # Set success message with reference number
                    messages.success(request, f'Thank you for applying! Your reference number is {loan.reference_number}. You can check your application status anytime here on our website!')
                    
                    # Clean up session data
                    if 'loan_id' in request.session:
                        del request.session['loan_id']
                    if 'loan_form_data' in request.session:
                        del request.session['loan_form_data']
                    # Clean up file data from session
                    for i in range(1, 12):
                        if f'step_{i}_files' in request.session:
                            del request.session[f'step_{i}_files']
                    
                    # Set submitted flag in session
                    request.session['submitted'] = True
                    
                    return redirect('loan_status', reference_number=loan.reference_number)
            
            # Redirect to next step if not the final step
            if step < len(FORMS):
                return redirect('loan_application_step', step=step + 1)
    else:
        # For GET requests, try to get saved data from session
        session_forms = request.session.get('loan_form_data', {})
        saved_data = session_forms.get(str(step))
        
        # Check if application was already submitted
        if request.session.get('submitted'):
            messages.warning(request, "Your application has already been submitted. You cannot modify it.")
            return redirect('home')
        
        if saved_data:
            form = form_class(initial=saved_data)
    
    context = {
        'form': form,
        'step': step,
        'total_steps': len(FORMS),
        'title': FORMS[step]['title'],
        'FORMS': FORMS,
        'submitted': request.session.get('submitted', False)
    }
    
    # For step 11, check if all previous steps have been completed
    if step == 11:
        session_forms = request.session.get('loan_form_data', {})
        completed_steps = [int(step_num) for step_num in session_forms.keys() if step_num.isdigit()]
        all_steps_completed = all(str(step_num) in session_forms for step_num in range(1, 11))
        context['all_steps_completed'] = all_steps_completed
        context['completed_steps'] = completed_steps
    
    # For dependents step, add dependents data from session
    if step == 4:
        session_forms = request.session.get('loan_form_data', {})
        step_data = session_forms.get(str(step), {})
        dependents_data_str = step_data.get('dependents_data', '[]')
        try:
            dependents_data = json.loads(dependents_data_str)
            context['dependents_data'] = dependents_data_str
            context['dependent_count'] = len(dependents_data)
        except json.JSONDecodeError:
            context['dependents_data'] = '[]'
            context['dependent_count'] = 0
    
    # For loan details step, add current interest rate
    if step == 9:  # Loan Details step
        from .models import InterestRate
        context['current_interest_rate'] = InterestRate.get_active_rate()
    
    return render(request, f'app/loan_application/{FORMS[step]["template"]}', context)

def loan_status(request, reference_number):
    try:
        loan = Borrower.objects.get(reference_number=reference_number)

        return render(request, 'app/loan_status.html', {
            'loan': loan
        })
    except Borrower.DoesNotExist:
        messages.error(request, 'No loan application found with this reference number.')
        return redirect('check_status')

@login_required
@role_required(['MARKETING'])
def marketing_officer_dashboard(request):
    # Get all loan applications
    loans = Borrower.objects.all().order_by('-created_at')
    return render(request, 'app/marketing_officer/dashboard.html', {'loans': loans})

@login_required
@role_required(['MARKETING'])
def loan_details_view(request, loan_id):
    loan = get_object_or_404(Borrower, loan_id=loan_id)
    
    # Check if marketing officer remarks exist, if not create a new one
    try:
        remarks = MarketingOfficerRemarks.objects.get(loan=loan)
    except MarketingOfficerRemarks.DoesNotExist:
        remarks = None
    
    if request.method == 'POST':
        form = MarketingOfficerRemarksForm(request.POST, instance=remarks)
        if form.is_valid():
            remarks = form.save(commit=False)
            remarks.loan = loan
            remarks.marketing_officer_name = f"{request.user.first_name} {request.user.last_name}" or request.user.username
            remarks.save()
            
            # Update loan status based on complete_documents field
            status = loan.status
            if form.cleaned_data['complete_documents'] == 'NO':
                status.status = 'HOLD'
                status.remarks = form.cleaned_data['remarks']
            elif form.cleaned_data['complete_documents'] == 'YES':
                status.status = 'PROCEED_CI'
                status.remarks = form.cleaned_data['remarks']
            else:
                status.status = 'CANCELLED'
                status.remarks = f"Application cancelled. {form.cleaned_data['remarks']}"
            
            status.save()
            
            messages.success(request, 'Remarks and status updated successfully.')
            return redirect('marketing_officer_dashboard')
    else:
        form = MarketingOfficerRemarksForm(instance=remarks)
    
    # Calculate loan amortization
    loan_amortization = None
    if hasattr(loan, 'loan_details'):
        loan_details = loan.loan_details
        
        loan_amortization = {
            'loan_amount': loan_details.loan_amount_applied,
            'interest_rate': loan_details.interest_rate,
            'term_months': loan_details.loan_amount_term,
            'monthly_payment': loan_details.monthly_amortization,
            'total_payment': loan_details.monthly_amortization * loan_details.loan_amount_term,
            'total_interest': (loan_details.monthly_amortization * loan_details.loan_amount_term) - loan_details.loan_amount_applied
        }
    
    context = {
        'loan': loan,
        'form': form,
        'loan_amortization': loan_amortization
    }
    
    return render(request, 'app/marketing_officer/loan_details.html', context)

def update_loan_status(request, reference_number):
    if request.method == 'POST':
        loan = get_object_or_404(Borrower, reference_number=reference_number)
        status = request.POST.get('status')
        remarks = request.POST.get('remarks', '')
        
        if status in dict(Borrower.LOAN_STATUS_CHOICES).keys():
            loan_status = loan.status
            loan_status.status = status
            loan_status.remarks = remarks
            loan_status.save()
            return JsonResponse({'status': 'success', 'message': f'Application status updated to {status}'})
        
    return JsonResponse({'status': 'error', 'message': 'Invalid request'}, status=400)

def check_status(request):
    if request.method == 'POST':
        reference_number = request.POST.get('reference_number', '').strip()
        
        # Validate reference number format
        if not reference_number.isdigit() or len(reference_number) != 12:
            messages.error(request, 'Invalid reference number format.')
            return render(request, 'app/check_status.html')
        
        try:
            loan = Borrower.objects.get(reference_number=reference_number)
            return redirect('loan_status', reference_number=loan.reference_number)
        except Borrower.DoesNotExist:
            messages.error(request, 'No loan application found with this reference number.')
    
    return render(request, 'app/check_status.html')

def home(request):
    return render(request, 'app/home.html')

@login_required
@role_required(['CREDIT'])
def credit_investigator_dashboard(request):
    # Get all loan applications including both pending and completed investigations
    loans = Borrower.objects.filter(
        models.Q(status__status='PROCEED_CI') |  # Pending investigations
        models.Q(credit_investigator_remarks__isnull=False)  # Completed investigations
    ).order_by('-created_at')
    return render(request, 'app/credit_investigator/dashboard.html', {'loans': loans})

@login_required
@role_required(['CREDIT'])
def credit_investigator_loan_details(request, loan_id):
    loan = get_object_or_404(Borrower, loan_id=loan_id)
    try:
        remarks = CreditInvestigatorRemarks.objects.get(loan=loan)
    except CreditInvestigatorRemarks.DoesNotExist:
        remarks = None
    
    if request.method == 'POST':
        form = CreditInvestigatorRemarksForm(request.POST, instance=remarks)
        if form.is_valid():
            remarks = form.save(commit=False)
            remarks.loan = loan
            remarks.credit_investigator_name = f"{request.user.first_name} {request.user.last_name}" or request.user.username
            remarks.save()
            
            messages.success(request, 'Credit investigation assessment saved successfully.')
            return redirect('credit_investigator_dashboard')
    else:
        form = CreditInvestigatorRemarksForm(instance=remarks)

    # Calculate DTI ratio for display purposes only
    dti_ratio = None
    if hasattr(loan, 'employment') and hasattr(loan, 'loan_details'):
        monthly_income = loan.employment.monthly_net_income
        monthly_payment = loan.loan_details.monthly_amortization
        if monthly_income > 0:
            dti_ratio = (monthly_payment / monthly_income) * 100

    # Get loan amortization details
    loan_amortization = None
    if hasattr(loan, 'loan_details'):
        loan_details = loan.loan_details
        
        loan_amortization = {
            'loan_amount': loan_details.loan_amount_applied,
            'interest_rate': loan_details.interest_rate,
            'term_months': loan_details.loan_amount_term,
            'monthly_payment': loan_details.monthly_amortization,
            'total_payment': loan_details.monthly_amortization * loan_details.loan_amount_term,
            'total_interest': (loan_details.monthly_amortization * loan_details.loan_amount_term) - loan_details.loan_amount_applied
        }

    context = {
        'loan': loan,
        'form': form,
        'loan_amortization': loan_amortization,
        'dti_ratio': dti_ratio
    }
    
    return render(request, 'app/credit_investigator/loan_details.html', context)

@login_required
@role_required(['APPROVAL'])
def loan_approval_officer_dashboard(request):
    # Get all loan applications that have been approved by credit investigator
    # and loans that have AI predictions
    loans = Borrower.objects.filter(
        Q(status__status='PROCEED_LAO') |  # Include loans pending approval
        Q(loan_approval_officer_remarks__isnull=False)  # Include loans with any remarks
    ).distinct().order_by('-created_at')
    
    # For each loan with AI prediction, extract the probability
    for loan in loans:
        if hasattr(loan, 'loan_approval_officer_remarks'):
            # Try to extract the probability from the remarks
            probability_match = re.search(r'with ([\d.]+) confidence', loan.loan_approval_officer_remarks.remarks)
            if probability_match:
                loan.prediction_probability = float(probability_match.group(1))
    
    return render(request, 'app/loan_approval_officer/dashboard.html', {'loans': loans})

def get_loan_prediction_data(loan):
    """Helper function to prepare loan data for prediction"""

    # Create a dictionary with all features in the EXACT ORDER required by the model
    data = {
        # 1. Age (int64)
        'Age': [int(loan.personal_info.age) if hasattr(loan, 'personal_info') else 35],
        
        # 2. ResidencyAndCitizenship (object)
        'ResidencyAndCitizenship': [loan.personal_info.residency_and_citizenship if hasattr(loan, 'personal_info') else 'Resident Filipino Citizen'],
        
        # 3. SourceOfFunds (object)
        'SourceOfFunds': [loan.employment.source_of_funds if hasattr(loan, 'employment') else 'Salary'],
        
        # 4. Employment Status&Nature of Business (object)
        'Employment Status&Nature of Business': [loan.employment.employment_status if hasattr(loan, 'employment') else 'Employee'],
        
        # 5. VehiclePrice (float64)
        'VehiclePrice': [float(loan.loan_details.estimated_vehicle_value) if hasattr(loan, 'loan_details') else 0.0],
        
        # 6. DownPayment (float64)
        'DownPayment': [float(loan.loan_details.down_payment_percentage) if hasattr(loan, 'loan_details') else 0.2],
        
        # 7. AmountApproved (float64)
        'AmountApproved': [float(loan.loan_details.loan_amount_applied) if hasattr(loan, 'loan_details') else 0.0],
        
        # 8. TotalIncome (float64)
        'TotalIncome': [float(loan.employment.monthly_net_income) if hasattr(loan, 'employment') else 0.0],
        
        # 9. TotalExpenses (int64)
        'TotalExpenses': [int(float(loan.cash_flow.total_expenses)) if hasattr(loan, 'cash_flow') else 0],
        
        # 10. NetDisposal (float64)
        'NetDisposal': [float(loan.cash_flow.net_disposal) if hasattr(loan, 'cash_flow') else 0.0],
        
        # 11. Terms (int64)
        'Terms': [int(loan.loan_details.loan_amount_term) if hasattr(loan, 'loan_details') else 12],
        
        # 12. MonthlyAmortization (int64)
        'MonthlyAmortization': [int(float(loan.loan_details.monthly_amortization)) if hasattr(loan, 'loan_details') else 0],
        
        # 13. Affordability (object) - Calculate based on DTI ratio
        'Affordability': ['Yes' if hasattr(loan, 'employment') and hasattr(loan, 'loan_details') and 
                          loan.employment.monthly_net_income > 0 and 
                          (float(loan.loan_details.monthly_amortization) / float(loan.employment.monthly_net_income) * 100) <= 40 
                          else 'No'],
        
        # 14. CreditRiskAssessment (object)
        'CreditRiskAssessment': [('Low Risk' if hasattr(loan, 'credit_investigator_remarks') and 
                                loan.credit_investigator_remarks.credit_risk_assessment == 'LOW' else
                                'Medium Risk' if hasattr(loan, 'credit_investigator_remarks') and 
                                loan.credit_investigator_remarks.credit_risk_assessment == 'MEDIUM' else
                                'High Risk' if hasattr(loan, 'credit_investigator_remarks') and 
                                loan.credit_investigator_remarks.credit_risk_assessment == 'HIGH' else
                                'Medium Risk')],
        
        # 15. CompleteDocuments (object)
        'CompleteDocuments': ['Yes' if hasattr(loan, 'marketing_officer_remarks') and 
                             loan.marketing_officer_remarks.complete_documents == 'YES' else 'No'],
        
        # 16. Verified (object)
        'Verified': ['Yes' if hasattr(loan, 'credit_investigator_remarks') and 
                    loan.credit_investigator_remarks.verified == 'YES' else 'No'],
    }
    
    # Debug: print data for verification
    print("\nPrepared Loan Prediction Data:")
    for key, value in data.items():
        print(f"{key}: {value}")
    
    return data

def predict_loan_approval(loan_data):
    """Make a prediction using the trained model"""
    try:
        # Load the model
        model_path = 'app/trained_models/loan_approval_model.joblib'
        if os.path.exists(model_path):
            # Load the trained model
            model = joblib.load(model_path)
            
            # Convert loan data to DataFrame with correct column order
            # Make sure these match exactly with what was used during training
            expected_columns = [
                'Age', 'ResidencyAndCitizenship', 'SourceOfFunds', 
                'Employment Status&Nature of Business', 'VehiclePrice', 
                'DownPayment', 'AmountApproved', 'TotalIncome', 
                'TotalExpenses', 'NetDisposal', 'Terms', 
                'MonthlyAmortization', 'Affordability', 
                'CreditRiskAssessment', 'CompleteDocuments', 'Verified'
            ]
            
            # Ensure all expected columns are present
            for col in expected_columns:
                if col not in loan_data:
                    print(f"Missing column: {col}")
                    loan_data[col] = None  # Add missing column with None values
            
            # Create DataFrame with expected columns only and in the correct order
            df = pd.DataFrame(loan_data, columns=expected_columns)
            
            # Make prediction
            try:
                prediction = model.predict(df)[0]
                probability = model.predict_proba(df)[0][1]  # Probability of approval
                result = 'Approved' if prediction == 1 else 'Declined'
                
                print(f"Prediction result: {result} with probability: {probability:.2f}")
                return result, probability
            except Exception as e:
                print(f"Error making prediction with model: {str(e)}")
        else:
            print(f"Model not found at {model_path}")
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return "Error in prediction", 0

@login_required
def quick_loan_prediction(request, loan_id):
    """API endpoint for quick loan prediction from dashboard"""
    if request.method == 'GET':
        try:
            loan = get_object_or_404(Borrower, loan_id=loan_id)
            
            # Get the loan data formatted for prediction
            loan_data = get_loan_prediction_data(loan)
            
            # Print loan data for debugging
            print("\nLoan Prediction Data:")
            print("=====================")
            for key, value in loan_data.items():
                print(f"{key}: {value}")
            print("=====================\n")
            
            # Make prediction using the voting classifier with polynomial features
            prediction_result, prediction_probability = predict_loan_approval(loan_data)
            
            # Print prediction results
            print("\nPrediction Results:")
            print("===================")
            print(f"Result: {prediction_result}")
            print(f"Confidence: {prediction_probability:.2f}")
            print("===================\n")
            
            # Save the prediction to the database
            from .models import LoanApprovalOfficerRemarks, LoanStatus
            from decimal import Decimal
            
            # Map prediction result to model choice
            model_prediction = 'APPROVED' if prediction_result == 'Approved' else 'DECLINED'
            
            # Create or update the LoanApprovalOfficerRemarks
            user_name = f"{request.user.first_name} {request.user.last_name}" or request.user.username
            remarks = f'AI Prediction: {prediction_result} with {prediction_probability:.2f} confidence'
            
            remarks_obj, created = LoanApprovalOfficerRemarks.objects.update_or_create(
                loan=loan,
                defaults={
                    'loan_approval_officer_name': user_name,
                    'remarks': remarks,
                    'prediction_result': model_prediction,
                    'prediction_probability': Decimal(str(round(prediction_probability * 100, 2)))
                }
            )
            
            # Ensure the loan status remains as 'PROCEED_LAO'
            loan_status, _ = LoanStatus.objects.get_or_create(loan=loan)
            if loan_status.status != 'PROCEED_LAO':
                loan_status.status = 'PROCEED_LAO'
                loan_status.remarks = 'Awaiting human review after AI prediction.'
                loan_status.save()
            
            return JsonResponse({
                'prediction': prediction_result,
                'probability': prediction_probability,
                'loan_id': loan_id,
                'saved': True,
                'remarks': remarks
            })
        except Exception as e:
            print(f"Error in loan prediction: {str(e)}")
            return JsonResponse({
                'error': str(e)
            }, status=400)
    return JsonResponse({'error': 'Invalid request'}, status=400)

@login_required
@role_required(['APPROVAL'])
def loan_approval_officer_loan_details(request, loan_id):
    loan = get_object_or_404(Borrower, loan_id=loan_id)
    try:
        remarks = LoanApprovalOfficerRemarks.objects.get(loan=loan)
    except LoanApprovalOfficerRemarks.DoesNotExist:
        remarks = None
    
    if request.method == 'POST':
        form = LoanApprovalOfficerRemarksForm(request.POST, instance=remarks)
        if form.is_valid():
            remarks = form.save(commit=False)
            remarks.loan = loan
            remarks.loan_approval_officer_name = f"{request.user.first_name} {request.user.last_name}" or request.user.username
            remarks.save()
            messages.success(request, 'Loan approval assessment saved successfully.')
            return redirect('loan_approval_officer_dashboard')
    else:
        form = LoanApprovalOfficerRemarksForm(instance=remarks)
    
    # Calculate loan metrics
    loan_amortization = None
    if hasattr(loan, 'loan_details'):
        loan_details = loan.loan_details

        loan_amortization = {
            'loan_amount': loan_details.loan_amount_applied,
            'interest_rate': loan_details.interest_rate,
            'term_months': loan_details.loan_amount_term,
            'monthly_payment': loan_details.monthly_amortization,
            'total_payment': loan_details.monthly_amortization * loan_details.loan_amount_term,
            'total_interest': (loan_details.monthly_amortization * loan_details.loan_amount_term) - loan_details.loan_amount_applied
        }

    # Calculate debt-to-income ratio
    dti_ratio = None
    if hasattr(loan, 'employment') and hasattr(loan, 'loan_details'):
        monthly_income = loan.employment.monthly_net_income
        monthly_payment = loan.loan_details.monthly_amortization
        if monthly_income > 0:
            dti_ratio = (monthly_payment / monthly_income) * 100
    
    # Get prediction information
    prediction_result = None
    prediction_probability = None
    
    if remarks and remarks.prediction_result:
        # Use saved prediction from remarks
        if remarks.prediction_result == 'APPROVED':
            prediction_result = 'Approved'
        elif remarks.prediction_result == 'DECLINED':
            prediction_result = 'Declined'
            
        # Use saved probability
        prediction_probability = remarks.prediction_probability
    
    context = {
        'loan': loan,
        'form': form,
        'loan_amortization': loan_amortization,
        'dti_ratio': dti_ratio,
        'prediction_result': prediction_result,
        'prediction_probability': prediction_probability
    }
    
    return render(request, 'app/loan_approval_officer/loan_details.html', context)

@login_required
@role_required(['DISBURSEMENT'])
def loan_disbursement_officer_dashboard(request):
    # Get all loan applications that are either pending disbursement or have been processed
    loans = Borrower.objects.filter(
        Q(status__status='PROCEED_LDO') |  # Pending disbursements
        Q(loan_disbursement_officer_remarks__isnull=False)  # Processed disbursements
    ).distinct().order_by('-created_at')
    
    return render(request, 'app/loan_disbursement_officer/dashboard.html', {'loans': loans})

@login_required
@role_required(['DISBURSEMENT'])
def loan_disbursement_officer_loan_details(request, loan_id):
    loan = get_object_or_404(Borrower, loan_id=loan_id)
    try:
        remarks = LoanDisbursementOfficerRemarks.objects.get(loan=loan)
    except LoanDisbursementOfficerRemarks.DoesNotExist:
        remarks = None
    
    if request.method == 'POST':
        form = LoanDisbursementOfficerRemarksForm(request.POST, instance=remarks)
        if form.is_valid():
            remarks = form.save(commit=False)
            remarks.loan = loan
            remarks.loan_disbursement_officer_name = f"{request.user.first_name} {request.user.last_name}" or request.user.username
            remarks.save()
            
            # If loan is completed (disbursed), update the loan disbursement statistics
            if remarks.status == 'COMPLETED':
                update_loan_disbursement_statistics(loan, remarks.disbursement_date)
            
            messages.success(request, 'Loan disbursement details saved successfully.')
            return redirect('loan_disbursement_officer_dashboard')
    else:
        form = LoanDisbursementOfficerRemarksForm(instance=remarks)
    
    # Calculate loan metrics
    loan_amortization = None
    if hasattr(loan, 'loan_details'):
        loan_details = loan.loan_details
 
        loan_amortization = {
            'loan_amount': loan_details.loan_amount_applied,
            'interest_rate': loan_details.interest_rate,
            'term_months': loan_details.loan_amount_term,
            'monthly_payment': loan_details.monthly_amortization,
            'total_payment': loan_details.monthly_amortization * loan_details.loan_amount_term,
            'total_interest': (loan_details.monthly_amortization * loan_details.loan_amount_term) - loan_details.loan_amount_applied
        }

    # Calculate debt-to-income ratio
    dti_ratio = None
    if hasattr(loan, 'employment') and hasattr(loan, 'loan_details'):
        monthly_income = loan.employment.monthly_net_income
        monthly_payment = loan.loan_details.monthly_amortization
        if monthly_income > 0:
            dti_ratio = (monthly_payment / monthly_income) * 100
    
    context = {
        'loan': loan,
        'form': form,
        'loan_amortization': loan_amortization,
        'dti_ratio': dti_ratio
    }
    
    return render(request, 'app/loan_disbursement_officer/loan_details.html', context)

def get_quota_settings():
    """
    Get quota settings from database.
    Returns a dictionary containing monthly and yearly quotas.
    """
    from decimal import Decimal
    
    # Get from the model
    try:
        monthly_quota = QuotaSettings.get_monthly_quota()
    except:
        # Fallback to default if there's any error
        monthly_quota = Decimal('30000.00')
    
    # Calculate yearly quota
    yearly_quota = monthly_quota * 12
    
    return {
        'monthly_quota': monthly_quota,
        'yearly_quota': yearly_quota
    }

def update_quota_settings(monthly_quota):
    """
    Update the quota settings in the database.
    Returns True if successful, False otherwise.
    """
    from decimal import Decimal
    
    try:
        monthly_quota = Decimal(str(monthly_quota))
        
        # Get or create the settings object
        quota_obj, created = QuotaSettings.objects.get_or_create(
            id=1,
            defaults={'monthly_quota': monthly_quota}
        )
        
        if not created:
            quota_obj.monthly_quota = monthly_quota
            quota_obj.save()
        
        # Also update any monthly loan disbursement records for the current month
        from .models import MonthlyLoanDisbursement
        from django.utils import timezone
        
        current_month = timezone.now().month
        current_year = timezone.now().year
        
        monthly_stat = MonthlyLoanDisbursement.objects.filter(
            year=current_year,
            month=current_month
        ).first()
        
        if monthly_stat:
            monthly_stat.monthly_quota = monthly_quota
            monthly_stat.quota_met = monthly_stat.total_amount >= monthly_quota
            
            # Avoid division by zero
            if monthly_quota > 0:
                monthly_stat.disbursement_percentage = (monthly_stat.total_amount / monthly_quota * 100).quantize(Decimal('0.01'))
            else:
                monthly_stat.disbursement_percentage = Decimal('0')
                
            monthly_stat.save()
        
        return True
    except Exception as e:
        print(f"Error updating quota settings: {e}")
        return False

def calculate_date_ranges():
    """
    Calculate and return various date ranges needed for statistics.
    Returns a dictionary containing all relevant dates and ranges.
    """
    today = timezone.now().date()
    return {
        'today': today,
        'thirty_days_ago': today - timedelta(days=30),
        'twelve_weeks_ago': today - timedelta(weeks=12),
        'twelve_months_ago': today - timedelta(days=365),
        'current_week_number': int(today.strftime('%W')),
        'current_month': today.month,
        'current_year': today.year
    }

def get_completed_loans():
    """
    Get all completed loans with necessary related fields.
    Returns a QuerySet of completed loans.
    """
    return Borrower.objects.filter(
        loan_disbursement_officer_remarks__status='COMPLETED'
    ).select_related(
        'loan_details',
        'loan_disbursement_officer_remarks',
        'personal_info',
        'vehicle',
        'marketing'
    )

def calculate_statistics(date_obj, loan_amount):
    """
    Calculate statistics for a given date and loan amount.
    Returns a dictionary containing all calculated statistics.
    """
    week_number = int(date_obj.strftime('%W'))
    week_start = date_obj - timedelta(days=date_obj.weekday())
    week_end = week_start + timedelta(days=6)
    
    quotas = get_quota_settings()
    monthly_quota = quotas['monthly_quota']
    yearly_quota = quotas['yearly_quota']
    
    return {
        'date': date_obj,
        'year': date_obj.year,
        'month': date_obj.month,
        'week_number': week_number,
        'week_start': week_start,
        'week_end': week_end,
        'amount': loan_amount,
        'monthly_quota': monthly_quota,
        'yearly_quota': yearly_quota,
        'disbursement_percentage': (loan_amount / monthly_quota * 100) if monthly_quota > 0 else 0
    }

def update_loan_disbursement_statistics(loan, disbursement_date):
    """
    Update loan disbursement statistics when a loan is disbursed.
    
    Args:
        loan: Borrower instance
        disbursement_date: Date of disbursement
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not hasattr(loan, 'loan_details'):
            print(f"Error: Loan {loan.loan_id} has no loan details")
            return False
        
        stats = calculate_statistics(disbursement_date, loan.loan_details.loan_amount_applied)
        
        # 1. Update daily statistics
        daily_stat, _ = DailyLoanDisbursement.objects.get_or_create(
            date=stats['date'],
            defaults={'total_amount': 0}
        )
        daily_stat.total_amount += stats['amount']
        # Calculate daily percentage based on monthly quota
        daily_stat.disbursement_percentage = (daily_stat.total_amount / stats['monthly_quota'] * 100).quantize(Decimal('0.01'))
        daily_stat.save()
        
        # 2. Update weekly statistics
        weekly_stat, _ = WeeklyLoanDisbursement.objects.get_or_create(
            year=stats['year'],
            month=stats['month'],
            week=stats['week_number'],
            defaults={
                'start_date': stats['week_start'],
                'end_date': stats['week_end'],
                'total_amount': 0
            }
        )
        weekly_stat.total_amount += stats['amount']
        # Calculate weekly percentage based on monthly quota
        weekly_stat.disbursement_percentage = (weekly_stat.total_amount / stats['monthly_quota'] * 100).quantize(Decimal('0.01'))
        weekly_stat.save()
        
        # 3. Update monthly statistics with current quota
        monthly_stat, _ = MonthlyLoanDisbursement.objects.get_or_create(
            year=stats['year'],
            month=stats['month'],
            defaults={
                'total_amount': 0,
                'monthly_quota': stats['monthly_quota'],
                'quota_met': False
            }
        )
        monthly_stat.total_amount += stats['amount']
        monthly_stat.monthly_quota = stats['monthly_quota']  # Always update the quota
        monthly_stat.quota_met = monthly_stat.total_amount >= stats['monthly_quota']
        monthly_stat.disbursement_percentage = (monthly_stat.total_amount / stats['monthly_quota'] * 100).quantize(Decimal('0.01'))
        monthly_stat.save()
        
        # 4. Update yearly statistics with current quota
        yearly_stat, _ = YearlyLoanDisbursement.objects.get_or_create(
            year=stats['year'],
            defaults={
                'total_amount': 0,
                'yearly_quota': stats['yearly_quota'],
                'quota_met': False
            }
        )
        yearly_stat.total_amount += stats['amount']
        yearly_stat.yearly_quota = stats['yearly_quota']  # Always update the quota
        yearly_stat.quota_met = yearly_stat.total_amount >= stats['yearly_quota']
        yearly_stat.disbursement_percentage = (yearly_stat.total_amount / stats['yearly_quota'] * 100).quantize(Decimal('0.01'))
        yearly_stat.save()
        
        return True
        
    except Exception as e:
        print(f"Error updating loan disbursement statistics: {str(e)}")
        return False

def recalculate_all_statistics():
    """
    Recalculate all loan disbursement statistics from scratch.
    Returns True if successful, False otherwise.
    """
    try:
        # Get quota settings
        quotas = get_quota_settings()
        monthly_quota = quotas['monthly_quota']
        yearly_quota = quotas['yearly_quota']
        
        # Clear existing statistics
        DailyLoanDisbursement.objects.all().delete()
        WeeklyLoanDisbursement.objects.all().delete()
        MonthlyLoanDisbursement.objects.all().delete()
        YearlyLoanDisbursement.objects.all().delete()
        
        # Get all completed loans
        completed_loans = Borrower.objects.filter(
            loan_disbursement_officer_remarks__status='COMPLETED'
        ).select_related(
            'loan_details',
            'loan_disbursement_officer_remarks'
        )
        
        # Process each loan
        for loan in completed_loans:
            if hasattr(loan, 'loan_disbursement_officer_remarks') and hasattr(loan, 'loan_details'):
                date_obj = loan.loan_disbursement_officer_remarks.disbursement_date
                amount = loan.loan_details.loan_amount_applied
                
                # Calculate week information
                week_number = int(date_obj.strftime('%W'))
                week_start = date_obj - timedelta(days=date_obj.weekday())
                week_end = week_start + timedelta(days=6)
                
                # 1. Update daily statistics
                daily_stat, _ = DailyLoanDisbursement.objects.get_or_create(
                    date=date_obj,
                    defaults={'total_amount': 0}
                )
                daily_stat.total_amount += amount
                daily_stat.disbursement_percentage = (daily_stat.total_amount / monthly_quota * 100).quantize(Decimal('0.01'))
                daily_stat.save()
                
                # 2. Update weekly statistics
                weekly_stat, _ = WeeklyLoanDisbursement.objects.get_or_create(
                    year=date_obj.year,
                    month=date_obj.month,
                    week=week_number,
                    defaults={
                        'start_date': week_start,
                        'end_date': week_end,
                        'total_amount': 0
                    }
                )
                weekly_stat.total_amount += amount
                weekly_stat.disbursement_percentage = (weekly_stat.total_amount / monthly_quota * 100).quantize(Decimal('0.01'))
                weekly_stat.save()
                
                # 3. Update monthly statistics
                monthly_stat, _ = MonthlyLoanDisbursement.objects.get_or_create(
                    year=date_obj.year,
                    month=date_obj.month,
                    defaults={
                        'total_amount': 0,
                        'monthly_quota': monthly_quota,
                        'quota_met': False
                    }
                )
                monthly_stat.total_amount += amount
                monthly_stat.monthly_quota = monthly_quota
                monthly_stat.quota_met = monthly_stat.total_amount >= monthly_quota
                monthly_stat.disbursement_percentage = (monthly_stat.total_amount / monthly_quota * 100).quantize(Decimal('0.01'))
                monthly_stat.save()
                
                # 4. Update yearly statistics with current quota
                yearly_stat, _ = YearlyLoanDisbursement.objects.get_or_create(
                    year=date_obj.year,
                    defaults={
                        'total_amount': 0,
                        'yearly_quota': yearly_quota,
                        'quota_met': False
                    }
                )
                yearly_stat.total_amount += amount
                yearly_stat.yearly_quota = yearly_quota
                yearly_stat.quota_met = yearly_stat.total_amount >= yearly_quota
                yearly_stat.disbursement_percentage = (yearly_stat.total_amount / yearly_quota * 100).quantize(Decimal('0.01'))
                yearly_stat.save()
        
        return True
        
    except Exception as e:
        print(f"Error recalculating statistics: {str(e)}")
        return False

def prepare_chart_data(daily_stats, weekly_stats, monthly_stats, yearly_stats):
    """
    Prepare data for charts from statistics.
    Optimized to process data in memory.
    """
    # Process daily data
    daily_data = {
        'labels': [],
        'data': []
    }
    for stat in daily_stats:
        daily_data['labels'].append(stat.date.strftime('%d %b'))
        daily_data['data'].append(float(stat.total_amount))
    
    # Process weekly data
    weekly_data = {
        'labels': [],
        'data': []
    }
    for stat in weekly_stats:
        weekly_data['labels'].append(f"Week {stat.week} ({stat.start_date.strftime('%d %b')} - {stat.end_date.strftime('%d %b')})")
        weekly_data['data'].append(float(stat.total_amount))
    
    # Process monthly data
    monthly_data = {
        'labels': [],
        'data': []
    }
    for stat in monthly_stats:
        monthly_data['labels'].append(f"{stat.get_month_name()} {stat.year}")
        monthly_data['data'].append(float(stat.total_amount))
    
    # Process yearly data
    yearly_data = {
        'labels': [],
        'data': []
    }
    for stat in yearly_stats:
        yearly_data['labels'].append(str(stat.year))
        yearly_data['data'].append(float(stat.total_amount))
    
    return {
        'daily': daily_data,
        'weekly': weekly_data,
        'monthly': monthly_data,
        'yearly': yearly_data
    }

@login_required
@role_required(['AREA'])
def area_manager_dashboard(request):
    """
    Dashboard view for Area Manager showing loan disbursement statistics.
    Optimized with caching and reduced database queries.
    """
    # Try to get cached data first
    cache_key = 'area_manager_dashboard_data'
    context = cache.get(cache_key)
    
    if context is None:
        try:
            # Get completed loans with select_related to reduce queries
            completed_loans = Borrower.objects.filter(
                loan_disbursement_officer_remarks__status='COMPLETED'
            ).select_related(
                'personal_info',
                'loan_details',
                'vehicle',
                'marketing',
                'loan_disbursement_officer_remarks'
            )
            
            # Calculate total disbursed in Python to reduce DB load
            total_disbursed = sum(loan.loan_details.loan_amount_applied for loan in completed_loans if hasattr(loan, 'loan_details'))
            
            # Get date ranges
            dates = calculate_date_ranges()
            
            # Get statistics with optimized queries
            daily_stats = DailyLoanDisbursement.objects.filter(
                date__gte=dates['thirty_days_ago']
            ).order_by('-date')
            
            weekly_stats = WeeklyLoanDisbursement.objects.filter(
                start_date__gte=dates['twelve_weeks_ago']
            ).order_by('-year', '-month', '-week')
            
            monthly_stats = MonthlyLoanDisbursement.objects.all().order_by('-year', '-month')[:12]
            
            yearly_stats = YearlyLoanDisbursement.objects.all().order_by('-year')
            
            # Prepare chart data in memory
            chart_data = prepare_chart_data(daily_stats, weekly_stats, monthly_stats, yearly_stats)
            
            # Get recent loans (limit to 10)
            recent_loans = completed_loans.order_by(
                '-loan_disbursement_officer_remarks__disbursement_date'
            )[:10]
            
            context = {
                'completed_loans_count': len(completed_loans),
                'total_disbursed': total_disbursed,
                # Overview card statistics
                'daily_stats_today': daily_stats.first(),
                'weekly_stats_current': weekly_stats.first(),
                'monthly_stats_current': monthly_stats.first(),
                'yearly_stats_current': yearly_stats.first(),
                # Detailed table statistics
                'daily_stats': daily_stats,
                'weekly_stats': weekly_stats,
                'monthly_stats': monthly_stats,
                'yearly_stats': yearly_stats,
                # Chart data (pre-processed)
                'daily_labels': json.dumps(chart_data['daily']['labels']),
                'daily_data': json.dumps(chart_data['daily']['data']),
                'weekly_labels': json.dumps(chart_data['weekly']['labels']),
                'weekly_data': json.dumps(chart_data['weekly']['data']),
                'monthly_labels': json.dumps(chart_data['monthly']['labels']),
                'monthly_data': json.dumps(chart_data['monthly']['data']),
                'yearly_labels': json.dumps(chart_data['yearly']['labels']),
                'yearly_data': json.dumps(chart_data['yearly']['data']),
                'recent_loans': recent_loans
            }
            
            # Cache the context for 5 minutes
            cache.set(cache_key, context, 300)
        
        except Exception as e:
            print(f"Error in area manager dashboard: {str(e)}")
            messages.error(request, "An error occurred while loading the dashboard.")
            return render(request, 'app/area_manager/dashboard.html', {
                'error': "Failed to load dashboard data"
            })
    
    return render(request, 'app/area_manager/dashboard.html', context)

@login_required
@role_required(['AREA'])
def area_manager_loan_details(request, loan_id):
    """
    Display the details of a specific loan for the area manager.
    
    Args:
        request: The HTTP request.
        loan_id: The ID of the loan to display.
        
    Returns:
        A rendered HTML page with the loan details.
    """
    try:
        loan = Borrower.objects.select_related(
            'personal_info',
            'contact_info',
            'education',
            'employment',
            'spouse',
            'cash_flow',
            'loan_details',
            'vehicle',
            'marketing',
            'loan_disbursement_officer_remarks',
            'status'
        ).get(loan_id=loan_id)
        
        # Calculate DTI (Debt-to-Income ratio)
        dti_ratio = 0
        if hasattr(loan, 'loan_details') and hasattr(loan, 'cash_flow') and loan.cash_flow.total_income > 0:
            dti_ratio = loan.loan_details.monthly_amortization / loan.cash_flow.total_income
        
        # Calculate loan status progress percentage
        status_percentage = 0
        if loan.status.status == 'PENDING':
            status_percentage = 25
        elif loan.status.status == 'PROCEED_CI':
            status_percentage = 50
        elif loan.status.status == 'PROCEED_LAO':
            status_percentage = 75
        elif loan.status.status == 'PROCEED_LDO':
            status_percentage = 90
        elif loan.status.status == 'COMPLETED':
            status_percentage = 100
                
        # Calculate loan amortization schedule
        loan_amortization = {}
        
        context = {
            'loan': loan,
            'dti_ratio': dti_ratio,
            'status_percentage': status_percentage,
            'loan_amortization': loan_amortization
        }
        
        return render(request, 'app/area_manager/loan_details.html', context)
    
    except Borrower.DoesNotExist:
        messages.error(request, "Loan not found.")
        return redirect('area_manager_dashboard')

@login_required
@role_required(['AREA'])
def area_manager_forecasting(request):
    """View to display forecasting data for Area Manager"""
    
    # Get most recent statistics
    daily_stats = DailyLoanDisbursement.objects.order_by('-date')[:30]
    weekly_stats = WeeklyLoanDisbursement.objects.order_by('-year', '-month', '-week')[:12]
    monthly_stats = MonthlyLoanDisbursement.objects.order_by('-year', '-month')[:12]
    yearly_stats = YearlyLoanDisbursement.objects.order_by('-year')[:5]
    
    context = {
        'daily_stats': daily_stats,
        'weekly_stats': weekly_stats,
        'monthly_stats': monthly_stats,
        'yearly_stats': yearly_stats,
    }
    
    return render(request, 'app/area_manager/forecasting.html', context)

@login_required
def regenerate_forecasts(request):
    """Regenerate forecasts with the latest data"""
    if request.method != 'GET':
        return JsonResponse({'success': False, 'error': 'Only GET requests are allowed'})
    
    try:
        # Train new models with latest data
        from .forecasting import train_and_save_models
        results = train_and_save_models()
        
        if results:
            return JsonResponse({'success': True})
        else:
            return JsonResponse({'success': False, 'error': 'Failed to generate forecasts'})
    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"Error regenerating forecasts: {error_msg}")
        print(traceback.format_exc())
        return JsonResponse({'success': False, 'error': error_msg})

def generate_future_forecast(model, last_sequence, n_steps, scaler, freq):
    """Generate future forecast using the pre-trained model"""
    curr_seq = last_sequence.copy()
    future_preds = []
    
    for _ in range(n_steps):
        # Reshape for prediction
        seq_reshaped = curr_seq.reshape(1, curr_seq.shape[0], curr_seq.shape[1])
        # Make prediction
        curr_pred = model.predict(seq_reshaped, verbose=0)
        future_preds.append(curr_pred[0, 0])
        
        # Update sequence by rolling and adding new prediction
        curr_seq = np.roll(curr_seq, -1, axis=0)
        curr_seq[-1, 0] = curr_pred[0, 0]
    
    # Convert predictions to original scale
    scaled_features = np.zeros((len(future_preds), scaler.scale_.shape[0]))
    scaled_features[:, 0] = future_preds
    future_preds_scaled = scaler.inverse_transform(scaled_features)[:, 0]
    
    # Create future dates
    last_date = datetime.now()
    
    # Create future dates based on frequency
    if freq == 'W':
        future_dates = pd.date_range(start=last_date, periods=n_steps, freq='W')
    elif freq == 'M':
        future_dates = pd.date_range(start=last_date, periods=n_steps, freq='MS')
    elif freq == 'Q':
        future_dates = pd.date_range(start=last_date, periods=n_steps, freq='QS')
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecast': future_preds_scaled
    }).set_index('Date')
    
    return forecast_df
    
# System Administrator Views
@login_required
@role_required(['ADMIN'])
def admin_dashboard(request):
    """
    Display the System Administrator dashboard with user accounts and interest rates.
    
    Args:
        request: The HTTP request.
        
    Returns:
        A rendered HTML page with the admin dashboard.
    """
    # Check if user is an admin
    if not request.user.is_superuser:
        messages.error(request, "You do not have permission to access this page.")
        return redirect('home')
    
    # Get total loan applications
    total_applications = Borrower.objects.count()
    
    # Get total loans and disbursed amount
    total_loans = Borrower.objects.filter(
        loan_disbursement_officer_remarks__status='COMPLETED'
    ).count()
    
    total_disbursed = Borrower.objects.filter(
        loan_disbursement_officer_remarks__status='COMPLETED'
    ).aggregate(
        total=Sum('loan_details__loan_amount_applied')
    )['total'] or 0
    
    # Get current month's quota information
    current_month = timezone.now().month
    current_year = timezone.now().year
    
    monthly_stats = MonthlyLoanDisbursement.objects.filter(
        year=current_year,
        month=current_month
    ).first()
    
    quotas = get_quota_settings()
    monthly_quota = quotas['monthly_quota']
    monthly_progress = monthly_stats.total_amount if monthly_stats else 0
    monthly_quota_met = monthly_stats.quota_met if monthly_stats else False
    
    # Get interest rates
    interest_rates = InterestRate.objects.all().order_by('-created_at')[:1]
    
    context = {
        'total_applications': total_applications,
        'total_loans': total_loans,
        'total_disbursed': total_disbursed,
        'monthly_quota': monthly_quota,
        'monthly_progress': monthly_progress,
        'monthly_quota_met': monthly_quota_met,
        'interest_rates': interest_rates,
    }
    
    return render(request, 'app/admin/dashboard.html', context)

@login_required
@role_required(['ADMIN'])
def interest_rate_list(request):
    """
    Display the current interest rate.
    
    Args:
        request: The HTTP request.
        
    Returns:
        A rendered HTML page with the current interest rate.
    """
    # Check if user is an admin
    if not request.user.is_superuser:
        messages.error(request, "You do not have permission to access this page.")
        return redirect('home')
    
    # Get the current interest rate (should be only one)
    interest_rate = InterestRate.objects.first() or InterestRate(rate=InterestRate.get_active_rate())
    
    context = {
        'interest_rate': interest_rate,
    }
    
    return render(request, 'app/admin/interest_rate.html', context)

@login_required
@role_required(['ADMIN'])
def interest_rate_update(request, pk):
    """
    Update or create an interest rate.
    
    Args:
        request: The HTTP request.
        pk: The primary key of the interest rate to update.
        
    Returns:
        A redirect to the interest rate list page.
    """
    # Check if user is an admin
    if not request.user.is_superuser:
        messages.error(request, "You do not have permission to access this page.")
        return redirect('home')
    
    if request.method == 'POST':
        try:
            # Get or create the interest rate
            interest_rate, created = InterestRate.objects.get_or_create(pk=pk)
            
            # Update the rate
            interest_rate.rate = Decimal(request.POST.get('rate'))
            interest_rate.save()
            
            messages.success(request, "Interest rate updated successfully.")
        except (ValueError, InvalidOperation) as e:
            messages.error(request, "Invalid interest rate value.")
        except Exception as e:
            messages.error(request, f"Error updating interest rate: {str(e)}")
    
    return redirect('interest_rate_list')

@login_required
@role_required(['ADMIN'])
def user_list(request):
    """
    Display a list of all user accounts.
    
    Args:
        request: The HTTP request.
        
    Returns:
        A rendered HTML page with the user account list.
    """
    # Check if user is an admin
    if not request.user.is_superuser:
        messages.error(request, "You do not have permission to access this page.")
        return redirect('home')
    
    users = User.objects.filter(is_active=True).select_related('user_account').order_by('username')
    
    return render(request, 'app/admin/user_list.html', {'users': users})

@login_required
@role_required(['ADMIN'])
def user_create(request):
    """
    Create a new user account.
    
    Args:
        request: The HTTP request.
        
    Returns:
        A redirect to the user list.
    """
    # Check if user is an admin
    if not request.user.is_superuser:
        messages.error(request, "You do not have permission to access this page.")
        return redirect('home')
    
    if request.method == 'POST':
        # Process the form data
        username = request.POST.get('username')
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        email = request.POST.get('email')
        password1 = request.POST.get('password1')
        password2 = request.POST.get('password2')
        position = request.POST.get('position')
        contact_number = request.POST.get('contact_number')
        address = request.POST.get('address')
        
        # Validate data
        if not username or not password1 or not password2 or not position:
            messages.error(request, "All fields are required.")
            return redirect('user_list')
        
        if password1 != password2:
            messages.error(request, "Passwords do not match.")
            return redirect('user_list')
        
        # Check if username or email already exists
        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists.")
            return redirect('user_list')
        
        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already exists.")
            return redirect('user_list')
        
        try:
            # Create User object - if position is ADMIN, create superuser
            if position == 'ADMIN':
                user = User.objects.create_superuser(
                    username=username,
                    email=email,
                    password=password1,
                    first_name=first_name,
                    last_name=last_name
                )
            else:
                user = User.objects.create_user(
                    username=username,
                    email=email,
                    password=password1,
                    first_name=first_name,
                    last_name=last_name
                )
            
            # Create UserAccount object
            UserAccount.objects.create(
                user=user,
                position=position,
                contact_number=contact_number,
                address=address
            )
            
            messages.success(request, f"User '{username}' created successfully.")
        except Exception as e:
            messages.error(request, f"Error creating user: {str(e)}")
    
    return redirect('user_list')

@login_required
@role_required(['ADMIN'])
def user_update(request, pk):
    """
    Update an existing user account.
    
    Args:
        request: The HTTP request.
        pk: The primary key of the user to update.
        
    Returns:
        A redirect to the user list.
    """
    # Check if user is an admin
    if not request.user.is_superuser:
        messages.error(request, "You do not have permission to access this page.")
        return redirect('home')
    
    user = get_object_or_404(User, pk=pk)
    
    if request.method == 'POST':
        try:
            # Get or initialize UserAccount
            user_account, created = UserAccount.objects.get_or_create(user=user)
            
            # Process the form data
            first_name = request.POST.get('first_name')
            last_name = request.POST.get('last_name')
            email = request.POST.get('email')
            position = request.POST.get('position')
            contact_number = request.POST.get('contact_number')
            address = request.POST.get('address')
            
            # Update User model
            user.first_name = first_name
            user.last_name = last_name
            user.email = email
            
            # Update superuser status based on position
            if position == 'ADMIN':
                user.is_superuser = True
                user.is_staff = True
            else:
                user.is_superuser = False
                user.is_staff = False
                
            user.save()
            
            # Update UserAccount model
            user_account.position = position
            user_account.contact_number = contact_number
            user_account.address = address
            user_account.save()
            
            messages.success(request, f"User '{user.username}' updated successfully.")
        except Exception as e:
            messages.error(request, f"Error updating user: {str(e)}")
    
    return redirect('user_list')

@login_required
@role_required(['ADMIN'])
def user_delete(request, pk):
    """
    Delete a user account.
    
    Args:
        request: The HTTP request.
        pk: The primary key of the user to delete.
        
    Returns:
        A redirect to the user list.
    """
    # Check if user is an admin
    if not request.user.is_superuser:
        messages.error(request, "You do not have permission to access this page.")
        return redirect('home')
    
    # Don't allow deleting yourself
    if request.user.pk == pk:
        messages.error(request, "You cannot delete your own account.")
        return redirect('user_list')
    
    user = get_object_or_404(User, pk=pk)
    username = user.username
    
    # Delete the user and its related UserAccount (CASCADE)
    user.delete()
    
    messages.success(request, f"User '{username}' deleted successfully.")
    return redirect('user_list')
   

@login_required
@role_required(['ADMIN'])
def loan_quota_list(request):
    """
    Display the current monthly loan quota.
    
    Args:
        request: The HTTP request.
        
    Returns:
        A rendered HTML page with the current monthly loan quota.
    """
    # Check if user is an admin
    if not request.user.is_superuser:
        messages.error(request, "You do not have permission to access this page.")
        return redirect('home')
    
    # Get the current quota settings
    quotas = get_quota_settings()
    monthly_quota = quotas['monthly_quota']
    yearly_quota = quotas['yearly_quota']
    
    # Get the current month and year stats if available
    current_month = timezone.now().month
    current_year = timezone.now().year
    
    monthly_stats = MonthlyLoanDisbursement.objects.filter(
        year=current_year,
        month=current_month
    ).first()
    
    context = {
        'monthly_quota': monthly_quota,
        'yearly_quota': yearly_quota,
        'monthly_stats': monthly_stats,
        'current_month': datetime(current_year, current_month, 1).strftime('%B'),
        'current_year': current_year
    }
    
    return render(request, 'app/admin/loan_quota.html', context)

@login_required
@role_required(['ADMIN'])
def loan_quota_update(request):
    """Update monthly loan quota"""
    if not request.user.is_superuser:
        messages.error(request, "You don't have permission to access this page.")
        return redirect('home')
    
    if request.method == 'POST':
        form = MonthlyQuotaForm(request.POST)
        if form.is_valid():
            monthly_quota = form.cleaned_data['monthly_quota']
            
            # Update the quota settings
            success = update_quota_settings(monthly_quota)
            
            if success:
                messages.success(request, "Monthly loan quota updated successfully.")
            else:
                messages.error(request, "Failed to update monthly loan quota.")
                
            return redirect('loan_quota_list')
    else:
        # Get current quota settings
        quotas = get_quota_settings()
        form = MonthlyQuotaForm(initial={'monthly_quota': quotas['monthly_quota']})
    
    return redirect('loan_quota_list')

def loan_computation(request):
    """
    Display the loan computation calculator page.
    """
    # Create a form instance
    form = LoanDetailsForm()

    from .models import InterestRate
    current_rate = InterestRate.get_active_rate()
    
    return render(request, 'app/loan_computation.html', {
        'form': form,
        'current_interest_rate': current_rate
    })

def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user_role = request.POST.get('user_role')
        next_url = request.POST.get('next', '')
        
        if not username or not password:
            messages.error(request, 'Please provide both username and password.')
            return render(request, 'app/login.html', {'position_choices': UserAccount.POSITION_CHOICES})
        
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            # Check if user has the selected role
            has_correct_role = False
            
            if user.is_superuser:
                # Superuser can access any role
                has_correct_role = True
            elif hasattr(user, 'user_account'):
                if not user_role or user.user_account.position == user_role:
                    has_correct_role = True
            
            if has_correct_role:
                login(request, user)
                messages.success(request, f'Welcome, {user.first_name if user.first_name else user.username}!')
                
                # Redirect to appropriate dashboard based on role
                if next_url:
                    return redirect(next_url)
                elif user.is_superuser:
                    return redirect('admin_dashboard')
                elif hasattr(user, 'user_account'):
                    if user.user_account.position == 'MARKETING':
                        return redirect('marketing_officer_dashboard')
                    elif user.user_account.position == 'CREDIT':
                        return redirect('credit_investigator_dashboard')
                    elif user.user_account.position == 'APPROVAL':
                        return redirect('loan_approval_officer_dashboard')
                    elif user.user_account.position == 'DISBURSEMENT':
                        return redirect('loan_disbursement_officer_dashboard')
                    elif user.user_account.position == 'AREA':
                        return redirect('area_manager_dashboard')
                
                # Default fallback
                return redirect('home')
            else:
                messages.error(request, 'You do not have access to the selected role.')
                return render(request, 'app/login.html', {'position_choices': UserAccount.POSITION_CHOICES})
        else:
            messages.error(request, 'Invalid username or password.')
            return render(request, 'app/login.html', {'position_choices': UserAccount.POSITION_CHOICES})
    
    return render(request, 'app/login.html', {'position_choices': UserAccount.POSITION_CHOICES})

def logout_view(request):
    logout(request)
    messages.success(request, 'You have been logged out successfully.')
    return redirect('home')

def contact_us(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        subject = request.POST.get('subject')
        message = request.POST.get('message')
        
        # Here you would typically send an email or save the contact form data
        # For now, we'll just redirect with a success message
        
        messages.success(request, "Thank you for your message! We'll get back to you soon.")
        return redirect('contact_us')
    
    return render(request, 'app/contact.html')

@login_required
@role_required(['AREA'])
def regenerate_forecasts(request):
    """Regenerate forecasts with the latest data"""
    if request.method != 'GET':
        return JsonResponse({'success': False, 'error': 'Only GET requests are allowed'})
    
    try:
        # Train new models with latest data
        from .forecasting import train_and_save_models
        results = train_and_save_models()
        
        if results:
            return JsonResponse({'success': True})
        else:
            return JsonResponse({'success': False, 'error': 'Failed to generate forecasts'})
    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"Error regenerating forecasts: {error_msg}")
        print(traceback.format_exc())
        return JsonResponse({'success': False, 'error': error_msg})

@login_required
def get_forecast_data_view(request, frequency):
    """API endpoint to get forecast data for charts"""
    freq_map = {
        'weekly': 'W',
        'monthly': 'M',
        'quarterly': 'Q'
    }
    
    freq = freq_map.get(frequency.lower())
    if not freq:
        return JsonResponse({'error': 'Invalid frequency'}, status=400)
        
    data = get_forecast_data(freq)
    if data is None:
        return JsonResponse({'error': 'Failed to generate forecast data'}, status=500)
        
    return JsonResponse(data)