from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    # Authentication URLs
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    
    path('apply/', views.loan_application_step, name='loan_application'),
    path('apply/step/<int:step>/', views.loan_application_step, name='loan_application_step'),
    path('loan-status/<uuid:reference_number>/', views.loan_status, name='loan_status'),
    path('update-loan-status/<uuid:reference_number>/', views.update_loan_status, name='update_loan_status'),
    path('loan-computation/', views.loan_computation, name='loan_computation'),
    path('check-status/', views.check_status, name='check_status'),
    
    # Marketing Officer URLs
    path('marketing-officer/', views.marketing_officer_dashboard, name='marketing_officer_dashboard'),
    path('marketing-officer/loan/<int:loan_id>/', views.loan_details_view, name='loan_details_view'),
    
    # Credit Investigator URLs
    path('credit-investigator/', views.credit_investigator_dashboard, name='credit_investigator_dashboard'),
    path('credit-investigator/loan/<int:loan_id>/', views.credit_investigator_loan_details, name='credit_investigator_loan_details'),
    
    # Loan Approval Officer URLs
    path('loan-approval-officer/', views.loan_approval_officer_dashboard, name='loan_approval_officer_dashboard'),
    path('loan-approval-officer/loan/<int:loan_id>/', views.loan_approval_officer_loan_details, name='loan_approval_officer_loan_details'),
    path('loan-approval-officer/predict/<int:loan_id>/', views.quick_loan_prediction, name='quick_loan_prediction'),
    
    # Loan Disbursement Officer URLs
    path('loan-disbursement-officer/', views.loan_disbursement_officer_dashboard, name='loan_disbursement_officer_dashboard'),
    path('loan-disbursement-officer/loan/<int:loan_id>/', views.loan_disbursement_officer_loan_details, name='loan_disbursement_officer_loan_details'),
    
    # Area Manager URLs
    path('area-manager/', views.area_manager_dashboard, name='area_manager_dashboard'),
    path('area-manager/loan/<int:loan_id>/', views.area_manager_loan_details, name='area_manager_loan_details'),
    path('area-manager/forecasting/', views.area_manager_forecasting, name='area_manager_forecasting'),
    path('area-manager/make-forecast/<str:freq>/<int:steps>/', views.make_new_forecast, name='make_new_forecast'),
    
    # System Administrator URLs
    path('system-admin/', views.admin_dashboard, name='admin_dashboard'),
    
    # User Management
    path('system-admin/users/', views.user_list, name='user_list'),
    path('system-admin/users/create/', views.user_create, name='user_create'),
    path('system-admin/users/update/<int:pk>/', views.user_update, name='user_update'),
    path('system-admin/users/delete/<int:pk>/', views.user_delete, name='user_delete'),
    
    # Interest Rate Management
    path('system-admin/interest-rates/', views.interest_rate_list, name='interest_rate_list'),
    path('system-admin/interest-rates/update/<int:pk>/', views.interest_rate_update, name='interest_rate_update'),
    
    # Monthly Loan Quota Management
    path('system-admin/loan-quotas/', views.loan_quota_list, name='loan_quota_list'),
    path('system-admin/loan-quotas/update/', views.loan_quota_update, name='loan_quota_update'),
]