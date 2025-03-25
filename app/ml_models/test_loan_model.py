import joblib
import pandas as pd
import numpy as np

# Load the saved model
print("Loading the model...")
model = joblib.load('loan_approval_model.joblib')

# Create test data based on real patterns from the dataset
test_data = pd.DataFrame({
    'Age': [37, 29, 41, 30, 38],  # Common age ranges from dataset
    'ResidencyAndCitizenship': ['Resident Filipino Citizen'] * 5,
    'SourceOfFunds': ['Salary', 'Income from Business', 'Salary', 'Income from Business', 'Salary'],
    'Employment Status&Nature of Business': ['Employee', 'Registered Business', 'Employee', 'Registered Business', 'Employee'],
    'VehiclePrice': [150000, 370000, 260000, 299000, 82500],  # Real price ranges from dataset
    'DownPayment': [30000, 74000, 52000, 59800, 16500],  # 20% of vehicle price
    'AmountApproved': [120000, 296000, 208000, 239200, 66000],  # 80% of vehicle price
    'TotalIncome': [112400, 72000, 38324, 70000, 56000],  # Real income ranges
    'TotalExpenses': [12000, 18000, 13000, 12000, 26000],  # Real expense ranges
    'NetDisposal': [70280, 37800, 17726, 40600, 21000],  # Real net disposal ranges
    'Terms': [18, 24, 24, 24, 12],  # Common loan terms
    'MonthlyAmortization': [8467, 16786, 11787, 13555, 6496],  # Real amortization amounts
    'Affordability': ['Yes', 'Yes', 'Yes', 'No', 'Yes'],
    'CreditRiskAssessment': ['Low Risk', 'Low Risk', 'Low Risk', 'High Risk', 'Low Risk'],
    'CompleteDocuments': ['Yes', 'Yes', 'Yes', 'No', 'Yes'],
    'Verified': ['Yes', 'Yes', 'Yes', 'Yes', 'Yes']
})

# Make predictions
print("\nMaking predictions for the following test cases:")
print("\nCase details:")
for i in range(len(test_data)):
    print(f"\nApplication {i+1}:")
    print(f"Age: {test_data.iloc[i]['Age']}")
    print(f"Source of Funds: {test_data.iloc[i]['SourceOfFunds']}")
    print(f"Employment: {test_data.iloc[i]['Employment Status&Nature of Business']}")
    print(f"Vehicle Price: ₱{test_data.iloc[i]['VehiclePrice']:,.2f}")
    print(f"Monthly Income: ₱{test_data.iloc[i]['TotalIncome']:,.2f}")
    print(f"Monthly Expenses: ₱{test_data.iloc[i]['TotalExpenses']:,.2f}")
    print(f"Net Disposal: ₱{test_data.iloc[i]['NetDisposal']:,.2f}")
    print(f"Monthly Amortization: ₱{test_data.iloc[i]['MonthlyAmortization']:,.2f}")
    print(f"Affordability: {test_data.iloc[i]['Affordability']}")
    print(f"Credit Risk: {test_data.iloc[i]['CreditRiskAssessment']}")
    print(f"Complete Documents: {test_data.iloc[i]['CompleteDocuments']}")

predictions = model.predict(test_data)
probabilities = model.predict_proba(test_data)

# Display results
print("\nPrediction Results:")
print("------------------")
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    status = "APPROVED" if pred == 1 else "DECLINED"
    confidence = prob[1] if pred == 1 else prob[0]
    print(f"\nApplication {i+1}:")
    print(f"Loan Status: {status}")
    print(f"Confidence: {confidence:.2%}")
    
    # Add explanation based on key factors
    print("Key Factors:")
    if status == "APPROVED":
        factors = []
        if test_data.iloc[i]['CreditRiskAssessment'] == 'Low Risk':
            factors.append("Low credit risk")
        if test_data.iloc[i]['Affordability'] == 'Yes':
            factors.append("Affordable monthly payments")
        if test_data.iloc[i]['CompleteDocuments'] == 'Yes':
            factors.append("Complete documentation")
        if test_data.iloc[i]['NetDisposal'] > test_data.iloc[i]['MonthlyAmortization'] * 2:
            factors.append("Strong debt service capacity")
        print("- " + "\n- ".join(factors))
    else:
        factors = []
        if test_data.iloc[i]['CreditRiskAssessment'] == 'High Risk':
            factors.append("High credit risk")
        if test_data.iloc[i]['Affordability'] == 'No':
            factors.append("Monthly payments may not be affordable")
        if test_data.iloc[i]['CompleteDocuments'] == 'No':
            factors.append("Incomplete documentation")
        if test_data.iloc[i]['NetDisposal'] < test_data.iloc[i]['MonthlyAmortization'] * 2:
            factors.append("Insufficient debt service capacity")
        print("- " + "\n- ".join(factors)) 