import gradio as gr
import matplotlib
matplotlib.use('Agg')
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. LOAD THE LENDINGCLUB MODEL PIPELINE ---
# In a real scenario, you would load your trained model. For this example,
# we will use a dummy logic as the .pkl file is not provided.
# To use your real model, uncomment the following lines and ensure the .pkl file is in the 'saved_model' directory.
#
# print("Loading LendingClub model pipeline...")
# with open('saved_model/lendingclub_model_pipeline.pkl', 'rb') as f:
#     pipeline = pickle.load(f)
# print("Model loaded successfully.")
pipeline = None # Using dummy logic instead of a real model for now.


# ==============================================================================
# --- BANK DATABASE AND SUGGESTION LOGIC ---
# ==============================================================================

# Simulated database of banks and their lending criteria.
BANK_DATABASE = [
    {
        "name": "State Bank of India (SBI)",
        "region": "India",
        "logo_url": "https://i.imgur.com/v8p4z2B.png", # Using a reliable imgur link for the logo
        "loan_types": ["Personal Loan", "Home Loan", "Car Loan"],
        "min_credit_score": 700,
        "max_loan_usd": 250000,
        "interest_range": (8.5, 12.0),
        "feature": "Competitive rates for government employees."
    },
    {
        "name": "HDFC Bank",
        "region": "India",
        "logo_url": "https://i.imgur.com/fA21n7l.png",
        "loan_types": ["Personal Loan", "Car Loan"],
        "min_credit_score": 750,
        "max_loan_usd": 60000,
        "interest_range": (10.5, 15.0),
        "feature": "Fast processing and pre-approved offers."
    },
    {
        "name": "Chase Bank",
        "region": "USA",
        "logo_url": "https://i.imgur.com/T0v8DSm.png",
        "loan_types": ["Home Loan", "Car Loan"],
        "min_credit_score": 680,
        "max_loan_usd": 500000,
        "interest_range": (6.5, 9.5),
        "feature": "Wide range of mortgage options available."
    },
    {
        "name": "SoFi",
        "region": "USA",
        "logo_url": "https://i.imgur.com/2sUh3bE.png",
        "loan_types": ["Personal Loan"],
        "min_credit_score": 680,
        "max_loan_usd": 100000,
        "interest_range": (8.9, 14.0),
        "feature": "No fees and flexible terms for personal loans."
    },
    {
        "name": "BNP Paribas",
        "region": "EU",
        "logo_url": "https://i.imgur.com/2U8A5Qy.png",
        "loan_types": ["Personal Loan", "Home Loan"],
        "min_credit_score": 700,
        "max_loan_usd": 300000,
        "interest_range": (4.5, 8.0),
        "feature": "Leading European bank with diverse loan products."
    },
]

def get_bank_suggestions(loan_type, credit_score, loan_amount_usd, currency):
    """Filters the bank database to find suitable lenders."""
    region_map = {'INR (₹)': "India", 'USD ($)': "USA", 'EUR (€)': "EU"}
    user_region = region_map.get(currency)
    
    if not user_region:
        return []

    suggestions = []
    for bank in BANK_DATABASE:
        if (bank["region"] == user_region and
            loan_type in bank["loan_types"] and
            credit_score >= bank["min_credit_score"] and
            loan_amount_usd <= bank["max_loan_usd"]):
            suggestions.append(bank)
            
    return suggestions

def format_bank_suggestions_html(suggestions):
    """Creates an HTML block to display bank suggestions."""
    if not suggestions:
        return ""

    html_string = "<div class='suggestions-section'><h3>🏦 Potential Bank Offers</h3><p>Based on your profile, you may qualify for offers from the following institutions:</p><div class='bank-grid'>"
    
    for bank in suggestions:
        html_string += f"""
        <div class='bank-card'>
            <img src='{bank['logo_url']}' alt='{bank['name']} Logo' class='bank-logo'>
            <div class='bank-details'>
                <h4>{bank['name']}</h4>
                <p><strong>Interest Rate:</strong> {bank['interest_range'][0]}% - {bank['interest_range'][1]}% (Est.)</p>
                <p class='feature'>✨ {bank['feature']}</p>
                <a href='#' class='learn-more-btn' target='_blank'>Learn More</a>
            </div>
        </div>
        """
    html_string += "</div></div>"
    return html_string


# --- 2. HELPER FUNCTIONS (Existing) ---
def calculate_emi(principal, tenure_years, rate_of_interest):
    """Calculates the Equated Monthly Installment (EMI) for a loan."""
    months = int(tenure_years * 12)
    if months == 0: return float('inf'), float('inf'), float('inf')
    monthly_interest_rate = rate_of_interest / (12 * 100)
    if monthly_interest_rate == 0: emi = principal / months
    else: emi = (principal * monthly_interest_rate * (1 + monthly_interest_rate) ** months) / ((1 + monthly_interest_rate) ** months - 1)
    total_payment = emi * months
    total_interest = total_payment - principal
    return emi, total_interest, total_payment

def create_amortization_chart(principal, tenure_years, rate_of_interest, currency_symbol):
    """Generates a loan amortization chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        tenure_years_int = int(tenure_years)
        if tenure_years_int <= 0:
             return None # Return None for invalid tenure
        
        emi, _, __ = calculate_emi(principal, tenure_years_int, rate_of_interest)
        monthly_interest_rate = rate_of_interest / (12 * 100)
        balance = principal
        months = tenure_years_int * 12
        
        principal_paid_per_year = [0] * tenure_years_int
        interest_paid_per_year = [0] * tenure_years_int
        years = list(range(1, tenure_years_int + 1))

        for month in range(1, months + 1):
            interest_payment = balance * monthly_interest_rate
            principal_payment = emi - interest_payment
            balance -= principal_payment
            current_year_index = (month - 1) // 12
            if current_year_index < tenure_years_int:
                principal_paid_per_year[current_year_index] += principal_payment
                interest_paid_per_year[current_year_index] += interest_payment
        
        ax.bar(years, principal_paid_per_year, label='Principal Paid', color='#2A69B8')
        ax.bar(years, interest_paid_per_year, bottom=principal_paid_per_year, label='Interest Paid', color='#A9C4E2')
        ax.set_title('Yearly Payment Breakdown (Principal vs. Interest)')
        ax.set_xlabel('Year')
        ax.set_ylabel(f'Amount Paid ({currency_symbol})')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
    except (ValueError, TypeError):
        return None # Return None if conversion fails
    return fig


def get_country_specific_docs(currency):
    """Returns a formatted HTML string of required documents."""
    if currency == 'INR (₹)':
        return """<h3>Required Documents for Verification</h3><ul><li><strong>Identity & Age Proof:</strong> PAN Card, Aadhaar Card, Passport, or Driver's License</li><li><strong>Address Proof:</strong> Aadhaar Card, recent Utility Bill, or Passport</li><li><strong>Income Proof:</strong> Latest 3 Months' Salary Slips and Latest 6 Months' Bank Statements</li></ul>"""
    elif currency == 'USD ($)':
        return """<h3>Required Documents for Verification</h3><ul><li><strong>Identity & Age Proof:</strong> Driver's License, State ID, or Passport</li><li><strong>Address Proof:</strong> Recent Utility Bill or Bank Statement</li><li><strong>Income Proof:</strong> Recent Pay Stubs, W-2 Forms, or Tax Returns</li><li><strong>Social Security Number (SSN) Verification</strong></li></ul>"""
    elif currency == 'EUR (€)':
        return """<h3>Required Documents for Verification</h3><ul><li><strong>Identity & Age Proof:</strong> National ID Card or Passport</li><li><strong>Address Proof:</strong> Recent Utility Bill or Residency Permit</li><li><strong>Income Proof:</strong> Recent Employment Contract, Payslips, or Tax Declaration</li><li><strong>Tax Identification Number (TIN) Verification</strong></li></ul>"""
    return ""

# --- 3. MAIN HYBRID PREDICTION FUNCTION ---
def predict_loan_approval(
    currency, monthly_income_local, co_applicant_income_local, current_emis_local, loan_amount_local, loan_tenure_years,
    credit_score, loan_type, home_ownership, employment_length):
    
    exchange_rates_to_usd = {'INR (₹)': 83.0, 'USD ($)': 1.0, 'EUR (€)': 0.92}
    currency_symbols = {'INR (₹)': '₹', 'USD ($)': '$', 'EUR (€)': '€'}
    selected_rate = exchange_rates_to_usd.get(currency, 83.0)
    selected_symbol = currency_symbols.get(currency, '₹')
    
    if not all([loan_amount_local, monthly_income_local, credit_score, current_emis_local is not None, co_applicant_income_local is not None, loan_tenure_years > 0]):
        message = "<div class='message-box rejected-box'><h2>❌ Invalid Input</h2><p>Please fill all required fields and ensure tenure is greater than zero.</p></div>"
        return gr.HTML(message), None, None, gr.update(visible=False, value=""), gr.update(visible=False)
    
    interest_rate_matrix = {'Personal Loan': [12.5, 14.5, 16.0], 'Car Loan': [9.5, 11.5, 13.0], 'Home Loan': [8.5, 10.0, 11.5]}
    provisional_interest_rate = interest_rate_matrix[loan_type][1]
    
    total_monthly_income = monthly_income_local + co_applicant_income_local
    new_emi_local, _, __ = calculate_emi(loan_amount_local, loan_tenure_years, provisional_interest_rate)
    dti_ratio = ((current_emis_local + new_emi_local) / total_monthly_income) * 100 if total_monthly_income > 0 else 101
    
    if credit_score < 550:
        message_html = f"<div class='message-box rejected-box'><h2 class='message-title'>❌ Loan Rejected</h2><p class='message-body'>Your credit score of {credit_score} is below the minimum requirement of 550.</p></div>"
        return gr.HTML(message_html), None, None, gr.update(visible=False, value=""), gr.update(visible=False)
    if dti_ratio > 60:
        message_html = f"<div class='message-box rejected-box'><h2 class='message-title'>❌ Loan Rejected</h2><p class='message-body'>Your Debt-to-Income (DTI) ratio is {dti_ratio:.2f}%, which is above the maximum limit of 60%.</p></div>"
        return gr.HTML(message_html), None, None, gr.update(visible=False, value=""), gr.update(visible=False)

    loan_amount_usd = loan_amount_local / selected_rate
    
    # Using dummy AI model logic since pipeline is None
    chance_of_approval_ml = (credit_score / 900) * (1 - dti_ratio / 100)
    
    rule_based_status = "Rejected"
    if credit_score >= 650 and dti_ratio < 40: rule_based_status = "Approved"
    elif credit_score >= 550 and dti_ratio < 50: rule_based_status = "Conditional"
    
    message_html, summary_html, bank_suggestions_html = "", "", ""
    plot = None
    show_summary_and_plot = False
    
    if rule_based_status == "Approved" and chance_of_approval_ml > 0.70:
        status_class, status_title, interest_rate = "approved-box", "✅ Loan Approved", interest_rate_matrix[loan_type][0]
        status_reason = f"Congratulations! Your strong profile (Credit Score: {credit_score}, DTI: {dti_ratio:.2f}%) meets all criteria."
        show_summary_and_plot = True
    elif rule_based_status != "Rejected" or chance_of_approval_ml > 0.50:
        status_class, status_title, interest_rate = "conditional-box", "⚠️ Conditional Approval", interest_rate_matrix[loan_type][1]
        status_reason = f"Your application requires further review. Your DTI ratio of {dti_ratio:.2f}% is in a cautionary range."
        show_summary_and_plot = True
    else:
        status_class, status_title, interest_rate = "rejected-box", "❌ Loan Rejected", 0
        status_reason = f"Your application does not meet the criteria due to a combination of factors including a DTI of {dti_ratio:.2f}%."
        show_summary_and_plot = False
    
    message_html = f"<div class='message-box {status_class}'><h2 class='message-title'>{status_title}</h2><p class='message-body'>{status_reason}</p></div>"
    
    if show_summary_and_plot:
        emi_local, total_interest_local, total_payment_local = calculate_emi(loan_amount_local, loan_tenure_years, interest_rate)
        summary_html = f"""<div class='summary-section'><h3>Summary of Your Loan</h3><div class='summary-container'><div class='summary-card highlight-card'><span class='label'>Your Monthly EMI</span><span class='value'>{selected_symbol}{emi_local:,.2f}</span></div><div class='summary-card'><span class='label'>Loan Amount</span><span class='value'>{selected_symbol}{loan_amount_local:,.2f}</span></div><div class='summary-card'><span class='label'>Loan Tenure</span><span class='value'>{int(loan_tenure_years)} years</span></div><div class='summary-card'><span class='label'>Interest Rate</span><span class='value'>{interest_rate}%</span></div><div class='summary-card'><span class='label'>Total Interest Payable</span><span class='value'>{selected_symbol}{total_interest_local:,.2f}</span></div><div class='summary-card'><span class='label'>Total Payment</span><span class='value'>{selected_symbol}{total_payment_local:,.2f}</span></div></div></div>"""
        plot = create_amortization_chart(loan_amount_local, loan_tenure_years, interest_rate, selected_symbol)
        
        bank_suggestions = get_bank_suggestions(loan_type, credit_score, loan_amount_usd, currency)
        bank_suggestions_html = format_bank_suggestions_html(bank_suggestions)
        
        localized_doc_list = get_country_specific_docs(currency)
        return gr.HTML(message_html + summary_html), plot, bank_suggestions_html, gr.HTML(localized_doc_list), gr.update(visible=True)
    else:
        return gr.HTML(message_html), None, None, gr.update(visible=False, value=""), gr.update(visible=False)

def update_currency_examples(currency):
    if currency == 'INR (₹)': return gr.update(value=75000), gr.update(value=500000)
    elif currency == 'USD ($)': return gr.update(value=5000), gr.update(value=20000)
    elif currency == 'EUR (€)': return gr.update(value=4500), gr.update(value=18000)
    return gr.update(), gr.update()
def toggle_co_applicant(checked): return gr.update(visible=checked)
def reset_all_fields():
    return 'INR (₹)', 'Personal Loan', 75000, False, 0, 0, 500000, 5, 3, 750, 'RENT', "", None, "", "", gr.update(visible=False)

# --- 4. GRADIO INTERFACE ---
css = """
body { background-color: #F0F2F6; } .gradio-container { font-family: 'IBM Plex Sans', sans-serif; border-radius: 12px !important; box-shadow: 0 4px 20px rgba(0,0,0,0.05); max-width: 1280px; margin: auto;}
#header_container { display: flex; align-items: center; justify-content: center; gap: 15px; background-color: #E6EEF7; padding: 1rem; border-radius: 10px 10px 0 0; }
#header_title { color: #1F497D; padding: 0; margin: 0; font-size: 1.8em; }
#submit_button { background-color: #2A69B8 !important; color: white !important; font-weight: bold; border-radius: 8px !important; }
#reset_button { background-color: #6c757d !important; }
.message-box { padding: 1rem 1.5rem; border-radius: 8px; margin-bottom: 1.5rem; border-left: 5px solid; }
.message-box .message-title { margin-top: 0; }
.approved-box { background-color: #E9F7EF; border-color: #4CAF50; color: #1E4620; }
.conditional-box { background-color: #FFF4E5; border-color: #FF9800; color: #663C00; }
.rejected-box { background-color: #FDEDED; border-color: #F44336; color: #5F2120; }
.summary-section h3, .suggestions-section h3 { color: #1F497D; margin-bottom: 1rem; border-top: 1px solid #ddd; padding-top: 1.5rem; }
.summary-container { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1rem; }
.summary-card { background-color: #ffffff; padding: 1rem; border-radius: 8px; border: 1px solid #E0E0E0; text-align: center; }
.summary-card .label { font-size: 0.9em; color: #666; } .summary-card .value { font-size: 1.4em; font-weight: 600; color: #1F497D; }
.highlight-card { grid-column: 1 / -1; background-color: #E6EEF7; border-color: #B4C7E7; }
/* --- CSS for Bank Suggestions --- */
.suggestions-section p { margin-bottom: 1.5rem; color: #333; }
.bank-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1.5rem; }
.bank-card { background-color: #fff; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.07); overflow: hidden; display: flex; flex-direction: column; transition: transform 0.2s ease, box-shadow 0.2s ease; border: 1px solid #eee;}
.bank-card:hover { transform: translateY(-5px); box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
.bank-logo { width: 100px; height: 50px; object-fit: contain; margin: 1.5rem auto 1rem; }
.bank-details { padding: 0 1.5rem 1.5rem; text-align: center; flex-grow: 1; display: flex; flex-direction: column; }
.bank-details h4 { margin-top: 0; margin-bottom: 0.5rem; color: #1F497D; }
.bank-details p { margin: 0.25rem 0; font-size: 0.9em; color: #555;}
.feature { color: #007BFF; font-style: italic; margin-top: 1rem; margin-bottom: 1.5rem; flex-grow: 1;}
.learn-more-btn { display: inline-block; background-color: #f0f2f6; color: #1F497D; padding: 0.6rem 1.2rem; border-radius: 6px; text-decoration: none; font-weight: bold; margin-top: auto; border: 1px solid #ddd; }
.learn-more-btn:hover { background-color: #e2e6ea; }
"""

with gr.Blocks(theme='gradio/soft', css=css) as demo:
    gr.HTML(f"<div id='header_container'><h1 id='header_title'>🪙 LoanSense - Hybrid Finance AI Expert System</h1></div>")
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("👤 Applicant & Financial Details"):
                    currency_dropdown = gr.Dropdown(['INR (₹)', 'USD ($)', 'EUR (€)'], label="Select Currency", value='INR (₹)')
                    loan_type = gr.Dropdown(['Personal Loan', 'Home Loan', 'Car Loan'], label="Type of Loan", value='Personal Loan')
                    monthly_income_local = gr.Number(label="Your Monthly Income", value=75000)
                    co_applicant_checkbox = gr.Checkbox(label="Add a Co-Applicant?", value=False)
                    co_applicant_income_local = gr.Number(label="Co-Applicant's Monthly Income", value=0, visible=False)
                    current_emis_local = gr.Number(label="Total Current Monthly EMIs", value=0)
                    loan_amount_local = gr.Number(label="Loan Amount Requested", value=500000)
                    loan_tenure_years = gr.Slider(minimum=1, maximum=30, step=1, label="Loan Tenure (Years)", value=5)
                    employment_length = gr.Slider(minimum=0, maximum=10, step=0.5, label="Employment Length (Years)", value=3)
                    credit_score = gr.Slider(minimum=300, maximum=900, step=1, label="Credit Score", value=750)
                    home_ownership = gr.Dropdown(['RENT', 'OWN', 'MORTGAGE'], label="Home Ownership Status", value='RENT')
            with gr.Row():
                reset_btn = gr.Button("Start Over", elem_id="reset_button")
                submit_btn = gr.Button("Evaluate My Application", variant="primary", elem_id="submit_button", scale=2)
        
        with gr.Column(scale=3):
            output_report = gr.HTML(label="Eligibility Report")
            visual_analysis_plot = gr.Plot(label="Loan Breakdown")
            bank_suggestions_output = gr.HTML()
            with gr.Accordion("📄 Next Steps & Required Documents", visible=False) as doc_upload_ui:
                doc_list_html = gr.HTML()

    inputs_list = [currency_dropdown, monthly_income_local, co_applicant_income_local, current_emis_local, loan_amount_local, loan_tenure_years, credit_score, loan_type, home_ownership, employment_length]
    outputs_list = [output_report, visual_analysis_plot, bank_suggestions_output, doc_list_html, doc_upload_ui]
    reset_list = [currency_dropdown, loan_type, monthly_income_local, co_applicant_checkbox, co_applicant_income_local, current_emis_local, loan_amount_local, loan_tenure_years, employment_length, credit_score, home_ownership, output_report, visual_analysis_plot, bank_suggestions_output, doc_list_html, doc_upload_ui]
        
    co_applicant_checkbox.change(fn=toggle_co_applicant, inputs=co_applicant_checkbox, outputs=co_applicant_income_local)
    currency_dropdown.change(fn=update_currency_examples, inputs=currency_dropdown, outputs=[monthly_income_local, loan_amount_local])
    submit_btn.click(fn=predict_loan_approval, inputs=inputs_list, outputs=outputs_list)
    reset_btn.click(fn=reset_all_fields, inputs=None, outputs=reset_list)
        
    gr.HTML("<div id='footer'><p>Powered by a Hybrid AI Model. For illustrative purposes only.</p></div>")

# --- 5. LAUNCH THE APP ---
print("Launching Gradio app...")
demo.launch(debug=True)