document.getElementById('risk-form').addEventListener('submit', async function (event) {
  event.preventDefault();

  // Collect data from the form
  const data = {
    // Personal & Financial
    monthly_income_rs: parseFloat(document.getElementById('monthly_income_rs').value),
    age: parseInt(document.getElementById('age').value),
    gender: document.getElementById('gender').value,
    income_stability_type: document.getElementById('income_stability_type').value,
    
    // Behavioral & Derived
    debt_burden_ratio: parseFloat(document.getElementById('debt_burden_ratio').value),
    income_consistency_score: parseFloat(document.getElementById('income_consistency_score').value),
    loan_repayment_consistency: parseFloat(document.getElementById('loan_repayment_consistency').value),
    bnpl_repayment_consistency: parseFloat(document.getElementById('bnpl_repayment_consistency').value),
    financial_coping_ability: parseInt(document.getElementById('financial_coping_ability').value),
    clickstream_volatility: parseFloat(document.getElementById('clickstream_volatility').value),
    app_diversity: parseInt(document.getElementById('app_diversity').value),

    // Economic & External
    peer_default_exposure: parseFloat(document.getElementById('peer_default_exposure').value),
    urbanization_score: parseFloat(document.getElementById('urbanization_score').value),
    local_unemployment_rate: parseFloat(document.getElementById('local_unemployment_rate').value),
    
    // Fill in other non-user-facing fields with defaults
    takes_menstrual_leave: 0,
    device_tier: 1,
    asset_diversity: 2,
    earner_density: 4,
    total_income: 896794,
    total_debits: 385089,
    avg_monthly_food: 4503,
    avg_monthly_ott: 316,
    income_tier: 2
  };

  const resultsDiv = document.getElementById('results');
  resultsDiv.textContent = 'Getting prediction...';

  try {
    const response = await fetch('http://localhost:8080/api/check-risk', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
        // Attempt to get a more detailed error message from the backend
        const errorData = await response.json().catch(() => ({error: `HTTP error! status: ${response.status}`}));
        throw new Error(errorData.error);
    }

    const result = await response.json();
    resultsDiv.textContent = JSON.stringify(result, null, 2);

  } catch (error) {
    resultsDiv.textContent = `Error: ${error.message}`;
    console.error('Error fetching prediction:', error);
  }
});