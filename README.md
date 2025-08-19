# FinSheild_Hackathon

# Next-Generation Credit Risk Management for the Underbanked

**Team:** altDataAlchemists
**Project:** An end-to-end, AI-powered system to assess credit risk using alternative data, designed to promote financial inclusion for underbanked populations in emerging markets.

---

## 1. The Problem: The "Credit Invisible" Population

A large part of the population in emerging markets remains unbanked or underbanked, lacking the formal credit history (e.g., a CIBIL score) required by traditional lenders. Traditional credit scoring models, which rely heavily on historical loan data and credit reports, automatically exclude these individuals. This creates a significant barrier to financial access.

Our challenge is to build a credit risk model that is:
- **Inclusive:** Functions without traditional financial records.
- **Accurate:** Predicts default probability using behavioral, digital, and contextual data.
- **Transparent:** Provides clear, explainable reasons for every decision.
- **Scalable & Private:** Ready for real-world lending scenarios while respecting user privacy.

---

## 2. Our Solution: A Multi-Pipeline, AI-Powered Ecosystem

We have engineered a complete risk management ecosystem built around two distinct, powerful pipelines: a **Semi-Supervised Training Pipeline** for offline model creation and a real-time **Inference & Tracking Pipeline** for live predictions.

This dual-pipeline approach ensures our model is both incredibly robust and operationally efficient.

### Key Innovations & Features:

*   **PDF to Prediction Pipeline:** Users can directly upload their bank/UPI statement as a PDF. Our system intelligently parses, classifies, and analyzes this unstructured data to generate a risk profile.
*   **Intelligent Transaction Classification:** We solve the problem of ambiguous or unbranded UPI transactions by using an intelligent classification engine to categorize spending into meaningful groups (e.g., Utilities, OTT, Peer-to-Peer, E-commerce, Gambling).
*   **Advanced Semi-Supervised Training:** Our unique training process uses a small, high-quality labeled dataset to teach the model the "ground truth" of default. It then refines this knowledge on a massive 100,000+ user synthetic dataset to learn the complex patterns of the general population. This creates a highly accurate and generalized final model.
*   **Total Transparency with SHAP:** Every prediction is accompanied by a full SHAP (SHapley Additive exPlanations) breakdown, showing exactly which behavioral and financial factors contributed to the decision. This is critical for lender trust and regulatory compliance.
*   **Dynamic Post-Loan Tracking:** Our system includes a dedicated ML-driven monitoring flow to track a borrower's risk profile *after* a loan has been disbursed. It re-uses our core ML model to detect and quantify behavioral drift, enabling proactive risk management.

### Hidden Features & Depth of the Synthetic Dataset

The synthetic dataset, which forms the bedrock of our model's knowledge, was engineered with deep, context-aware logic to simulate realistic human behavior. It's more than just random numbers; it's a simulated digital economy.

*   **Contextual Risk Modeling:** A user's spending risk is not absolute; it's **conditional on their income tier**. For example, high spending on OTT services is flagged as a high-risk signal for a low-income user but is considered normal for a high-income user.
*   **Inter-Feature Correlations:** The features are not independent. We used **conditional sampling** to create realistic relationships. For example, a user with a low-tier device is statistically more likely to have higher clickstream volatility. Users with unstable income show different UPI transaction frequencies.
*   **Behavioral Anomaly Simulation:** The data includes simulated "financial shocks." A user might have a stable history but suddenly exhibit patterns of high debt or erratic spending, which the model learns to identify as warning signs.
*   **Psychometric & Social Proxies:** Features like `financial_coping_ability` and `peer_default_exposure` are not just numbers; they are proxies for a user's resilience and the financial health of their social circle, adding a layer of psychometric analysis to the model.

---

## 3. System Architecture & Technical Stack

The platform is designed as a scalable, containerized system using a microservices architecture. This ensures modularity, resilience, and ease of deployment.

![System Architecture Diagram](https://architecture-diagram.png)  <!-- You would create and link a diagram here -->

### The Pipeline Services:

1.  **Frontend (UI):** A clean, professional web interface where a user uploads their PDF statement and provides supplementary data.
    - **Stack:** HTML, CSS, JavaScript

2.  **API Gateway (Node.js):** The central entry point for the application. It receives requests from the frontend and orchestrates the calls to the various backend services.
    - **Stack:** Node.js, Express.js, Axios, Multer

3.  **PDF Processing Service (Python):** A dedicated microservice for handling file uploads.
    - **Role:** Extracts raw text from PDF documents.
    - **Stack:** Python, Flask, PyMuPDF

4.  **Feature Engineering Service (Python):** The core intelligence unit for data transformation.
    - **Role:** Classifies raw transactions and calculates a rich set of 35+ "alternative data" features (e.g., `utility_bill_payment_ratio`, `ott_spending_tier`).
    - **Stack:** Python, Flask, Pandas

5.  **Prediction Service (Python):** The final ML inference engine.
    - **Role:** Loads the trained XGBoost model and SHAP explainer to generate the final prediction and explanation.
    - **Stack:** Python, Flask, XGBoost, SHAP, Scikit-learn

### Technical Stack Summary:

-   **Frontend:** HTML5, CSS3, JavaScript
-   **Backend & APIs:** Node.js, Python (Flask)
-   **Machine Learning:** XGBoost, Scikit-learn, SHAP, Pandas
-   **Containerization & Orchestration:** Docker, Docker Compose

---

## 4. Project Pipelines and Development Phases

Our project followed a structured, five-phase development lifecycle, encapsulating two core operational pipelines: **Training** and **Inference**.

### The Offline Training & Modeling Pipeline

This pipeline is executed in **Phase 2** to produce the final, high-performance ML model.

**`Step 1: Supervised Foundation Training`**
-   **Input:** A small, high-quality, manually-labeled dataset (`organic_data_500.csv`) with a known `loan_default` column.
-   **Process:** An initial XGBoost model is trained exclusively on this data.
-   **Output:** A `foundation_model.joblib`. This model is accurate but specialized, understanding the core drivers of default.

**`Step 2: Unsupervised Knowledge Refinement`**
-   **Input:** The massive 100,000+ user synthetic dataset (from **Phase 1**) and the `foundation_model.joblib`.
-   **Process:**
    1.  KMeans clustering is used on the synthetic data to identify inherent behavioral risk groups, creating a `cluster_risk_label`.
    2.  The `foundation_model` is then **fine-tuned** by continuing its training on this massive, cluster-labeled dataset.
-   **Output:** The `final_credit_risk_model.joblib` and its `final_shap_explainer.joblib`. This model now combines the specific default knowledge with a broad, generalized understanding of population behavior.

### The Real-time Inference & Tracking Pipeline

This pipeline is what runs live in our deployed application (**Phases 3, 4, and 5**).

**`Step 1: Data Ingestion & Processing`**
-   The **Frontend** allows a user to upload a PDF.
-   The **API Gateway** receives this file and orchestrates the backend workflow.
-   The **PDF Processing Service** extracts raw text from the document.

**`Step 2: Intelligent Feature Engineering`**
-   The **Feature Engineering Service** receives the raw text.
-   It performs **Transaction Classification**, turning ambiguous text into clean categories (Utility, Food, etc.).
-   It then calculates over 35 advanced behavioral and financial features.

**`Step 3: ML-Powered Prediction & Explanation`**
-   The **Prediction Service** receives the final feature vector.
-   It uses the `final_credit_risk_model.joblib` to calculate the Probability of Default.
-   It uses the `final_shap_explainer.joblib` to determine the reasons for the decision.

**`Step 4: Post-Loan Monitoring (Dynamic Risk Tracking)`**
-   A scheduled task periodically feeds a borrower's new transaction data back into this pipeline (Steps 1-3).
-   The new Probability of Default is compared to their original score.
-   A significant increase in this ML-driven score (`Risk Drift`) triggers an alert for a loan officer, enabling proactive intervention.

---

## 5. How to Run the Project

### Prerequisites:

*   [Docker](https://www.docker.com/products/docker-desktop/) installed and running.
*   A terminal or command prompt.

### Running the Application:

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd Fintech_Hackathon
    ```

2.  **Ensure Model Artifacts are in Place:**
    Before the first run, ensure the final trained model and its artifacts from `Phase 2` are copied into the `prediction_service` directory in `Phase 3`:
    -   `final_credit_risk_model.joblib` -> `prediction_service/credit_risk_model.joblib`
    -   `final_shap_explainer.joblib` -> `prediction_service/shap_explainer.joblib`
    -   `final_model_config.json` -> `prediction_service/model_config.json`

3.  **Build and Run with Docker Compose:**
    From the root directory (`Fintech_Hackathon/`), run the following command:
    ```bash
    docker-compose up --build
    ```
    This command will build the Docker images for all five services and start them in the correct order.

4.  **Access the Application:**
    Once the containers are running (you will see logs from all the services in your terminal), open your web browser and navigate to:
    [http://localhost:8080](http://localhost:8080)

You can now upload a sample bank statement PDF and see the ML-driven credit risk assessment in action.

---

## 6. Future Roadmap & Improvements

This project provides a robust foundation. Our roadmap includes several advanced enhancements:

1.  **Time-Series Behavioral Modeling:** Transition the post-loan tracking system from a drift detection model to a predictive forecasting model using LSTMs or RNNs to predict future risk.
2.  **Fairness Auditing & Bias Mitigation:** Implement adversarial debiasing techniques to rigorously test and ensure the model is fair across all demographic segments (age, gender, region).
3.  **Federated Learning:** Enhance privacy further by implementing a federated learning approach, allowing the model to be trained on user data locally on their devices without centralizing sensitive information.
4.  **Token Interoperability:** Develop a standardized schema for "behavioral tokens" that would allow users to port their verified creditworthiness across different financial institutions, further promoting financial inclusion.
