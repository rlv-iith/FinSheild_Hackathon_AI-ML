# Project: AI-Powered Credit Risk Assessment with MLOps

This project is a comprehensive, end-to-end system for assessing credit risk using machine learning. It follows a multi-phase development process, starting from synthetic data generation and culminating in a user-facing application with a continuous feedback loop for model improvement.

## Project Goal

The primary goal is to build an intelligent system that can predict the likelihood of a loan applicant defaulting. The system starts by processing raw financial data, uses clustering to generate initial insights in an unsupervised manner, trains a predictive model, and serves these predictions through an API. Crucially, it incorporates user feedback to continuously refine the model's accuracy over time.

## Workflow & Project Phases

### Phase 1: Data Foundation (Python)

*   **Input:** Raw financial data, such as bank statements.
*   **Process:** The system ingests this data, classifies transactions, and structures it into a meaningful format.
*   **Augmentation:** It enriches the dataset with synthetically generated data to create a robust foundation for training.
*   **Output:** A clean, feature-engineered, and "ML-ready" dataset (e.g., `ml_ready_for_raw_credit_database_100k.db`).

### Phase 2: Core Intelligence - ML Modeling & Explainability (Python)

This is a two-step modeling process due to the absence of a pre-existing target variable ("default" or "not default").

*   **Step A: Label Generation (Unsupervised Learning):**
    *   A clustering algorithm (e.g., K-Means) is applied to the ML-ready dataset.
    *   The goal is to group users into distinct clusters based on their financial behavior. The hypothesis is that one or more of these clusters will represent "high-risk" or "potential defaulter" profiles. These cluster assignments become the target labels for the next step.

*   **Step B: Predictive Modeling (Supervised Learning):**
    *   Powerful classification models like **XGBoost** and **Random Forest** are trained on the dataset, using the cluster-generated labels as the target.
    *   The system is designed to be interactive, asking the user if the model should be retrained or if a specific train/test split should be used.
    *   **Explainability:** For each prediction made on the test set, **SHAP (SHapley Additive exPlanations)** files and plots are generated. These explain *why* the model made a certain prediction by showing the contribution of each financial feature.

### Phase 3: System Backbone - Backend & API (Node.js)

*   **Architecture:** A microservices architecture using Docker, featuring an **API Gateway** and a dedicated **ML Service**.
*   **Prediction Flow:**
    1.  The backend exposes an API endpoint to receive new applicant data.
    2.  This data is passed to a dedicated Python prediction script.
    3.  The script loads the trained ML model (from Phase 2), makes a prediction, and generates a new SHAP explanation for that specific applicant.
    4.  The prediction and the SHAP explanation are returned to the backend, which can then pass it to the frontend.

### Phase 4: User Experience & Feedback Loop

*   **Frontend:** A user interface (to be developed) that displays the model's prediction (e.g., "High Risk") and the corresponding SHAP plot that explains the decision.
*   **Human-in-the-Loop:** The UI includes a critical feedback mechanism. An analyst or user can validate the model's prediction (e.g., "This prediction was correct" or "This prediction was incorrect").
*   **Feedback Storage:** This feedback is collected and stored in a new, separate dataset within this phase.
*   **Feedback feed:** This feedback is refead into ML if seems fit.
### Phase 5: Finalization, Presentation & Retraining

*   **Manual Inspection:** The feedback data collected in Phase 4 is manually inspected to ensure its quality.
*   **Continuous Improvement:** This verified feedback data is then used to **retrain and improve** the machine learning model in Phase 2. This creates a powerful MLOps cycle where the model gets smarter and more accurate over time based on real-world outcomes.
*   **Presentation:** This phase involves preparing the final project showcase, documentation, and presentation.