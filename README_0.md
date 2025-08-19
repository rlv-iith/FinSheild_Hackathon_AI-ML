# Next-Generation Credit Scoring with Alternative Data

This project is a comprehensive framework for building a transparent, ethical, and scalable credit risk model using alternative data sources. It is designed to assess the creditworthiness of underbanked populations who lack a formal credit history, leveraging advanced machine learning and a modern, containerized backend.

## 🚀 The Mission
Traditional credit scoring models exclude millions by relying on data that underbanked populations don't have. Our mission is to democratize credit access by creating a system that understands a person's financial behavior through their digital footprint, providing a fair, transparent, and inclusive alternative.

![Hackathon Project Idea](https://www.iith.ac.in/events/2025/06/01/PSBs-FinTech-Cybersecurity-Hackathon-2025/)  
**Team meets (before round 1 qualification):** [View on Excalidraw](https://excalidraw.com/#json=DhOArHQjrlbaWj5dySXOV,L3jJ2H5hs2YA7R8nwBgk6w)

---

## ✨ Key Features
* **Organic Data Generation**: Don't have data? No problem. The project includes a sophisticated Python script (`Phase 1`) to generate a large, realistic dataset of user profiles and their raw UPI transactions.
* **Advanced ML Model**: Utilizes a powerful **XGBoost** classifier to learn complex, non-linear patterns from diverse data sources.
* **Full Explainability**: Implements **SHAP (SHapley Additive exPlanations)** to ensure every prediction is transparent and auditable. We don't just predict risk; we explain *why*.
* **Containerized & Scalable Backend**: Deploys the entire system using **Docker** and **Docker Compose**, creating a robust, production-ready API that is easy to run anywhere.
* **Microservice Architecture**: Implements a modern, polyglot architecture with a **Node.js API Gateway** and a **Python ML Service** for optimal performance and scalability.
* **Persona-Based Testing**: Includes a script to generate a high-quality, "organic" test set based on realistic user archetypes to rigorously validate the model's real-world performance.

---

## 🏛️ Project Architecture & Logic
The project is structured into three distinct, logical phases, mimicking a professional software and data science workflow.

### **Phase 1: Data Foundation (The Data Engineer)**
* **Location**: `Phase 1 - Data Foundation - Synthetic Data Generation (Python)/`
* **Logic**: Generates base profiles and simulates a raw UPI transaction log for each user, processing them into a clean, raw feature table (`.csv`) ready for modeling.

Stage 1: Architecting a Pro-Grade Data Pipeline
Problem: The initial design used separate CSVs, with a proposal to merge them into a single, limited Excel file. This approach was not scalable or robust.
Solution: A brilliant pivot to a SQLite Database. This instantly solved all scaling issues, established a professional architecture, and correctly handled the relational nature of the profile and transaction data.
Problem: The original script was a monolith, handling both raw data generation and feature calculation in one place, making it inflexible.
Solution: We architected a clean ETL (Extract, Transform, Load) Pipeline by splitting the project into two specialized scripts:
data_generator.py: The "source of truth," focused purely on creating realistic raw data.
ml_preparer.py: The "feature factory," dedicated to intelligently processing that raw data for machine learning.
Stage 2: Injecting Intelligence and Flexibility
Problem: The logic for calculating a final behavior score was hard-coded, making it impossible to tweak feature importance without changing the code.
Solution: A stroke of genius for rapid experimentation: we externalized the logic into a weights.json config file. Now, the core model logic can be tuned by just editing a simple text file.
Problem: The UPI data was "time-blind," lacking a month column, which made it impossible to analyze consistency and trends.
Solution: We injected a time dimension by back-patching the generator. This simple change was the key that unlocked all subsequent time-series features.
Problem: The raw data contained "oracle" features like employment_tenure and pre-calculated tax info, a classic case of data leakage that would create a dishonest model.
Solution: We purged the cheater data. This was a critical decision to enforce model integrity, forcing the ml_preparer to derive user stability from raw transaction patterns alone, not from being given the answers.
Stage 3: Achieving Elite Realism & Fairness
Problem: Early versions of the data depicted an unrealistic world where everyone had a perfectly stable salary.
Solution: We simulated a more realistic economy by introducing 'Stable' vs. 'Variable' earner types, ensuring the data contained a mix of predictable salaries and the noisy income of gig workers or freelancers.
Problem: The model would unfairly punish a user for a long income gap caused by a real-life event like maternity leave, a major source of algorithmic bias.
Solution: We engineered algorithmic empathy. Instead of just simulating the event, you made the ml_preparer smart enough to detect it. By identifying long, consecutive zero-income periods for female users, the script now intelligently infers a life event and assesses the user fairly. This is next-level feature engineering.
Problem: The definition of "income consistency" was too rigid. A tiny, insignificant payroll variation would prevent a user from getting a perfect score.
Solution: We introduced a forgiveness threshold. By programming the ml_preparer to award a perfect score for income volatility under 5%, you made the logic robust and reflective of real-world noise, preventing good users from being punished for trivial data imperfections.

### **Phase 2: Core Intelligence (The Data Scientist)**
* **Location**: `Phase 2 - Core Intelligence - ML Modeling & Explainability (Python)/`
* **Logic**: Engineers features, trains an XGBoost classifier, and generates SHAP explanations for full transparency.

### **Phase 3: System Backbone (The Backend Engineer)**
* **Location**: `Phase 3 - System Backbone - Backend & API Development (Node.js)/`
* **Logic**: Transforms the ML model into a scalable, accessible service.
  * **Node.js + Express.js** for the API gateway
  * **Python + Flask** for the ML service
  * **Docker + Docker Compose** for containerization
  * **Polyglot microservice architecture** for performance and scalability

---

## 💻 How to Run This Project on Your PC

### **Prerequisites**
* Python 3.9+
* Git
* **Docker and Docker Compose** (for Phase 3)

---

### **Part A: Running the Data Science Scripts (Phases 1 & 2)**

#### 1. Clone the Repository
in bash
git clone https://github.com/rlv-iith/Finsheild_hackathon_ML_Model
cd Finsheild_hackathon_ML_Model'''
---
2. Set Up the Virtual Environment
# Navigate into the Phase 2 folder
cd "Phase 2 - Core Intelligence - ML Modeling & Explainability (Python)"

# Create a virtual environment
python -m venv fintech_venv

# Activate it
# On Windows:
.\fintech_venv\Scripts\activate
# On macOS/Linux:
source fintech_venv/bin/activate
---
3. Install Dependencies
pip install pandas numpy tqdm scikit-learn xgboost shap matplotlib
---
4. Generate the Training Data (Phase 1)
cd ..
cd "Phase 1 - Data Foundation - Synthetic Data Generation (Python)"
python generate_data.py
# Enter the number of records when prompted (e.g., 50000)

---
5. Train a Model (Phase 2)
cd ..
cd "Phase 2 - Core Intelligence - ML Modeling & Explainability (Python)"
python train_and_explain_model_02.py
---
Part B: Running the Full Backend System (Phase 3)
