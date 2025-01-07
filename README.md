### **Modular Programming for Machine Learning: Customer Churn classification use case**
---

### Overview
This repository focuses on building a modular and reusable Python codebase for customer Churn classification in the telecom industry. The project involves data ingestion, data preprocessing and model training while saving outputs as serialized `.pkl` files for reuse in subsequent stages.

### Features
- Modular Python scripts for data ingestion, data preprocessing and model training.
- Comprehensive EDA through Jupyter notebooks.
- Main entry module for reliability.
- Saved best machine learning model and preprocessing pipeline in `.pkl` format.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/makina0928/Modular-Programming-for-Machine-Learning-Customer-Churn-classification-use-case.git

   cd Modular-Programming-for-Machine-Learning-Customer-Churn-classification-use-case
   ```

2. Git Configuration
   ```
   git config --global user.name "USER_NAME"
   ```

   ```
   git config --global user.email "USER_EMAIL"
   ```
3. Create a Virtual Environment
   ```
   conda create -p venv python==3.12.5 -y
   ```

   ```
   conda activate venv/
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
- Run the data ingestion module:
  ```bash
  python src/data_ingestion.py
  ```
- Run the preprocessing module:
  ```bash
  python src/data_transformation.py
  ```
- Train and save models:
  ```bash
  python src/model_trainer.py
  ```

### Project Structure

```
├── artifacts/
├── logs/
├── mlproject.egg-info/
├── notebooks/
│   ├── EDA_Notebook.ipynb
│   ├── Model_Training_Notebook.ipynb
│   └── Telco-Customer-Churn.xlsx
├── src/
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── eda_functions.py
│   │   └── model_trainer.py
│   ├── __init__.py
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
├── main.py
├── README.md
├── requirements.txt
└── setup.py
```

### Explanation:
- **artifacts/**: Directory for saved models, intermediate files, or other outputs.
- **logs/**: Logs generated during the execution.
- **mlproject.egg-info/**: Metadata related to the project package.
- **notebooks/**: Contains Jupyter notebooks for EDA and model training.
- **src/**: Contains the main source code of the project.
  - **components/**: Modular scripts for data ingestion, transformation, EDA, and model training.
  - **exception.py**: Custom exception handling.
  - **logger.py**: Logging utilities.
  - **utils.py**: General utility functions.
- **main.py**: Entry point for the project.
- **requirements.txt**: Dependencies for the project.
- **setup.py**: Script for packaging the project.
