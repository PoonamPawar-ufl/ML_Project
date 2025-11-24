
SmartApply: Automated LinkedIn Messaging for Job Hunters
Project Description

Job hunting is often repetitive and time-consuming. SmartApply automates the process by combining job discovery, resume tailoring, and personalized outreach into one system. The system scans LinkedIn for fresh job postings, analyzes the job description, compares it to your resume, and generates tailored LinkedIn messages for recruiters or hiring managers.

This project emphasizes responsible automation, giving users full control over generated messages while saving hours of manual effort.

Installation and Setup

1. Clone the repository
2. Create a Python environment 
3. Install dependencies

Running the scripts

streamlit run app.py


Dataset

Source: Kaggle Job Postings Dataset

Type: Tabular CSV, text-based fields (job title, company, description)

Size: ~100,000 postings

Location in Repo: Place CSV file in data/kaggle_jobs.csv

Preprocessing Steps:

Fill missing values for job title, company, and description.

Clean text by removing special characters and stopwords.

Tokenize and numericalize text for model input.

How It Works

Upload Resume: Users upload a PDF or text file.

Preprocessing: Text is extracted, cleaned, and tokenized.

Job Matching: Job descriptions are embedded and compared using cosine similarity.

Result Display: SmartApply UI shows top matching jobs with similarity scores.

Message Generation: The app generates a personalized draft message for each matched role.

Author

Name: Poonam Pawar
Email: p.pawar@ufl.edu

Project Structure

SmartApply/
│
├── data/                   # Kaggle dataset CSV
│   ├── postings.csv
│   ├── cleaned_jobs.csv
│   ├── Poonam_Kishor_Pawar_Resume.txt
├── results/                # Plots and intermediate outputs
├── src/                    # Python scripts for preprocessing
│   ├── Job_matcher
│   ├── EDA_smartApply.py
├── requirements.txt
├── UI
│   ├── app.py
├── docs
│   ├── Technical Blueprint Report
│   ├── Report_Deliverable_2
└── README.md
