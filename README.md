Zepto Data & AI Platform
This repository contains the capstone project for the Zepto Data & AI Platform, consisting of three modules:

Data Pipeline (/data_pipeline): Scrapes book data, cleans it, and stores it in a SQLite database, along with SQL analytics.
Analytics Pipeline (/analytics): End-to-end data profiling, visualization, and predictive modeling on the Titanic dataset.
Support Assistant (/support_assistant): A locally-run RAG (Retrieval-Augmented Generation) application built with LangGraph and FastAPI, answering questions based on Zepto's policies.
Setup Instructions
Ensure you have Python 3.9+ installed.
Clone this repository.
Install dependencies: pip install -r requirements.txt
Running the Modules
Data Pipeline: Run cd data_pipeline && python scraper.py. This will fetch the data, create books.db, and run SQL queries.
Analytics Pipeline: Run cd analytics && python 01_eda_and_modeling.py. This will fetch the Titanic dataset, save it, produce plots in the charts/ folder, and save the final modeling pipeline.
Support Assistant: Run cd support_assistant && uvicorn main:app --host 0.0.0.0 --port 7860. Then make POST requests to http://localhost:7860/ask.
See the README.md inside each module folder for more specific design decisions and outputs.
