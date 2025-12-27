# 📉 Customer Churn Prediction System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)

A complete End-to-End Machine Learning application that predicts whether a customer is likely to churn (leave the service). This project demonstrates a **Dual-Interface Architecture**, serving both business users (via a Web Dashboard) and software systems (via a REST API).

---

## Architecture

This application is containerized using Docker and provides two entry points to the same Machine Learning "Brain" (Random Forest Model):

1.  **The Dashboard (Streamlit):** An interactive UI for managers to input customer data, visualize risk probabilities, and get retention suggestions.
2.  **The API (FastAPI):** A high-performance REST endpoint for integrating predictions into other software or mobile apps.

```mermaid
graph TD
    User[User / Manager] -->|Browser| Streamlit[Streamlit Dashboard :8501]
    System[External System] -->|HTTP POST| API[FastAPI :8000]

    subgraph Docker Container
        Streamlit --> Logic[Preprocessing Logic]
        API --> Logic
        Logic --> Model[Random Forest Model (.pkl)]
    end
```
