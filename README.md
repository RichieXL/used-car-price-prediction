<h1 align="center">
  Used Car Resale Price Prediction
</h1>

<p align="center">
  <img src="UsedCar_price_image.png" alt="Used Car Price Prediction" width="850">
</p>


<p align="center">
  <strong>A machine learning system for estimating used vehicle resale prices using vehicle characteristics and historical market data.</strong>
</p>

<p align="center">
  BANA 7075 — Machine Learning Systems Design
</p>

<h3 align="center">
  Team: Derek Weaver, Jake Bayman, Adam Ouahidy, Richie James, Bryn Biemiller
</h3>

## Project Overview

This project develops a machine learning system to predict the resale price of used vehicles based on factors such as mileage, age, make, model, fuel type, transmission, condition, and other vehicle characteristics.

The goal is to provide a consistent, data-driven approach to used vehicle pricing that can support dealerships, private sellers, online marketplaces, and buyers.

## Machine Learning Approach

The project is structured as a supervised regression problem.

We will begin with regularized regression models as baseline models and then evaluate tree-based ensemble models to capture more complex relationships within the data.

Models under consideration include:

- Ridge Regression
- Lasso Regression
- Random Forest
- XGBoost

Experiments and model performance will be tracked using MLflow.

## Data

Three datasets were considered:

1. **Craigslist Private Sales** — A large dataset containing 400K+ vehicle listings and more than 20 features.
2. **Cars.com Retail Listings** — A smaller dataset containing 4,009 retail vehicle listings and 9 features.
3. **Vehicle Sales & Market Trends** — A structured dataset containing vehicle characteristics, condition, odometer readings, selling price, sale date, and Manheim Market Report (MMR) values.

The Vehicle Sales & Market Trends dataset was selected as the primary dataset due to its size, structure, and inclusion of an industry pricing benchmark (MMR). The cleaned version used for modeling is stored as National_data_2_locations_Clean.csv.

## Data Pipeline

Data is processed in batches. Raw data is ingested, cleaned, and validated before being versioned with DVC, which ensures that every model can be traced back to the exact dataset used to train it. See DATA_VERSIONING.md for details on how datasets are tracked.

## System Architecture

The system connects data ingestion, processing, model development, experiment tracking, and deployment into a single end-to-end workflow.

<p align="center">
  <img src="ArcFlow.png" alt="Used Car Price Prediction System Architecture" width="1000">
</p>
