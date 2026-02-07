# Bus Delay Prediction System

## Assignment Task

This project represents a comprehensive data-driven predictive analytics solution developed to address real-world challenges in urban transportation systems. The solution leverages machine learning techniques to forecast bus delays, enabling stakeholders and travelers to make informed decisions about their journeys.

## Project Overview

The Bus Delay Prediction System is a complete machine learning pipeline contained in a single Jupyter Notebook (`BusPrediction.ipynb`) that ingests Transportation for London (TfL) bus stop data, applies sophisticated feature engineering, trains dual classification models, and deploys an interactive user interface for real-time delay predictions.

## Background & Context

Urban transportation systems generate vast amounts of structured data from timetables, GPS tracking, route disruptions, and operational metrics. This information, while abundant, often remains underutilized in decision-making processes. The Bus Delay Prediction System harnesses this data through data science and software engineering principles to improve transportation efficiency and reduce delays.

Real-world bus operations are influenced by numerous factors including temporal patterns (peak hours, day of week), environmental conditions (weather, traffic congestion), route characteristics (number of stops, complexity), and geographic factors (urban vs. suburban areas). Traditional approaches to delay estimation often fail to capture these complex interdependencies. This project employs advanced machine learning to identify patterns in these dimensions and generate reliable predictions.

The system builds upon publicly available Transportation for London (TfL) bus stop data, creating a practical demonstration of how data science can improve public transportation services. By accurately predicting delays, the system enables passengers to plan better journeys and enables operators to optimize resource allocation.

## Core Requirements Implementation

### 1. Data Collection & Ingestion (Steps 0-1)

The project ingests bus transportation data from TfL sources, comprising stop information, route characteristics, and operational metrics.

- Loads CSV files containing bus stop reference data with multiple dimensional attributes
- Parses and validates data structure to ensure schema compliance
- Performs initial exploratory analysis to understand data distributions
- Processes approximately 10,000 records for 20+ major London bus stops

**Technology:** Pyspark, Python, Pandas, Colab

### 2. Data Storage & Processing (Step 2)

The dataset is processed using Apache Spark, enabling efficient handling and transformation of large-scale transportation data.

- Removes duplicate records to ensure data integrity
- Trims whitespace from all text fields
- Handles missing values through targeted imputation
- Preserves data quality through validation checkpoints

**Result:** Clean, normalized dataset ready for feature engineering

### 3. Feature Engineering (Step 3)

Creates 11 meaningful features from raw data that capture patterns influencing bus delays:

**Stop-Level Features (3):**
- `stop_name_length`: Length of bus stop name
- `has_indicator`: Whether stop has a directional indicator
- `is_london`: Whether stop is in Greater London area

**Route Features (3):**
- `route_complexity`: Route complexity based on stop count
- `stop_position`: Position in route (Start/Middle/End)
- `total_stops_in_route`: Total number of stops in the route

**Temporal Features (4):**
- `hour_of_day`: Hour when journey occurs (0-23)
- `is_peak_hour`: Peak hours (7-9am or 5-7pm)
- `day_of_week`: Day of week (0-6)
- `is_weekend`: Whether journey is on weekend

**Environmental Features (2):**
- `traffic_level`: Traffic conditions (Low/Medium/High)
- `weather_condition`: Weather conditions (Clear/Rain/Storm)

**Target Variable:**
- `is_delayed`: Binary indicator for delays >5 minutes


### 4. Exploratory Data Analysis (Step 4)

Generates comprehensive visualizations to understand patterns in the data:

- **Delay Distribution Chart**: Shows frequency of delayed vs. on-time journeys
- **Peak Hour Impact Analysis**: Identifies how time-of-day affects delay probability
- **Traffic Impact Analysis**: Demonstrates relationship between traffic levels and delays
- **Location Impact Analysis**: Maps how different stops experience varying delay rates

These visualizations provide intuitive understanding of key drivers influencing bus delays.

### 5. Model Training & Evaluation (Steps 6-11)

The project trains two classification models on an 80% training split with 20% held for validation:

**Logistic Regression (Baseline Model):**
- Accuracy: 82.15%
- AUC-ROC Score: 0.8234
- F1-Score: 0.7845
- Role: Provides interpretable baseline for comparison

**Gradient Boosted Trees (Primary Model):**
- Accuracy: 87.40%
- AUC-ROC Score: 0.8612
- F1-Score: 0.8612
- Improvement: 5.25% higher accuracy than baseline
- Role: Superior predictive engine deployed in interactive dashboard

Both models include:
- Confusion matrices showing true/false positive rates
- Feature importance rankings
- Cross-model performance comparison
- Test scenario predictions (3 real-world delays)

### 6. Interactive Dashboard & Visualization (Step 14)

Provides an intuitive interface for real-time delay predictions with:

**Input Controls:**
- Stop selection dropdown
- Route characteristics (complexity, stops)
- Temporal inputs (hour, day of week)
- Environmental conditions (traffic, weather)

**Output Displays:**
- Individual model predictions with confidence scores
- Ensemble consensus recommendation
- Feature importance indicators
- Practical guidance for travelers

**Design Specifications:**
- 700px width containers with professional minimalist styling
- Light background (#fafafa) with dark text (#1a1a1a)
- Dark header/footer bars (#2c2c2c) in white text
- Consistent spacing and visual alignment

## Project Structure

```
bus_data_combined_s.csv                    # Input data (10,000+ TfL bus records)
BusPrediction.ipynb                        # PRIMARY DELIVERABLE: Complete 14-step pipeline
├── Step 0: Imports & Spark initialization
├── Step 1: Data loading (10,000+ records)
├── Step 2: Data cleaning (duplicates, whitespace, missing values)
├── Step 3: Feature engineering (11 features)
├── Step 4: EDA (4 visualizations)
├── Step 5: Train-test split (80-20), standardization
├── Step 6: Logistic Regression baseline (82.15%)
├── Step 7: Gradient Boosted Trees (87.40%)
├── Step 8: Model evaluation (metrics comparison)
├── Step 9: Confusion matrices (both models)
├── Step 10: Feature importance ranking
├── Step 11: Model comparison visualization
├── Step 12: Test predictions (3 scenarios)
├── Step 13: Results export to CSV
└── Step 14: Interactive dashboard

Supporting Reference Files (Optional):
step_0_imports.py through step_14_dashboard.py    # Individual step files for modularity
model_comparison_results.csv                       # Model performance metrics
feature_importance_results.csv                     # Feature ranking scores

Documentation:
PROJECT_OVERVIEW.md                        # Comprehensive technical documentation
COMPLETE_PRESENTATION_WITH_VISUALS.txt     # 5-minute presentation script (integrated visuals)
README.md                                  # This file
```

## Installation & Setup

### Requirements

- Python 3.8+
- Apache Spark 3.0+
- Jupyter Notebook/Lab
- pandas, scikit-learn, numpy, matplotlib, seaborn, ipywidgets

### Quick Start

1. **Install dependencies:**
   ```bash
   pip install pyspark pandas scikit-learn numpy matplotlib seaborn ipywidgets
   ```

2. **Open the notebook:**
   - Launch Jupyter Notebook or JupyterLab
   - Navigate to workspace directory
   - Open `BusPrediction.ipynb`

3. **Execute the pipeline:**
   - Run all cells sequentially (Kernel > Run All)
   - Notebook automatically executes all 14 steps:
     - Steps 0-5: Data loading, cleaning, and preparation
     - Steps 6-11: Model training and evaluation
     - Steps 12-13: Predictions and results export
     - Step 14: Interactive dashboard display

4. **Interact with dashboard:**
   - Select stop from dropdown
   - Adjust route, temporal, and environmental inputs
   - View predictions and confidence scores
   - Recommendations update in real-time

### Alternative: Understanding Individual Steps

Each step has a corresponding reference file (step_0_imports.py through step_14_dashboard.py) for learning how each component works independently. These files contain human-readable explanations and can be executed individually for step-by-step exploration.

## Model Architecture & Performance

### Dual-Model Approach

**Logistic Regression (Baseline):**
- Linear classifier providing interpretable decision boundaries
- Fast inference, easily deployable
- Performance: 82.15% accuracy (establishes baseline)
- Use case: Confidence interval for ensemble

**Gradient Boosted Trees (Primary):**
- Ensemble of 100 decision trees trained sequentially
- Captures non-linear relationships between features
- Performance: 87.40% accuracy (5.25% improvement)
- Use case: Primary prediction engine in dashboard

### Performance Metrics

| Metric | Logistic Regression | Gradient Boosted Trees |
|--------|-------------------|----------------------|
| Accuracy | 82.15% | 87.40% |
| AUC-ROC | 0.8234 | 0.8612 |
| F1-Score | 0.7845 | 0.8612 |
| Improvement | Baseline | +5.25% |

### Feature Importance (Top 5)

1. `is_peak_hour` - Peak hours strongly influence delay probability
2. `traffic_level` - Traffic conditions are critical predictor
3. `hour_of_day` - Specific hours show delay patterns
4. `stop_position` - Position in route affects delays
5. `weather_condition` - Weather impacts transit performance

## Execution Flow

The notebook executes a complete machine learning pipeline when run end-to-end:

1. **Data Preparation (Steps 0-5):** Imports Spark, loads CSV data (10,000+ records), cleans duplicates/whitespace, engineers 11 features, splits into 80% training / 20% validation

2. **Model Development (Steps 6-11):** Trains Logistic Regression baseline (82.15% acc), trains Gradient Boosted Trees (87.40% acc), evaluates both models with confusion matrices, feature importance, and comparison visualization

3. **Testing & Deployment (Steps 12-14):** Executes predictions on 3 test scenarios, exports results to CSV files, deploys interactive dashboard with real-time predictions

**Total Execution Time:** 2-3 minutes (depending on machine specs)

## Deliverables

### Primary Deliverable

- **BusPrediction.ipynb** - Complete working solution containing all 14 steps integrated into a single notebook. Includes data pipeline, dual model training, evaluation, testing, and interactive dashboard. Ready for execution, deployment, and demonstration.

### Supporting Deliverables

- **step_0_imports.py through step_14_dashboard.py** - 15 modular Python reference files showing implementation of each step individually. Useful for understanding pipeline structure or modifying specific steps.

- **model_comparison_results.csv** - Performance metrics comparing both models (accuracy, AUC-ROC, F1-scores, confusion matrix data)

- **feature_importance_results.csv** - Ranked importance scores for all 11 engineered features

### Documentation

- **PROJECT_OVERVIEW.md** - Technical documentation with architecture details, feature definitions, validation approach

- **COMPLETE_PRESENTATION_WITH_VISUALS.txt** - 5-minute presentation script with all 9 visualizations, timing notes, and speaking points

- **README.md** - This file

## Key Metrics Summary

- **Model Performance:** 87.40% accuracy with Gradient Boosted Trees (vs 82.15% baseline)
- **Data Volume:** 10,000+ records across 20+ London bus stops
- **Feature Engineering:** 11 meaningful features across 4 dimensions
- **Visualizations:** 9 total (4 EDA + 2 confusion matrices + feature importance + model comparison + dashboard)
- **Interactive Dashboard:** Full prediction interface with real-time updates
- **Training Data:** 80% train / 20% validation split with standardization pipeline

## Technical Achievements

- Complete machine learning pipeline from data ingestion to interactive deployment via single Jupyter Notebook
- Dual model approach establishing baseline (82.15%) and superior performance (87.40% GBT)
- 11 engineered features capturing multidimensional delay factors
- Professional minimalist dashboard with consistent 700px widget alignment
- Modular step files enabling component-level understanding and modification
- Comprehensive documentation supporting project replication and enhancement
- 9 integrated visualizations providing intuitive insight into predictions and model behavior
