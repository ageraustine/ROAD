# Research Framework and Expected Deliverables

Our main aim is to develop a **Physics-informed, KARST-aware Hybrid Graph Neural Network (GNN)** for streamflow prediction in ungauged catchments by integrating hydrological observations, meteorological forcing, river network topology, basin characteristics, and subsurface groundwater connectivity. The framework is designed as a two-phase study that shares a common graph-based learning architecture while addressing two contrasting hydrological environments.

## Before Starting

You should have a brief overview of KARST characteristics in the Normandie part of downstream Seine watershed. Challenges for KARST-dominated watershed streamflow prediction are as follows:

- Highly nonlinear, temporally lagged hydrological responses that deviate from the assumptions
- Streamflow may exhibit rapid flash responses during intense precipitation events
- Complicate predictions of downstream streamflow response
- Contrasting dynamics complicate hydrograph prediction — **reduce the reliability of conceptual rainfall–runoff models**

## Phase 1: Seine Watershed

Phase 1 focuses on the Seine watershed, initially considering the **La Risle** and **La Eure** river systems before extending to the wider basin. A comprehensive hydrological database will be developed by integrating hydrometric observations, meteorological data (e.g., SAFRAN/Météo-France), digital elevation models, land use, river networks, watershed characteristics, and karst hydrogeological information (more input data can be included in the future, so the model should be scalable in this aspect).

We have done some initial data collection for the 27 stations on the La Risle and La Eure rivers. The hydrometric data (1960–2026) were downloaded in both JSON and CSV formats using the Hub'Eau Hydrométrie API. The dataset includes:

### Real-time Observations

Measurements updated approximately every five minutes, covering about one month of recent data. The two main variables are:

- **H (Water level):** the measured height of the river at the monitoring station.
- **Q (River discharge):** the volume of water flowing through the river cross-section.

### Historical Hydrological Statistics

Processed long-term indicators, including:

- **QmnJ / QmM:** daily and monthly mean discharge
- **QINnJ / QINM:** daily and monthly minimum instantaneous discharge
- **QIXnJ / QIXM:** daily and monthly maximum instantaneous discharge
- **HIXnJ / HIXM:** daily and monthly maximum instantaneous water level

### SAFRAN Meteorological Reanalysis Data

In addition, we have collected the SAFRAN meteorological reanalysis data (1960–2026) for each station using their geographic coordinates. The selected variables include:

- **DLI_Q:** atmospheric radiation
- **DRAINC_Q:** drainage
- **ETP_Q:** potential evapotranspiration
- **FF_Q:** wind speed
- **HU_Q:** relative humidity
- **PRELIQ_Q:** liquid precipitation
- **PRENEI_Q:** solid precipitation (snowfall)
- **RESR_NEIGE_Q:** snowpack water equivalent
- **RUNC_Q:** runoff
- **SSI_Q:** visible radiation
- **SWI_Q:** soil moisture index
- **T_Q:** average air temperature

These datasets are stored locally in both JSON and CSV formats for further processing and analysis.

These datasets will be used to construct a physics-informed river graph where nodes represent gauged stations, hydrological sub-basins, and synthetically generated virtual ungauged stations, while edges represent both surface river connectivity and subsurface karst groundwater pathways. Edge attributes will incorporate physically meaningful information such as river distance, elevation gradient, travel time, drainage area, and hydrogeological connectivity.

A hybrid spatio-temporal GNN combining graph learning with temporal sequence modeling (e.g., GCN/GAT with LSTM or Transformer, or STGNN) will be developed and constrained using hydrological principles, including mass conservation, flow continuity, and groundwater–surface water interactions. The model will be trained using observed streamflow and validated at gauged stations before being evaluated for prediction at ungauged virtual stations.

### Graph Construction (Physics-Informed)

- **Nodes:** gauged stations, virtual ungauged stations, and hydrological sub-basins
- **Edges:** surface river connectivity (upstream–downstream), karst groundwater connections (subsurface flow paths)
- **Edge weights:** informed by distance, elevation gradient, hydrogeological connectivity

**Important point in model architecture — Physics-informed constraints:** mass conservation, flow continuity, or groundwater–surface interaction consistency.

**Output:** streamflow prediction at gauged stations (validation) as well as ungauged virtual stations (generalization test).

## Before Starting Phase 2

A **dry valley** is a river channel with little or no permanent surface flow. It is very common in karst limestone regions. Here, water infiltrates underground instead of flowing on the surface, creating hidden hydrological pathways.

**Why is this important for this study?**

- Hidden groundwater routing
- Missing surface observations
- Complex runoff generation
- Increased uncertainty at ungauged stations

## Phase 2: Dry Valley Extension

Phase 2 extends the framework to dry valley systems, where hydrological observations are sparse, intermittent, or unavailable. To address data scarcity, we might use sinkhole data, or synthetic hydrological time series can be generated using physically constrained rainfall–runoff simulations or stochastic hydrological models.

The pretrained Seine model will then be adapted through transfer learning and domain adaptation techniques to account for intermittent flow conditions. The graph structure will be modified to represent ephemeral channels, disconnected flow paths, and stronger rainfall–runoff dependence, while the learning framework will be enhanced using zero-inflated loss functions, event-based learning, and temporal flow/no-flow classification to improve prediction of dry periods, transition events, and flash-flow responses.

## Research Workflow

The research follows a progressive workflow consisting of data collection, river network analysis, dry valley and karst characterization, graph construction, virtual station generation, baseline GNN development, hybrid physics-informed GNN modeling, model validation, and ungauged streamflow prediction. Performance will be assessed using standard hydrological metrics, including RMSE, MAE, NSE, KGE, and R², together with sensitivity and uncertainty analyses.

### Overall Research Workflow (Diagram)

```
                          Data Collection
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
    Hydrometric           Meteorological         Spatial Data and
    (Hub'Eau)                (SAFRAN)          Watershed Attributes
        │                       │            (DEM, Land Use, River)
        └───────────────────────┼───────────────────────┘
                                │
                        Data Preprocessing
                                │
                    River Network Construction
                                │
                     Dry Valley Identification
                                │
                Basin Feature Extraction (Nodes & Edges)
                                │
                 Synthetic Virtual Station Generation
                                │
                   Graph Construction (Physics-Based)
                                │
                ┌───────────────┴───────────────┐
                │                               │
        Baseline GNN                     Hybrid GNN
   (River Network + Dry Valley)    (River Network + Dry Valley)
    (Streamflow + Network)      (+ Remote Sensing + Physics-based
                                         Constraints)
                │                               │
                └───────────────┬───────────────┘
                                │
                Model Training & Hyperparameter Tuning
                                │
                    Validation at Gauged Stations
                                │
                 Prediction at Ungauged Virtual Stations
                                │
                  Performance & Sensitivity Analysis
                                │
                     Hydrological Interpretation
```

## Major Deliverables

1. An integrated hydrological and meteorological geodatabase.
2. A physics-based river network graph with physically meaningful node and edge attributes.
3. Dry valley and karst connectivity assessment.
4. A methodology for synthetic virtual station generation.
5. Baseline and hybrid GNN models for streamflow prediction.
6. A novel physics-informed graph learning framework for karst hydrology.
7. An ungauged streamflow prediction methodology, scalable for other input data and transferable to other watersheds.
8. **An interactive AI-based Decision Support System (DSS)** that enables users to create virtual gauging stations anywhere along the river network or dry valleys and obtain real-time streamflow predictions through the trained physics-informed Hybrid GNN.
9. Two research articles generated from the framework — one on the novel physics-informed graph learning framework for karst hydrology, and another on its novel dry valley extension using transfer learning.

Collectively, this framework advances graph-based hydrological modeling by integrating physical laws, karst hydrogeology, and artificial intelligence into a unified methodology for distributed streamflow prediction in complex and data-scarce river basins.
