# Physics-Informed KARST-Aware Hybrid GNN for Streamflow Prediction

**Senior ML Engineering Implementation - IBM Research**

## Project Structure

```
gnn_flowstream/
├── README.md                          # This file
├── project.md                         # Project requirements
├── requirements.txt                   # Python dependencies
├── notebooks/
│   ├── 01_data_exploration.ipynb     # EDA and data analysis
│   ├── 02_graph_construction.ipynb    # Physics-informed graph building
│   ├── 03_model_training.ipynb        # Training pipeline
│   └── 04_evaluation.ipynb            # Model evaluation and results
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders.py                # Data loading utilities
│   │   └── preprocessing.py           # Data preprocessing
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── construction.py            # Graph construction
│   │   └── physics.py                 # Physics-informed constraints
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gat_lstm.py               # GAT + LSTM hybrid model
│   │   ├── gcn_transformer.py         # GCN + Transformer variant
│   │   └── physics_layers.py          # Physics-informed layers
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                 # Training pipeline
│   │   └── metrics.py                 # Hydrological metrics
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py           # Plotting utilities
│       └── config.py                  # Configuration management
├── configs/
│   ├── phase1_seine.yaml             # Phase 1 configuration
│   └── phase2_dryvalley.yaml         # Phase 2 configuration
├── data/
│   ├── raw/                          # Raw data from Hub'Eau, SAFRAN
│   ├── processed/                    # Preprocessed data
│   └── graphs/                       # Saved graph structures
└── experiments/
    ├── runs/                         # Training runs
    ├── checkpoints/                  # Model checkpoints
    └── results/                      # Evaluation results
```

## Quick Start

### 1. Installation

```bash
# Clone repository
cd gnn_flowstream

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
# Place your data in the data/raw directory:
# - Hub'Eau hydrometric data (JSON/CSV)
# - SAFRAN meteorological reanalysis data
# - Spatial data (DEM, land use, river networks)

python src/data/preprocessing.py --config configs/phase1_seine.yaml
```

### 3. Graph Construction

```bash
# Build physics-informed river network graph
python src/graph/construction.py --config configs/phase1_seine.yaml
```

### 4. Model Training

```bash
# Train GAT-LSTM hybrid model
python src/training/trainer.py \
    --config configs/phase1_seine.yaml \
    --model gat_lstm \
    --epochs 200 \
    --batch_size 32
```

### 5. Evaluation

```bash
# Evaluate on gauged and ungauged stations
python src/training/evaluate.py \
    --checkpoint experiments/checkpoints/best_model.pt \
    --test_data data/processed/test_graph.pt
```

## Key Features

### 🌊 Physics-Informed Graph Construction
- **Surface connectivity**: River network topology (upstream-downstream)
- **Subsurface connectivity**: Karst groundwater pathways
- **Edge attributes**: Distance, elevation gradient, travel time, drainage area
- **Node features**: Basin characteristics, meteorological forcing, hydrometric observations

### 🧠 Hybrid Spatio-Temporal GNN
- **Spatial**: Graph Attention Network (GAT) for adaptive spatial aggregation
- **Temporal**: LSTM for sequential hydrological dynamics
- **Physics constraints**: Mass conservation, flow continuity, groundwater-surface interaction

### 📊 Multi-Modal Input Integration
- **Hydrometric**: H, Q, QmnJ, QmM, QINnJ, QINM, QIXnJ, QIXM
- **Meteorological**: 12 SAFRAN variables (precipitation, temperature, ET, etc.)
- **Spatial**: DEM, land use, drainage area, slope

### 🎯 Ungauged Station Prediction
- Virtual station generation using spatial interpolation
- Transfer learning for dry valley systems (Phase 2)
- Zero-inflated loss functions for intermittent flow

## Model Architecture

```
Input: Multi-modal time series + River network graph
       │
       ├─→ Node Features [B, N, T, F_node]
       │   ├─ Hydrometric: H, Q, statistics
       │   ├─ Meteorological: SAFRAN 12 variables
       │   └─ Static: Basin characteristics
       │
       ├─→ Edge Features [B, E, F_edge]
       │   ├─ Distance, elevation gradient
       │   └─ Hydrogeological connectivity
       │
       ↓
   ┌─────────────────────────────┐
   │  Temporal Encoding (LSTM)   │
   │  [B, N, T, F_node] → [B, N, H_lstm]
   └─────────────────────────────┘
       │
       ↓
   ┌─────────────────────────────┐
   │  Spatial Aggregation (GAT)  │
   │  [B, N, H_lstm] → [B, N, H_gat]
   │  • Multi-head attention
   │  • River topology awareness
   │  • Karst connectivity
   └─────────────────────────────┘
       │
       ↓
   ┌─────────────────────────────┐
   │  Physics-Informed Layer     │
   │  • Mass conservation        │
   │  • Flow continuity          │
   │  • GW-SW interaction        │
   └─────────────────────────────┘
       │
       ↓
   ┌─────────────────────────────┐
   │  Output Decoder             │
   │  [B, N, H_gat] → [B, N, T_pred, 1]
   └─────────────────────────────┘
       │
       ↓
Output: Streamflow prediction Q(t)
```

## Evaluation Metrics

### Standard Hydrological Metrics
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **NSE**: Nash-Sutcliffe Efficiency
- **KGE**: Kling-Gupta Efficiency
- **R²**: Coefficient of Determination

### Custom KARST Metrics
- **Flash response accuracy**: Performance during intense precipitation
- **Low-flow bias**: Prediction accuracy during dry periods
- **Peak timing error**: Temporal lag in flash flood prediction

## Configuration

### Phase 1: Seine Watershed (`configs/phase1_seine.yaml`)

```yaml
data:
  stations: 27  # La Risle + La Eure
  date_range: [1960-01-01, 2026-01-01]
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15

graph:
  node_types: [gauged, ungauged, subbasin]
  edge_types: [river, karst]
  max_distance: 50  # km
  karst_connectivity_threshold: 0.3

model:
  spatial:
    type: GAT
    hidden_dim: 128
    num_layers: 3
    heads: 4
  temporal:
    type: LSTM
    hidden_dim: 128
    num_layers: 2
  physics:
    mass_conservation_weight: 0.1
    flow_continuity_weight: 0.1

training:
  epochs: 200
  batch_size: 32
  learning_rate: 0.001
  optimizer: AdamW
  scheduler: CosineAnnealing
  early_stopping_patience: 20
```

## Expected Results (Phase 1)

### Gauged Stations
- **NSE**: >0.75
- **KGE**: >0.70
- **RMSE**: <2.0 m³/s

### Ungauged Virtual Stations
- **NSE**: >0.60
- **KGE**: >0.55
- **RMSE**: <3.0 m³/s

## Research Deliverables

✅ Integrated hydrological-meteorological database
✅ Physics-based river network graph
✅ Virtual station generation methodology
✅ Baseline and hybrid GNN models
✅ Ungauged streamflow prediction framework
✅ Interactive Decision Support System (DSS)
📄 Two research articles (karst hydrology + dry valley extension)

## Next Steps (Phase 2: Dry Valley Extension)

1. **Transfer Learning**: Adapt Seine model to dry valley systems
2. **Intermittent Flow Handling**: Zero-inflated loss functions
3. **Sinkhole Data Integration**: Incorporate karst feature data
4. **Event-Based Learning**: Focus on flash flood events
5. **Temporal Classification**: Flow/no-flow prediction

## References

- PyTorch Geometric Documentation
- Karst Hydrology Literature
- Hub'Eau API Documentation
- SAFRAN Meteorological Reanalysis

## License

IBM Research - Academic Use

## Contact

Senior ML Engineer, IBM Research
