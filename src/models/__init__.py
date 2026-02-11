# Machine learning models for RUL prediction
#
# Baseline models:
#   - LinearRULPredictor (baseline.py)
#   - RandomForestRUL, GradientBoostingRUL (ensemble.py)
#
# Neural network models:
#   - SimpleNeuralNetwork (neural_network.py)
#   - LSTMModel (neural_network.py)
#
# Advanced deep learning models (GTX 1650 optimized):
#   - TCNModel (tcn.py) — Primary workhorse, best ROI
#   - BiLSTMAttentionModel (bilstm_attention.py) — High quality, slower
#   - CompactTransformerModel (transformer.py) — Possible but painful
