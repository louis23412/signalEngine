# NeuralSignalEngine

## Overview
NeuralSignalEngine is a JavaScript-based tool for generating simulated trading signals using an ensemble of 128 shallow transformer neural networks, collectively named `HiveMind`, integrated with technical indicators computed by `IndicatorProcessor` and Q-learning. It processes financial market candlestick data to generate buy/hold signals with trade sizing, target prices, stop-loss levels, and entry prices. All trades are simulated and not executed in real markets.

## Features
- **HiveMind Ensemble**: Utilizes 128 shallow transformer models with multi-head attention, collaborating via weight sharing to predict robust trading signals.
- **Technical Indicators**: Computes Relative Strength Index (RSI), Moving Average Convergence Divergence (MACD), Average True Range (ATR), and volume Z-score to analyze market conditions.
- **Market Phase Analysis**: Identifies trending, ranging, or volatile markets based on RSI and MACD.
- **Q-Learning**: Optimizes trading decisions using a Q-table based on historical patterns.
- **SQLite Databases**: Persists `HiveMind` parameters in `hivemind_state.db` and trading patterns, simulated trades, and candlestick data in `neural_engine.db`.
- **Dynamic Thresholding**: Adapts confidence thresholds based on market volatility, using a 60% base threshold for buy signals.
- **Signal Output**: Produces a single `suggestedAction` ("buy" or "hold") with `entryPrice`, `sellPrice`, `stopLoss`, `multiplier`, `expectedReward`, `rawConfidence`, `rawThreshold`, `filteredConfidence`, and `filteredThreshold`.

## Requirements
- **Node.js**: Version 16 or higher.
- **Dependencies**: Install `better-sqlite3` via `npm install better-sqlite3`.
- **Writable Directory**: Ensure the `state` directory (auto-created in project root) is writable for SQLite databases (`neural_engine.db` and `hivemind_state.db`).

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/louis23412/signalEngine.git
   cd signalEngine
   ```
2. Install dependencies:
   ```bash
   npm install better-sqlite3
   ```

## Project Structure
- **`src/`**:
  - `neuralSignalEngine.js`: Core logic with `NeuralSignalEngine` class, integrating `HiveMind` and `IndicatorProcessor`.
  - `hiveMind.js`: Implementation of the `HiveMind` ensemble with 128 transformer models and persistence logic.
  - `indicatorProcessor.js`: Computes technical indicators (RSI, MACD, ATR, volume Z-score) and market phase.
- **`test/`**:
  - `testEngine.js`: Script for training and testing the engine.
  - `candles.jsonl`: Input file with candlestick data (must be created).
- **`state/`**: Directory for SQLite databases (`neural_engine.db` and `hivemind_state.db`).
- **`README.md`**: Project documentation.

## Technical Indicators
The `IndicatorProcessor` computes the following indicators, requiring at least 11 valid candles:
- **Relative Strength Index (RSI)**:
  - **Period**: 14
  - **Description**: Measures momentum (0–100). Values >70 indicate overbought, <30 oversold.
  - **Output**: Array of RSI values; latest value used for market phase.
- **Moving Average Convergence Divergence (MACD)**:
  - **Parameters**: Fast EMA (8), Slow EMA (21), Signal Line (5)
  - **Description**: Tracks trend direction and strength via MACD line, signal line, and histogram.
  - **Output**: Array of objects `{ MACD, signal, histogram }`.
- **Average True Range (ATR)**:
  - **Period**: 14
  - **Description**: Measures volatility based on true range (high-low, high-close, low-close).
  - **Output**: Array of ATR values; latest value used for stop-loss and target price calculations.
- **Volume Z-Score**:
  - **Description**: Normalizes volume relative to mean and standard deviation, capped at ±3.
  - **Output**: Single value for the latest candle.
- **Market Phase**:
  - **Logic**:
    - **Trending**: MACD > signal line.
    - **Ranging**: RSI between 30 and 70.
    - **Volatile**: Neither trending nor ranging.
  - **Output**: `marketPhase` ("trending", "ranging", "volatile"), `isTrending` (0 or 1), `isRanging` (0 or 1).
- **Additional Metrics**:
  - Min/max values for RSI, MACD (MACD - signal), and ATR over the input candles.

### Example Indicator Output
```javascript
{
  close: [100, 102, ...], // Array of closing prices
  high: [105, 107, ...], // Array of high prices
  low: [95, 97, ...], // Array of low prices
  volume: [1000, 1200, ...], // Array of volumes
  lastClose: 102, // Latest closing price
  volumeZScore: 1.2, // Volume Z-score for latest candle
  rsi: [50, 52, ...], // RSI values
  macd: [{ MACD: 0.1, signal: 0.05, histogram: 0.05 }, ...], // MACD values
  atr: [1.5, 1.6, ...], // ATR values
  isTrending: 1, // 1 if trending, 0 otherwise
  isRanging: 0, // 1 if ranging, 0 otherwise
  marketPhase: "trending", // "trending", "ranging", or "volatile"
  rsiMin: 45, // Minimum RSI
  rsiMax: 75, // Maximum RSI
  macdMin: -0.2, // Minimum MACD - signal
  macdMax: 0.3, // Maximum MACD - signal
  atrMin: 1.0, // Minimum ATR
  atrMax: 2.0 // Maximum ATR
}
```

## Training and Testing
The `testEngine.js` script trains the `HiveMind` ensemble and tests signal generation using candlestick data from `test/candles.jsonl`. Training requires at least 100,000 candles covering varied market conditions for reliable signals. The `HiveMind` learns incrementally, updating transformer weights, ensemble weights, and the Q-table based on simulated trade outcomes.

The script maintains a sliding window of the latest 1,000 candles, mimicking real-world usage where the engine processes each new candle to generate a signal. This ensures efficient memory usage and focuses on recent market data.

### Steps
1. **Prepare `test/candles.jsonl`**:
   Create a `candles.jsonl` file in `test/`, with each line as a JSON object representing a candlestick. Example:
   ```json
   {"timestamp":1672531200000,"open":100.0,"high":105.0,"low":95.0,"close":102.0,"volume":1000}
   ```
   - **Fields**: `timestamp` (milliseconds, e.g., from `Date.now()`), `open`, `high`, `low`, `close` (numbers), `volume` (non-negative number).
   - **Validation**: `IndicatorProcessor` requires numeric fields and non-negative volume. Invalid candles are filtered out, but at least 11 valid candles are needed.
   - **Tip**: Use a JSONL validator to ensure correct formatting.
2. **Run `testEngine.js`**:
   ```bash
   node test/testEngine.js
   ```
   - Processes `test/candles.jsonl`, updates the sliding window, and generates signals.
   - Displays progress, processing time, and ETA.
   - Example signal output:
     ```javascript
     {
       suggestedAction: "buy", // "buy" or "hold"
       multiplier: 1.789, // Trade sizing multiplier
       entryPrice: 102.00, // Last closing price
       sellPrice: 105.23, // Target sell price
       stopLoss: 98.45, // Stop-loss price
       expectedReward: 0.029, // Expected reward ratio
       rawConfidence: 75.123, // Raw confidence score
       rawThreshold: 60.456, // Raw dynamic threshold
       filteredConfidence: 70.234, // Adjusted confidence score
       filteredThreshold: 62.789 // Adjusted buy threshold
     }
     ```
3. **Training Process**:
   - `HiveMind` trains on simulated trade outcomes (positive: 1, negative: 0).
   - Transformers periodically share weights, with high-performing models contributing more.
   - Large datasets (100,000+ candles) improve signal accuracy.
   - **Minimum Candles**: 11 candles required for indicators; 100,000+ recommended for training.

## Usage
To use the engine programmatically:
1. **Import and Initialize**:
   ```javascript
   import NeuralSignalEngine from './src/neuralSignalEngine.js';
   const engine = new NeuralSignalEngine();
   ```
2. **Provide Candlestick Data**:
   ```javascript
   const candles = [
     { timestamp: 1672531200000, open: 100.0, high: 105.0, low: 95.0, close: 102.0, volume: 1000 },
     // At least 11 candles
   ];
   ```
3. **Generate Signal**:
   ```javascript
   const signal = engine.getSignal(candles);
   console.log(signal);
   ```

## Configuration
Adjust parameters in `src/neuralSignalEngine.js` within the `#config` object:
- `minMultiplier`: Minimum trade sizing (default: 1).
- **maxMultiplier**: Maximum trade sizing (default: 2.5).
- **baseConfidenceThreshold**: Base confidence threshold (default: 60).
- **atrFactor**: ATR multiplier for target prices (default: 10).
- **stopFactor**: ATR multiplier for stop-loss (default: 2.5).
- **learningRate**: Q-table learning rate (default: 0.25).

For advanced users:
- **HiveMind**: Modify parameters in `src/hiveMind.js` (e.g., `#learningRate: 0.01`, `#dropoutRate: 0.15`, `#numHeads: 4`). Test thoroughly, as changes impact model convergence.
- **IndicatorProcessor**: Adjust indicator periods in `src/indicatorProcessor.js` (e.g., RSI period: 14, MACD: 8/21/5). Ensure sufficient candles for new periods (e.g., slow EMA + signal period + 1).

## Database Schemas
The project uses two SQLite databases:

### `neural_engine.db`
Managed by `NeuralSignalEngine`, stores:
- **`qtable`**: Q-learning values (`state_key`, `buy`, `hold`).
- **`patterns`**: Trading patterns (`bucket_key`, `features`, `score`, `usage_count`, `win_count`).
- **`open_trades`**: Simulated trades (`timestamp`, `sellPrice`, `stopLoss`, `entryPrice`, `confidence`, `patternScore`, `features`, `stateKey`, `dynamicThreshold`).
- **`candles`**: Historical candles (`timestamp`, `open`, `high`, `low`, `close`, `volume`).

### `hivemind_state.db`
Managed by `HiveMind`, stores transformer parameters:
- **`metadata`**: Training step count (`key`, `value`).
- **`ensemble_weights`**: Weights for each transformer (`idx`, `weight`).
- **`performance_scores`**: Performance scores (`idx`, `score`).
- **`agreement_scores`**: Agreement scores (`idx`, `score`).
- **`specialization_scores`**: Specialization scores (`idx`, `score`).
- **`historical_performance`**: Historical performance (`idx`, `step`, `score`).
- **`trust_scores_history`**: Historical trust scores (`idx`, `step`, `score`).
- **`adaptive_learning_rate`**: Adaptive learning rates (`idx`, `rate`).
- **`attention_weight_matrix`**: Attention weights (`idx`, `row`, `value`).
- **`attention_bias`**: Attention biases (`idx`, `row`, `value`).
- **`specialization_weights`**: Specialization weights (`idx`, `row`, `col`, `value`).
- **`attention_memory`**: Attention memory (`idx`, `window`, `seq`, `dim`, `value`).
- **`transformers`**: Transformer weights (`idx`, `layer`, `weight_type`, `row`, `col`, `value`).
- **`transformer_biases`**: Transformer biases (`idx`, `layer`, `bias_type`, `row`, `value`).
- **`transformer_layer_norm`**: Layer normalization (`idx`, `layer`, `norm_type`, `row`, `value`).
- **`momentum_weights`**: Momentum for weights (`idx`, `layer`, `weight_type`, `row`, `col`, `value`).
- **`momentum_biases`**: Momentum for biases (`idx`, `layer`, `bias_type`, `row`, `value`).
- **`gradient_accumulation`**: Gradient accumulation for weights (`idx`, `layer`, `weight_type`, `row`, `col`, `value`).
- **`gradient_biases`**: Gradient accumulation for biases (`idx`, `layer`, `bias_type`, `row`, `value`).

## Performance Tips
- **Data Size**: Use 100,000+ candles for robust training.
- **Database Management**: The engine retains 1,000 recent candles in `neural_engine.db`. Clear old data if the database grows large. Monitor `hivemind_state.db` size due to extensive transformer parameters.
- **Input Validation**: Ensure `test/candles.jsonl` has valid JSONL with numeric fields and non-negative volume.
- **Sliding Window**: Adjust the 1,000-candle window in `neuralSignalEngine.js` if needed, balancing memory and performance.
- **Indicator Periods**: Ensure sufficient candles for indicator calculations (e.g., MACD requires 26 candles: slow EMA 21 + signal 5).

## Troubleshooting
- **Invalid `candles.jsonl`**:
  - **Error**: `IndicatorProcessor` returns `{ error: true }` if <11 valid candles or if fields are non-numeric or volume is negative.
  - **Fix**: Validate JSONL, ensuring `timestamp`, `open`, `high`, `low`, `close` are numbers, and `volume` ≥ 0.
- **Database Corruption**:
  - **Symptoms**: Errors accessing `neural_engine.db` or `hivemind_state.db`.
  - **Fix**: Delete the database to reset (loses learned parameters; retraining required).
- **Unreliable Signals**:
  - **Cause**: Insufficient training data.
  - **Fix**: Use 100,000+ candles with varied market conditions.
- **Slow Performance**:
  - **Cause**: Large candle datasets or large sliding window.
  - **Fix**: Monitor `testEngine.js` output for processing times; reduce window size or candle volume.

## Notes
- **Minimum Candles**: 11 candles required for indicators; 100,000+ recommended for training.
- **Persistence**: `neural_engine.db` and `hivemind_state.db` allow resuming learning across sessions. Deleting `hivemind_state.db` resets transformer weights, requiring retraining.
- **Node.js Compatibility**: `HiveMind` uses `import.meta.dirname`, which may require Node.js experimental features (e.g., `--experimental-import-meta-resolve`). Use a compatible Node.js version or adjust path resolution if errors occur.
- **Indicator Customization**: Modify periods in `indicatorProcessor.js` for different market conditions, but ensure sufficient candles.

## Disclaimer
This tool is for educational and research purposes only. It generates simulated signals, not real trades. Signals may not predict future market performance. Use at your own risk and consult a financial advisor before making decisions. The author is not liable for any losses.

## License
MIT License. See [LICENSE](LICENSE) for details.