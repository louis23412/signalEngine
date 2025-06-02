import fs from 'fs';
import path from 'path';
import Database from 'better-sqlite3';

const directoryPath = path.join(import.meta.dirname, '..', 'state')

const truncateToDecimals = (value, decimals) => {
  const factor = Math.pow(10, decimals);
  return Math.floor(value * factor) / factor;
};

const isValidNumber = (value) => {
  if (value == null) return false;
  const num = typeof value === 'string' ? Number(value) : value;
  return typeof num === 'number' && !isNaN(num) && isFinite(num);
};

const sigmoid = (x) => isValidNumber(x) ? 1 / (1 + Math.exp(-Math.min(Math.max(x, -100), 100))) : 0;

const softmax = (arr) => {
  if (!arr.every(isValidNumber)) return arr.map(() => 1 / arr.length);
  const max = Math.max(...arr);
  const exp = arr.map(x => Math.exp(x - max));
  const sum = exp.reduce((a, b) => a + b, 0) || 1;
  return exp.map(x => x / sum);
};

class HiveMind {
  #inputSize = 6;
  #hiddenSize = 16;
  #outputSize = 1;
  #numHeads = 4;
  #numLayers = 2;
  #feedForwardSize = 32;
  #dropoutRate = 0.15;
  #learningRate = 0.005;
  #ensembleSize = 128;
  #transformers = [];
  #ensembleWeights = [];
  #weightSharingRate = 0.1;
  #performanceScores = Array(this.#ensembleSize).fill(0);
  #agreementScores = Array(this.#ensembleSize).fill(0);
  #trainingStepCount = 0;
  #historicalPerformance = Array(this.#ensembleSize).fill().map(() => []);
  #momentumWeights = Array(this.#ensembleSize).fill().map(() => ({
    outputWeights: Array(this.#hiddenSize).fill().map(() => Array(this.#outputSize).fill(0)),
    outputBias: Array(this.#outputSize).fill(0),
    attentionWeights: Array(this.#numLayers).fill().map(() => ({
      Wq: Array(this.#hiddenSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
      Wk: Array(this.#hiddenSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
      Wv: Array(this.#hiddenSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
      Wo: Array(this.#hiddenSize).fill().map(() => Array(this.#hiddenSize).fill(0))
    })),
    ffnWeights: Array(this.#numLayers).fill().map(() => ({
      W1: Array(this.#hiddenSize).fill().map(() => Array(this.#feedForwardSize).fill(0)),
      W2: Array(this.#feedForwardSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
      b1: Array(this.#feedForwardSize).fill(0),
      b2: Array(this.#hiddenSize).fill(0)
    })),
    layerNormWeights: Array(this.#numLayers).fill().map(() => ({
      gamma1: Array(this.#hiddenSize).fill(0),
      beta1: Array(this.#hiddenSize).fill(0),
      gamma2: Array(this.#hiddenSize).fill(0),
      beta2: Array(this.#hiddenSize).fill(0)
    }))
  }));
  #attentionWeightMatrix = Array(this.#ensembleSize).fill().map(() => Array(this.#hiddenSize).fill(0));
  #attentionBias = Array(this.#ensembleSize).fill().map(() => Array(this.#hiddenSize).fill(0));
  #diversityWeight = 0.2;
  #maxPerformanceHistory = 100;
  #contextWindow = 10;
  #knowledgeDistillationLoss = 0.1;
  #attentionScalingFactor = 1.0;
  #gradientClippingThreshold = 2.0;
  #swarmIntelligenceFactor = 0.3;
  #adaptiveLearningRate = Array(this.#ensembleSize).fill(this.#learningRate);
  #trustScoresHistory = Array(this.#ensembleSize).fill().map(() => []);
  #maxTrustHistory = 50;
  #attentionMemory = Array(this.#ensembleSize).fill().map(() => Array(this.#contextWindow).fill().map(() => Array(this.#hiddenSize).fill(0)));
  #specializationScores = Array(this.#ensembleSize).fill(0);
  #gradientAccumulation = Array(this.#ensembleSize).fill().map(() => ({
    outputWeights: Array(this.#hiddenSize).fill().map(() => Array(this.#outputSize).fill(0)),
    outputBias: Array(this.#outputSize).fill(0),
    attentionWeights: Array(this.#numLayers).fill().map(() => ({
      Wq: Array(this.#hiddenSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
      Wk: Array(this.#hiddenSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
      Wv: Array(this.#hiddenSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
      Wo: Array(this.#hiddenSize).fill().map(() => Array(this.#hiddenSize).fill(0))
    })),
    ffnWeights: Array(this.#numLayers).fill().map(() => ({
      W1: Array(this.#hiddenSize).fill().map(() => Array(this.#feedForwardSize).fill(0)),
      W2: Array(this.#feedForwardSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
      b1: Array(this.#feedForwardSize).fill(0),
      b2: Array(this.#hiddenSize).fill(0)
    })),
    layerNormWeights: Array(this.#numLayers).fill().map(() => ({
      gamma1: Array(this.#hiddenSize).fill(0),
      beta1: Array(this.#hiddenSize).fill(0),
      gamma2: Array(this.#hiddenSize).fill(0),
      beta2: Array(this.#hiddenSize).fill(0)
    })),
    attentionBias: Array(this.#hiddenSize).fill(0)
  }));
  #specializationWeights = Array(this.#ensembleSize).fill().map(() => Array(this.#hiddenSize).fill().map(() => Array(this.#hiddenSize).fill(0)));

  constructor() {
    const xavierInit = (rows, cols) => Array(rows).fill().map(() => Array(cols).fill().map(() => (Math.random() - 0.5) * Math.sqrt(2 / (rows + cols))));
    for (let i = 0; i < this.#ensembleSize; i++) {
      const transformer = {
        positionalEncoding: Array(this.#inputSize).fill().map((_, idx) => {
          const pos = idx / (10000 ** (2 * Math.floor(idx / 2) / this.#hiddenSize));
          return Array(this.#hiddenSize).fill().map((_, d) => d % 2 === 0 ? Math.sin(pos) : Math.cos(pos));
        }),
        attentionWeights: Array(this.#numLayers).fill().map(() => ({
          Wq: xavierInit(this.#hiddenSize, this.#hiddenSize),
          Wk: xavierInit(this.#hiddenSize, this.#hiddenSize),
          Wv: xavierInit(this.#hiddenSize, this.#hiddenSize),
          Wo: xavierInit(this.#hiddenSize, this.#hiddenSize)
        })),
        ffnWeights: Array(this.#numLayers).fill().map(() => ({
          W1: xavierInit(this.#hiddenSize, this.#feedForwardSize),
          W2: xavierInit(this.#feedForwardSize, this.#hiddenSize),
          b1: Array(this.#feedForwardSize).fill(0),
          b2: Array(this.#hiddenSize).fill(0)
        })),
        layerNormWeights: Array(this.#numLayers).fill().map(() => ({
          gamma1: Array(this.#hiddenSize).fill(1),
          beta1: Array(this.#hiddenSize).fill(0),
          gamma2: Array(this.#hiddenSize).fill(1),
          beta2: Array(this.#hiddenSize).fill(0)
        })),
        outputWeights: xavierInit(this.#hiddenSize, this.#outputSize),
        outputBias: Array(this.#outputSize).fill(0)
      };
      this.#transformers.push(transformer);
      this.#ensembleWeights.push(1 / this.#ensembleSize);
      this.#historicalPerformance[i] = [0];
      this.#trustScoresHistory[i] = [0];
      for (let j = 0; j < this.#hiddenSize; j++) {
        for (let k = 0; k < this.#hiddenSize; k++) {
          this.#specializationWeights[i][j][k] = (Math.random() - 0.5) * Math.sqrt(2 / (this.#hiddenSize + this.#hiddenSize));
        }
      }
    }
    this.#normalizeEnsembleWeights();
    for (let i = 0; i < this.#ensembleSize; i++) {
      for (let j = 0; j < this.#hiddenSize; j++) {
        this.#attentionWeightMatrix[i][j] = (Math.random() - 0.5) * Math.sqrt(2 / (this.#ensembleSize + this.#hiddenSize));
      }
    }
    for (let t = 0; t < this.#ensembleSize; t++) {
      for (let i = 0; i < this.#hiddenSize; i++) {
        this.#attentionBias[t][i] = (Math.random() - 0.5) * Math.sqrt(2 / this.#hiddenSize);
      }
    }
  }

  /**
   * Determines whether transformers should share weights based on performance variance.
   * Returns true if variance exceeds a dynamic threshold, encouraging communication
   * among transformers when performance diverges significantly.
   * @returns {boolean} True if weight sharing should occur, false otherwise.
   */
  #shouldCommunicate() {
      // Compute mean performance
      const mean = this.#performanceScores.reduce((sum, score) => 
          sum + (isValidNumber(score) ? score : 0), 0) / this.#ensembleSize;

      // Compute variance of performance scores
      const variance = this.#performanceScores.reduce((sum, score) => {
          const val = isValidNumber(score) ? score : 0;
          return sum + Math.pow(val - mean, 2);
      }, 0) / this.#ensembleSize;

      // Check for invalid variance
      if (!isValidNumber(variance)) return false;

      // Compute standard deviation
      const performanceStd = Math.sqrt(variance);

      // Dynamic threshold decreases as training progresses
      const progress = Math.min(this.#trainingStepCount / 5000, 1);
      const threshold = 0.1 * (1 - progress) + 0.05;

      return performanceStd > threshold;
  }

  /**
   * Computes ensemble weights for each transformer by integrating attention mechanisms,
   * historical performance, trust scores, and specialization scores. Processes input features
   * and transformer outputs to calculate attention scores, ensuring robust contextual understanding
   * across all input positions and transformers. Returns normalized weights summing to 1.
   * Handles invalid inputs by returning uniform weights and ensures numerical stability.
   * @param {number[]} inputs - Array of input features (length: #inputSize, typically 6).
   * @param {number[]} outputs - Array of transformer outputs (length: #ensembleSize, typically 128).
   * @returns {number[]} Array of ensemble weights (length: #ensembleSize), normalized to sum to 1.
   */
  #computeAttentionWeights(inputs, outputs) {
      // Validate inputs and outputs
      if (
          !Array.isArray(inputs) ||
          inputs.length !== this.#inputSize ||
          !inputs.every(isValidNumber) ||
          !Array.isArray(outputs) ||
          outputs.length !== this.#ensembleSize ||
          !outputs.every(isValidNumber)
      ) {
          return Array(this.#ensembleSize).fill(1 / this.#ensembleSize); // Uniform weights for invalid inputs
      }

      // Transform inputs into feature vectors for attention computation
      const inputFeatures = inputs.map(x =>
          Array(this.#hiddenSize).fill(isValidNumber(x) ? x : 0)
      );

      // Compute attention scores for each transformer
      const attentionScores = Array(this.#ensembleSize).fill(0);
      for (let t = 0; t < this.#ensembleSize; t++) {
          // Ensure attention memory is valid
          if (
              !Array.isArray(this.#attentionMemory[t]) ||
              this.#attentionMemory[t].length === 0 ||
              !this.#attentionMemory[t][0].every(row => Array.isArray(row) && row.length === this.#hiddenSize)
          ) {
              this.#attentionMemory[t] = Array(this.#contextWindow).fill().map(() =>
                  Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0))
              );
          }

          // Compute queries, keys, and values
          const queries = inputFeatures.map(row =>
              row.map((val, i) =>
                  Array(this.#hiddenSize).fill().reduce((sum, _, j) =>
                      sum + (
                          isValidNumber(val) && isValidNumber(this.#attentionWeightMatrix[t][j])
                              ? val * this.#attentionWeightMatrix[t][j]
                              : 0
                      ),
                      0
                  )
              )
          );

          const keys = inputFeatures.map(row =>
              row.map((val, i) =>
                  Array(this.#hiddenSize).fill().reduce((sum, _, j) =>
                      sum + (
                          isValidNumber(val) && isValidNumber(this.#attentionWeightMatrix[t][j])
                              ? val * this.#attentionWeightMatrix[t][j]
                              : 0
                      ),
                      0
                  )
              )
          );

          const values = Array(this.#inputSize).fill().map(() =>
              Array(this.#hiddenSize).fill(isValidNumber(outputs[t]) ? outputs[t] : 0)
          );

          // Compute attention scores across all input positions
          let score = 0;
          for (let i = 0; i < this.#inputSize; i++) {
              for (let j = 0; j < this.#inputSize; j++) {
                  let dotProduct = 0;
                  for (let k = 0; k < this.#hiddenSize; k++) {
                      dotProduct += isValidNumber(queries[i][k]) && isValidNumber(keys[j][k])
                          ? queries[i][k] * keys[j][k]
                          : 0;
                  }
                  const attentionWeight = softmax(
                      Array(this.#inputSize).fill().map((_, k) => {
                          let dp = 0;
                          for (let m = 0; m < this.#hiddenSize; m++) {
                              dp += isValidNumber(queries[i][m]) && isValidNumber(keys[k][m])
                                  ? queries[i][m] * keys[k][m]
                                  : 0;
                          }
                          return isValidNumber(dp)
                              ? dp / Math.sqrt(this.#hiddenSize) * this.#attentionScalingFactor
                              : 0;
                      })
                  )[j];

                  for (let k = 0; k < this.#hiddenSize; k++) {
                      score += isValidNumber(attentionWeight) && isValidNumber(values[j][k])
                          ? attentionWeight * values[j][k]
                          : 0;
                  }
              }
          }

          // Scale score and add bias
          const biasSum = this.#attentionBias[t].reduce((sum, val) => sum + (isValidNumber(val) ? val : 0), 0);
          score = score / this.#inputSize + (isValidNumber(biasSum) ? biasSum / this.#hiddenSize : 0);

          // Incorporate historical performance, trust, specialization, and swarm intelligence
          const historicalWeight = this.#historicalPerformance[t].length > 0
              ? this.#historicalPerformance[t].reduce((sum, val) => sum + (isValidNumber(val) ? val : 0), 0) /
                this.#historicalPerformance[t].length
              : 0;
          const trustScore = this.#trustScoresHistory[t].length > 0
              ? this.#trustScoresHistory[t].reduce((sum, val) => sum + (isValidNumber(val) ? val : 0), 0) /
                this.#trustScoresHistory[t].length
              : 0.5;
          const specializationBoost = 1 + (isValidNumber(this.#specializationScores[t])
              ? this.#specializationScores[t] * this.#swarmIntelligenceFactor
              : 0);

          attentionScores[t] = score * (0.5 + 0.2 * historicalWeight + 0.2 * trustScore + 0.1 * specializationBoost);
      }

      // Apply softmax to normalize attention scores
      const weights = softmax(attentionScores.map(score => isValidNumber(score) ? score : 0));

      // Compute final ensemble weights
      const finalWeights = weights.map((w, idx) => {
          const performanceScore = isValidNumber(this.#performanceScores[idx]) ? this.#performanceScores[idx] : 0;
          const specializationFactor = 1 + (isValidNumber(this.#specializationScores[idx])
              ? this.#specializationScores[idx] * this.#swarmIntelligenceFactor
              : 0);
          const trustScore = this.#trustScoresHistory[idx].length > 0
              ? this.#trustScoresHistory[idx][this.#trustScoresHistory[idx].length - 1]
              : 0.5;
          const weight = 0.4 * w +
              0.3 * performanceScore +
              0.2 * specializationFactor +
              0.1 * trustScore;
          return isValidNumber(weight) && weight >= 0 ? weight : 1 / this.#ensembleSize;
      });

      // Normalize final weights
      const sum = finalWeights.reduce((s, w) => s + (isValidNumber(w) && w >= 0 ? w : 0), 0);
      if (sum <= 1e-6) {
          return Array(this.#ensembleSize).fill(1 / this.#ensembleSize);
      }
      return finalWeights.map(w => (isValidNumber(w) && w >= 0 ? w / sum : 1 / this.#ensembleSize));
  }

  #updateTrustScores() {
    const performanceMean = this.#performanceScores.reduce((sum, score) => sum + (isValidNumber(score) ? score : 0), 0) / this.#ensembleSize || 1;
    const performanceStd = Math.sqrt(
      this.#performanceScores.reduce((sum, score) => sum + ((isValidNumber(score) ? score : 0) - performanceMean) ** 2, 0) / this.#ensembleSize
    ) || 1;

    this.#trustScoresHistory = this.#trustScoresHistory.map((history, idx) => {
      const normalizedScore = isValidNumber(this.#performanceScores[idx]) && performanceStd > 0
        ? (this.#performanceScores[idx] - performanceMean) / performanceStd
        : 0;
      const agreementFactor = isValidNumber(this.#agreementScores[idx]) ? this.#agreementScores[idx] : 0.5;
      const historicalTrend = this.#historicalPerformance[idx].length > 1
        ? (this.#historicalPerformance[idx][this.#historicalPerformance[idx].length - 1] - this.#historicalPerformance[idx][0]) /
          this.#historicalPerformance[idx].length
        : 0;
      const specializationBoost = 1 + this.#specializationScores[idx] * this.#swarmIntelligenceFactor;
      const trustScore = sigmoid(normalizedScore * (0.6 + 0.2 * agreementFactor + 0.1 * historicalTrend + 0.1 * specializationBoost));
      history.push(isValidNumber(trustScore) ? trustScore : 0.5);
      if (history.length > this.#maxTrustHistory) history.shift();
      return history;
    });

    const trustMomentum = 0.6;
    this.#performanceScores = this.#performanceScores.map((score, idx) => {
      const recentTrust = this.#trustScoresHistory[idx].slice(-5);
      const avgTrust = recentTrust.length > 0
        ? recentTrust.reduce((sum, val) => sum + (isValidNumber(val) ? val : 0), 0) / recentTrust.length
        : 0.5;
      const historicalWeight = this.#trustScoresHistory[idx].length / this.#maxTrustHistory;
      const specializationFactor = 1 + this.#specializationScores[idx] * this.#swarmIntelligenceFactor;
      return trustMomentum * (isValidNumber(score) ? score : 0) + (1 - trustMomentum) * avgTrust * (0.7 + 0.2 * historicalWeight + 0.1 * specializationFactor);
    });

    this.#ensembleWeights = this.#trustScoresHistory.map((history, idx) => {
      const recentTrust = history.slice(-5);
      const avgTrust = recentTrust.length > 0
        ? recentTrust.reduce((sum, val) => sum + (isValidNumber(val) ? val : 0), 0) / recentTrust.length
        : 1 / this.#ensembleSize;
      const specializationFactor = 1 + this.#specializationScores[idx] * this.#swarmIntelligenceFactor;
      return avgTrust * (0.8 + 0.2 * specializationFactor);
    });
    this.#normalizeEnsembleWeights();
  }

  /**
   * Computes specialization scores for each transformer based on correlations between input features
   * and transformer outputs. Updates specialization weights to reflect feature importance, incorporating
   * trust scores, performance, and swarm intelligence. Handles invalid inputs by skipping updates and
   * normalizing scores to ensure numerical stability.
   * @param {number[]} inputs - Array of input features (length: #inputSize, typically 6).
   * @param {number[]} outputs - Array of transformer outputs (length: #ensembleSize, typically 128).
   */
  #computeSpecializationScores(inputs, outputs) {
      // Validate inputs and outputs to ensure they are usable
      if (
          !Array.isArray(inputs) ||
          inputs.length !== this.#inputSize ||
          !inputs.every(isValidNumber) ||
          !Array.isArray(outputs) ||
          outputs.length !== this.#ensembleSize ||
          !outputs.every(isValidNumber)
      ) {
          // Skip updates if inputs or outputs are invalid to prevent corruption
          return;
      }

      // Initialize feature correlations array for each transformer and input feature
      const featureCorrelations = Array(this.#ensembleSize).fill().map(() => Array(this.#inputSize).fill(0));

      // Compute mean of inputs and outputs for correlation calculation
      const inputMean = inputs.reduce((sum, v) => sum + (isValidNumber(v) ? v : 0), 0) / inputs.length || 0;
      const outputMean = outputs.reduce((sum, v) => sum + (isValidNumber(v) ? v : 0), 0) / outputs.length || 0;

      // Calculate correlations between each transformer's output and each input feature
      for (let i = 0; i < this.#ensembleSize; i++) {
          for (let j = 0; j < this.#inputSize; j++) {
              if (isValidNumber(inputs[j]) && isValidNumber(outputs[i])) {
                  const inputDiff = inputs[j] - inputMean;
                  const outputDiff = outputs[i] - outputMean;
                  const inputStd = Math.sqrt(
                      inputs.reduce((sum, v) => sum + ((isValidNumber(v) ? v : 0) - inputMean) ** 2, 0) / inputs.length
                  ) || 1e-6;
                  const outputStd = Math.sqrt(
                      outputs.reduce((sum, v) => sum + ((isValidNumber(v) ? v : 0) - outputMean) ** 2, 0) / outputs.length
                  ) || 1e-6;
                  const numerator = inputDiff * outputDiff;
                  // Compute correlation, handling division by zero
                  featureCorrelations[i][j] = isValidNumber(numerator) && inputStd > 1e-6 && outputStd > 1e-6
                      ? Math.min(Math.max(numerator / (inputStd * outputStd), -1), 1)
                      : 0;
              }
          }
      }

      // Compute specialization scores using correlations, trust scores, and performance
      this.#specializationScores = featureCorrelations.map((corr, idx) => {
          const meanCorr = corr.reduce((sum, val) => sum + (isValidNumber(val) ? Math.abs(val) : 0), 0) / corr.length || 0.5;
          const trustFactor = this.#trustScoresHistory[idx].length > 0
              ? this.#trustScoresHistory[idx][this.#trustScoresHistory[idx].length - 1]
              : 0.5;
          const performanceFactor = isValidNumber(this.#performanceScores[idx]) ? this.#performanceScores[idx] : 0.5;
          // Combine factors with sigmoid for bounded scores, incorporating swarm intelligence
          const input = Math.min(meanCorr * (1 + this.#swarmIntelligenceFactor) * (0.5 + 0.3 * trustFactor + 0.2 * performanceFactor), 5);
          return sigmoid(input);
      });

      // Normalize specialization scores to ensure sum to 1
      const sumScores = this.#specializationScores.reduce((sum, score) => sum + (isValidNumber(score) ? score : 0), 0) || 1;
      this.#specializationScores = this.#specializationScores.map(score => isValidNumber(score) ? score / sumScores : 0);

      // Update specialization weights based on correlations, mapping to hidden dimensions
      for (let i = 0; i < this.#ensembleSize; i++) {
          const specializationFactor = 1 + this.#specializationScores[i] * this.#swarmIntelligenceFactor;
          for (let j = 0; j < this.#hiddenSize; j++) {
              for (let k = 0; k < this.#hiddenSize; k++) {
                  // Map hidden dimension to input feature cyclically
                  const inputIdx = j % this.#inputSize;
                  const corr = featureCorrelations[i][inputIdx];
                  if (isValidNumber(corr)) {
                      // Update weights with learning rate, correlation, and specialization influence
                      const update = isValidNumber(this.#adaptiveLearningRate[i]) && isValidNumber(corr) && isValidNumber(specializationFactor)
                          ? Math.min(Math.max(this.#adaptiveLearningRate[i] * corr * specializationFactor, -0.1), 0.1)
                          : 0;
                      this.#specializationWeights[i][j][k] += update;
                      // Clip weights to prevent numerical instability
                      this.#specializationWeights[i][j][k] = Math.min(Math.max(this.#specializationWeights[i][j][k], -1), 1);
                  }
              }
          }
      }

      // Update attention weight matrix to reflect specialization
      for (let i = 0; i < this.#ensembleSize; i++) {
          const meanCorr = featureCorrelations[i].reduce((sum, val) => sum + (isValidNumber(val) ? Math.abs(val) : 0), 0) / this.#inputSize || 0;
          for (let j = 0; j < this.#hiddenSize; j++) {
              // Apply correlation-based update to attention weights, scaled by specialization score
              const update = isValidNumber(this.#adaptiveLearningRate[i]) && isValidNumber(meanCorr) && isValidNumber(this.#specializationScores[i])
                  ? Math.min(Math.max(this.#adaptiveLearningRate[i] * meanCorr * (1 + this.#specializationScores[i] * this.#swarmIntelligenceFactor), -0.1), 0.1)
                  : 0;
              this.#attentionWeightMatrix[i][j] += update;
              // Clip weights to prevent numerical instability
              this.#attentionWeightMatrix[i][j] = Math.min(Math.max(this.#attentionWeightMatrix[i][j], -1), 1);
          }
      }
  }

  #updateAdaptiveLearningRates() {
    const performanceMean = this.#performanceScores.reduce((sum, score) => 
    sum + (isValidNumber(score) && score >= 0 && score <= 1 ? score : 0), 0) / this.#ensembleSize || 1;
    
    this.#adaptiveLearningRate = this.#adaptiveLearningRate.map((lr, idx) => {
      const performanceDiff = isValidNumber(this.#performanceScores[idx]) 
        ? Math.min(Math.max(this.#performanceScores[idx] - performanceMean, -10), 10) 
        : 0;
      const adjustment = 1 + 0.1 * sigmoid(performanceDiff);
      const newLr = this.#learningRate * adjustment;
      return Math.min(Math.max(newLr, this.#learningRate * 0.5), this.#learningRate * 2);
    });
  }

  /**
   * Shares weights among lower-performing transformers to improve their performance by adopting
   * knowledge from top performers, while preserving the weights of top performers to maintain
   * specialization. Uses trust scores, specialization scores, and swarm intelligence to weight
   * contributions, with momentum to smooth updates. Handles invalid values and ensures numerical
   * stability through clipping and validation.
   */
  #shareWeights() {
      // Compute trust scores based on performance, normalized to sum to 1
      const performanceSum = this.#performanceScores.reduce((sum, score) => sum + (isValidNumber(score) ? score : 0), 0) || 1;
      const trustScores = this.#performanceScores.map(score => (isValidNumber(score) ? score : 0) / performanceSum);
      const momentumFactor = 0.7;

      // Helper function to compute weighted average of weights
      const avgWeights = (weights, trustWeights) => {
          const result = weights[0].map(row => row.map(() => 0));
          for (let i = 0; i < result.length; i++) {
              for (let j = 0; j < result[i].length; j++) {
                  let weightedSum = 0, totalWeight = 0;
                  for (let t = 0; t < weights.length; t++) {
                      if (isValidNumber(weights[t][i][j]) && isValidNumber(trustWeights[t])) {
                          weightedSum += weights[t][i][j] * trustWeights[t] * (1 + this.#swarmIntelligenceFactor * this.#specializationScores[t]);
                          totalWeight += trustWeights[t];
                      }
                  }
                  result[i][j] = totalWeight > 1e-6 ? weightedSum / totalWeight : 0;
              }
          }
          return result;
      };

      // Helper function to compute weighted average of biases
      const avgBias = (biases, trustWeights) => {
          const result = biases[0].map(() => 0);
          for (let i = 0; i < result.length; i++) {
              let weightedSum = 0, totalWeight = 0;
              for (let t = 0; t < biases.length; t++) {
                  if (isValidNumber(biases[t][i]) && isValidNumber(trustWeights[t])) {
                      weightedSum += biases[t][i] * trustWeights[t] * (1 + this.#swarmIntelligenceFactor * this.#specializationScores[t]);
                      totalWeight += trustWeights[t];
                  }
              }
              result[i] = totalWeight > 1e-6 ? weightedSum / totalWeight : 0;
          }
          return result;
      };

      // Identify top performers (top 25%)
      const topPerformers = this.#performanceScores
          .map((score, idx) => ({ score: isValidNumber(score) ? score : 0, idx }))
          .sort((a, b) => b.score - a.score)
          .slice(0, Math.floor(this.#ensembleSize * 0.25))
          .map(({ idx }) => idx);

      // Compute averaged weights and biases for all layers
      for (let layer = 0; layer < this.#numLayers; layer++) {
          const allWq = this.#transformers.map(t => t.attentionWeights[layer].Wq);
          const allWk = this.#transformers.map(t => t.attentionWeights[layer].Wk);
          const allWv = this.#transformers.map(t => t.attentionWeights[layer].Wv);
          const allWo = this.#transformers.map(t => t.attentionWeights[layer].Wo);
          const allW1 = this.#transformers.map(t => t.ffnWeights[layer].W1);
          const allW2 = this.#transformers.map(t => t.ffnWeights[layer].W2);
          const allB1 = this.#transformers.map(t => t.ffnWeights[layer].b1);
          const allB2 = this.#transformers.map(t => t.ffnWeights[layer].b2);
          const allGamma1 = this.#transformers.map(t => t.layerNormWeights[layer].gamma1);
          const allBeta1 = this.#transformers.map(t => t.layerNormWeights[layer].beta1);
          const allGamma2 = this.#transformers.map(t => t.layerNormWeights[layer].gamma2);
          const allBeta2 = this.#transformers.map(t => t.layerNormWeights[layer].beta2);

          const avgWq = avgWeights(allWq, trustScores);
          const avgWk = avgWeights(allWk, trustScores);
          const avgWv = avgWeights(allWv, trustScores);
          const avgWo = avgWeights(allWo, trustScores);
          const avgW1 = avgWeights(allW1, trustScores);
          const avgW2 = avgWeights(allW2, trustScores);
          const avgB1 = avgBias(allB1, trustScores);
          const avgB2 = avgBias(allB2, trustScores);
          const avgGamma1 = avgBias(allGamma1, trustScores);
          const avgBeta1 = avgBias(allBeta1, trustScores);
          const avgGamma2 = avgBias(allGamma2, trustScores);
          const avgBeta2 = avgBias(allBeta2, trustScores);

          this.#transformers.forEach((t, idx) => {
              // Skip top performers
              if (topPerformers.includes(idx)) return;

              const sharingRate = trustScores[idx] < 0.5 ? this.#weightSharingRate + 0.1 * (0.5 - trustScores[idx]) : this.#weightSharingRate;
              const swarmInfluence = 1 + this.#swarmIntelligenceFactor * (isValidNumber(this.#specializationScores[idx]) ? this.#specializationScores[idx] : 0);

              // Update attention weights
              for (let i = 0; i < t.attentionWeights[layer].Wq.length; i++) {
                  for (let j = 0; j < t.attentionWeights[layer].Wq[i].length; j++) {
                      const specializationFactor = isValidNumber(this.#specializationWeights[idx][i % this.#hiddenSize][j])
                          ? 1 + this.#specializationScores[idx] * this.#specializationWeights[idx][i % this.#hiddenSize][j]
                          : 1;
                      const update = isValidNumber(t.attentionWeights[layer].Wq[i][j]) && isValidNumber(avgWq[i][j])
                          ? (1 - sharingRate) * t.attentionWeights[layer].Wq[i][j] + sharingRate * avgWq[i][j] * swarmInfluence * specializationFactor
                          : t.attentionWeights[layer].Wq[i][j] || 0;
                      this.#momentumWeights[idx].attentionWeights[layer].Wq[i][j] = momentumFactor * this.#momentumWeights[idx].attentionWeights[layer].Wq[i][j] +
                          (1 - momentumFactor) * Math.min(Math.max(update, -1), 1);
                      t.attentionWeights[layer].Wq[i][j] = this.#momentumWeights[idx].attentionWeights[layer].Wq[i][j];

                      const wkUpdate = isValidNumber(t.attentionWeights[layer].Wk[i][j]) && isValidNumber(avgWk[i][j])
                          ? (1 - sharingRate) * t.attentionWeights[layer].Wk[i][j] + sharingRate * avgWk[i][j] * swarmInfluence * specializationFactor
                          : t.attentionWeights[layer].Wk[i][j] || 0;
                      this.#momentumWeights[idx].attentionWeights[layer].Wk[i][j] = momentumFactor * this.#momentumWeights[idx].attentionWeights[layer].Wk[i][j] +
                          (1 - momentumFactor) * Math.min(Math.max(wkUpdate, -1), 1);
                      t.attentionWeights[layer].Wk[i][j] = this.#momentumWeights[idx].attentionWeights[layer].Wk[i][j];

                      const wvUpdate = isValidNumber(t.attentionWeights[layer].Wv[i][j]) && isValidNumber(avgWv[i][j])
                          ? (1 - sharingRate) * t.attentionWeights[layer].Wv[i][j] + sharingRate * avgWv[i][j] * swarmInfluence * specializationFactor
                          : t.attentionWeights[layer].Wv[i][j] || 0;
                      this.#momentumWeights[idx].attentionWeights[layer].Wv[i][j] = momentumFactor * this.#momentumWeights[idx].attentionWeights[layer].Wv[i][j] +
                          (1 - momentumFactor) * Math.min(Math.max(wvUpdate, -1), 1);
                      t.attentionWeights[layer].Wv[i][j] = this.#momentumWeights[idx].attentionWeights[layer].Wv[i][j];

                      const woUpdate = isValidNumber(t.attentionWeights[layer].Wo[i][j]) && isValidNumber(avgWo[i][j])
                          ? (1 - sharingRate) * t.attentionWeights[layer].Wo[i][j] + sharingRate * avgWo[i][j] * swarmInfluence * specializationFactor
                          : t.attentionWeights[layer].Wo[i][j] || 0;
                      this.#momentumWeights[idx].attentionWeights[layer].Wo[i][j] = momentumFactor * this.#momentumWeights[idx].attentionWeights[layer].Wo[i][j] +
                          (1 - momentumFactor) * Math.min(Math.max(woUpdate, -1), 1);
                      t.attentionWeights[layer].Wo[i][j] = this.#momentumWeights[idx].attentionWeights[layer].Wo[i][j];
                  }
              }

              // Update feed-forward weights
              for (let i = 0; i < t.ffnWeights[layer].W1.length; i++) {
                  for (let j = 0; j < t.ffnWeights[layer].W1[i].length; j++) {
                      const specializationFactor = isValidNumber(this.#specializationWeights[idx][i % this.#hiddenSize][j % this.#hiddenSize])
                          ? 1 + this.#specializationScores[idx] * this.#specializationWeights[idx][i % this.#hiddenSize][j % this.#hiddenSize]
                          : 1;
                      const update = isValidNumber(t.ffnWeights[layer].W1[i][j]) && isValidNumber(avgW1[i][j])
                          ? (1 - sharingRate) * t.ffnWeights[layer].W1[i][j] + sharingRate * avgW1[i][j] * swarmInfluence * specializationFactor
                          : t.ffnWeights[layer].W1[i][j] || 0;
                      this.#momentumWeights[idx].ffnWeights[layer].W1[i][j] = momentumFactor * this.#momentumWeights[idx].ffnWeights[layer].W1[i][j] +
                          (1 - momentumFactor) * Math.min(Math.max(update, -1), 1);
                      t.ffnWeights[layer].W1[i][j] = this.#momentumWeights[idx].ffnWeights[layer].W1[i][j];
                  }
              }
              for (let i = 0; i < t.ffnWeights[layer].W2.length; i++) {
                  for (let j = 0; j < t.ffnWeights[layer].W2[i].length; j++) {
                      const specializationFactor = isValidNumber(this.#specializationWeights[idx][j % this.#hiddenSize][i % this.#hiddenSize])
                          ? 1 + this.#specializationScores[idx] * this.#specializationWeights[idx][j % this.#hiddenSize][i % this.#hiddenSize]
                          : 1;
                      const update = isValidNumber(t.ffnWeights[layer].W2[i][j]) && isValidNumber(avgW2[i][j])
                          ? (1 - sharingRate) * t.ffnWeights[layer].W2[i][j] + sharingRate * avgW2[i][j] * swarmInfluence * specializationFactor
                          : t.ffnWeights[layer].W2[i][j] || 0;
                      this.#momentumWeights[idx].ffnWeights[layer].W2[i][j] = momentumFactor * this.#momentumWeights[idx].ffnWeights[layer].W2[i][j] +
                          (1 - momentumFactor) * Math.min(Math.max(update, -1), 1);
                      t.ffnWeights[layer].W2[i][j] = this.#momentumWeights[idx].ffnWeights[layer].W2[i][j];
                  }
              }

              // Update feed-forward biases
              for (let i = 0; i < t.ffnWeights[layer].b1.length; i++) {
                  const update = isValidNumber(t.ffnWeights[layer].b1[i]) && isValidNumber(avgB1[i])
                      ? (1 - sharingRate) * t.ffnWeights[layer].b1[i] + sharingRate * avgB1[i] * swarmInfluence
                      : t.ffnWeights[layer].b1[i] || 0;
                  this.#momentumWeights[idx].ffnWeights[layer].b1[i] = momentumFactor * this.#momentumWeights[idx].ffnWeights[layer].b1[i] +
                      (1 - momentumFactor) * Math.min(Math.max(update, -1), 1);
                  t.ffnWeights[layer].b1[i] = this.#momentumWeights[idx].ffnWeights[layer].b1[i];
              }
              for (let i = 0; i < t.ffnWeights[layer].b2.length; i++) {
                  const update = isValidNumber(t.ffnWeights[layer].b2[i]) && isValidNumber(avgB2[i])
                      ? (1 - sharingRate) * t.ffnWeights[layer].b2[i] + sharingRate * avgB2[i] * swarmInfluence
                      : t.ffnWeights[layer].b2[i] || 0;
                  this.#momentumWeights[idx].ffnWeights[layer].b2[i] = momentumFactor * this.#momentumWeights[idx].ffnWeights[layer].b2[i] +
                      (1 - momentumFactor) * Math.min(Math.max(update, -1), 1);
                  t.ffnWeights[layer].b2[i] = this.#momentumWeights[idx].ffnWeights[layer].b2[i];
              }

              // Update layer normalization weights
              for (let i = 0; i < t.layerNormWeights[layer].gamma1.length; i++) {
                  const gamma1Update = isValidNumber(t.layerNormWeights[layer].gamma1[i]) && isValidNumber(avgGamma1[i])
                      ? (1 - sharingRate) * t.layerNormWeights[layer].gamma1[i] + sharingRate * avgGamma1[i] * swarmInfluence
                      : t.layerNormWeights[layer].gamma1[i] || 1;
                  this.#momentumWeights[idx].layerNormWeights[layer].gamma1[i] = momentumFactor * this.#momentumWeights[idx].layerNormWeights[layer].gamma1[i] +
                      (1 - momentumFactor) * Math.min(Math.max(gamma1Update, -1), 1);
                  t.layerNormWeights[layer].gamma1[i] = this.#momentumWeights[idx].layerNormWeights[layer].gamma1[i];

                  const beta1Update = isValidNumber(t.layerNormWeights[layer].beta1[i]) && isValidNumber(avgBeta1[i])
                      ? (1 - sharingRate) * t.layerNormWeights[layer].beta1[i] + sharingRate * avgBeta1[i] * swarmInfluence
                      : t.layerNormWeights[layer].beta1[i] || 0;
                  this.#momentumWeights[idx].layerNormWeights[layer].beta1[i] = momentumFactor * this.#momentumWeights[idx].layerNormWeights[layer].beta1[i] +
                      (1 - momentumFactor) * Math.min(Math.max(beta1Update, -1), 1);
                  t.layerNormWeights[layer].beta1[i] = this.#momentumWeights[idx].layerNormWeights[layer].beta1[i];

                  const gamma2Update = isValidNumber(t.layerNormWeights[layer].gamma2[i]) && isValidNumber(avgGamma2[i])
                      ? (1 - sharingRate) * t.layerNormWeights[layer].gamma2[i] + sharingRate * avgGamma2[i] * swarmInfluence
                      : t.layerNormWeights[layer].gamma2[i] || 1;
                  this.#momentumWeights[idx].layerNormWeights[layer].gamma2[i] = momentumFactor * this.#momentumWeights[idx].layerNormWeights[layer].gamma2[i] +
                      (1 - momentumFactor) * Math.min(Math.max(gamma2Update, -1), 1);
                  t.layerNormWeights[layer].gamma2[i] = this.#momentumWeights[idx].layerNormWeights[layer].gamma2[i];

                  const beta2Update = isValidNumber(t.layerNormWeights[layer].beta2[i]) && isValidNumber(avgBeta2[i])
                      ? (1 - sharingRate) * t.layerNormWeights[layer].beta2[i] + sharingRate * avgBeta2[i] * swarmInfluence
                      : t.layerNormWeights[layer].beta2[i] || 0;
                  this.#momentumWeights[idx].layerNormWeights[layer].beta2[i] = momentumFactor * this.#momentumWeights[idx].layerNormWeights[layer].beta2[i] +
                      (1 - momentumFactor) * Math.min(Math.max(beta2Update, -1), 1);
                  t.layerNormWeights[layer].beta2[i] = this.#momentumWeights[idx].layerNormWeights[layer].beta2[i];
              }
          });
      }

      // Update output weights and biases for non-top performers
      const allOutputWeights = this.#transformers.map(t => t.outputWeights);
      const allOutputBias = this.#transformers.map(t => t.outputBias);
      const avgOutputWeights = avgWeights(allOutputWeights, trustScores);
      const avgOutputBias = avgBias(allOutputBias, trustScores);

      this.#transformers.forEach((t, idx) => {
          // Skip top performers
          if (topPerformers.includes(idx)) return;

          const sharingRate = trustScores[idx] < 0.5 ? this.#weightSharingRate + 0.1 * (0.5 - trustScores[idx]) : this.#weightSharingRate;
          const swarmInfluence = 1 + this.#swarmIntelligenceFactor * (isValidNumber(this.#specializationScores[idx]) ? this.#specializationScores[idx] : 0);

          // Update output weights
          for (let i = 0; i < t.outputWeights.length; i++) {
              for (let j = 0; j < t.outputWeights[i].length; j++) {
                  const specializationFactor = isValidNumber(this.#specializationWeights[idx][i % this.#hiddenSize][j])
                      ? 1 + this.#specializationScores[idx] * this.#specializationWeights[idx][i % this.#hiddenSize][j]
                      : 1;
                  const update = isValidNumber(t.outputWeights[i][j]) && isValidNumber(avgOutputWeights[i][j])
                      ? (1 - sharingRate) * t.outputWeights[i][j] + sharingRate * avgOutputWeights[i][j] * swarmInfluence * specializationFactor
                      : t.outputWeights[i][j] || 0;
                  this.#momentumWeights[idx].outputWeights[i][j] = momentumFactor * this.#momentumWeights[idx].outputWeights[i][j] +
                      (1 - momentumFactor) * Math.min(Math.max(update, -1), 1);
                  t.outputWeights[i][j] = this.#momentumWeights[idx].outputWeights[i][j];
              }
          }

          // Update output biases
          for (let i = 0; i < t.outputBias.length; i++) {
              const update = isValidNumber(t.outputBias[i]) && isValidNumber(avgOutputBias[i])
                  ? (1 - sharingRate) * t.outputBias[i] + sharingRate * avgOutputBias[i] * swarmInfluence
                  : t.outputBias[i] || 0;
              this.#momentumWeights[idx].outputBias[i] = momentumFactor * this.#momentumWeights[idx].outputBias[i] +
                  (1 - momentumFactor) * Math.min(Math.max(update, -1), 1);
              t.outputBias[i] = this.#momentumWeights[idx].outputBias[i];
          }
      });
  }

  /**
   * Normalizes the ensemble weights to ensure they sum to 1, handling edge cases like invalid or zero weights.
   * This method adjusts the weights of each transformer in the ensemble to maintain a valid probability distribution.
   */
  #normalizeEnsembleWeights() {
      // Sum all valid weights, ignoring invalid or negative values
      const sum = this.#ensembleWeights.reduce((s, w) => s + (isValidNumber(w) && w >= 0 ? w : 0), 0);

      // If sum is effectively zero, assign uniform weights to avoid division by zero
      if (sum <= 1e-6) {
          this.#ensembleWeights = Array(this.#ensembleSize).fill(1 / this.#ensembleSize);
          return;
      }

      // Normalize weights, replacing invalid or negative values with zero
      this.#ensembleWeights = this.#ensembleWeights.map(w => 
          isValidNumber(w) && w >= 0 ? w / sum : 0
      );

      // Ensure sum is exactly 1 by adjusting the largest weight if necessary
      const finalSum = this.#ensembleWeights.reduce((s, w) => s + w, 0);
      if (Math.abs(finalSum - 1) > 1e-6) {
          const maxIndex = this.#ensembleWeights.indexOf(Math.max(...this.#ensembleWeights));
          this.#ensembleWeights[maxIndex] += 1 - finalSum;
      }
  }

  /**
   * Applies the GELU activation function to a single input value, ensuring numerical stability.
   * Returns 0 for invalid inputs to prevent propagation of NaN or Infinity.
   * @param {number} x - Input value to apply GELU activation.
   * @returns {number} GELU-activated value or 0 if input is invalid.
   */
  #gelu(x) {
      if (!isValidNumber(x)) return 0;
      // GELU approximation: x * Φ(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
      return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));
  }

  /**
   * Computes the derivative of the GELU activation function for backpropagation.
   * Handles invalid inputs by returning 0 and uses an approximation for numerical stability.
   * @param {number} x - Input value for which to compute GELU derivative.
   * @returns {number} Derivative of GELU at x, or 0 if input is invalid.
   */
  #geluDerivative(x) {
      if (!isValidNumber(x)) return 0;
      // Compute cumulative distribution function (CDF) for GELU
      const cdf = 0.5 * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));
      // Approximate normal PDF for derivative: exp(-x^2/2) / sqrt(2π)
      const pdf = Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
      // GELU derivative: CDF + x * PDF
      return isValidNumber(cdf) && isValidNumber(pdf) ? cdf + x * pdf : 0;
  }

  /**
   * Applies layer normalization to an input vector, stabilizing training by normalizing
   * across the hidden dimensions. Handles invalid inputs by returning a zero-filled array.
   * @param {number[]} x - Input vector to normalize (length: hiddenSize).
   * @param {number[]} gamma - Scaling parameters (length: hiddenSize).
   * @param {number[]} beta - Shift parameters (length: hiddenSize).
   * @param {number} [eps=1e-6] - Small constant to prevent division by zero.
   * @returns {number[]} Normalized output vector (length: hiddenSize).
   */
  #layerNorm(x, gamma, beta, eps = 1e-6) {
      // Validate inputs, gamma, and beta
      if (
          !Array.isArray(x) || x.length !== this.#hiddenSize ||
          !Array.isArray(gamma) || gamma.length !== this.#hiddenSize ||
          !Array.isArray(beta) || beta.length !== this.#hiddenSize ||
          !x.every(isValidNumber) || !gamma.every(isValidNumber) || !beta.every(isValidNumber)
      ) {
          return Array(this.#hiddenSize).fill(0);
      }

      // Compute mean and variance of input
      const mean = x.reduce((sum, val) => sum + val, 0) / x.length;
      const variance = x.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / x.length;

      // Check for invalid variance
      if (!isValidNumber(variance) || !isValidNumber(mean)) {
          return Array(this.#hiddenSize).fill(0);
      }

      // Normalize and apply gamma and beta
      const output = x.map((val, i) => {
          const normalized = (val - mean) / Math.sqrt(variance + eps);
          return isValidNumber(normalized) ? gamma[i] * normalized + beta[i] : 0;
      });

      return output;
  }

  /**
   * Computes context-aware input embeddings by combining current inputs with historical
   * attention outputs, enhancing temporal dependencies. Projects inputs to hidden size,
   * adds positional encodings, blends with historical context, and applies layer normalization.
   * @param {number[]} inputs - Array of input values (length: inputSize).
   * @param {number} transformerIdx - Index of the transformer in the ensemble.
   * @returns {number[][]} Array of shape [inputSize, hiddenSize] with context-enhanced embeddings.
   */
  #contextAwareAttention(inputs, transformerIdx) {
      // Validate inputs and transformer index
      if (
          !Array.isArray(inputs) ||
          inputs.length !== this.#inputSize ||
          !inputs.every(isValidNumber) ||
          !Number.isInteger(transformerIdx) ||
          transformerIdx < 0 ||
          transformerIdx >= this.#ensembleSize
      ) {
          return Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
      }

      const transformer = this.#transformers[transformerIdx];

      // Project inputs to hidden size using specialization weights
      const inputProjection = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
      for (let i = 0; i < this.#inputSize; i++) {
          for (let j = 0; j < this.#hiddenSize; j++) {
              const specWeight = isValidNumber(this.#specializationWeights[transformerIdx][i % this.#hiddenSize][j])
                  ? 1 + this.#specializationScores[transformerIdx] * this.#specializationWeights[transformerIdx][i % this.#hiddenSize][j]
                  : 1;
              inputProjection[i][j] = isValidNumber(inputs[i])
                  ? inputs[i] * specWeight * (1 + this.#swarmIntelligenceFactor)
                  : 0;
          }
      }

      // Add positional encodings
      const output = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
      for (let i = 0; i < this.#inputSize; i++) {
          for (let j = 0; j < this.#hiddenSize; j++) {
              output[i][j] = isValidNumber(inputProjection[i][j]) && isValidNumber(transformer.positionalEncoding[i][j])
                  ? inputProjection[i][j] + transformer.positionalEncoding[i][j]
                  : inputProjection[i][j];
          }
      }

      // Incorporate historical context from attention memory
      if (
          this.#attentionMemory[transformerIdx].length > 0 &&
          Array.isArray(this.#attentionMemory[transformerIdx][this.#attentionMemory[transformerIdx].length - 1]) &&
          this.#attentionMemory[transformerIdx][this.#attentionMemory[transformerIdx].length - 1].length === this.#inputSize &&
          this.#attentionMemory[transformerIdx][this.#attentionMemory[transformerIdx].length - 1].every(row => 
              Array.isArray(row) && row.length === this.#hiddenSize && row.every(isValidNumber)
          )
      ) {
          const recentAttention = this.#attentionMemory[transformerIdx][this.#attentionMemory[transformerIdx].length - 1];
          const contextWeight = 0.3;

          for (let i = 0; i < this.#inputSize; i++) {
              for (let j = 0; j < this.#hiddenSize; j++) {
                  const historicalValue = isValidNumber(recentAttention[i][j]) ? recentAttention[i][j] : 0;
                  const specializationFactor = isValidNumber(this.#specializationWeights[transformerIdx][i % this.#hiddenSize][j])
                      ? 1 + this.#specializationScores[transformerIdx] * this.#specializationWeights[transformerIdx][i % this.#hiddenSize][j]
                      : 1;
                  output[i][j] = isValidNumber(output[i][j])
                      ? (1 - contextWeight) * output[i][j] + contextWeight * historicalValue * specializationFactor
                      : historicalValue * specializationFactor;
              }
          }
      }

      // Apply layer normalization
      for (let i = 0; i < this.#inputSize; i++) {
          output[i] = this.#layerNorm(
              output[i],
              transformer.layerNormWeights[0].gamma1,
              transformer.layerNormWeights[0].beta1
          );
      }

      return output;
  }

  /**
   * Performs multi-head attention for a single transformer layer, processing input embeddings
   * through multiple attention heads to capture diverse patterns. Integrates specialization weights
   * to enhance attention focus, updates attention memory for historical context, and applies dropout
   * during training. Returns the attention output and intermediates for backpropagation, handling
   * invalid inputs by returning zero-filled arrays.
   * @param {number[][]} x - Input embeddings of shape [inputSize, hiddenSize].
   * @param {object} layer - Transformer layer object containing attention weights (Wq, Wk, Wv, Wo).
   * @param {number} transformerIdx - Index of the transformer in the ensemble (0 to ensembleSize-1).
   * @returns {object} Object containing attention output of shape [inputSize, hiddenSize] and intermediates (Q, K, V, scores, probs).
   */
  #multiHeadAttention(x, layer, transformerIdx) {
      // Validate inputs, layer, and transformer index
      if (
          !Array.isArray(x) ||
          x.length !== this.#inputSize ||
          !x.every(row => Array.isArray(row) && row.length === this.#hiddenSize && row.every(isValidNumber)) ||
          !Number.isInteger(transformerIdx) ||
          transformerIdx < 0 ||
          transformerIdx >= this.#ensembleSize ||
          !layer || !layer.Wq || !layer.Wk || !layer.Wv || !layer.Wo ||
          !layer.Wq.every(row => Array.isArray(row) && row.length === this.#hiddenSize && row.every(isValidNumber)) ||
          !layer.Wk.every(row => Array.isArray(row) && row.length === this.#hiddenSize && row.every(isValidNumber)) ||
          !layer.Wv.every(row => Array.isArray(row) && row.length === this.#hiddenSize && row.every(isValidNumber)) ||
          !layer.Wo.every(row => Array.isArray(row) && row.length === this.#hiddenSize && row.every(isValidNumber))
      ) {
          return {
              output: Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
              Q: Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
              K: Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
              V: Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
              scores: Array(this.#numHeads).fill().map(() =>
                  Array(this.#inputSize).fill().map(() => Array(this.#inputSize).fill(0))
              ),
              probs: Array(this.#numHeads).fill().map(() =>
                  Array(this.#inputSize).fill().map(() => Array(this.#inputSize).fill(0))
              )
          };
      }

      const headSize = this.#hiddenSize / this.#numHeads;
      if (!Number.isInteger(headSize) || headSize <= 0) {
          return {
              output: Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
              Q: Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
              K: Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
              V: Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
              scores: Array(this.#numHeads).fill().map(() =>
                  Array(this.#inputSize).fill().map(() => Array(this.#inputSize).fill(0))
              ),
              probs: Array(this.#numHeads).fill().map(() =>
                  Array(this.#inputSize).fill().map(() => Array(this.#inputSize).fill(0))
              )
          };
      }

      // Compute query (Q), key (K), and value (V) matrices
      const Q = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
      const K = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
      const V = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
      for (let i = 0; i < this.#inputSize; i++) {
          for (let j = 0; j < this.#hiddenSize; j++) {
              for (let k = 0; k < this.#hiddenSize; k++) {
                  const specWeight = isValidNumber(this.#specializationWeights[transformerIdx][i % this.#hiddenSize][k])
                      ? 1 + this.#specializationScores[transformerIdx] * this.#specializationWeights[transformerIdx][i % this.#hiddenSize][k]
                      : 1;
                  Q[i][j] += isValidNumber(x[i][k]) && isValidNumber(layer.Wq[k][j])
                      ? x[i][k] * layer.Wq[k][j] * specWeight
                      : 0;
                  K[i][j] += isValidNumber(x[i][k]) && isValidNumber(layer.Wk[k][j])
                      ? x[i][k] * layer.Wk[k][j] * specWeight
                      : 0;
                  V[i][j] += isValidNumber(x[i][k]) && isValidNumber(layer.Wv[k][j])
                      ? x[i][k] * layer.Wv[k][j] * specWeight
                      : 0;
              }
          }
      }

      // Compute attention scores for each head
      const attentionScores = Array(this.#numHeads).fill().map(() =>
          Array(this.#inputSize).fill().map(() => Array(this.#inputSize).fill(0))
      );
      const attentionProbs = Array(this.#numHeads).fill().map(() =>
          Array(this.#inputSize).fill().map(() => Array(this.#inputSize).fill(0))
      );
      for (let h = 0; h < this.#numHeads; h++) {
          for (let i = 0; i < this.#inputSize; i++) {
              for (let j = 0; j < this.#inputSize; j++) {
                  let sum = 0;
                  for (let k = 0; k < headSize; k++) {
                      const idx = h * headSize + k;
                      sum += isValidNumber(Q[i][idx]) && isValidNumber(K[j][idx])
                          ? Q[i][idx] * K[j][idx]
                          : 0;
                  }
                  attentionScores[h][i][j] = isValidNumber(sum)
                      ? sum / Math.sqrt(headSize) * this.#attentionScalingFactor * (1 + this.#specializationScores[transformerIdx])
                      : 0;
              }
              attentionProbs[h][i] = softmax(attentionScores[h][i].map(score => isValidNumber(score) ? score : 0));
          }
      }

      // Compute attention output
      const output = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
      for (let h = 0; h < this.#numHeads; h++) {
          for (let i = 0; i < this.#inputSize; i++) {
              for (let j = 0; j < this.#inputSize; j++) {
                  for (let k = 0; k < headSize; k++) {
                      const idx = h * headSize + k;
                      output[i][idx] += isValidNumber(attentionProbs[h][i][j]) && isValidNumber(V[j][idx])
                          ? attentionProbs[h][i][j] * V[j][idx]
                          : 0;
                  }
              }
          }
      }

      // Apply output projection with specialization weights
      const finalOutput = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
      for (let i = 0; i < this.#inputSize; i++) {
          for (let j = 0; j < this.#hiddenSize; j++) {
              for (let k = 0; k < this.#hiddenSize; k++) {
                  const specWeight = isValidNumber(this.#specializationWeights[transformerIdx][i % this.#hiddenSize][k])
                      ? 1 + this.#specializationScores[transformerIdx] * this.#specializationWeights[transformerIdx][i % this.#hiddenSize][k]
                      : 1;
                  finalOutput[i][j] += isValidNumber(output[i][k]) && isValidNumber(layer.Wo[k][j])
                      ? output[i][k] * layer.Wo[k][j] * specWeight
                      : 0;
              }
              // Apply dropout during training
              finalOutput[i] = this.#dropout(finalOutput[i], this.#dropoutRate, true);
          }
      }

      // Update attention memory
      if (this.#attentionMemory[transformerIdx].length >= this.#contextWindow) {
          this.#attentionMemory[transformerIdx].shift();
      }
      this.#attentionMemory[transformerIdx].push(finalOutput.map(row => row.slice()));

      // Return output and intermediates for backpropagation
      return {
          output: finalOutput,
          Q,
          K,
          V,
          scores: attentionScores,
          probs: attentionProbs
      };
  }

  /**
   * Computes the feed-forward network output for a transformer layer, applying
   * linear transformations, GELU activation, and bias terms. Handles invalid inputs
   * by returning a zero-filled array and ensures numerical stability.
   * @param {number[]} x - Input vector (length: hiddenSize).
   * @param {object} layer - Feed-forward layer object with W1, W2, b1, b2.
   * @returns {number[]} Output vector (length: hiddenSize).
   */
  #feedForward(x, layer) {
      // Validate inputs and layer parameters
      if (
          !Array.isArray(x) || x.length !== this.#hiddenSize ||
          !x.every(isValidNumber) ||
          !layer || !layer.W1 || !layer.W2 || !layer.b1 || !layer.b2 ||
          !layer.W1.every(row => Array.isArray(row) && row.length === this.#feedForwardSize && row.every(isValidNumber)) ||
          !layer.W2.every(row => Array.isArray(row) && row.length === this.#hiddenSize && row.every(isValidNumber)) ||
          !Array.isArray(layer.b1) || layer.b1.length !== this.#feedForwardSize || !layer.b1.every(isValidNumber) ||
          !Array.isArray(layer.b2) || layer.b2.length !== this.#hiddenSize || !layer.b2.every(isValidNumber)
      ) {
          return Array(this.#hiddenSize).fill(0);
      }

      // First linear transformation: x * W1 + b1
      const hidden = Array(this.#feedForwardSize).fill(0);
      for (let j = 0; j < this.#feedForwardSize; j++) {
          for (let i = 0; i < this.#hiddenSize; i++) {
              hidden[j] += isValidNumber(x[i]) && isValidNumber(layer.W1[i][j]) ? x[i] * layer.W1[i][j] : 0;
          }
          hidden[j] = isValidNumber(hidden[j]) && isValidNumber(layer.b1[j]) ? hidden[j] + layer.b1[j] : hidden[j];
      }

      // Apply GELU activation
      const activated = hidden.map(val => this.#gelu(val));

      // Second linear transformation: activated * W2 + b2
      const output = Array(this.#hiddenSize).fill(0);
      for (let j = 0; j < this.#hiddenSize; j++) {
          for (let i = 0; i < this.#feedForwardSize; i++) {
              output[j] += isValidNumber(activated[i]) && isValidNumber(layer.W2[i][j]) ? activated[i] * layer.W2[i][j] : 0;
          }
          output[j] = isValidNumber(output[j]) && isValidNumber(layer.b2[j]) ? output[j] + layer.b2[j] : output[j];
      }

      return output;
  }

  /**
   * Applies dropout to an input vector during training to prevent overfitting.
   * Returns the input unchanged during inference or if the rate is invalid.
   * @param {number[]} x - Input vector to apply dropout (length: hiddenSize).
   * @param {number} rate - Dropout rate (0 to 1).
   * @param {boolean} training - Whether the model is in training mode.
   * @returns {number[]} Output vector with dropout applied (length: hiddenSize).
   */
  #dropout(x, rate, training = false) {
      // Validate inputs and rate
      if (
          !Array.isArray(x) || 
          !x.every(isValidNumber) || 
          !isValidNumber(rate) || 
          rate < 0 || 
          rate >= 1
      ) {
          return x.slice(); // Return a copy of input to avoid modifying original
      }

      if (!training) return x.slice();

      // Apply dropout: scale kept values by 1/(1-rate) to maintain expected value
      return x.map(val => {
          if (!isValidNumber(val)) return 0;
          return Math.random() >= rate ? val / (1 - rate) : 0;
      });
  }

  // Computes the diversity loss for the ensemble to encourage varied transformer outputs,
  // promoting specialization while maintaining collaboration. Calculates the variance of
  // transformer outputs and scales it by the diversity weight, incorporating trust scores
  // to prioritize high-performing transformers. Handles invalid inputs by returning a
  // fallback loss of 0 to ensure training stability.
  // Args:
  //   outputs: Array of transformer outputs (length: #ensembleSize, typically 128).
  // Returns:
  //   A single number representing the diversity loss, bounded and numerically stable.
  #computeDiversityLoss(outputs) {
      // Validate inputs to ensure they are an array of valid numbers
      if (
          !Array.isArray(outputs) ||
          outputs.length !== this.#ensembleSize ||
          !outputs.every(isValidNumber)
      ) {
          return 0; // Return 0 loss for invalid inputs to avoid disrupting training
      }

      // Compute the mean of transformer outputs
      const meanOutput = outputs.reduce((sum, val) => sum + val, 0) / this.#ensembleSize;

      // Calculate variance of outputs, weighted by trust scores to emphasize high-performing transformers
      let variance = 0;
      let totalTrustWeight = 0;
      for (let i = 0; i < this.#ensembleSize; i++) {
          const trustWeight = this.#trustScoresHistory[i].length > 0
              ? this.#trustScoresHistory[i][this.#trustScoresHistory[i].length - 1]
              : 0.5; // Default to 0.5 if no trust history
          const diff = outputs[i] - meanOutput;
          variance += trustWeight * diff * diff;
          totalTrustWeight += trustWeight;
      }

      // Normalize variance by total trust weight, default to 0 if total weight is zero
      variance = totalTrustWeight > 0 ? variance / totalTrustWeight : 0;

      // Scale variance by diversity weight and apply sigmoid to bound the loss
      const diversityLoss = this.#diversityWeight * sigmoid(variance);

      // Ensure the loss is a valid number and non-negative
      return isValidNumber(diversityLoss) && diversityLoss >= 0 ? diversityLoss : 0;
  }

  /**
   * Performs knowledge distillation to align lower-performing transformers with top performers
   * using KL divergence and diversity loss. Updates transformer parameters in-place, incorporating
   * specialization scores, swarm intelligence, and adaptive learning rates. Skips updates for
   * invalid inputs to ensure numerical stability and preserves top performers' weights.
   * @param {number[]} outputs - Array of transformer outputs (length: #ensembleSize, typically 128).
   * @param {number} target - Target output value for distillation.
   */
  #distillKnowledge(outputs, target) {
      // Validate inputs and target
      if (
          !Array.isArray(outputs) ||
          outputs.length !== this.#ensembleSize ||
          !outputs.every(isValidNumber) ||
          !isValidNumber(target)
      ) {
          return; // Skip updates for invalid inputs
      }

      // Identify top and bottom performers
      const sortedIndices = this.#performanceScores
          .map((score, idx) => ({ score: isValidNumber(score) ? score : 0, idx }))
          .sort((a, b) => b.score - a.score);
      const topPerformers = sortedIndices.slice(0, Math.floor(this.#ensembleSize * 0.25)).map(({ idx }) => idx);
      const bottomPerformers = sortedIndices.slice(Math.floor(this.#ensembleSize * 0.25)).map(({ idx }) => idx);

      // Compute target distribution from top performers
      const topOutputs = topPerformers.map(idx => outputs[idx]);
      const topWeights = topPerformers.map(idx => this.#ensembleWeights[idx]);
      const weightSum = topWeights.reduce((sum, w) => sum + (isValidNumber(w) ? w : 0), 0) || 1;
      const normalizedTopWeights = topWeights.map(w => (isValidNumber(w) ? w : 0) / weightSum);
      let targetOutput = topOutputs.reduce((sum, output, i) =>
          sum + (isValidNumber(output) && isValidNumber(normalizedTopWeights[i]) ? output * normalizedTopWeights[i] : 0), 0
      );
      targetOutput = isValidNumber(targetOutput) ? targetOutput : target;

      // Compute diversity loss
      const diversityLoss = this.#computeDiversityLoss(outputs);

      this.#transformers.forEach((transformer, idx) => {
          // Skip top performers to preserve their specialized weights
          if (topPerformers.includes(idx)) return;

          // Compute KL divergence loss
          const output = outputs[idx];
          const klLoss = isValidNumber(output) && isValidNumber(targetOutput)
              ? output * Math.log((output + 1e-6) / (targetOutput + 1e-6))
              : 0;
          const totalLoss = this.#knowledgeDistillationLoss * klLoss + diversityLoss;
          const adjustedLearningRate = this.#adaptiveLearningRate[idx];
          let grad = isValidNumber(totalLoss) ? totalLoss * adjustedLearningRate : 0;

          // Update output layer
          for (let i = 0; i < this.#hiddenSize; i++) {
              for (let j = 0; j < this.#outputSize; j++) {
                  const specializationFactor = isValidNumber(this.#specializationWeights[idx][i % this.#hiddenSize][j])
                      ? 1 + this.#specializationScores[idx] * this.#specializationWeights[idx][i % this.#hiddenSize][j]
                      : 1;
                  const gradUpdate = grad * transformer.outputWeights[i][j] * specializationFactor;
                  const clippedUpdate = Math.min(
                      Math.max(adjustedLearningRate * gradUpdate, -this.#gradientClippingThreshold),
                      this.#gradientClippingThreshold
                  );
                  transformer.outputWeights[i][j] = isValidNumber(transformer.outputWeights[i][j])
                      ? transformer.outputWeights[i][j] - clippedUpdate
                      : 0;
                  this.#gradientAccumulation[idx].outputWeights[i][j] += clippedUpdate;
              }
          }
          for (let i = 0; i < this.#outputSize; i++) {
              const gradUpdate = grad;
              const clippedUpdate = Math.min(
                  Math.max(adjustedLearningRate * gradUpdate, -this.#gradientClippingThreshold),
                  this.#gradientClippingThreshold
              );
              transformer.outputBias[i] = isValidNumber(transformer.outputBias[i])
                  ? transformer.outputBias[i] - clippedUpdate
                  : 0;
              this.#gradientAccumulation[idx].outputBias[i] += clippedUpdate;
          }

          // Backpropagate through transformer layers
          for (let layer = this.#numLayers - 1; layer >= 0; layer--) {
              // Validate attention memory
              if (
                  !Array.isArray(this.#attentionMemory[idx]) ||
                  this.#attentionMemory[idx].length === 0 ||
                  !this.#attentionMemory[idx][this.#attentionMemory[idx].length - 1].every(row => Array.isArray(row) && row.length === this.#hiddenSize)
              ) {
                  this.#attentionMemory[idx] = Array(this.#contextWindow).fill().map(() =>
                      Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0))
                  );
              }
              const attentionInput = this.#attentionMemory[idx][this.#attentionMemory[idx].length - 1];
              const headSize = this.#hiddenSize / this.#numHeads;

              // Compute queries, keys, and values
              const Q = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
              const K = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0)) ;
              const V = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
              for (let i = 0; i < this.#inputSize; i++) {
                  for (let j = 0; j < this.#hiddenSize; j++) {
                      for (let k = 0; k < this.#hiddenSize; k++) {
                          const specWeight = isValidNumber(this.#specializationWeights[idx][i % this.#hiddenSize][k])
                              ? 1 + this.#specializationScores[idx] * this.#specializationWeights[idx][i % this.#hiddenSize][k]
                              : 1;
                          Q[i][j] += isValidNumber(attentionInput[i]?.[k]) && isValidNumber(transformer.attentionWeights[layer].Wq[k][j])
                              ? attentionInput[i][k] * transformer.attentionWeights[layer].Wq[k][j] * specWeight
                              : 0;
                          K[i][j] += isValidNumber(attentionInput[i]?.[k]) && isValidNumber(transformer.attentionWeights[layer].Wk[k][j])
                              ? attentionInput[i][k] * transformer.attentionWeights[layer].Wk[k][j] * specWeight
                              : 0;
                          V[i][j] += isValidNumber(attentionInput[i]?.[k]) && isValidNumber(transformer.attentionWeights[layer].Wv[k][j])
                              ? attentionInput[i][k] * transformer.attentionWeights[layer].Wv[k][j] * specWeight
                              : 0;
                      }
                  }
              }

              // Compute attention scores and probabilities
              const attentionScores = Array(this.#numHeads).fill().map(() =>
                  Array(this.#inputSize).fill().map(() => Array(this.#inputSize).fill(0))
              );
              const attentionProbs = Array(this.#numHeads).fill().map(() =>
                  Array(this.#inputSize).fill().map(() => Array(this.#inputSize).fill(0))
              );
              for (let h = 0; h < this.#numHeads; h++) {
                  for (let i = 0; i < this.#inputSize; i++) {
                      for (let j = 0; j < this.#inputSize; j++) {
                          let sum = 0;
                          for (let k = 0; k < headSize; k++) {
                              sum += isValidNumber(Q[i][h * headSize + k]) && isValidNumber(K[j][h * headSize + k])
                                  ? Q[i][h * headSize + k] * K[j][h * headSize + k]
                                  : 0;
                          }
                          attentionScores[h][i][j] = isValidNumber(sum)
                              ? sum / Math.sqrt(headSize) * (1 + this.#specializationScores[idx])
                              : 0;
                      }
                      attentionProbs[h][i] = softmax(attentionScores[h][i].map(score => isValidNumber(score) ? score : 0));
                  }
              }

              // Compute attention output
              const attentionOutput = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
              for (let h = 0; h < this.#numHeads; h++) {
                  for (let i = 0; i < this.#inputSize; i++) {
                      for (let j = 0; j < this.#inputSize; j++) {
                          for (let k = 0; k < headSize; k++) {
                              attentionOutput[i][h * headSize + k] += isValidNumber(attentionProbs[h][i][j]) && isValidNumber(V[j][h * headSize + k])
                                  ? attentionProbs[h][i][j] * V[j][h * headSize + k]
                                  : 0;
                          }
                      }
                  }
              }

              // Compute gradients for output projection
              const woGrad = Array(this.#hiddenSize).fill().map(() => Array(this.#hiddenSize).fill(0));
              const attentionOutputGrad = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(grad));
              for (let i = 0; i < this.#inputSize; i++) {
                  for (let j = 0; j < this.#hiddenSize; j++) {
                      for (let k = 0; k < this.#hiddenSize; k++) {
                          woGrad[k][j] += isValidNumber(attentionOutputGrad[i][j]) && isValidNumber(attentionOutput[i][k])
                              ? attentionOutputGrad[i][j] * attentionOutput[i][k]
                              : 0;
                      }
                  }
              }

              // Compute gradients for values
              const vGrad = Array(this.#numHeads).fill().map(() =>
                  Array(this.#inputSize).fill().map(() => Array(headSize).fill(0))
              );
              const scoreGrad = Array(this.#numHeads).fill().map(() =>
                  Array(this.#inputSize).fill().map(() => Array(this.#inputSize).fill(0))
              );
              for (let h = 0; h < this.#numHeads; h++) {
                  for (let i = 0; i < this.#inputSize; i++) {
                      for (let j = 0; j < this.#inputSize; j++) {
                          for (let k = 0; k < headSize; k++) {
                              const gradIndex = h * headSize + k;
                              scoreGrad[h][i][j] += isValidNumber(attentionOutputGrad[i][gradIndex]) && isValidNumber(V[j][gradIndex])
                                  ? attentionOutputGrad[i][gradIndex] * V[j][gradIndex]
                                  : 0;
                              vGrad[h][j][k] += isValidNumber(attentionProbs[h][i][j]) && isValidNumber(attentionOutputGrad[i][gradIndex])
                                  ? attentionProbs[h][i][j] * attentionOutputGrad[i][gradIndex]
                                  : 0;
                          }
                      }
                  }
              }

              // Compute gradients for queries and keys
              const qGrad = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
              const kGrad = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
              for (let h = 0; h < this.#numHeads; h++) {
                  for (let i = 0; i < this.#inputSize; i++) {
                      for (let j = 0; j < this.#inputSize; j++) {
                          for (let k = 0; k < headSize; k++) {
                              const scaledScore = isValidNumber(scoreGrad[h][i][j])
                                  ? scoreGrad[h][i][j] / Math.sqrt(headSize)
                                  : 0;
                              qGrad[i][h * headSize + k] += isValidNumber(scaledScore) && isValidNumber(K[j][h * headSize + k])
                                  ? scaledScore * K[j][h * headSize + k]
                                  : 0;
                              kGrad[j][h * headSize + k] += isValidNumber(scaledScore) && isValidNumber(Q[i][h * headSize + k])
                                  ? scaledScore * Q[i][h * headSize + k]
                                  : 0;
                          }
                      }
                  }
              }

              // Update attention weights
              for (let i = 0; i < this.#hiddenSize; i++) {
                  for (let j = 0; j < this.#hiddenSize; j++) {
                      const specializationFactor = isValidNumber(this.#specializationWeights[idx][i % this.#hiddenSize][j])
                          ? 1 + this.#specializationScores[idx] * this.#specializationWeights[idx][i % this.#hiddenSize][j]
                          : 1;
                      const wqUpdate = qGrad.reduce((sum, row) => sum + (isValidNumber(row[j]) && isValidNumber(attentionInput[i % this.#inputSize]?.[i]) ? row[j] * attentionInput[i % this.#inputSize][i] : 0), 0);
                      const wkUpdate = kGrad.reduce((sum, row) => sum + (isValidNumber(row[j]) && isValidNumber(attentionInput[i % this.#inputSize]?.[i]) ? row[j] * attentionInput[i % this.#inputSize][i] : 0), 0);
                      const wvUpdate = vGrad.reduce((sum, head) => sum + head[i % this.#inputSize].reduce((s, v) => s + (isValidNumber(v[j % headSize]) ? v[j % headSize] : 0), 0), 0);
                      const woUpdate = isValidNumber(woGrad[i][j]) ? woGrad[i][j] : 0;

                      if (isValidNumber(wqUpdate)) {
                          const clippedUpdate = Math.min(
                              Math.max(adjustedLearningRate * wqUpdate * specializationFactor, -this.#gradientClippingThreshold),
                              this.#gradientClippingThreshold
                          );
                          transformer.attentionWeights[layer].Wq[i][j] = isValidNumber(transformer.attentionWeights[layer].Wq[i][j])
                              ? transformer.attentionWeights[layer].Wq[i][j] - clippedUpdate
                              : 0;
                          this.#gradientAccumulation[idx].attentionWeights[layer].Wq[i][j] += clippedUpdate;
                      }
                      if (isValidNumber(wkUpdate)) {
                          const clippedUpdate = Math.min(
                              Math.max(adjustedLearningRate * wkUpdate * specializationFactor, -this.#gradientClippingThreshold),
                              this.#gradientClippingThreshold
                          );
                          transformer.attentionWeights[layer].Wk[i][j] = isValidNumber(transformer.attentionWeights[layer].Wk[i][j])
                              ? transformer.attentionWeights[layer].Wk[i][j] - clippedUpdate
                              : 0;
                          this.#gradientAccumulation[idx].attentionWeights[layer].Wk[i][j] += clippedUpdate;
                      }
                      if (isValidNumber(wvUpdate)) {
                          const clippedUpdate = Math.min(
                              Math.max(adjustedLearningRate * wvUpdate * specializationFactor, -this.#gradientClippingThreshold),
                              this.#gradientClippingThreshold
                          );
                          transformer.attentionWeights[layer].Wv[i][j] = isValidNumber(transformer.attentionWeights[layer].Wv[i][j])
                              ? transformer.attentionWeights[layer].Wv[i][j] - clippedUpdate
                              : 0;
                          this.#gradientAccumulation[idx].attentionWeights[layer].Wv[i][j] += clippedUpdate;
                      }
                      if (isValidNumber(woUpdate)) {
                          const clippedUpdate = Math.min(
                              Math.max(adjustedLearningRate * woUpdate * specializationFactor, -this.#gradientClippingThreshold),
                              this.#gradientClippingThreshold
                          );
                          transformer.attentionWeights[layer].Wo[i][j] = isValidNumber(transformer.attentionWeights[layer].Wo[i][j])
                              ? transformer.attentionWeights[layer].Wo[i][j] - clippedUpdate
                              : 0;
                          this.#gradientAccumulation[idx].attentionWeights[layer].Wo[i][j] += clippedUpdate;
                      }
                  }
              }

              // Update feed-forward weights
              const ffnInput = attentionInput[0] || Array(this.#hiddenSize).fill(0);
              const hidden = Array(this.#feedForwardSize).fill(0);
              for (let i = 0; i < this.#hiddenSize; i++) {
                  for (let j = 0; j < this.#feedForwardSize; j++) {
                      hidden[j] += isValidNumber(ffnInput[i]) && isValidNumber(transformer.ffnWeights[layer].W1[i][j])
                          ? ffnInput[i] * transformer.ffnWeights[layer].W1[i][j]
                          : 0;
                  }
              }
              const activated = hidden.map((val, i) => this.#gelu(val + (isValidNumber(transformer.ffnWeights[layer].b1[i]) ? transformer.ffnWeights[layer].b1[i] : 0)));
              let ffnGrad = Array(this.#feedForwardSize).fill(0);
              for (let i = 0; i < this.#feedForwardSize; i++) {
                  for (let j = 0; j < this.#hiddenSize; j++) {
                      ffnGrad[i] += isValidNumber(grad) && isValidNumber(transformer.ffnWeights[layer].W2[i][j])
                          ? grad * transformer.ffnWeights[layer].W2[i][j]
                          : 0;
                  }
                  ffnGrad[i] = isValidNumber(ffnGrad[i]) ? ffnGrad[i] * this.#geluDerivative(hidden[i]) : 0;
              }
              for (let i = 0; i < this.#hiddenSize; i++) {
                  for (let j = 0; j < this.#feedForwardSize; j++) {
                      const update = adjustedLearningRate * ffnGrad[j] * ffnInput[i];
                      if (isValidNumber(update)) {
                          const clippedUpdate = Math.min(
                              Math.max(update, -this.#gradientClippingThreshold),
                              this.#gradientClippingThreshold
                          );
                          transformer.ffnWeights[layer].W1[i][j] = isValidNumber(transformer.ffnWeights[layer].W1[i][j])
                              ? transformer.ffnWeights[layer].W1[i][j] - clippedUpdate
                              : 0;
                          this.#gradientAccumulation[idx].ffnWeights[layer].W1[i][j] += clippedUpdate;
                      }
                  }
              }
              for (let i = 0; i < this.#feedForwardSize; i++) {
                  const update = adjustedLearningRate * ffnGrad[i];
                  if (isValidNumber(update)) {
                      const clippedUpdate = Math.min(
                          Math.max(update, -this.#gradientClippingThreshold),
                          this.#gradientClippingThreshold
                      );
                      transformer.ffnWeights[layer].b1[i] = isValidNumber(transformer.ffnWeights[layer].b1[i])
                          ? transformer.ffnWeights[layer].b1[i] - clippedUpdate
                          : 0;
                      this.#gradientAccumulation[idx].ffnWeights[layer].b1[i] += clippedUpdate;
                  }
              }
              for (let i = 0; i < this.#feedForwardSize; i++) {
                  for (let j = 0; j < this.#hiddenSize; j++) {
                      const update = adjustedLearningRate * grad * activated[i];
                      if (isValidNumber(update)) {
                          const clippedUpdate = Math.min(
                              Math.max(update, -this.#gradientClippingThreshold),
                              this.#gradientClippingThreshold
                          );
                          transformer.ffnWeights[layer].W2[i][j] = isValidNumber(transformer.ffnWeights[layer].W2[i][j])
                              ? transformer.ffnWeights[layer].W2[i][j] - clippedUpdate
                              : 0;
                          this.#gradientAccumulation[idx].ffnWeights[layer].W2[i][j] += clippedUpdate;
                      }
                  }
              }
              for (let i = 0; i < this.#hiddenSize; i++) {
                  const update = adjustedLearningRate * grad;
                  if (isValidNumber(update)) {
                      const clippedUpdate = Math.min(
                          Math.max(update, -this.#gradientClippingThreshold),
                          this.#gradientClippingThreshold
                      );
                      transformer.ffnWeights[layer].b2[i] = isValidNumber(transformer.ffnWeights[layer].b2[i])
                          ? transformer.ffnWeights[layer].b2[i] - clippedUpdate
                          : 0;
                      this.#gradientAccumulation[idx].ffnWeights[layer].b2[i] += clippedUpdate;
                  }
              }

              // Update layer normalization weights
              for (let i = 0; i < this.#hiddenSize; i++) {
                  const gammaUpdate = adjustedLearningRate * grad;
                  if (isValidNumber(gammaUpdate)) {
                      const clippedGammaUpdate = Math.min(
                          Math.max(gammaUpdate, -this.#gradientClippingThreshold),
                          this.#gradientClippingThreshold
                      );
                      transformer.layerNormWeights[layer].gamma1[i] = isValidNumber(transformer.layerNormWeights[layer].gamma1[i])
                          ? transformer.layerNormWeights[layer].gamma1[i] - clippedGammaUpdate
                          : 1;
                      transformer.layerNormWeights[layer].beta1[i] = isValidNumber(transformer.layerNormWeights[layer].beta1[i])
                          ? transformer.layerNormWeights[layer].beta1[i] - clippedGammaUpdate
                          : 0;
                      transformer.layerNormWeights[layer].gamma2[i] = isValidNumber(transformer.layerNormWeights[layer].gamma2[i])
                          ? transformer.layerNormWeights[layer].gamma2[i] - clippedGammaUpdate
                          : 1;
                      transformer.layerNormWeights[layer].beta2[i] = isValidNumber(transformer.layerNormWeights[layer].beta2[i])
                          ? transformer.layerNormWeights[layer].beta2[i] - clippedGammaUpdate
                          : 0;
                      this.#gradientAccumulation[idx].layerNormWeights[layer].gamma1[i] += clippedGammaUpdate;
                      this.#gradientAccumulation[idx].layerNormWeights[layer].beta1[i] += clippedGammaUpdate;
                      this.#gradientAccumulation[idx].layerNormWeights[layer].gamma2[i] += clippedGammaUpdate;
                      this.#gradientAccumulation[idx].layerNormWeights[layer].beta2[i] += clippedGammaUpdate;
                  }
              }
          }
      });

      // Reset gradient accumulation
      this.#gradientAccumulation = this.#gradientAccumulation.map(() => ({
          outputWeights: Array(this.#hiddenSize).fill().map(() => Array(this.#outputSize).fill(0)),
          outputBias: Array(this.#outputSize).fill(0),
          attentionWeights: Array(this.#numLayers).fill().map(() => ({
              Wq: Array(this.#hiddenSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
              Wk: Array(this.#hiddenSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
              Wv: Array(this.#hiddenSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
              Wo: Array(this.#hiddenSize).fill().map(() => Array(this.#hiddenSize).fill(0))
          })),
          ffnWeights: Array(this.#numLayers).fill().map(() => ({
              W1: Array(this.#hiddenSize).fill().map(() => Array(this.#feedForwardSize).fill(0)),
              W2: Array(this.#feedForwardSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
              b1: Array(this.#feedForwardSize).fill(0),
              b2: Array(this.#hiddenSize).fill(0)
          })),
          layerNormWeights: Array(this.#numLayers).fill().map(() => ({
              gamma1: Array(this.#hiddenSize).fill(0),
              beta1: Array(this.#hiddenSize).fill(0),
              gamma2: Array(this.#hiddenSize).fill(0),
              beta2: Array(this.#hiddenSize).fill(0)
          })),
          attentionBias: Array(this.#hiddenSize).fill(0)
      }));
  }

  /**
   * Performs a forward pass through the HiveMind model, processing input data through each transformer
   * in the ensemble and combining their outputs using ensemble weights. Returns a final output array
   * representing the model's prediction, handling invalid inputs gracefully by returning a zero-filled array.
   * Ensures attention memory is properly initialized and used, and outputs are aggregated correctly.
   * @param {number[]} inputs - Array of input features (length: #inputSize, typically 6).
   * @returns {number[]} Final output array (length: #outputSize, typically 1).
   */
  forward(inputs) {
      // Validate inputs to ensure they are an array of valid numbers with correct length
      if (
          !Array.isArray(inputs) ||
          inputs.length !== this.#inputSize ||
          !inputs.every(isValidNumber)
      ) {
          return Array(this.#outputSize).fill(0); // Return zero-filled array for invalid inputs
      }

      // Initialize outputs array for all transformers
      const outputs = this.#transformers.map((transformer, idx) => {
          // Ensure attention memory is initialized for this transformer
          if (
              !Array.isArray(this.#attentionMemory[idx]) ||
              this.#attentionMemory[idx].length === 0 ||
              !this.#attentionMemory[idx][0].every(row => Array.isArray(row) && row.length === this.#hiddenSize)
          ) {
              this.#attentionMemory[idx] = Array(this.#contextWindow).fill().map(() =>
                  Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0))
              );
          }

          // Compute context-aware input embeddings
          let x = this.#contextAwareAttention(inputs, idx);

          // Process through transformer layers
          for (let layer = 0; layer < this.#numLayers; layer++) {
              // Apply first layer normalization
              const normX = x.map(row => this.#layerNorm(row, transformer.layerNormWeights[layer].gamma1, transformer.layerNormWeights[layer].beta1));

              // Perform multi-head attention
              const attentionResult = this.#multiHeadAttention(normX, transformer.attentionWeights[layer], idx);
              const attentionOutput = attentionResult.output;

              // Apply residual connection
              const attentionResidual = x.map((row, i) => row.map((val, j) =>
                  isValidNumber(val) && isValidNumber(attentionOutput[i][j]) ? val + attentionOutput[i][j] : val
              ));

              // Apply second layer normalization
              const normAttention = attentionResidual.map(row => this.#layerNorm(row, transformer.layerNormWeights[layer].gamma2, transformer.layerNormWeights[layer].beta2));

              // Compute feed-forward output for the first sequence element
              const ffnOutput = this.#feedForward(normAttention[0], transformer.ffnWeights[layer]);

              // Update x with residual connection
              x = attentionResidual.map((row, i) => row.map((val, j) =>
                  isValidNumber(val) && isValidNumber(ffnOutput[j]) ? val + ffnOutput[j] : val
              ));
          }

          // Compute final output for this transformer
          let output = Array(this.#outputSize).fill(0);
          for (let i = 0; i < this.#hiddenSize; i++) {
              for (let j = 0; j < this.#outputSize; j++) {
                  output[j] += isValidNumber(x[0][i]) && isValidNumber(transformer.outputWeights[i][j])
                      ? x[0][i] * transformer.outputWeights[i][j]
                      : 0;
              }
          }

          // Add output bias and apply sigmoid activation
          output = output.map((val, i) => isValidNumber(val) && isValidNumber(transformer.outputBias[i])
              ? val + transformer.outputBias[i]
              : val
          );
          return sigmoid(output[0]);
      });

      // Compute ensemble weights based on inputs and transformer outputs
      const ensembleWeights = this.#computeAttentionWeights(inputs, outputs);
      this.#ensembleWeights = ensembleWeights; // Update ensemble weights
      this.#normalizeEnsembleWeights(); // Ensure weights sum to 1

      // Combine transformer outputs using ensemble weights
      const finalOutput = Array(this.#outputSize).fill(0);
      for (let i = 0; i < this.#ensembleSize; i++) {
          if (isValidNumber(outputs[i]) && isValidNumber(this.#ensembleWeights[i])) {
              finalOutput[0] += outputs[i] * this.#ensembleWeights[i];
          }
      }

      // Return the final output, ensuring it's a valid number
      return [isValidNumber(finalOutput[0]) ? finalOutput[0] : 0];
  }

  /**
   * Trains the HiveMind model by processing input data and a target value, computing gradients
   * for each transformer, and updating parameters using backpropagation, knowledge distillation,
   * and weight sharing. Integrates performance scores, trust scores, and adaptive learning rates
   * to guide updates, ensuring robust learning while handling invalid inputs gracefully.
   * @param {number[]} inputs - Array of input features (length: #inputSize, typically 6).
   * @param {number} target - Target output value for training.
   * @param {number} [winRate=0.5] - Optional win rate to adjust learning rate (0 to 1).
   */
  train(inputs, target, winRate = 0.5) {
      // Validate inputs, target, and winRate
      if (
          !Array.isArray(inputs) ||
          inputs.length !== this.#inputSize ||
          !inputs.every(isValidNumber) ||
          !isValidNumber(target) ||
          !isValidNumber(winRate) ||
          winRate < 0 ||
          winRate > 1
      ) {
          return; // Early exit for invalid inputs to prevent errors
      }
      this.#trainingStepCount++;

      // Compute forward pass and individual transformer outputs
      const individualOutputs = this.#transformers.map((transformer, idx) => {
          let x = this.#contextAwareAttention(inputs, idx); // Use context-aware attention for input embeddings
          const layerOutputs = [x];
          const activations = [];
          const attentionIntermediates = [];

          // Process through transformer layers
          for (let layer = 0; layer < this.#numLayers; layer++) {
              const normX = x.map(row => this.#layerNorm(row, transformer.layerNormWeights[layer].gamma1, transformer.layerNormWeights[layer].beta1));
              const attentionResult = this.#multiHeadAttention(normX, transformer.attentionWeights[layer], idx);
              // Use attentionResult.output instead of attentionResult directly
              const attentionOutput = attentionResult.output;
              const attentionResidual = x.map((row, i) => row.map((val, j) =>
                  isValidNumber(val) && isValidNumber(attentionOutput[i][j]) ? val + attentionOutput[i][j] : val
              ));
              const normAttention = attentionResidual.map(row => this.#layerNorm(row, transformer.layerNormWeights[layer].gamma2, transformer.layerNormWeights[layer].beta2));
              const ffnOutput = this.#feedForward(normAttention[0], transformer.ffnWeights[layer]);
              x = attentionResidual.map((row, i) => row.map((val, j) =>
                  isValidNumber(val) && isValidNumber(ffnOutput[j]) ? val + ffnOutput[j] : val
              ));
              layerOutputs.push(x);
              activations.push({ normX, attentionOutput, normAttention });
              attentionIntermediates.push({
                  Q: attentionResult.Q,
                  K: attentionResult.K,
                  V: attentionResult.V,
                  attentionScores: attentionResult.scores,
                  attentionProbs: attentionResult.probs
              });
          }

          // Compute final output
          let output = Array(this.#outputSize).fill(0);
          for (let i = 0; i < this.#hiddenSize; i++) {
              for (let j = 0; j < this.#outputSize; j++) {
                  output[j] += isValidNumber(x[0][i]) && isValidNumber(transformer.outputWeights[i][j])
                      ? x[0][i] * transformer.outputWeights[i][j]
                      : 0;
              }
          }
          output = output.map((val, i) => isValidNumber(val) && isValidNumber(transformer.outputBias[i]) ? val + transformer.outputBias[i] : val);
          return sigmoid(output[0]);
      });

      // Compute ensemble output
      const output = individualOutputs.reduce((sum, out, idx) =>
          sum + (isValidNumber(out) ? out * this.#ensembleWeights[idx] : 0), 0
      );
      const error = target - output;

      // Update performance and agreement scores
      this.#performanceScores = this.#performanceScores.map((score, idx) => {
          const individualError = Math.abs(target - individualOutputs[idx]);
          const newScore = 0.9 * score + 0.1 * (1 - individualError);
          this.#historicalPerformance[idx].push(isValidNumber(newScore) ? newScore : 0);
          if (this.#historicalPerformance[idx].length > this.#maxPerformanceHistory) {
              this.#historicalPerformance[idx].shift();
          }
          return isValidNumber(newScore) ? newScore : 0;
      });

      this.#agreementScores = this.#agreementScores.map((score, idx) => {
          const agreement = 1 - Math.abs(individualOutputs[idx] - output);
          const newScore = 0.9 * score + 0.1 * (isValidNumber(agreement) ? agreement : 0);
          return isValidNumber(newScore) ? newScore : 0;
      });

      // Update trust scores and adaptive learning rates
      this.#updateTrustScores();
      this.#computeSpecializationScores(inputs, individualOutputs);
      this.#updateAdaptiveLearningRates();

      // Compute gradients and update parameters
      this.#transformers.forEach((transformer, idx) => {
          let x = this.#contextAwareAttention(inputs, idx);
          let grad = Array(this.#hiddenSize).fill(0);
          const adjustedLearningRate = this.#adaptiveLearningRate[idx] * (0.5 + 0.5 * winRate);
          const delta = Math.min(Math.max(error * output * (1 - output) * adjustedLearningRate, -1), 1);

          // Compute output layer gradients
          for (let i = 0; i < this.#hiddenSize; i++) {
              for (let j = 0; j < this.#outputSize; j++) {
                  const gradUpdate = isValidNumber(delta) && isValidNumber(this.#ensembleWeights[idx]) && isValidNumber(x[0][i])
                      ? delta * this.#ensembleWeights[idx] * x[0][i]
                      : 0;
                  const clippedUpdate = Math.min(Math.max(gradUpdate, -this.#gradientClippingThreshold), this.#gradientClippingThreshold);
                  this.#gradientAccumulation[idx].outputWeights[i][j] += clippedUpdate;
                  grad[i] += isValidNumber(delta) && isValidNumber(transformer.outputWeights[i][j])
                      ? delta * transformer.outputWeights[i][j]
                      : 0;
              }
          }
          this.#gradientAccumulation[idx].outputBias[0] += isValidNumber(delta) ? delta : 0;

          for (let i = 0; i < this.#hiddenSize; i++) {
            const biasGrad = isValidNumber(delta) ? delta / this.#inputSize : 0;
            const clippedBiasUpdate = Math.min(Math.max(adjustedLearningRate * biasGrad, -this.#gradientClippingThreshold), this.#gradientClippingThreshold);
            this.#gradientAccumulation[idx].attentionBias[i] += clippedBiasUpdate;
          }

          // Backpropagate through layers
          for (let layer = this.#numLayers - 1; layer >= 0; layer--) {
              const normX = x.map(row => this.#layerNorm(row, transformer.layerNormWeights[layer].gamma1, transformer.layerNormWeights[layer].beta1));
              const attentionResult = this.#multiHeadAttention(normX, transformer.attentionWeights[layer], idx);
              const attentionOutput = attentionResult.output; // Use attentionResult.output
              const attentionResidual = x.map((row, i) => row.map((val, j) =>
                  isValidNumber(val) && isValidNumber(attentionOutput[i][j]) ? val + attentionOutput[i][j] : val
              ));
              const normAttention = attentionResidual.map(row => this.#layerNorm(row, transformer.layerNormWeights[layer].gamma2, transformer.layerNormWeights[layer].beta2));
              const ffnInput = normAttention[0];
              const hidden = Array(this.#feedForwardSize).fill(0);
              for (let i = 0; i < this.#hiddenSize; i++) {
                  for (let j = 0; j < this.#feedForwardSize; j++) {
                      hidden[j] += isValidNumber(ffnInput[i]) && isValidNumber(transformer.ffnWeights[layer].W1[i][j])
                          ? ffnInput[i] * transformer.ffnWeights[layer].W1[i][j]
                          : 0;
                  }
              }
              const activated = hidden.map((val, i) => this.#gelu(val + (isValidNumber(transformer.ffnWeights[layer].b1[i]) ? transformer.ffnWeights[layer].b1[i] : 0)));
              let ffnGrad = Array(this.#feedForwardSize).fill(0);
              for (let i = 0; i < this.#feedForwardSize; i++) {
                  for (let j = 0; j < this.#hiddenSize; j++) {
                      ffnGrad[i] += isValidNumber(grad[j]) && isValidNumber(transformer.ffnWeights[layer].W2[i][j])
                          ? grad[j] * transformer.ffnWeights[layer].W2[i][j]
                          : 0;
                  }
                  ffnGrad[i] = isValidNumber(ffnGrad[i]) ? ffnGrad[i] * this.#geluDerivative(hidden[i]) : 0;
              }
              for (let i = 0; i < this.#hiddenSize; i++) {
                  for (let j = 0; j < this.#feedForwardSize; j++) {
                      const update = adjustedLearningRate * ffnGrad[j] * ffnInput[i];
                      this.#gradientAccumulation[idx].ffnWeights[layer].W1[i][j] += isValidNumber(update)
                          ? Math.min(Math.max(update, -this.#gradientClippingThreshold), this.#gradientClippingThreshold)
                          : 0;
                  }
              }
              for (let i = 0; i < this.#feedForwardSize; i++) {
                  const update = adjustedLearningRate * ffnGrad[i];
                  this.#gradientAccumulation[idx].ffnWeights[layer].b1[i] += isValidNumber(update)
                      ? Math.min(Math.max(update, -this.#gradientClippingThreshold), this.#gradientClippingThreshold)
                      : 0;
              }
              for (let i = 0; i < this.#feedForwardSize; i++) {
                  for (let j = 0; j < this.#hiddenSize; j++) {
                      const update = adjustedLearningRate * grad[j] * activated[i];
                      this.#gradientAccumulation[idx].ffnWeights[layer].W2[i][j] += isValidNumber(update)
                          ? Math.min(Math.max(update, -this.#gradientClippingThreshold), this.#gradientClippingThreshold)
                          : 0;
                  }
              }
              for (let i = 0; i < this.#hiddenSize; i++) {
                  const update = adjustedLearningRate * grad[i];
                  this.#gradientAccumulation[idx].ffnWeights[layer].b2[i] += isValidNumber(update)
                      ? Math.min(Math.max(update, -this.#gradientClippingThreshold), this.#gradientClippingThreshold)
                      : 0;
              }
              x = normAttention; // Update x for next layer
              grad = ffnGrad; // Propagate gradient backward
          }
      });

      // Apply accumulated gradients
      this.#transformers.forEach((transformer, idx) => {
          for (let i = 0; i < this.#hiddenSize; i++) {
              for (let j = 0; j < this.#outputSize; j++) {
                  transformer.outputWeights[i][j] -= this.#gradientAccumulation[idx].outputWeights[i][j];
              }
          }
          for (let j = 0; j < this.#outputSize; j++) {
              transformer.outputBias[j] -= this.#gradientAccumulation[idx].outputBias[j];
          }
          for (let i = 0; i < this.#hiddenSize; i++) {
            this.#attentionBias[idx][i] -= this.#gradientAccumulation[idx].attentionBias[i];
          }
          for (let layer = 0; layer < this.#numLayers; layer++) {
              for (let i = 0; i < this.#hiddenSize; i++) {
                  for (let j = 0; j < this.#feedForwardSize; j++) {
                      transformer.ffnWeights[layer].W1[i][j] -= this.#gradientAccumulation[idx].ffnWeights[layer].W1[i][j];
                  }
              }
              for (let i = 0; i < this.#feedForwardSize; i++) {
                  for (let j = 0; j < this.#hiddenSize; j++) {
                      transformer.ffnWeights[layer].W2[i][j] -= this.#gradientAccumulation[idx].ffnWeights[layer].W2[i][j];
                  }
              }
              for (let i = 0; i < this.#feedForwardSize; i++) {
                  transformer.ffnWeights[layer].b1[i] -= this.#gradientAccumulation[idx].ffnWeights[layer].b1[i];
              }
              for (let i = 0; i < this.#hiddenSize; i++) {
                  transformer.ffnWeights[layer].b2[i] -= this.#gradientAccumulation[idx].ffnWeights[layer].b2[i];
              }
          }
      });

      // Apply knowledge distillation and weight sharing
      this.#distillKnowledge(individualOutputs, target);
      this.#ensembleWeights = this.#computeAttentionWeights(inputs, individualOutputs);
      this.#normalizeEnsembleWeights();
      if (this.#shouldCommunicate()) {
          this.#shareWeights();
      }
  }

  // Getter / Setter methods
  // -----------------------

  getParameters(idx) {
    return this.#transformers[idx];
  }

  setParameters(idx, params) {
    const validateArray = (arr, defaultValue) => arr.map(val => isValidNumber(val) ? val : defaultValue);
    const validatedParams = {
      ...params,
      ffnWeights: params.ffnWeights.map(layer => ({
        W1: layer.W1.map(row => validateArray(row, 0)),
        W2: layer.W2.map(row => validateArray(row, 0)),
        b1: validateArray(layer.b1, 0),
        b2: validateArray(layer.b2, 0)
      })),
      layerNormWeights: params.layerNormWeights.map(layer => ({
        gamma1: validateArray(layer.gamma1, 1),
        beta1: validateArray(layer.beta1, 0),
        gamma2: validateArray(layer.gamma2, 1),
        beta2: validateArray(layer.beta2, 0)
      })),
      attentionWeights: params.attentionWeights.map(layer => ({
        Wq: layer.Wq.map(row => validateArray(row, 0)),
        Wk: layer.Wk.map(row => validateArray(row, 0)),
        Wv: layer.Wv.map(row => validateArray(row, 0)),
        Wo: layer.Wo.map(row => validateArray(row, 0))
      })),
      positionalEncoding: params.positionalEncoding.map(row => validateArray(row, 0)),
      outputWeights: params.outputWeights.map(row => validateArray(row, 0)),
      outputBias: validateArray(params.outputBias, 0)
    };
    this.#transformers[idx] = validatedParams;
  }

  getEnsembleWeight(idx) {
    return this.#ensembleWeights[idx];
  }

  setEnsembleWeight(idx, weight) {
    if (isValidNumber(weight) && weight >= 0) {
      this.#ensembleWeights[idx] = weight;
      this.#normalizeEnsembleWeights();
    }
  }

  getPerformanceScore(idx) {
    return this.#performanceScores[idx];
  }

  setPerformanceScore(idx, score) {
    if (isValidNumber(score) && score >= 0 && score <= 1) {
      this.#performanceScores[idx] = score;
    }
  }

  getAgreementScore(idx) {
    return this.#agreementScores[idx];
  }

  setAgreementScore(idx, score) {
    if (isValidNumber(score) && score >= 0 && score <= 1) {
      this.#agreementScores[idx] = score;
    }
  }

  getHistoricalPerformance(idx) {
    return this.#historicalPerformance[idx];
  }

  setHistoricalPerformance(idx, history) {
    if (Array.isArray(history) && history.every(isValidNumber)) {
      this.#historicalPerformance[idx] = history.slice(0, this.#maxPerformanceHistory);
    }
  }

  getTrustScoresHistory(idx) {
    return this.#trustScoresHistory[idx];
  }

  setTrustScoresHistory(idx, value) {
    if (Array.isArray(value) && value.every(isValidNumber)) {
      this.#trustScoresHistory[idx] = value.slice(0, this.#maxTrustHistory);
    }
  }

  getSpecializationScore(idx) {
    return this.#specializationScores[idx];
  }

  setSpecializationScore(idx, value) {
    if (isValidNumber(value) && value >= 0 && value <= 1) {
      this.#specializationScores[idx] = value;
    }
  }

  getSpecializationWeights(idx) {
    return this.#specializationWeights[idx];
  }

  setSpecializationWeights(idx, weights) {
    if (
      Array.isArray(weights) &&
      weights.length === this.#hiddenSize &&
      weights.every(
        w => Array.isArray(w) &&
        w.length === this.#hiddenSize &&
        w.every(row => isValidNumber(row))
      )
    ) {
      this.#specializationWeights[idx] = weights.map(row => row.map(val => isValidNumber(val) ? val : 0));
    }
  }

  getAttentionWeightMatrix(idx) {
    return this.#attentionWeightMatrix[idx];
  }

  setAttentionWeightMatrix(idx, matrix) {
    if (
      Array.isArray(matrix) &&
      matrix.length === this.#hiddenSize &&
      matrix.every(row => Array.isArray(row) && row.length === this.#hiddenSize && row.every(isValidNumber))
    ) {
      this.#attentionWeightMatrix[idx] = matrix.map(row => row.map(val => isValidNumber(val) ? val : 0));
    }
  }

  getAttentionBias(idx) {
    if (
      Number.isInteger(idx) &&
      idx >= 0 &&
      idx < this.#ensembleSize
    ) {
      return this.#attentionBias[idx];
    }
    return Array(this.#hiddenSize).fill(0);
  }

  setAttentionBias(idx, bias) {
    if (
      Number.isInteger(idx) &&
      idx >= 0 &&
      idx < this.#ensembleSize &&
      Array.isArray(bias) &&
      bias.length === this.#hiddenSize &&
      bias.every(isValidNumber)
    ) {
      this.#attentionBias[idx] = bias.map(val => isValidNumber(val) ? val : 0);
    }
  }

  getAttentionMemory(idx) {
    return this.#attentionMemory[idx];
  }

  setAttentionMemory(idx, memory) {
    if (
      Array.isArray(memory) &&
      memory.length === this.#contextWindow &&
      memory.every(
        m => Array.isArray(m) &&
        m.length === this.#hiddenSize &&
        m.every(row => Array.isArray(row) && row.length === this.#hiddenSize && row.every(isValidNumber))
      )
    ) {
      this.#attentionMemory[idx] = memory.map(m => m.map(row => row.map(val => isValidNumber(val) ? val : 0)));
    }
  }

  getAdaptiveLearningRate(idx) {
    return this.#adaptiveLearningRate[idx];
  }

  setAdaptiveLearningRate(idx, rate) {
    if (isValidNumber(rate) && rate >= 0) {
      this.#adaptiveLearningRate[idx] = rate;
    }
  }
}

class IndicatorProcessor {
  compute(candles) {
    if (!Array.isArray(candles) || candles.length < 11) return { error: true };
    const validCandles = candles.filter(c =>
      isValidNumber(c.close) && isValidNumber(c.high) && isValidNumber(c.low) &&
      isValidNumber(c.volume) && c.volume >= 0 && isValidNumber(c.timestamp)
    );
    if (validCandles.length < 11) {
      return { error: true };
    }
    const close = validCandles.map(c => Number(c.close));
    const high = validCandles.map(c => Number(c.high));
    const low = validCandles.map(c => Number(c.low));
    const volume = validCandles.map(c => Number(c.volume));
    const lastClose = close[close.length - 1];
    const volumeMean = volume.reduce((sum, v) => sum + (isValidNumber(v) ? v : 0), 0) / volume.length || 1;
    const volumeStd = Math.sqrt(
      volume.reduce((sum, v) => sum + (isValidNumber(v) ? (v - volumeMean) ** 2 : 0), 0) / volume.length
    ) || 1;
    const volumeZScore = isValidNumber(volume[volume.length - 1])
      ? (volume[volume.length - 1] - volumeMean) / volumeStd
      : 0;
    const indicators = {
      close,
      high,
      low,
      volume,
      lastClose,
      volumeZScore: Math.min(Math.max(volumeZScore, -3), 3),
      rsi: this.#computeRSI(close, 14),
      macd: this.#computeMACD(close, 8, 21, 5),
      atr: this.#computeATR(high, low, close, 14),
      isTrending: 0,
      isRanging: 0,
      marketPhase: 'neutral'
    };
    const lastMacd = indicators.macd[indicators.macd.length - 1];
    const lastRsi = indicators.rsi[indicators.rsi.length - 1];
    indicators.isTrending = lastMacd && isValidNumber(lastMacd.MACD) && isValidNumber(lastMacd.signal) && lastMacd.MACD > lastMacd.signal ? 1 : 0;
    indicators.isRanging = isValidNumber(lastRsi) && lastRsi > 30 && lastRsi < 70 ? 1 : 0;
    indicators.marketPhase = indicators.isTrending ? 'trending' : indicators.isRanging ? 'ranging' : 'volatile';
    indicators.rsiMin = Math.min(...indicators.rsi.filter(isValidNumber)) || 0;
    indicators.rsiMax = Math.max(...indicators.rsi.filter(isValidNumber)) || 100;
    indicators.macdMin = Math.min(...indicators.macd.map(m => m.MACD - m.signal).filter(isValidNumber)) || -1;
    indicators.macdMax = Math.max(...indicators.macd.map(m => m.MACD - m.signal).filter(isValidNumber)) || 1;
    indicators.atrMin = Math.min(...indicators.atr.filter(isValidNumber)) || 0;
    indicators.atrMax = Math.max(...indicators.atr.filter(isValidNumber)) || 1;
    return indicators;
  }

  #computeRSI(values, period) {
    if (!Array.isArray(values) || values.length < period + 1 || !values.every(isValidNumber)) {
      return new Array(Math.max(0, values.length - period)).fill(50);
    }
    let gains = 0, losses = 0;
    const rsi = new Array(values.length - period);
    for (let i = 1; i <= period; i++) {
      const delta = values[i] - values[i - 1];
      if (!isValidNumber(delta)) return new Array(values.length - period).fill(50);
      gains += delta > 0 ? delta : 0;
      losses += delta < 0 ? -delta : 0;
    }
    let avgGain = gains / period;
    let avgLoss = losses / period;
    if (!isValidNumber(avgGain) || !isValidNumber(avgLoss)) return new Array(values.length - period).fill(50);
    for (let i = period, j = 0; j < rsi.length; i++, j++) {
      const delta = values[i] - values[i - 1];
      const gain = delta > 0 ? delta : 0;
      const loss = delta < 0 ? -delta : 0;
      avgGain = (avgGain * (period - 1) + gain) / period;
      avgLoss = (avgLoss * (period - 1) + loss) / period;
      if (!isValidNumber(avgGain) || !isValidNumber(avgLoss)) {
        rsi[j] = 50;
        continue;
      }
      const rs = avgLoss <= 0 ? 100 : avgGain / avgLoss;
      rsi[j] = isValidNumber(rs) ? 100 - (100 / (1 + rs)) : 50;
    }
    return rsi;
  }

  #computeMACD(values, fastPeriod, slowPeriod, signalPeriod) {
    if (!Array.isArray(values) || values.length < slowPeriod + signalPeriod || !values.every(isValidNumber)) {
      return new Array(Math.max(0, values.length - slowPeriod - signalPeriod + 1)).fill({ MACD: 0, signal: 0, histogram: 0 });
    }
    const median = values.slice().sort((a, b) => a - b)[Math.floor(values.length / 2)] || 1;
    const normalizedValues = values.map(v => v / median);
    const fastEMA = this.#computeEMA(normalizedValues, fastPeriod);
    const slowEMA = this.#computeEMA(normalizedValues, slowPeriod);
    const macdLine = new Array(normalizedValues.length - (slowPeriod - fastPeriod)).fill(0);
    for (let i = slowPeriod - fastPeriod, j = 0; i < normalizedValues.length; i++, j++) {
      const diff = fastEMA[i] - slowEMA[i - (slowPeriod - fastPeriod)];
      macdLine[j] = isValidNumber(diff) ? Math.min(Math.max(diff, -1000), 1000) : 0;
    }
    const signalLine = this.#computeEMA(macdLine.slice(0, macdLine.length - signalPeriod + 1), signalPeriod);
    const macd = new Array(macdLine.length - signalPeriod + 1).fill({ MACD: 0, signal: 0, histogram: 0 });
    for (let i = signalPeriod - 1, j = 0; i < macdLine.length; i++, j++) {
      const signal = signalLine[j] || 0;
      const macdVal = macdLine[i];
      const hist = macdVal - signal;
      macd[j] = {
        MACD: isValidNumber(macdVal) ? macdVal : 0,
        signal: isValidNumber(signal) ? signal : 0,
        histogram: isValidNumber(hist) ? hist : 0
      };
    }
    return macd;
  }

  #computeEMA(values, period) {
    if (!Array.isArray(values) || values.length < period || !values.every(isValidNumber)) return new Array(values.length).fill(0);
    const alpha = 2 / (period + 1);
    const ema = new Array(values.length);
    const initialValues = values.slice(0, period);
    const initialAvg = initialValues.reduce((sum, v) => sum + v, 0) / period;
    if (!isValidNumber(initialAvg)) return new Array(values.length).fill(0);
    ema[period - 1] = Math.min(Math.max(initialAvg, -10000), 10000);
    for (let i = period; i < values.length; i++) {
      const next = alpha * values[i] + (1 - alpha) * ema[i - 1];
      ema[i] = isValidNumber(next) ? Math.min(Math.max(next, -10000), 10000) : 0;
    }
    for (let i = 0; i < period - 1; i++) {
      ema[i] = 0;
    }
    return ema;
  }

  #computeATR(high, low, close, period) {
    if (
      !Array.isArray(high) || high.length < period + 1 ||
      high.length !== low.length || low.length !== close.length ||
      !high.every(isValidNumber) || !low.every(isValidNumber) || !close.every(isValidNumber)
    ) return new Array(Math.max(0, high.length - period)).fill(0);
    const tr = new Array(close.length - 1);
    for (let i = 1; i < close.length; i++) {
      const highLow = high[i] - low[i];
      const highClose = Math.abs(high[i] - close[i - 1]);
      const lowClose = Math.abs(low[i] - close[i - 1]);
      tr[i - 1] = isValidNumber(highLow) && isValidNumber(highClose) && isValidNumber(lowClose)
        ? Math.max(highLow, highClose, lowClose)
        : 0;
    }
    const atr = new Array(tr.length - period + 1);
    let sum = tr.slice(0, period).reduce((s, v) => s + (isValidNumber(v) ? v : 0), 0);
    atr[0] = isValidNumber(sum) ? Math.min(Math.max(sum / period, 0), 1000) : 0;
    for (let i = period, j = 1; i < tr.length; i++, j++) {
      const next = (atr[j - 1] * (period - 1) + (isValidNumber(tr[i]) ? tr[i] : 0)) / period;
      atr[j] = isValidNumber(next) ? Math.min(Math.max(next, 0), 1000) : 0;
    }
    return atr;
  }
}

class NeuralSignalEngine {
  #transformer = new HiveMind();
  #indicators = new IndicatorProcessor();
  #db;
  #config = {
    minMultiplier: 1,
    maxMultiplier: 2.5,
    baseConfidenceThreshold: 60,
    atrFactor: 10,
    stopFactor: 2.5,
    learningRate: 0.25
  };

  constructor() {
    fs.mkdirSync(directoryPath, { recursive: true });
    this.#db = new Database(path.join(directoryPath, 'neural_engine.db'), { fileMustExist: false });
    this.#initDatabase();
    this.#loadState();
  }

  #initDatabase() {
    this.#db.exec(`
      CREATE TABLE IF NOT EXISTS qtable (
        state_key TEXT PRIMARY KEY,
        buy REAL NOT NULL,
        hold REAL NOT NULL
      );
      CREATE TABLE IF NOT EXISTS patterns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        bucket_key TEXT NOT NULL,
        features TEXT NOT NULL,
        score REAL NOT NULL,
        usage_count INTEGER NOT NULL DEFAULT 0,
        win_count INTEGER NOT NULL DEFAULT 0,
        UNIQUE(bucket_key, features)
      );
      CREATE TABLE IF NOT EXISTS open_trades (
        timestamp TEXT PRIMARY KEY,
        sellPrice REAL NOT NULL,
        stopLoss REAL NOT NULL,
        entryPrice REAL NOT NULL,
        confidence REAL NOT NULL,
        patternScore REAL NOT NULL,
        features TEXT NOT NULL,
        stateKey TEXT NOT NULL,
        dynamicThreshold REAL NOT NULL
      );
      CREATE TABLE IF NOT EXISTS candles (
        timestamp TEXT PRIMARY KEY,
        open REAL NOT NULL,
        high REAL NOT NULL,
        low REAL NOT NULL,
        close REAL NOT NULL,
        volume REAL NOT NULL
      );
      CREATE TABLE IF NOT EXISTS transformer_parameters (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        transformer_id TEXT NOT NULL,
        parameters TEXT NOT NULL,
        ensemble_weight REAL NOT NULL,
        performance_score REAL NOT NULL,
        agreement_score REAL NOT NULL,
        historical_performance TEXT NOT NULL,
        trust_scores TEXT NOT NULL,
        specialization_score REAL NOT NULL,
        specialization_weights TEXT NOT NULL,
        attention_weight_matrix TEXT NOT NULL,
        attention_bias TEXT NOT NULL,
        attention_memory TEXT NOT NULL,
        adaptive_learning_rate REAL NOT NULL,
        updated_at INTEGER NOT NULL,
        UNIQUE(transformer_id)
      );
      CREATE INDEX IF NOT EXISTS idx_bucket_key ON patterns(bucket_key);
      CREATE INDEX IF NOT EXISTS idx_open_trades_sellPrice ON open_trades(sellPrice);
      CREATE INDEX IF NOT EXISTS idx_open_trades_stopLoss ON open_trades(stopLoss);
      CREATE INDEX IF NOT EXISTS idx_candles_timestamp ON candles(timestamp);
      CREATE INDEX IF NOT EXISTS idx_transformer_id ON transformer_parameters(transformer_id);
    `);
  }

  #loadState() {
    const stmt = this.#db.prepare(`
      SELECT transformer_id, parameters, ensemble_weight, performance_score, 
             agreement_score, historical_performance, trust_scores, specialization_score,
             specialization_weights, attention_weight_matrix, attention_bias, 
             attention_memory, adaptive_learning_rate
      FROM transformer_parameters
    `);
    const params = stmt.all();
    
    if (params.length === 0) {
      for (let i = 0; i < 128; i++) {
        const transformerId = `transformer_${i + 1}`;
        const parameters = this.#transformer.getParameters(i);
        const weight = this.#transformer.getEnsembleWeight(i);
        const performanceScore = this.#transformer.getPerformanceScore(i);
        const agreementScore = this.#transformer.getAgreementScore(i);
        const historicalPerformance = this.#transformer.getHistoricalPerformance(i);
        const trustScores = this.#transformer.getTrustScoresHistory(i);
        const specializationScore = this.#transformer.getSpecializationScore(i);
        const specializationWeights = this.#transformer.getSpecializationWeights(i);
        const attentionWeightMatrix = this.#transformer.getAttentionWeightMatrix(i);
        const attentionBias = this.#transformer.getAttentionBias();
        const attentionMemory = this.#transformer.getAttentionMemory(i);
        const adaptiveLearningRate = this.#transformer.getAdaptiveLearningRate(i);
        this.#db.prepare(`
          INSERT OR REPLACE INTO transformer_parameters (
            transformer_id, parameters, ensemble_weight, performance_score, 
            agreement_score, historical_performance, trust_scores, specialization_score,
            specialization_weights, attention_weight_matrix, attention_bias,
            attention_memory, adaptive_learning_rate, updated_at
          )
          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `).run(
          transformerId, 
          JSON.stringify(parameters), 
          weight, 
          performanceScore, 
          agreementScore, 
          JSON.stringify(historicalPerformance), 
          JSON.stringify(trustScores), 
          specializationScore,
          JSON.stringify(specializationWeights),
          JSON.stringify(attentionWeightMatrix),
          JSON.stringify(attentionBias),
          JSON.stringify(attentionMemory),
          adaptiveLearningRate,
          Date.now()
        );
      }
    } else {
      params.forEach(param => {
        if (/^transformer_[1-128]$/.test(param.transformer_id)) {
          const idx = parseInt(param.transformer_id.split('_')[1]) - 1;
          this.#transformer.setParameters(idx, JSON.parse(param.parameters));
          this.#transformer.setEnsembleWeight(idx, param.ensemble_weight);
          this.#transformer.setPerformanceScore(idx, param.performance_score);
          this.#transformer.setAgreementScore(idx, param.agreement_score);
          this.#transformer.setHistoricalPerformance(idx, JSON.parse(param.historical_performance));
          this.#transformer.setTrustScoresHistory(idx, JSON.parse(param.trust_scores));
          this.#transformer.setSpecializationScore(idx, param.specialization_score);
          this.#transformer.setSpecializationWeights(idx, JSON.parse(param.specialization_weights));
          this.#transformer.setAttentionWeightMatrix(idx, JSON.parse(param.attention_weight_matrix));
          this.#transformer.setAttentionBias(JSON.parse(param.attention_bias));
          this.#transformer.setAttentionMemory(idx, JSON.parse(param.attention_memory));
          this.#transformer.setAdaptiveLearningRate(idx, param.adaptive_learning_rate);
        }
      });
    }
  }

  #saveState() {
    const transaction = this.#db.transaction(() => {
      for (let i = 0; i < 128; i++) {
        const transformerId = `transformer_${i + 1}`;
        const parameters = this.#transformer.getParameters(i);
        const weight = this.#transformer.getEnsembleWeight(i);
        const performanceScore = this.#transformer.getPerformanceScore(i);
        const agreementScore = this.#transformer.getAgreementScore(i);
        const historicalPerformance = this.#transformer.getHistoricalPerformance(i);
        const trustScores = this.#transformer.getTrustScoresHistory(i);
        const specializationScore = this.#transformer.getSpecializationScore(i);
        const specializationWeights = this.#transformer.getSpecializationWeights(i);
        const attentionWeightMatrix = this.#transformer.getAttentionWeightMatrix(i);
        const attentionBias = this.#transformer.getAttentionBias();
        const attentionMemory = this.#transformer.getAttentionMemory(i);
        const adaptiveLearningRate = this.#transformer.getAdaptiveLearningRate(i);
        
        this.#db.prepare(`
          INSERT OR REPLACE INTO transformer_parameters (
            transformer_id, parameters, ensemble_weight, performance_score, 
            agreement_score, historical_performance, trust_scores, specialization_score,
            specialization_weights, attention_weight_matrix, attention_bias,
            attention_memory, adaptive_learning_rate, updated_at
          )
          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `).run(
          transformerId, 
          JSON.stringify(parameters), 
          weight, 
          performanceScore, 
          agreementScore, 
          JSON.stringify(historicalPerformance), 
          JSON.stringify(trustScores), 
          specializationScore,
          JSON.stringify(specializationWeights),
          JSON.stringify(attentionWeightMatrix),
          JSON.stringify(attentionBias),
          JSON.stringify(attentionMemory),
          adaptiveLearningRate,
          Date.now()
        );
      }
    });
    transaction();
  }

  #getRecentCandles(candles) {
    if (!Array.isArray(candles) || candles.length === 0) {
      return { error: 'Invalid candle array type or length', candles: [] };
    }

    const newCandles = candles.filter(c =>
      isValidNumber(c.timestamp) &&
      isValidNumber(c.open) &&
      isValidNumber(c.high) &&
      isValidNumber(c.low) &&
      isValidNumber(c.close) &&
      isValidNumber(c.volume) &&
      c.volume >= 0
    );

    let recentCandles = [];
    const transaction = this.#db.transaction(() => {
      if (newCandles.length > 0) {
        const insertCandleStmt = this.#db.prepare(`
          INSERT OR IGNORE INTO candles (timestamp, open, high, low, close, volume)
          VALUES (?, ?, ?, ?, ?, ?)
        `);
        for (const candle of newCandles) {
          insertCandleStmt.run(
            candle.timestamp,
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume
          );
        }
      }

      const fetchCandlesStmt = this.#db.prepare(`SELECT * FROM candles ORDER BY timestamp ASC LIMIT 1000`);
      recentCandles = fetchCandlesStmt.all();

      const cleanupStmt = this.#db.prepare(`DELETE FROM candles WHERE timestamp NOT IN (SELECT timestamp FROM candles ORDER BY timestamp DESC LIMIT 1000)`);
      cleanupStmt.run();
    });
    transaction();

    if (recentCandles.length === 0) {
      return { error: 'No valid candles available', candles: [] };
    }

    return { error: null, candles: recentCandles };
  }

  #extractFeatures(data) {
    const normalize = (value, min, max) => {
      if (!isValidNumber(value) || !isValidNumber(min) || !isValidNumber(max) || max === min) return 0;
      return Math.min(1, Math.max(0, (value - min) / (max - min)));
    };
    return [
      truncateToDecimals(normalize(data.rsi[data.rsi.length - 1], data.rsiMin, data.rsiMax), 6),
      truncateToDecimals(normalize(data.macd[data.macd.length - 1].MACD - data.macd[data.macd.length - 1].signal, data.macdMin, data.macdMax), 6),
      truncateToDecimals(normalize(data.atr[data.atr.length - 1], data.atrMin, data.atrMax), 6),
      truncateToDecimals(Math.min(1, Math.max(-1, data.volumeZScore / 3)), 6),
      data.isTrending,
      data.isRanging
    ];
  }

  #generateFeatureKey(features) {
    if (!Array.isArray(features) || features.length !== 6) return 'default';
    const quantized = features.map(f => isValidNumber(f) ? Math.round(f * 10000) / 10000 : 0);
    return quantized.join('|');
  }

  #scorePattern(features) {
    const key = this.#generateFeatureKey(features);
    const stmt = this.#db.prepare(`SELECT score, features, usage_count, win_count FROM patterns WHERE bucket_key = ?`);
    const patterns = stmt.all(key);
    if (!patterns || patterns.length === 0) return 0;
    let totalScore = 0, matchCount = 0;
    for (const pattern of patterns) {
      const patternFeatures = JSON.parse(pattern.features);
      if (features.every((f, i) => isValidNumber(f) && isValidNumber(patternFeatures[i]) && Math.abs(f - patternFeatures[i]) < 0.1)) {
        const pseudoWins = 1;
        const pseudoUses = 2;
        const winRate = isValidNumber(pattern.usage_count) && pattern.usage_count > 0
          ? (pattern.win_count + pseudoWins) / (pattern.usage_count + pseudoUses)
          : 0.5;
        totalScore += pattern.score * (0.5 + 0.5 * winRate);
        matchCount++;
      }
    }
    return matchCount > 0 ? totalScore / matchCount : 0;
  }

  #computeDynamicThreshold(data, confidence, baseThreshold = this.#config.baseConfidenceThreshold, winRate = 0.5) {
    const normalize = (value, min, max) => {
      if (!isValidNumber(value) || !isValidNumber(min) || !isValidNumber(max) || max === min) return 0.5;
      return Math.min(1, Math.max(0, (value - min) / (max - min)));
    };
    const atrNorm = normalize(data.atr[data.atr.length - 1], data.atrMin, data.atrMax);
    const rsiNorm = normalize(data.rsi[data.rsi.length - 1], data.rsiMin, data.rsiMax);
    const volumeNorm = isValidNumber(data.volumeZScore) ? Math.abs(data.volumeZScore) / 3 : 0;
    const volatilityScore = (atrNorm + volumeNorm + rsiNorm * 0.5) / 2.5;
    const marketCondition = data.isTrending ? 0.8 : data.isRanging ? 1.2 : 1.0;
    let dynamicThreshold = baseThreshold * volatilityScore * marketCondition * (1 - 0.2 * winRate);
    dynamicThreshold = Math.max(40, Math.min(80, isValidNumber(dynamicThreshold) ? dynamicThreshold : 60));
    if (!isValidNumber(confidence)) return parseFloat(dynamicThreshold.toFixed(3));
    const confidenceProximity = Math.abs(confidence - dynamicThreshold) / 100;
    return parseFloat((dynamicThreshold * (1 - 0.1 * confidenceProximity)).toFixed(3));
  }

  #updateOpenTrades(candles) {
    if (!Array.isArray(candles) || candles.length === 0) return;

    const minLow = Math.min(...candles.map(c => isValidNumber(c.low) ? c.low : Infinity));
    const maxHigh = Math.max(...candles.map(c => isValidNumber(c.high) ? c.high : -Infinity));
    if (!isValidNumber(minLow) || !isValidNumber(maxHigh)) return;

    const tradesStmt = this.#db.prepare(`
      SELECT timestamp, sellPrice, stopLoss, entryPrice, confidence, patternScore, features, stateKey, dynamicThreshold
      FROM open_trades
      WHERE sellPrice BETWEEN ? AND ? OR stopLoss BETWEEN ? AND ?
    `);
    const trades = tradesStmt.all(minLow, maxHigh, minLow, maxHigh);

    const closedTrades = [];
    const keysToDelete = new Set();

    for (const trade of trades) {
      const features = JSON.parse(trade.features);
      for (const candle of candles) {
        if (!candle || !isValidNumber(candle.high) || !isValidNumber(candle.low)) continue;

        if (candle.high >= trade.sellPrice || candle.low <= trade.stopLoss) {
          const exitPrice = candle.high >= trade.sellPrice ? trade.sellPrice : trade.stopLoss;
          const outcome = (exitPrice - trade.entryPrice) / trade.entryPrice;
          closedTrades.push({
            timestamp: Date.now(),
            entryPrice: trade.entryPrice,
            exitPrice,
            confidence: trade.confidence,
            outcome: Math.min(Math.max(outcome, -1), 1),
            reward: outcome * (trade.confidence / 100),
            patternScore: trade.patternScore,
            features,
            stateKey: trade.stateKey,
            dynamicThreshold: trade.dynamicThreshold
          });
          keysToDelete.add(trade.timestamp);
          break;
        }
      }
    }

    if (closedTrades.length > 0) {
      const transaction = this.#db.transaction(() => {
        const patternStmt = this.#db.prepare(`SELECT score, usage_count, win_count FROM patterns WHERE bucket_key = ? AND features = ?`);
        const insertPatternStmt = this.#db.prepare(`
          INSERT OR REPLACE INTO patterns (bucket_key, features, score, usage_count, win_count)
          VALUES (?, ?, ?, ?, ?)
        `);
        const updateQTableStmt = this.#db.prepare(`
          INSERT OR REPLACE INTO qtable (state_key, buy, hold)
          VALUES (?, ?, COALESCE((SELECT hold FROM qtable WHERE state_key = ?), 0))
        `);
        const deleteTradeStmt = this.#db.prepare(`DELETE FROM open_trades WHERE timestamp = ?`);

        for (const trade of closedTrades) {
          const key = this.#generateFeatureKey(trade.features);
          const pattern = patternStmt.get(key, JSON.stringify(trade.features));
          const isWin = trade.outcome > 0 ? 1 : 0;
          const usageCount = pattern ? pattern.usage_count + 1 : 1;
          const winCount = pattern ? pattern.win_count + isWin : isWin;
          insertPatternStmt.run(key, JSON.stringify(trade.features), trade.reward, usageCount, winCount);

          const existingQ = this.#db.prepare(`SELECT buy, hold FROM qtable WHERE state_key = ?`).get(trade.stateKey) || { buy: 0, hold: 0 };
          updateQTableStmt.run(trade.stateKey, existingQ.buy + this.#config.learningRate * (trade.reward - existingQ.buy), trade.stateKey);

          const winRate = pattern && pattern.usage_count > 0 ? (pattern.win_count + isWin) / usageCount : isWin;
          this.#transformer.train(trade.features, trade.outcome > 0 ? (0.5 + 0.5 * winRate) : 0);
        }

        for (const key of keysToDelete) {
          deleteTradeStmt.run(key);
        }
      });
      transaction();
      this.#saveState();
    }
  }

  #computeAdvancedAction(qValues, confidence, dynamicThreshold, features, patternScore, winRate) {
    if (
      !qValues ||
      !isValidNumber(confidence) ||
      !isValidNumber(dynamicThreshold) ||
      !Array.isArray(features) ||
      features.length !== 6 ||
      !features.every(isValidNumber) ||
      !isValidNumber(patternScore) ||
      !isValidNumber(winRate)
    ) {
      return 'hold';
    }

    const qBuy = isValidNumber(qValues.buy) ? qValues.buy : 0;
    const qHold = isValidNumber(qValues.hold) ? qValues.hold : 0;
    const maxQ = Math.max(qBuy, qHold, 1e-6);
    const temperature = 0.5 + 0.5 * winRate;
    const expBuy = Math.exp((qBuy - maxQ) / temperature);
    const expHold = Math.exp((qHold - maxQ) / temperature);
    const sumExp = expBuy + expHold || 1;
    const probBuy = expBuy / sumExp;
    const probHold = expHold / sumExp;

    const renyiEntropy = -Math.log2(probBuy ** 2 + probHold ** 2 + 1e-6) / Math.log2(2);
    const normalizedEntropy = Math.min(1, renyiEntropy);

    const priorConfidence = 0.5;
    const evidenceStrength = 1 + patternScore * (0.5 + 0.5 * winRate);
    const bayesianConfidence = (confidence * evidenceStrength + priorConfidence * 0.1) / (evidenceStrength + 0.1);
    const riskAdjustedConfidence = bayesianConfidence * (1 - 0.3 * normalizedEntropy * (1 - winRate));

    const featureNorm = Math.sqrt(features.reduce((sum, f) => sum + f ** 2, 0)) || 1;
    const normalizedFeatures = features.map(f => f / featureNorm);
    const idealBuyFeature = [1, 1, 0.5, 0, 1, 0];
    const idealHoldFeature = [0.5, 0, 0.5, 0, 0, 1];
    const buySimilarity = normalizedFeatures.reduce((sum, f, i) => sum + f * idealBuyFeature[i], 0) / (Math.sqrt(idealBuyFeature.reduce((sum, f) => sum + f ** 2, 0)) || 1);
    const holdSimilarity = normalizedFeatures.reduce((sum, f, i) => sum + f * idealHoldFeature[i], 0) / (Math.sqrt(idealHoldFeature.reduce((sum, f) => sum + f ** 2, 0)) || 1);
    const contextScore = buySimilarity / (buySimilarity + holdSimilarity + 1e-6);

    const featureVariance = features.reduce((sum, f) => sum + (f - 0.5) ** 2, 0) / features.length;
    const volatilityScore = Math.sqrt(featureVariance) * (1 + 0.2 * normalizedEntropy);
    const patternReliability = patternScore > 0 ? 1 + 0.3 * patternScore : 1;
    const marketStability = winRate > 0.5 ? 1 + 0.2 * (winRate - 0.5) : 1 - 0.2 * (0.5 - winRate);
    const riskScore = 0.4 * volatilityScore + 0.4 * (1 - patternReliability) + 0.2 * (1 - marketStability);

    const baseThreshold = dynamicThreshold * patternReliability * (1 - 0.2 * normalizedEntropy);
    const logisticAdjustment = 1 / (1 + Math.exp(-10 * (contextScore - 0.5)));
    const adaptiveThreshold = baseThreshold * (0.6 + 0.4 * logisticAdjustment) * (0.7 + 0.3 * winRate);

    const decisionScore = (
      0.5 * riskAdjustedConfidence * probBuy +
      0.3 * contextScore +
      0.2 * patternScore * winRate
    ) * (1 - 0.1 * riskScore);

    const hysteresisFactor = 1.05;
    const buyThreshold = adaptiveThreshold * (probBuy > probHold ? 1 : hysteresisFactor);
    
    return decisionScore >= buyThreshold ? 'buy' : 'hold';
  }

  getSignal(candles) {
    const { error, candles: recentCandles } = this.#getRecentCandles(candles);
    if (error) {
      return { error };
    }

    this.#updateOpenTrades(recentCandles);

    const indicators = this.#indicators.compute(recentCandles);
    if (indicators.error) {
      return { error: 'Indicators error' };
    }

    const features = this.#extractFeatures(indicators);
    const patternScore = this.#scorePattern(features);
    const confidence = this.#transformer.forward(features)[0] * 100 * (1 + patternScore);

    const key = this.#generateFeatureKey(features);
    const patternStmt = this.#db.prepare(`SELECT usage_count, win_count FROM patterns WHERE bucket_key = ? AND features = ?`);
    const pattern = patternStmt.get(key, JSON.stringify(features));
    const pseudoWins = 1;
    const pseudoUses = 2;
    const winRate = pattern && pattern.usage_count > 0
      ? (pattern.win_count + pseudoWins) / (pattern.usage_count + pseudoUses)
      : 0.5;

    const dynamicThreshold = this.#computeDynamicThreshold(indicators, confidence, this.#config.baseConfidenceThreshold, winRate);
    const multiplier = this.#config.minMultiplier + (this.#config.maxMultiplier - this.#config.minMultiplier) * Math.max(0, (confidence - dynamicThreshold) / (100 - dynamicThreshold));
    let sellPrice = indicators.lastClose + this.#config.atrFactor * (indicators.atr[indicators.atr.length - 1] || 0);
    let stopLoss = indicators.lastClose - this.#config.stopFactor * (indicators.atr[indicators.atr.length - 1] || 0);
    sellPrice = isValidNumber(sellPrice) && sellPrice > indicators.lastClose && sellPrice <= indicators.lastClose * 1.3 ? sellPrice : indicators.lastClose * 1.001;
    stopLoss = isValidNumber(stopLoss) && stopLoss < indicators.lastClose && stopLoss > 0 && stopLoss >= indicators.lastClose * 0.7 ? stopLoss : indicators.lastClose * 0.999;

    const fee = 0.0021;
    const adjustedSellPrice = sellPrice * (1 - fee);
    const expectedReward = isValidNumber(adjustedSellPrice) && isValidNumber(indicators.lastClose) && indicators.lastClose !== 0
      ? (adjustedSellPrice - indicators.lastClose) / indicators.lastClose
      : 0;

    const stateKey = this.#generateFeatureKey(features);
    const qTableStmt = this.#db.prepare(`SELECT buy, hold FROM qtable WHERE state_key = ?`);
    let qValues = qTableStmt.get(stateKey);
    if (!qValues) {
      this.#db.prepare(`INSERT OR IGNORE INTO qtable (state_key, buy, hold) VALUES (?, 0, 0)`).run(stateKey);
      qValues = { buy: 0, hold: 0 };
    }

    const entryPrice = indicators.lastClose;
    const timestamp = Date.now().toString();

    if (!pattern) {
      const insertPatternStmt = this.#db.prepare(`
        INSERT OR IGNORE INTO patterns (bucket_key, features, score, usage_count, win_count)
        VALUES (?, ?, ?, 0, 0)
      `);
      insertPatternStmt.run(key, JSON.stringify(features), patternScore);
    }

    const insertTradeStmt = this.#db.prepare(`
      INSERT INTO open_trades (timestamp, sellPrice, stopLoss, entryPrice, confidence, patternScore, features, stateKey, dynamicThreshold)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);
    insertTradeStmt.run(
      timestamp,
      truncateToDecimals(sellPrice, 2),
      truncateToDecimals(stopLoss, 2),
      entryPrice,
      confidence,
      patternScore,
      JSON.stringify(features),
      stateKey,
      dynamicThreshold
    );

    const suggestedAction = this.#computeAdvancedAction(qValues, confidence, dynamicThreshold, features, patternScore, winRate);

    return {
      suggestedAction,
      multiplier: isValidNumber(multiplier) ? truncateToDecimals(multiplier, 3) : this.#config.minMultiplier,
      entryPrice,
      sellPrice: isValidNumber(sellPrice) ? truncateToDecimals(sellPrice, 2) : 0,
      stopLoss: isValidNumber(stopLoss) ? truncateToDecimals(stopLoss, 2) : 0,
      expectedReward: truncateToDecimals(expectedReward, 8)
    };
  }
}

export default NeuralSignalEngine;