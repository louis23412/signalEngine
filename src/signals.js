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
  #learningRate = 0.001;
  #ensembleSize = 128;
  #transformers = [];
  #ensembleWeights = [];
  #communicationFrequency = 10;
  #weightSharingRate = 0.05;
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
  #attentionBias = Array(this.#hiddenSize).fill(0);
  #diversityWeight = 0.1;
  #maxPerformanceHistory = 100;
  #contextWindow = 10;
  #knowledgeDistillationLoss = 0.05;
  #attentionScalingFactor = 1.0;
  #gradientClippingThreshold = 1.0;
  #swarmIntelligenceFactor = 0.2;
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
    }))
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
    for (let i = 0; i < this.#hiddenSize; i++) {
      this.#attentionBias[i] = 0;
    }
  }

  #shareWeights() {
    const validateArray = (arr, defaultValue) => arr.map(val => isValidNumber(val) ? val : defaultValue);
    const performanceSum = this.#performanceScores.reduce((sum, score) => sum + (isValidNumber(score) ? score : 0), 0) || 1;
    const trustScores = this.#performanceScores.map(score => (isValidNumber(score) ? score : 0) / performanceSum);
    const momentumFactor = 0.9;

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
          result[i][j] = totalWeight > 0 ? weightedSum / totalWeight : 0;
        }
      }
      return result;
    };

    const topPerformers = this.#performanceScores
      .map((score, idx) => ({ score: isValidNumber(score) ? score : 0, idx }))
      .sort((a, b) => b.score - a.score)
      .slice(0, Math.floor(this.#ensembleSize * 0.25))
      .map(({ idx }) => idx);

    for (let layer = 0; layer < this.#numLayers; layer++) {
      const allWq = this.#transformers.map(t => t.attentionWeights[layer].Wq);
      const allWk = this.#transformers.map(t => t.attentionWeights[layer].Wk);
      const allWv = this.#transformers.map(t => t.attentionWeights[layer].Wv);
      const allWo = this.#transformers.map(t => t.attentionWeights[layer].Wo);
      const allW1 = this.#transformers.map(t => t.ffnWeights[layer].W1);
      const allW2 = this.#transformers.map(t => t.ffnWeights[layer].W2);
      const allB1 = this.#transformers.map(t => t.ffnWeights[layer].b1);
      const allB2 = this.#transformers.map(t => t.ffnWeights[layer].b2);

      const avgWq = avgWeights(allWq, trustScores);
      const avgWk = avgWeights(allWk, trustScores);
      const avgWv = avgWeights(allWv, trustScores);
      const avgWo = avgWeights(allWo, trustScores);
      const avgW1 = avgWeights(allW1, trustScores);
      const avgW2 = avgWeights(allW2, trustScores);
      const avgB1 = validateArray(
        allB1.reduce((sum, b) => sum.map((v, i) => v + (isValidNumber(b[i]) ? b[i] * trustScores[allB1.indexOf(b)] : 0)), Array(allB1[0].length).fill(0))
          .map(v => v / (trustScores.reduce((sum, p) => sum + p, 0) || 1)), 0
      );
      const avgB2 = validateArray(
        allB2.reduce((sum, b) => sum.map((v, i) => v + (isValidNumber(b[i]) ? b[i] * trustScores[allB2.indexOf(b)] : 0)), Array(allB2[0].length).fill(0))
          .map(v => v / (trustScores.reduce((sum, p) => sum + p, 0) || 1)), 0
      );

      this.#transformers.forEach((t, idx) => {
        const sharingRate = trustScores[idx] < 0.5 ? this.#weightSharingRate + 0.1 * (0.5 - trustScores[idx]) : this.#weightSharingRate;
        const isTopPerformer = topPerformers.includes(idx) ? 1.2 : 1.0;

        for (let i = 0; i < t.attentionWeights[layer].Wq.length; i++) {
          for (let j = 0; j < t.attentionWeights[layer].Wq[i].length; j++) {
            const swarmInfluence = isTopPerformer * (1 + this.#swarmIntelligenceFactor * this.#specializationScores[idx]);
            this.#momentumWeights[idx].attentionWeights[layer].Wq[i][j] = momentumFactor * this.#momentumWeights[idx].attentionWeights[layer].Wq[i][j] +
              (1 - momentumFactor) * (isValidNumber(t.attentionWeights[layer].Wq[i][j]) && isValidNumber(avgWq[i][j])
                ? (1 - sharingRate) * t.attentionWeights[layer].Wq[i][j] + sharingRate * avgWq[i][j] * swarmInfluence
                : t.attentionWeights[layer].Wq[i][j]);
            t.attentionWeights[layer].Wq[i][j] = this.#momentumWeights[idx].attentionWeights[layer].Wq[i][j];
            this.#momentumWeights[idx].attentionWeights[layer].Wk[i][j] = momentumFactor * this.#momentumWeights[idx].attentionWeights[layer].Wk[i][j] +
              (1 - momentumFactor) * (isValidNumber(t.attentionWeights[layer].Wk[i][j]) && isValidNumber(avgWk[i][j])
                ? (1 - sharingRate) * t.attentionWeights[layer].Wk[i][j] + sharingRate * avgWk[i][j] * swarmInfluence
                : t.attentionWeights[layer].Wk[i][j]);
            t.attentionWeights[layer].Wk[i][j] = this.#momentumWeights[idx].attentionWeights[layer].Wk[i][j];
            this.#momentumWeights[idx].attentionWeights[layer].Wv[i][j] = momentumFactor * this.#momentumWeights[idx].attentionWeights[layer].Wv[i][j] +
              (1 - momentumFactor) * (isValidNumber(t.attentionWeights[layer].Wv[i][j]) && isValidNumber(avgWv[i][j])
                ? (1 - sharingRate) * t.attentionWeights[layer].Wv[i][j] + sharingRate * avgWv[i][j] * swarmInfluence
                : t.attentionWeights[layer].Wv[i][j]);
            t.attentionWeights[layer].Wv[i][j] = this.#momentumWeights[idx].attentionWeights[layer].Wv[i][j];
            this.#momentumWeights[idx].attentionWeights[layer].Wo[i][j] = momentumFactor * this.#momentumWeights[idx].attentionWeights[layer].Wo[i][j] +
              (1 - momentumFactor) * (isValidNumber(t.attentionWeights[layer].Wo[i][j]) && isValidNumber(avgWo[i][j])
                ? (1 - sharingRate) * t.attentionWeights[layer].Wo[i][j] + sharingRate * avgWo[i][j] * swarmInfluence
                : t.attentionWeights[layer].Wo[i][j]);
            t.attentionWeights[layer].Wo[i][j] = this.#momentumWeights[idx].attentionWeights[layer].Wo[i][j];
          }
        }
        for (let i = 0; i < t.ffnWeights[layer].W1.length; i++) {
          for (let j = 0; j < t.ffnWeights[layer].W1[i].length; j++) {
            const swarmInfluence = isTopPerformer * (1 + this.#swarmIntelligenceFactor * this.#specializationScores[idx]);
            this.#momentumWeights[idx].ffnWeights[layer].W1[i][j] = momentumFactor * this.#momentumWeights[idx].ffnWeights[layer].W1[i][j] +
              (1 - momentumFactor) * (isValidNumber(t.ffnWeights[layer].W1[i][j]) && isValidNumber(avgW1[i][j])
                ? (1 - sharingRate) * t.ffnWeights[layer].W1[i][j] + sharingRate * avgW1[i][j] * swarmInfluence
                : t.ffnWeights[layer].W1[i][j]);
            t.ffnWeights[layer].W1[i][j] = this.#momentumWeights[idx].ffnWeights[layer].W1[i][j];
          }
        }
        for (let i = 0; i < t.ffnWeights[layer].W2.length; i++) {
          for (let j = 0; j < t.ffnWeights[layer].W2[i].length; j++) {
            const swarmInfluence = isTopPerformer * (1 + this.#swarmIntelligenceFactor * this.#specializationScores[idx]);
            this.#momentumWeights[idx].ffnWeights[layer].W2[i][j] = momentumFactor * this.#momentumWeights[idx].ffnWeights[layer].W2[i][j] +
              (1 - momentumFactor) * (isValidNumber(t.ffnWeights[layer].W2[i][j]) && isValidNumber(avgW2[i][j])
                ? (1 - sharingRate) * t.ffnWeights[layer].W2[i][j] + sharingRate * avgW2[i][j] * swarmInfluence
                : t.ffnWeights[layer].W2[i][j]);
            t.ffnWeights[layer].W2[i][j] = this.#momentumWeights[idx].ffnWeights[layer].W2[i][j];
          }
        }
        t.ffnWeights[layer].b1 = t.ffnWeights[layer].b1.map((v, i) => {
          const swarmInfluence = isTopPerformer * (1 + this.#swarmIntelligenceFactor * this.#specializationScores[idx]);
          this.#momentumWeights[idx].ffnWeights[layer].b1[i] = momentumFactor * this.#momentumWeights[idx].ffnWeights[layer].b1[i] +
            (1 - momentumFactor) * (isValidNumber(v) && isValidNumber(avgB1[i]) ? (1 - sharingRate) * v + sharingRate * avgB1[i] * swarmInfluence : v);
          return this.#momentumWeights[idx].ffnWeights[layer].b1[i];
        });
        t.ffnWeights[layer].b2 = t.ffnWeights[layer].b2.map((v, i) => {
          const swarmInfluence = isTopPerformer * (1 + this.#swarmIntelligenceFactor * this.#specializationScores[idx]);
          this.#momentumWeights[idx].ffnWeights[layer].b2[i] = momentumFactor * this.#momentumWeights[idx].ffnWeights[layer].b2[i] +
            (1 - momentumFactor) * (isValidNumber(v) && isValidNumber(avgB2[i]) ? (1 - sharingRate) * v + sharingRate * avgB2[i] * swarmInfluence : v);
          return this.#momentumWeights[idx].ffnWeights[layer].b2[i];
        });
      });
    }
  }

  #normalizeEnsembleWeights() {
    const sum = this.#ensembleWeights.reduce((s, w) => s + w, 0) || 1;
    this.#ensembleWeights = this.#ensembleWeights.map(w => w / sum);
  }

  #gelu(x) {
    return isValidNumber(x) ? 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3))) : 0;
  }

  #geluDerivative(x) {
    if (!isValidNumber(x)) return 0;
    const cdf = 0.5 * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
    const pdf = Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
    return cdf + x * pdf;
  }

  #layerNorm(x, gamma, beta, eps = 1e-6) {
    if (!x.every(isValidNumber) || !gamma.every(isValidNumber) || !beta.every(isValidNumber)) {
      return Array(x.length).fill(0);
    }
    const mean = x.reduce((sum, val) => sum + val, 0) / x.length;
    const variance = x.reduce((sum, val) => sum + (val - mean) ** 2, 0) / x.length;
    return x.map((val, i) => isValidNumber(val) && isValidNumber(variance) ? gamma[i] * (val - mean) / Math.sqrt(variance + eps) + beta[i] : 0);
  }

  #multiHeadAttention(x, layer, transformerIdx) {
    const contextEnhancedInput = this.#contextAwareAttention(x, transformerIdx);
    const headSize = this.#hiddenSize / this.#numHeads;
    const Q = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
    const K = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
    const V = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
    
    for (let i = 0; i < this.#inputSize; i++) {
      for (let j = 0; j < this.#hiddenSize; j++) {
        for (let k = 0; k < this.#hiddenSize; k++) {
          Q[i][j] += isValidNumber(contextEnhancedInput[i][k]) && isValidNumber(layer.Wq[k][j]) 
            ? contextEnhancedInput[i][k] * layer.Wq[k][j] 
            : 0;
          K[i][j] += isValidNumber(contextEnhancedInput[i][k]) && isValidNumber(layer.Wk[k][j]) 
            ? contextEnhancedInput[i][k] * layer.Wk[k][j] 
            : 0;
          V[i][j] += isValidNumber(contextEnhancedInput[i][k]) && isValidNumber(layer.Wv[k][j]) 
            ? contextEnhancedInput[i][k] * layer.Wv[k][j] 
            : 0;
        }
      }
    }
    
    const attentionScores = Array(this.#numHeads).fill().map(() => 
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
          attentionScores[h][i][j] = isValidNumber(sum) ? sum / Math.sqrt(headSize) : 0;
        }
        attentionScores[h] = attentionScores[h].map(row => softmax(row));
      }
    }
    
    const output = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
    for (let h = 0; h < this.#numHeads; h++) {
      for (let i = 0; i < this.#inputSize; i++) {
        for (let j = 0; j < this.#inputSize; j++) {
          for (let k = 0; k < headSize; k++) {
            output[i][h * headSize + k] += isValidNumber(attentionScores[h][i][j]) && isValidNumber(V[j][h * headSize + k]) 
              ? attentionScores[h][i][j] * V[j][h * headSize + k] 
              : 0;
          }
        }
      }
    }
    
    const finalOutput = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
    for (let i = 0; i < this.#inputSize; i++) {
      for (let j = 0; j < this.#hiddenSize; j++) {
        for (let k = 0; k < this.#hiddenSize; k++) {
          finalOutput[i][j] += isValidNumber(output[i][k]) && isValidNumber(layer.Wo[k][j]) 
            ? output[i][k] * layer.Wo[k][j] 
            : 0;
        }
      }
    }
    
    return finalOutput;
  }

  #feedForward(x, layer) {
    const hidden = Array(this.#feedForwardSize).fill(0);
    for (let i = 0; i < this.#hiddenSize; i++) {
      for (let j = 0; j < this.#feedForwardSize; j++) {
        hidden[j] += isValidNumber(x[i]) && isValidNumber(layer.W1[i][j]) ? x[i] * layer.W1[i][j] : 0;
      }
    }
    const activated = hidden.map((val, i) => this.#gelu(val + (isValidNumber(layer.b1[i]) ? layer.b1[i] : 0)));
    const output = Array(this.#hiddenSize).fill(0);
    for (let i = 0; i < this.#feedForwardSize; i++) {
      for (let j = 0; j < this.#hiddenSize; j++) {
        output[j] += isValidNumber(activated[i]) && isValidNumber(layer.W2[i][j]) ? activated[i] * layer.W2[i][j] : 0;
      }
    }
    return output.map((val, i) => val + (isValidNumber(layer.b2[i]) ? layer.b2[i] : 0));
  }

  #dropout(x, rate, training = false) {
    if (!training) return x;
    return x.map(val => isValidNumber(val) && Math.random() >= rate ? val / (1 - rate) : 0);
  }

  #distillKnowledge(outputs, target) {
    if (!isValidNumber(target) || !outputs.every(isValidNumber)) return;
    
    const sortedIndices = this.#performanceScores
      .map((score, idx) => ({ score, idx }))
      .sort((a, b) => b.score - a.score)
      .map(({ idx }) => idx);
      
    const topCount = Math.floor(this.#ensembleSize * 0.25);
    const bottomCount = Math.floor(this.#ensembleSize * 0.25);
    const topTransformers = sortedIndices.slice(0, topCount);
    const bottomTransformers = sortedIndices.slice(-bottomCount);
    
    const teacherOutputs = topTransformers.map(idx => outputs[idx]);
    const teacherMean = teacherOutputs.reduce((sum, val) => sum + val, 0) / teacherOutputs.length;
    
    bottomTransformers.forEach(idx => {
      const studentOutput = outputs[idx];
      const klLoss = isValidNumber(teacherMean) && isValidNumber(studentOutput)
        ? -teacherMean * Math.log(studentOutput + 1e-6) - (1 - teacherMean) * Math.log(1 - studentOutput + 1e-6)
        : 0;
      
      const distillationLoss = this.#knowledgeDistillationLoss * klLoss;
      const grad = isValidNumber(distillationLoss) && isValidNumber(studentOutput)
        ? distillationLoss * (studentOutput - teacherMean)
        : 0;
      
      const transformer = this.#transformers[idx];
      for (let layer = 0; layer < this.#numLayers; layer++) {
        for (let i = 0; i < this.#hiddenSize; i++) {
          for (let j = 0; j < this.#hiddenSize; j++) {
            transformer.attentionWeights[layer].Wq[i][j] += isValidNumber(grad) 
              ? -this.#adaptiveLearningRate[idx] * grad * this.#specializationWeights[idx][i][j] 
              : 0;
            transformer.attentionWeights[layer].Wk[i][j] += isValidNumber(grad) 
              ? -this.#adaptiveLearningRate[idx] * grad * this.#specializationWeights[idx][i][j] 
              : 0;
            transformer.attentionWeights[layer].Wv[i][j] += isValidNumber(grad) 
              ? -this.#adaptiveLearningRate[idx] * grad * this.#specializationWeights[idx][i][j] 
              : 0;
            transformer.attentionWeights[layer].Wo[i][j] += isValidNumber(grad) 
              ? -this.#adaptiveLearningRate[idx] * grad * this.#specializationWeights[idx][i][j] 
              : 0;
          }
        }
      }
      
      for (let i = 0; i < this.#hiddenSize; i++) {
        for (let j = 0; j < this.#outputSize; j++) {
          transformer.outputWeights[i][j] += isValidNumber(grad) 
            ? -this.#adaptiveLearningRate[idx] * grad * this.#specializationWeights[idx][i][j] 
            : 0;
        }
      }
      transformer.outputBias[0] += isValidNumber(grad) ? -this.#adaptiveLearningRate[idx] * grad : 0;
    });
  }

  #shouldCommunicate() {
    const performanceStd = Math.sqrt(
      this.#performanceScores.reduce((sum, score) => sum + (score - this.#performanceScores.reduce((s, v) => s + v, 0) / this.#ensembleSize) ** 2, 0) / this.#ensembleSize
    );
    const progress = Math.min(this.#trainingStepCount / 1000, 1);
    const threshold = 0.1 * (1 - progress) + 0.05;
    return isValidNumber(performanceStd) && performanceStd > threshold;
  }

  #computeAttentionWeights(inputs, outputs) {
    if (!inputs.every(isValidNumber) || !outputs.every(isValidNumber)) {
      return Array(this.#ensembleSize).fill(1 / this.#ensembleSize);
    }
    const inputFeatures = inputs.map(x => Array(this.#hiddenSize).fill(x));
    const queries = inputFeatures.map(row => row.map((val, i) => val * this.#attentionWeightMatrix[0][i]));
    const keys = outputs.map((output, idx) => this.#attentionWeightMatrix[idx].map(w => w * output));
    const attentionScores = Array(this.#ensembleSize).fill(0);
    for (let i = 0; i < this.#ensembleSize; i++) {
      let score = 0;
      for (let j = 0; j < this.#hiddenSize; j++) {
        score += isValidNumber(queries[0][j]) && isValidNumber(keys[i][j]) ? queries[0][j] * keys[i][j] : 0;
      }
      attentionScores[i] = score / Math.sqrt(this.#hiddenSize) + (isValidNumber(this.#attentionBias[i % this.#hiddenSize]) ? this.#attentionBias[i % this.#hiddenSize] : 0);
    }
    const weights = softmax(attentionScores);
    return weights.map((w, idx) => 0.5 * w + 0.5 * this.#performanceScores[idx]);
  }

  #computeDiversityLoss(outputs) {
    if (!outputs.every(isValidNumber)) return 0;
    const meanOutput = outputs.reduce((sum, val) => sum + val, 0) / outputs.length;
    const variance = outputs.reduce((sum, val) => sum + (val - meanOutput) ** 2, 0) / outputs.length;
    const diversityLoss = -this.#diversityWeight * Math.log(variance + 1e-6);
    return isValidNumber(diversityLoss) ? diversityLoss : 0;
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
      const trustScore = sigmoid(normalizedScore);
      history.push(trustScore);
      if (history.length > this.#maxTrustHistory) history.shift();
      return history;
    });

    const trustMomentum = 0.8;
    this.#performanceScores = this.#performanceScores.map((score, idx) => {
      const recentTrust = this.#trustScoresHistory[idx].slice(-5);
      const avgTrust = recentTrust.length > 0 
        ? recentTrust.reduce((sum, val) => sum + val, 0) / recentTrust.length 
        : 0.5;
      return trustMomentum * score + (1 - trustMomentum) * avgTrust;
    });
  }

  #computeSpecializationScores(inputs, outputs) {
    const featureCorrelations = Array(this.#ensembleSize).fill().map(() => Array(this.#inputSize).fill(0));
    for (let i = 0; i < this.#ensembleSize; i++) {
      for (let j = 0; j < this.#inputSize; j++) {
        if (isValidNumber(inputs[j]) && isValidNumber(outputs[i])) {
          featureCorrelations[i][j] = Math.abs(inputs[j] - outputs[i]);
        }
      }
    }

    this.#specializationScores = featureCorrelations.map(corr => {
      const meanCorr = corr.reduce((sum, val) => sum + val, 0) / corr.length;
      return sigmoid(meanCorr);
    });

    const specializationSum = this.#specializationScores.reduce((sum, score) => sum + score, 0) || 1;
    this.#specializationScores = this.#specializationScores.map(score => score / specializationSum);
  }

  #updateAdaptiveLearningRates() {
    const performanceMean = this.#performanceScores.reduce((sum, score) => sum + (isValidNumber(score) ? score : 0), 0) / this.#ensembleSize || 1;
    this.#adaptiveLearningRate = this.#adaptiveLearningRate.map((lr, idx) => {
      const performanceDiff = isValidNumber(this.#performanceScores[idx]) 
        ? this.#performanceScores[idx] - performanceMean 
        : 0;
      const adjustment = 1 + 0.1 * sigmoid(performanceDiff);
      const newLr = this.#learningRate * adjustment;
      return Math.min(Math.max(newLr, this.#learningRate * 0.5), this.#learningRate * 2);
    });
  }

  #contextAwareAttention(inputs, transformerIdx) {
    const memory = this.#attentionMemory[transformerIdx];
    const context = memory.slice(-this.#contextWindow);
    const contextWeights = context.map(() => Array(this.#hiddenSize).fill(0));
    
    for (let i = 0; i < context.length; i++) {
      for (let j = 0; j < this.#hiddenSize; j++) {
        contextWeights[i][j] = isValidNumber(context[i][j]) ? context[i][j] * this.#attentionScalingFactor : 0;
      }
    }
    
    const aggregatedContext = contextWeights.reduce((sum, weights) => 
      sum.map((row, i) => row.map((val, j) => val + weights[i][j])), 
      Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0))
    );
    
    const output = inputs.map((row, i) => 
      row.map((val, j) => isValidNumber(val) && isValidNumber(aggregatedContext[i][j]) 
        ? val + aggregatedContext[i][j] 
        : val)
    );
    
    this.#attentionMemory[transformerIdx].push(inputs[inputs.length - 1]);
    if (this.#attentionMemory[transformerIdx].length > this.#contextWindow) {
      this.#attentionMemory[transformerIdx].shift();
    }
    
    return output;
  }

  forward(inputs) {
    if (inputs.length !== this.#inputSize || !inputs.every(isValidNumber)) return [0];
    
    const ensembleOutputs = this.#transformers.map((transformer, idx) => {
      let x = inputs.map((val, i) => transformer.positionalEncoding[i].map(pos => 
        isValidNumber(val) && isValidNumber(pos) ? val + pos : 0
      ));
      
      for (let layer = 0; layer < this.#numLayers; layer++) {
        const normX = x.map(row => this.#layerNorm(row, transformer.layerNormWeights[layer].gamma1, transformer.layerNormWeights[layer].beta1));
        const attentionOutput = this.#multiHeadAttention(normX, transformer.attentionWeights[layer], idx);
        x = x.map((row, i) => row.map((v, j) => 
          isValidNumber(v) && isValidNumber(attentionOutput[i][j]) 
            ? v + attentionOutput[i][j] * (1 + this.#specializationScores[idx]) 
            : v
        ));
        const normAttention = x.map(row => this.#layerNorm(row, transformer.layerNormWeights[layer].gamma2, transformer.layerNormWeights[layer].beta2));
        x = normAttention.map(row => {
          const ffnOutput = this.#feedForward(row, transformer.ffnWeights[layer]);
          return row.map((v, j) => 
            isValidNumber(v) && isValidNumber(ffnOutput[j]) 
              ? v + ffnOutput[j] * (1 + this.#specializationScores[idx]) 
              : v
          );
        });
      }
      
      const finalOutput = Array(this.#outputSize).fill(0);
      for (let i = 0; i < this.#hiddenSize; i++) {
        for (let j = 0; j < this.#outputSize; j++) {
          finalOutput[j] += isValidNumber(x[x.length - 1][i]) && isValidNumber(transformer.outputWeights[i][j]) 
            ? x[x.length - 1][i] * transformer.outputWeights[i][j] * (1 + this.#specializationScores[idx]) 
            : 0;
        }
      }
      
      return sigmoid(finalOutput[0] + (isValidNumber(transformer.outputBias[0]) ? transformer.outputBias[0] : 0));
    });
    
    this.#computeSpecializationScores(inputs, ensembleOutputs);
    return [ensembleOutputs.reduce((sum, output, idx) => 
      sum + (isValidNumber(output) ? output * this.#ensembleWeights[idx] : 0), 0
    )];
  }

  train(inputs, target, winRate = 0.5) {
    if (inputs.length !== this.#inputSize || !inputs.every(isValidNumber) || !isValidNumber(target) || !isValidNumber(winRate)) return;
    this.#trainingStepCount++;
    
    const output = this.forward(inputs)[0];
    const error = target - output;
    this.#updateAdaptiveLearningRates();
    
    const individualOutputs = this.#transformers.map((transformer, idx) => {
      let x = inputs.map((val, i) => transformer.positionalEncoding[i].map(pos => 
        isValidNumber(val) && isValidNumber(pos) ? val + pos : 0
      ));
      const layerOutputs = [x];
      const activations = [];
      const attentionIntermediates = [];
      
      for (let layer = 0; layer < this.#numLayers; layer++) {
        const normX = x.map(row => this.#layerNorm(row, transformer.layerNormWeights[layer].gamma1, transformer.layerNormWeights[layer].beta1));
        const headSize = this.#hiddenSize / this.#numHeads;
        const Q = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
        const K = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
        const V = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
        
        for (let i = 0; i < this.#inputSize; i++) {
          for (let j = 0; j < this.#hiddenSize; j++) {
            for (let k = 0; k < this.#hiddenSize; k++) {
              Q[i][j] += isValidNumber(normX[i][k]) && isValidNumber(transformer.attentionWeights[layer].Wq[k][j]) 
                ? normX[i][k] * transformer.attentionWeights[layer].Wq[k][j] 
                : 0;
              K[i][j] += isValidNumber(normX[i][k]) && isValidNumber(transformer.attentionWeights[layer].Wk[k][j]) 
                ? normX[i][k] * transformer.attentionWeights[layer].Wk[k][j] 
                : 0;
              V[i][j] += isValidNumber(normX[i][k]) && isValidNumber(transformer.attentionWeights[layer].Wv[k][j]) 
                ? normX[i][k] * transformer.attentionWeights[layer].Wv[k][j] 
                : 0;
            }
          }
        }
        
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
              attentionScores[h][i][j] = isValidNumber(sum) ? Math.min(Math.max(sum / Math.sqrt(headSize), -100), 100) : 0;
            }
            attentionProbs[h] = attentionScores[h].map(row => softmax(row));
          }
        }
        
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
        
        const finalAttentionOutput = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
        for (let i = 0; i < this.#inputSize; i++) {
          for (let j = 0; j < this.#hiddenSize; j++) {
            for (let k = 0; k < this.#hiddenSize; k++) {
              finalAttentionOutput[i][j] += isValidNumber(attentionOutput[i][k]) && isValidNumber(transformer.attentionWeights[layer].Wo[k][j]) 
                ? attentionOutput[i][k] * transformer.attentionWeights[layer].Wo[k][j] 
                : 0;
            }
          }
        }
        
        x = x.map((row, i) => row.map((v, j) => 
          isValidNumber(v) && isValidNumber(finalAttentionOutput[i][j]) 
            ? v + finalAttentionOutput[i][j] 
            : v
        ));
        const normAttention = x.map(row => this.#layerNorm(row, transformer.layerNormWeights[layer].gamma2, transformer.layerNormWeights[layer].beta2));
        x = normAttention.map(row => {
          const ffnOutput = this.#feedForward(row, transformer.ffnWeights[layer]);
          return this.#dropout(row.map((v, j) => 
            isValidNumber(v) && isValidNumber(ffnOutput[j]) ? v + ffnOutput[j] : v
          ), this.#dropoutRate, true);
        });
        layerOutputs.push(x);
        activations.push({ normX, attentionOutput: finalAttentionOutput, normAttention });
        attentionIntermediates.push({ Q, K, V, attentionScores, attentionProbs });
      }
      
      const finalOutput = Array(this.#outputSize).fill(0);
      for (let i = 0; i < this.#hiddenSize; i++) {
        for (let j = 0; j < this.#outputSize; j++) {
          finalOutput[j] += isValidNumber(x[x.length - 1][i]) && isValidNumber(transformer.outputWeights[i][j]) 
            ? x[x.length - 1][i] * transformer.outputWeights[i][j] 
            : 0;
        }
      }
      
      return sigmoid(finalOutput[0] + (isValidNumber(transformer.outputBias[0]) ? transformer.outputBias[0] : 0));
    });

    this.#performanceScores = this.#performanceScores.map((score, idx) => {
      const individualError = Math.abs(target - individualOutputs[idx]);
      const newScore = 0.9 * score + 0.1 * (1 - individualError);
      this.#historicalPerformance[idx].push(newScore);
      if (this.#historicalPerformance[idx].length > this.#maxPerformanceHistory) {
        this.#historicalPerformance[idx].shift();
      }
      return newScore;
    });
    
    this.#agreementScores = this.#agreementScores.map((score, idx) => {
      const agreement = 1 - Math.abs(individualOutputs[idx] - output);
      return 0.9 * score + 0.1 * agreement;
    });

    this.#updateTrustScores();
    const diversityLoss = this.#computeDiversityLoss(individualOutputs);
    const regularization = this.#performanceScores.map(score => 0.01 * (1 - score));

    const sharedGradients = {
      outputWeights: Array(this.#ensembleSize).fill().map(() => 
        Array(this.#hiddenSize).fill().map(() => Array(this.#outputSize).fill(0))
      ),
      outputBias: Array(this.#ensembleSize).fill().map(() => Array(this.#outputSize).fill(0)),
      attentionWeights: Array(this.#ensembleSize).fill().map(() => 
        Array(this.#numLayers).fill().map(() => ({
          Wq: Array(this.#hiddenSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
          Wk: Array(this.#hiddenSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
          Wv: Array(this.#hiddenSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
          Wo: Array(this.#hiddenSize).fill().map(() => Array(this.#hiddenSize).fill(0))
        }))
      ),
      ffnWeights: Array(this.#ensembleSize).fill().map(() => 
        Array(this.#numLayers).fill().map(() => ({
          W1: Array(this.#hiddenSize).fill().map(() => Array(this.#feedForwardSize).fill(0)),
          W2: Array(this.#feedForwardSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
          b1: Array(this.#feedForwardSize).fill(0),
          b2: Array(this.#hiddenSize).fill(0)
        }))
      ),
      layerNormWeights: Array(this.#ensembleSize).fill().map(() => 
        Array(this.#numLayers).fill().map(() => ({
          gamma1: Array(this.#hiddenSize).fill(0),
          beta1: Array(this.#hiddenSize).fill(0),
          gamma2: Array(this.#hiddenSize).fill(0),
          beta2: Array(this.#hiddenSize).fill(0)
        }))
      )
    };

    this.#transformers.forEach((transformer, idx) => {
      let x = inputs.map((val, i) => transformer.positionalEncoding[i].map(pos => 
        isValidNumber(val) && isValidNumber(pos) ? val + pos : 0
      ));
      const layerOutputs = [x];
      const activations = [];
      const attentionIntermediates = [];
      
      for (let layer = 0; layer < this.#numLayers; layer++) {
        const normX = x.map(row => this.#layerNorm(row, transformer.layerNormWeights[layer].gamma1, transformer.layerNormWeights[layer].beta1));
        const headSize = this.#hiddenSize / this.#numHeads;
        const Q = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
        const K = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
        const V = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
        
        for (let i = 0; i < this.#inputSize; i++) {
          for (let j = 0; j < this.#hiddenSize; j++) {
            for (let k = 0; k < this.#hiddenSize; k++) {
              Q[i][j] += isValidNumber(normX[i][k]) && isValidNumber(transformer.attentionWeights[layer].Wq[k][j]) 
                ? normX[i][k] * transformer.attentionWeights[layer].Wq[k][j] 
                : 0;
              K[i][j] += isValidNumber(normX[i][k]) && isValidNumber(transformer.attentionWeights[layer].Wk[k][j]) 
                ? normX[i][k] * transformer.attentionWeights[layer].Wk[k][j] 
                : 0;
              V[i][j] += isValidNumber(normX[i][k]) && isValidNumber(transformer.attentionWeights[layer].Wv[k][j]) 
                ? normX[i][k] * transformer.attentionWeights[layer].Wv[k][j] 
                : 0;
            }
          }
        }
        
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
              attentionScores[h][i][j] = isValidNumber(sum) ? Math.min(Math.max(sum / Math.sqrt(headSize), -100), 100) : 0;
            }
            attentionProbs[h] = attentionScores[h].map(row => softmax(row));
          }
        }
        
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
        
        const finalAttentionOutput = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
        for (let i = 0; i < this.#inputSize; i++) {
          for (let j = 0; j < this.#hiddenSize; j++) {
            for (let k = 0; k < this.#hiddenSize; k++) {
              finalAttentionOutput[i][j] += isValidNumber(attentionOutput[i][k]) && isValidNumber(transformer.attentionWeights[layer].Wo[k][j]) 
                ? attentionOutput[i][k] * transformer.attentionWeights[layer].Wo[k][j] 
                : 0;
            }
          }
        }
        
        x = x.map((row, i) => row.map((v, j) => 
          isValidNumber(v) && isValidNumber(finalAttentionOutput[i][j]) 
            ? v + finalAttentionOutput[i][j] 
            : v
        ));
        const normAttention = x.map(row => this.#layerNorm(row, transformer.layerNormWeights[layer].gamma2, transformer.layerNormWeights[layer].beta2));
        x = normAttention.map(row => {
          const ffnOutput = this.#feedForward(row, transformer.ffnWeights[layer]);
          return this.#dropout(row.map((v, j) => 
            isValidNumber(v) && isValidNumber(ffnOutput[j]) ? v + ffnOutput[j] : v
          ), this.#dropoutRate, true);
        });
        layerOutputs.push(x);
        activations.push({ normX, attentionOutput: finalAttentionOutput, normAttention });
        attentionIntermediates.push({ Q, K, V, attentionScores, attentionProbs });
      }
      
      let outputGrad = Array(this.#hiddenSize).fill(0);
      const adjustedLearningRate = this.#adaptiveLearningRate[idx] * (0.5 + 0.5 * winRate);
      const delta = Math.min(Math.max(error * output * (1 - output) * adjustedLearningRate, -1), 1);
      
      for (let i = 0; i < this.#hiddenSize; i++) {
        outputGrad[i] = isValidNumber(delta) && isValidNumber(this.#ensembleWeights[idx]) && isValidNumber(transformer.outputWeights[i][0]) 
          ? delta * this.#ensembleWeights[idx] * transformer.outputWeights[i][0] 
          : 0;
        sharedGradients.outputWeights[idx][i][0] = isValidNumber(adjustedLearningRate) && isValidNumber(outputGrad[i]) && isValidNumber(x[x.length - 1][i])
          ? Math.min(Math.max(adjustedLearningRate * outputGrad[i] * x[x.length - 1][i] - regularization[idx] * transformer.outputWeights[i][0], -this.#gradientClippingThreshold), this.#gradientClippingThreshold)
          : 0;
        this.#gradientAccumulation[idx].outputWeights[i][0] += sharedGradients.outputWeights[idx][i][0];
      }
      
      sharedGradients.outputBias[idx][0] = isValidNumber(adjustedLearningRate) && isValidNumber(delta) && isValidNumber(this.#ensembleWeights[idx])
        ? Math.min(Math.max(adjustedLearningRate * delta * this.#ensembleWeights[idx], -this.#gradientClippingThreshold), this.#gradientClippingThreshold)
        : 0;
      this.#gradientAccumulation[idx].outputBias[0] += sharedGradients.outputBias[idx][0];
      
      let grad = outputGrad;
      for (let layer = this.#numLayers - 1; layer >= 0; layer--) {
        const ffnInput = activations[layer].normAttention;
        for (let j = 0; j < this.#feedForwardSize; j++) {
          let ffnGrad = 0;
          for (let k = 0; k < this.#hiddenSize; k++) {
            ffnGrad += isValidNumber(grad[k]) && isValidNumber(transformer.ffnWeights[layer].W2[j][k]) 
              ? grad[k] * transformer.ffnWeights[layer].W2[j][k] 
              : 0;
          }
          ffnGrad = isValidNumber(ffnGrad) ? ffnGrad * this.#geluDerivative(ffnInput[ffnInput.length - 1][j]) : 0;
          
          for (let i = 0; i < this.#hiddenSize; i++) {
            sharedGradients.ffnWeights[idx][layer].W1[i][j] = isValidNumber(adjustedLearningRate) && isValidNumber(ffnGrad) && isValidNumber(ffnInput[ffnInput.length - 1][i])
              ? Math.min(Math.max(adjustedLearningRate * ffnGrad * ffnInput[ffnInput.length - 1][i] - regularization[idx] * transformer.ffnWeights[layer].W1[i][j], -this.#gradientClippingThreshold), this.#gradientClippingThreshold)
              : 0;
            this.#gradientAccumulation[idx].ffnWeights[layer].W1[i][j] += sharedGradients.ffnWeights[idx][layer].W1[i][j];
          }
          
          sharedGradients.ffnWeights[idx][layer].b1[j] = isValidNumber(adjustedLearningRate) && isValidNumber(ffnGrad)
            ? Math.min(Math.max(adjustedLearningRate * ffnGrad - regularization[idx] * transformer.ffnWeights[layer].b1[j], -this.#gradientClippingThreshold), this.#gradientClippingThreshold)
            : 0;
          this.#gradientAccumulation[idx].ffnWeights[layer].b1[j] += sharedGradients.ffnWeights[idx][layer].b1[j];
        }
        
        const w2Grad = Array(this.#hiddenSize).fill(0);
        for (let j = 0; j < this.#hiddenSize; j++) {
          for (let i = 0; i < this.#feedForwardSize; i++) {
            const activatedInput = this.#gelu(ffnInput[ffnInput.length - 1][i] + (isValidNumber(transformer.ffnWeights[layer].b1[i]) ? transformer.ffnWeights[layer].b1[i] : 0));
            w2Grad[j] += isValidNumber(grad[j]) && isValidNumber(activatedInput) ? grad[j] * activatedInput : 0;
            sharedGradients.ffnWeights[idx][layer].W2[i][j] = isValidNumber(adjustedLearningRate) && isValidNumber(grad[j]) && isValidNumber(activatedInput)
              ? Math.min(Math.max(adjustedLearningRate * grad[j] * activatedInput - regularization[idx] * transformer.ffnWeights[layer].W2[i][j], -this.#gradientClippingThreshold), this.#gradientClippingThreshold)
              : 0;
            this.#gradientAccumulation[idx].ffnWeights[layer].W2[i][j] += sharedGradients.ffnWeights[idx][layer].W2[i][j];
          }
        }
        
        grad = w2Grad;
        const { Q, K, V, attentionScores, attentionProbs } = attentionIntermediates[layer];
        const headSize = this.#hiddenSize / this.#numHeads;
        const attentionGrad = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
        
        for (let i = 0; i < this.#inputSize; i++) {
          for (let j = 0; j < this.#hiddenSize; j++) {
            attentionGrad[i][j] = isValidNumber(grad[j]) ? grad[j] : 0;
          }
        }
        
        const woGrad = Array(this.#hiddenSize).fill().map(() => Array(this.#hiddenSize).fill(0));
        const attentionOutputGrad = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
        for (let i = 0; i < this.#inputSize; i++) {
          for (let j = 0; j < this.#hiddenSize; j++) {
            for (let k = 0; k < this.#hiddenSize; k++) {
              woGrad[k][j] += isValidNumber(attentionGrad[i][j]) && isValidNumber(activations[layer].attentionOutput[i][k]) 
                ? attentionGrad[i][j] * activations[layer].attentionOutput[i][k] 
                : 0;
              attentionOutputGrad[i][k] += isValidNumber(attentionGrad[i][j]) && isValidNumber(transformer.attentionWeights[layer].Wo[k][j]) 
                ? attentionGrad[i][j] * transformer.attentionWeights[layer].Wo[k][j] 
                : 0;
            }
          }
        }
        
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
                scoreGrad[h][i][j] += isValidNumber(attentionOutputGrad[i][h * headSize + k]) && isValidNumber(V[j][h * headSize + k]) 
                  ? attentionOutputGrad[i][h * headSize + k] * V[j][h * headSize + k] 
                  : 0;
                vGrad[h][j][k] += isValidNumber(attentionProbs[h][i][j]) && isValidNumber(attentionOutputGrad[i][h * headSize + k]) 
                  ? attentionProbs[h][i][j] * attentionOutputGrad[i][h * headSize + k] 
                  : 0;
              }
            }
          }
          
          const recomputedProbs = attentionScores[h].map(row => {
            if (!row.every(isValidNumber)) return row.map(() => 1 / row.length);
            const max = Math.max(...row);
            const exp = row.map(x => Math.exp(Math.min(Math.max(x - max, -100), 100)));
            const sum = exp.reduce((a, b) => a + b, 0) || 1;
            return exp.map(x => x / sum);
          });
          
          for (let i = 0; i < this.#inputSize; i++) {
            for (let j = 0; j < this.#inputSize; j++) {
              let softmaxGrad = 0;
              for (let k = 0; k < this.#inputSize; k++) {
                const prob = recomputedProbs[i][k];
                const delta = (j === k ? 1 : 0);
                softmaxGrad += isValidNumber(prob) && isValidNumber(scoreGrad[h][i][k]) 
                  ? prob * (delta - prob) * scoreGrad[h][i][k] 
                  : 0;
              }
              scoreGrad[h][i][j] = isValidNumber(softmaxGrad) ? Math.min(Math.max(softmaxGrad, -1), 1) : 0;
            }
          }
        }
        
        const qGrad = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
        const kGrad = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
        for (let h = 0; h < this.#numHeads; h++) {
          for (let i = 0; i < this.#inputSize; i++) {
            for (let j = 0; j < this.#inputSize; j++) {
              for (let k = 0; k < headSize; k++) {
                const scaledScore = isValidNumber(scoreGrad[h][i][j]) ? scoreGrad[h][i][j] / Math.sqrt(headSize) : 0;
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
        
        const wqGrad = Array(this.#hiddenSize).fill().map(() => Array(this.#hiddenSize).fill(0));
        const wkGrad = Array(this.#hiddenSize).fill().map(() => Array(this.#hiddenSize).fill(0));
        const wvGrad = Array(this.#hiddenSize).fill().map(() => Array(this.#hiddenSize).fill(0));
        for (let i = 0; i < this.#inputSize; i++) {
          for (let j = 0; j < this.#hiddenSize; j++) {
            for (let k = 0; k < this.#hiddenSize; k++) {
              wqGrad[k][j] += isValidNumber(qGrad[i][j]) && isValidNumber(activations[layer].normX[i][k]) 
                ? qGrad[i][j] * activations[layer].normX[i][k] 
                : 0;
              wkGrad[k][j] += isValidNumber(kGrad[i][j]) && isValidNumber(activations[layer].normX[i][k]) 
                ? kGrad[i][j] * activations[layer].normX[i][k] 
                : 0;
              wvGrad[k][j] += isValidNumber(vGrad.reduce((sum, head) => sum + head[i][j % headSize], 0)) && isValidNumber(activations[layer].normX[i][k]) 
                ? vGrad.reduce((sum, head) => sum + head[i][j % headSize], 0) * activations[layer].normX[i][k] 
                : 0;
            }
          }
        }
        
        for (let i = 0; i < this.#hiddenSize; i++) {
          for (let j = 0; j < this.#hiddenSize; j++) {
            sharedGradients.attentionWeights[idx][layer].Wq[i][j] = isValidNumber(adjustedLearningRate) && isValidNumber(wqGrad[i][j])
              ? Math.min(Math.max(adjustedLearningRate * wqGrad[i][j] - regularization[idx] * transformer.attentionWeights[layer].Wq[i][j], -this.#gradientClippingThreshold), this.#gradientClippingThreshold)
              : 0;
            sharedGradients.attentionWeights[idx][layer].Wk[i][j] = isValidNumber(adjustedLearningRate) && isValidNumber(wkGrad[i][j])
              ? Math.min(Math.max(adjustedLearningRate * wkGrad[i][j] - regularization[idx] * transformer.attentionWeights[layer].Wk[i][j], -this.#gradientClippingThreshold), this.#gradientClippingThreshold)
              : 0;
            sharedGradients.attentionWeights[idx][layer].Wv[i][j] = isValidNumber(adjustedLearningRate) && isValidNumber(wvGrad[i][j])
              ? Math.min(Math.max(adjustedLearningRate * wvGrad[i][j] - regularization[idx] * transformer.attentionWeights[layer].Wv[i][j], -this.#gradientClippingThreshold), this.#gradientClippingThreshold)
              : 0;
            sharedGradients.attentionWeights[idx][layer].Wo[i][j] = isValidNumber(adjustedLearningRate) && isValidNumber(woGrad[i][j])
              ? Math.min(Math.max(adjustedLearningRate * woGrad[i][j] - regularization[idx] * transformer.attentionWeights[layer].Wo[i][j], -this.#gradientClippingThreshold), this.#gradientClippingThreshold)
              : 0;
            this.#gradientAccumulation[idx].attentionWeights[layer].Wq[i][j] += sharedGradients.attentionWeights[idx][layer].Wq[i][j];
            this.#gradientAccumulation[idx].attentionWeights[layer].Wk[i][j] += sharedGradients.attentionWeights[idx][layer].Wk[i][j];
            this.#gradientAccumulation[idx].attentionWeights[layer].Wv[i][j] += sharedGradients.attentionWeights[idx][layer].Wv[i][j];
            this.#gradientAccumulation[idx].attentionWeights[layer].Wo[i][j] += sharedGradients.attentionWeights[idx][layer].Wo[i][j];
          }
        }
        
        for (let i = 0; i < this.#hiddenSize; i++) {
          grad[i] = isValidNumber(grad[i]) ? Math.min(Math.max(grad[i], -this.#gradientClippingThreshold), this.#gradientClippingThreshold) : 0;
          sharedGradients.layerNormWeights[idx][layer].gamma1[i] = isValidNumber(adjustedLearningRate) && isValidNumber(grad[i])
            ? Math.min(Math.max(adjustedLearningRate * grad[i] - regularization[idx] * transformer.layerNormWeights[layer].gamma1[i], -this.#gradientClippingThreshold), this.#gradientClippingThreshold)
            : 0;
          sharedGradients.layerNormWeights[idx][layer].beta1[i] = isValidNumber(adjustedLearningRate) && isValidNumber(grad[i])
            ? Math.min(Math.max(adjustedLearningRate * grad[i] - regularization[idx] * transformer.layerNormWeights[layer].beta1[i], -this.#gradientClippingThreshold), this.#gradientClippingThreshold)
            : 0;
          sharedGradients.layerNormWeights[idx][layer].gamma2[i] = isValidNumber(adjustedLearningRate) && isValidNumber(grad[i])
            ? Math.min(Math.max(adjustedLearningRate * grad[i] - regularization[idx] * transformer.layerNormWeights[layer].gamma2[i], -this.#gradientClippingThreshold), this.#gradientClippingThreshold)
            : 0;
          sharedGradients.layerNormWeights[idx][layer].beta2[i] = isValidNumber(adjustedLearningRate) && isValidNumber(grad[i])
            ? Math.min(Math.max(adjustedLearningRate * grad[i] - regularization[idx] * transformer.layerNormWeights[layer].beta2[i], -this.#gradientClippingThreshold), this.#gradientClippingThreshold)
            : 0;
          this.#gradientAccumulation[idx].layerNormWeights[layer].gamma1[i] += sharedGradients.layerNormWeights[idx][layer].gamma1[i];
          this.#gradientAccumulation[idx].layerNormWeights[layer].beta1[i] += sharedGradients.layerNormWeights[idx][layer].beta1[i];
          this.#gradientAccumulation[idx].layerNormWeights[layer].gamma2[i] += sharedGradients.layerNormWeights[idx][layer].gamma2[i];
          this.#gradientAccumulation[idx].layerNormWeights[layer].beta2[i] += sharedGradients.layerNormWeights[idx][layer].beta2[i];
        }
        
        grad = qGrad.map((row, i) => row.map((v, j) => 
          isValidNumber(v) && isValidNumber(kGrad[i][j]) && isValidNumber(vGrad.reduce((sum, head) => sum + head[i][j % headSize], 0))
            ? v + kGrad[i][j] + vGrad.reduce((sum, head) => sum + head[i][j % headSize], 0)
            : 0
        ));
      }
      
      return { layerOutputs, activations, attentionIntermediates };
    });

    // Apply accumulated gradients
    this.#transformers.forEach((transformer, idx) => {
      for (let i = 0; i < this.#hiddenSize; i++) {
        for (let j = 0; j < this.#outputSize; j++) {
          transformer.outputWeights[i][j] += this.#gradientAccumulation[idx].outputWeights[i][j];
        }
      }
      for (let j = 0; j < this.#outputSize; j++) {
        transformer.outputBias[j] += this.#gradientAccumulation[idx].outputBias[j];
      }
      
      for (let layer = 0; layer < this.#numLayers; layer++) {
        for (let i = 0; i < this.#hiddenSize; i++) {
          for (let j = 0; j < this.#hiddenSize; j++) {
            transformer.attentionWeights[layer].Wq[i][j] += this.#gradientAccumulation[idx].attentionWeights[layer].Wq[i][j];
            transformer.attentionWeights[layer].Wk[i][j] += this.#gradientAccumulation[idx].attentionWeights[layer].Wk[i][j];
            transformer.attentionWeights[layer].Wv[i][j] += this.#gradientAccumulation[idx].attentionWeights[layer].Wv[i][j];
            transformer.attentionWeights[layer].Wo[i][j] += this.#gradientAccumulation[idx].attentionWeights[layer].Wo[i][j];
          }
        }
        
        for (let i = 0; i < this.#hiddenSize; i++) {
          for (let j = 0; j < this.#feedForwardSize; j++) {
            transformer.ffnWeights[layer].W1[i][j] += this.#gradientAccumulation[idx].ffnWeights[layer].W1[i][j];
          }
        }
        
        for (let i = 0; i < this.#feedForwardSize; i++) {
          for (let j = 0; j < this.#hiddenSize; j++) {
            transformer.ffnWeights[layer].W2[i][j] += this.#gradientAccumulation[idx].ffnWeights[layer].W2[i][j];
          }
        }
        
        for (let i = 0; i < this.#feedForwardSize; i++) {
          transformer.ffnWeights[layer].b1[i] += this.#gradientAccumulation[idx].ffnWeights[layer].b1[i];
        }
        
        for (let i = 0; i < this.#hiddenSize; i++) {
          transformer.ffnWeights[layer].b2[i] += this.#gradientAccumulation[idx].ffnWeights[layer].b2[i];
          transformer.layerNormWeights[layer].gamma1[i] += this.#gradientAccumulation[idx].layerNormWeights[layer].gamma1[i];
          transformer.layerNormWeights[layer].beta1[i] += this.#gradientAccumulation[idx].layerNormWeights[layer].beta1[i];
          transformer.layerNormWeights[layer].gamma2[i] += this.#gradientAccumulation[idx].layerNormWeights[layer].gamma2[i];
          transformer.layerNormWeights[layer].beta2[i] += this.#gradientAccumulation[idx].layerNormWeights[layer].beta2[i];
        }
      }
      
      // Reset gradient accumulation
      this.#gradientAccumulation[idx] = {
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
      };
    });

    this.#distillKnowledge(individualOutputs, target);
    this.#ensembleWeights = this.#computeAttentionWeights(inputs, individualOutputs);
    this.#normalizeEnsembleWeights();

    if (this.#shouldCommunicate()) {
      this.#shareWeights();
    }
  }

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
    this.#trustScoresHistory[idx] = value;
  }

  getSpecializationScore(idx) {
    return this.#specializationScores[idx];
  }

  setSpecializationScore(idx, value) {
    this.#specializationScores[idx] = value;
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
            agreement_score, historical_performance, trust_scores, specialization_score 
      FROM transformer_parameters
    `);
    const params = stmt.all();
    
    if (params.length === 0) {
      for (let i = 0; i < 128; i++) {
        const transformerId = `transformer_${i + 1}`;
        const parameters = this.#transformer.getParameters(i);
        const weight = this.#transformer.getEnsembleWeight(i);
        this.#db.prepare(`
          INSERT OR REPLACE INTO transformer_parameters (
            transformer_id, parameters, ensemble_weight, performance_score, 
            agreement_score, historical_performance, trust_scores, specialization_score, updated_at
          )
          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        `).run(
          transformerId, 
          JSON.stringify(parameters), 
          weight, 
          0, 
          0, 
          JSON.stringify([0]), 
          JSON.stringify([0]), 
          0, 
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
        const historicalPerformance = JSON.stringify(this.#transformer.getHistoricalPerformance(i));
        const trustScores = JSON.stringify(this.#transformer.getTrustScoresHistory(i));
        const specializationScore = this.#transformer.getSpecializationScore(i);
        
        this.#db.prepare(`
          INSERT OR REPLACE INTO transformer_parameters (
            transformer_id, 
            parameters, 
            ensemble_weight, 
            performance_score, 
            agreement_score, 
            historical_performance, 
            trust_scores, 
            specialization_score, 
            updated_at
          )
          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        `).run(
          transformerId, 
          JSON.stringify(parameters), 
          weight, 
          performanceScore, 
          agreementScore, 
          historicalPerformance, 
          trustScores, 
          specializationScore, 
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
    // Input validation with robust fallback
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

    // Normalize Q-values using tempered softmax for numerical stability
    const qBuy = isValidNumber(qValues.buy) ? qValues.buy : 0;
    const qHold = isValidNumber(qValues.hold) ? qValues.hold : 0;
    const maxQ = Math.max(qBuy, qHold, 1e-6);
    const temperature = 0.5 + 0.5 * winRate; // Adaptive temperature based on win rate
    const expBuy = Math.exp((qBuy - maxQ) / temperature);
    const expHold = Math.exp((qHold - maxQ) / temperature);
    const sumExp = expBuy + expHold || 1;
    const probBuy = expBuy / sumExp;
    const probHold = expHold / sumExp;

    // Compute information-theoretic uncertainty (Rényi entropy, order 2)
    const renyiEntropy = -Math.log2(probBuy ** 2 + probHold ** 2 + 1e-6) / Math.log2(2);
    const normalizedEntropy = Math.min(1, renyiEntropy);

    // Bayesian-inspired confidence update
    const priorConfidence = 0.5; // Neutral prior
    const evidenceStrength = 1 + patternScore * (0.5 + 0.5 * winRate); // Pattern reliability
    const bayesianConfidence = (confidence * evidenceStrength + priorConfidence * 0.1) / (evidenceStrength + 0.1);
    const riskAdjustedConfidence = bayesianConfidence * (1 - 0.3 * normalizedEntropy * (1 - winRate));

    // Feature-space clustering for context awareness (cosine similarity-based)
    const featureNorm = Math.sqrt(features.reduce((sum, f) => sum + f ** 2, 0)) || 1;
    const normalizedFeatures = features.map(f => f / featureNorm);
    const idealBuyFeature = [1, 1, 0.5, 0, 1, 0]; // Hypothetical buy signal feature vector
    const idealHoldFeature = [0.5, 0, 0.5, 0, 0, 1]; // Hypothetical hold signal feature vector
    const buySimilarity = normalizedFeatures.reduce((sum, f, i) => sum + f * idealBuyFeature[i], 0) / (Math.sqrt(idealBuyFeature.reduce((sum, f) => sum + f ** 2, 0)) || 1);
    const holdSimilarity = normalizedFeatures.reduce((sum, f, i) => sum + f * idealHoldFeature[i], 0) / (Math.sqrt(idealHoldFeature.reduce((sum, f) => sum + f ** 2, 0)) || 1);
    const contextScore = buySimilarity / (buySimilarity + holdSimilarity + 1e-6);

    // Multi-factor risk assessment
    const featureVariance = features.reduce((sum, f) => sum + (f - 0.5) ** 2, 0) / features.length;
    const volatilityScore = Math.sqrt(featureVariance) * (1 + 0.2 * normalizedEntropy);
    const patternReliability = patternScore > 0 ? 1 + 0.3 * patternScore : 1;
    const marketStability = winRate > 0.5 ? 1 + 0.2 * (winRate - 0.5) : 1 - 0.2 * (0.5 - winRate);
    const riskScore = 0.4 * volatilityScore + 0.4 * (1 - patternReliability) + 0.2 * (1 - marketStability);

    // Dynamic decision boundary with logistic adjustment
    const baseThreshold = dynamicThreshold * patternReliability * (1 - 0.2 * normalizedEntropy);
    const logisticAdjustment = 1 / (1 + Math.exp(-10 * (contextScore - 0.5))); // Sigmoid for smooth boundary
    const adaptiveThreshold = baseThreshold * (0.6 + 0.4 * logisticAdjustment) * (0.7 + 0.3 * winRate);

    // Decision score with weighted contributions
    const decisionScore = (
      0.5 * riskAdjustedConfidence * probBuy +
      0.3 * contextScore +
      0.2 * patternScore * winRate
    ) * (1 - 0.1 * riskScore);

    // Robust decision rule with hysteresis to prevent oscillation
    const hysteresisFactor = 1.05; // Slight bias to maintain current action
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