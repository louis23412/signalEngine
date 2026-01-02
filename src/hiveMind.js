import fs from 'fs';
import path from 'path';
import Database from 'better-sqlite3';

import { isValidNumber } from './utils.js';

class HiveMind {
    #directoryPath;
    #ensembleSize;
    #inputSize;
    #numLayers;
    #numHeads;
    #headDim;
    #hiddenSize;
    #feedForwardSize;
    #contextWindow;
    #adaptiveWindow;
    #maxTrustHistory;
    #maxPerformanceHistory;
    #learningRate;
    #learningRateDecay;
    #dropoutRate;
    #swarmIntelligenceFactor;
    #gradientResetFrequency;
    #trainingStepCount = 0;
    #adaptiveLearningRate = [];
    #performanceScores = [];
    #agreementScores = [];
    #specializationScores = [];
    #historicalPerformance = [];
    #trustScoresHistory = [];
    #ensembleWeights = [];
    #attentionWeightMatrix = [];
    #attentionBias = [];
    #attentionMemory = [];
    #adaptiveContext = [];
    #specializationWeights = [];
    #transformers = [];
    #gradientAccumulation = [];

    constructor(dp, es, is) {
        this.#scaleAndSetDimensions(dp, es, is);

        const loadStatus = this.#loadState();

        if (!loadStatus.status && loadStatus.error) {
            console.log(`Load state failed! Error: ${loadStatus.error}. Trace: ${loadStatus.trace}`);
            process.exit();
        }
    }

    #scaleAndSetDimensions (dp, es, is) {
        this.#directoryPath = dp;
        this.#ensembleSize = es;
        this.#inputSize = is;

        const layerDecay = Math.min(7.0, 2.5 * Math.log10(this.#ensembleSize || 1));
        this.#numLayers = Math.max(1, Math.min(8, Math.round(8 - layerDecay)));

        const headDecay = Math.min(6.0, 2.0 * Math.log10(this.#ensembleSize || 1));
        this.#numHeads = Math.max(2, Math.min(8, Math.round(8 - headDecay)));

        const complexity = this.#ensembleSize * this.#numLayers * this.#numHeads;
        const safeComplexity = Math.max(1, complexity);
        const logComplexity = Number((Math.log10(safeComplexity)).toPrecision(6));

        const hsExponent = Math.max(2, Math.min(4, 4 - Math.log2(Math.sqrt(this.#ensembleSize || 1))));
        const hs = Math.pow(2, Math.floor(hsExponent));
        this.#headDim = hs * Math.ceil(this.#numHeads / 4);
        this.#hiddenSize = this.#numHeads * this.#headDim;

        const ffnDecay = Math.min(2.0, Math.log10(this.#ensembleSize || 1));
        const ffnMultiplier = 4.0 - ffnDecay;
        this.#feedForwardSize = Math.round(this.#hiddenSize * Math.max(2, Math.min(4, ffnMultiplier)));

        const memoryFactor = 1 + 0.5 * Math.log10(this.#ensembleSize || 1);
        this.#contextWindow = Math.round(this.#hiddenSize * 2 * memoryFactor);
        this.#adaptiveWindow = Math.max(1, Math.round(this.#contextWindow * 0.25));
        this.#maxTrustHistory = Math.round(this.#contextWindow * 2);
        this.#maxPerformanceHistory = Math.round(this.#contextWindow * 4);

        const baseLrUnscaled = 0.012 / Math.max(this.#inputSize, 1);
        const sizeScale = Math.pow(this.#hiddenSize, -0.18);
        this.#learningRate = Number(Math.min(0.0025, baseLrUnscaled * sizeScale).toPrecision(6));

        this.#learningRateDecay = Number((this.#learningRate / 10).toPrecision(6));

        const rawDropout = 0.05 * logComplexity;
        this.#dropoutRate = Number(Math.max(0.05, Math.min(0.3, rawDropout)).toFixed(6));

        const baseGrf = 50 + 12 * logComplexity;
        this.#gradientResetFrequency = Math.max(50, Math.min(600, Math.round(baseGrf)));

        const ensembleFactor = this.#ensembleSize > 1 
            ? Math.min(2.0, Math.log10(this.#ensembleSize))
            : 1;
        const rawSwarm = 0.4 + 0.4 * (ensembleFactor / 2.0);
        this.#swarmIntelligenceFactor = Number(rawSwarm.toFixed(6));

        this.#performanceScores = Array(this.#ensembleSize).fill(0);
        this.#agreementScores = Array(this.#ensembleSize).fill(0);
        this.#specializationScores = Array(this.#ensembleSize).fill(0);
        this.#trustScoresHistory = Array(this.#ensembleSize).fill().map(() => [0]);
        this.#historicalPerformance = Array(this.#ensembleSize).fill().map(() => [0]);
        this.#adaptiveLearningRate = Array(this.#ensembleSize).fill(this.#learningRate);

        this.#ensembleWeights = Array(this.#ensembleSize).fill().map(() => 
            Math.max(0, 1 / this.#ensembleSize + (Math.random() - 0.5) * 0.1 / this.#ensembleSize)
        );
        this.#normalizeEnsembleWeights();

        this.#attentionMemory = Array(this.#ensembleSize).fill().map(() =>
            Array(this.#contextWindow).fill().map(() => Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0)))
        );

        this.#adaptiveContext = Array(this.#ensembleSize).fill().map(() =>
            Array(this.#adaptiveWindow).fill().map(() => Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0)))
        );

        this.#attentionWeightMatrix = Array(this.#ensembleSize).fill().map(() =>
            this.#dynamicGeluInit(this.#hiddenSize, 1, 0, 1).map(row => row[0])
        );

        this.#attentionBias = Array(this.#ensembleSize).fill().map(() =>
            Array(this.#hiddenSize).fill().map(() => (Math.random() - 0.5) * Math.sqrt(4 / this.#hiddenSize))
        );

        this.#specializationWeights = Array(this.#ensembleSize).fill().map(() =>
            Array(this.#hiddenSize).fill().map((_, j) =>
                Array(this.#hiddenSize).fill().map((_, k) => {
                    const baseWeights = this.#dynamicGeluInit(this.#hiddenSize, this.#hiddenSize, 0, 1);
                    const scale = 1 + 0.1 * (j + k / this.#hiddenSize);
                    return baseWeights[j][k] * scale;
                })
            )
        );

        this.#transformers = this.#setTransformerStructure();

        this.#gradientAccumulation = this.#setGradientStructure();
    }

    #dynamicGeluInit (rows, cols, layerIndex, totalLayers, customK = null) {
        let baseK = customK !== null ? customK : 2.2;
        let fanInScale = Math.min(1.0, 1000 / rows);
        let depthScale = Math.pow(totalLayers, -layerIndex / totalLayers);
        let k = baseK * fanInScale * depthScale;
        k = Math.max(1.5, Math.min(k, 3.0));
        
        return Array(rows).fill().map(() =>
            Array(cols).fill().map(() => (Math.random() - 0.5) * Math.sqrt(k / rows))
        );
    }

    #createPositionalEncoding () {
        const base = 10000 + (Math.random() - 0.5) * 2000;
        return Array(this.#inputSize).fill().map((_, pos) =>
            Array(this.#hiddenSize).fill().map((_, d) => {
                const exponent = 2 * Math.floor(d / 2) / this.#hiddenSize;
                const freq = 1 / (base ** exponent);
                return d % 2 === 0 ? Math.sin(pos * freq) : Math.cos(pos * freq);
            })
        );
    }

    #setTransformerStructure () {
        return Array(this.#ensembleSize).fill().map(() => ({
            positionalEncoding: this.#createPositionalEncoding(),
            attentionWeights: Array(this.#numLayers).fill().map((_, layerIndex) => ({
                Wq: this.#dynamicGeluInit(this.#hiddenSize, this.#hiddenSize, layerIndex, this.#numLayers),
                Wk: this.#dynamicGeluInit(this.#hiddenSize, this.#hiddenSize, layerIndex, this.#numLayers),
                Wv: this.#dynamicGeluInit(this.#hiddenSize, this.#hiddenSize, layerIndex, this.#numLayers),
                Wo: this.#dynamicGeluInit(this.#hiddenSize, this.#hiddenSize, layerIndex, this.#numLayers)
            })),
            ffnWeights: Array(this.#numLayers).fill().map((_, layerIndex) => ({
                W1: this.#dynamicGeluInit(this.#hiddenSize, this.#feedForwardSize, layerIndex, this.#numLayers),
                W2: this.#dynamicGeluInit(this.#feedForwardSize, this.#hiddenSize, layerIndex, this.#numLayers),
                b1: Array(this.#feedForwardSize).fill(0),
                b2: Array(this.#hiddenSize).fill(0)
            })),
            layerNormWeights: Array(this.#numLayers).fill().map(() => ({
                gamma1: Array(this.#hiddenSize).fill(1),
                beta1: Array(this.#hiddenSize).fill(0),
                gamma2: Array(this.#hiddenSize).fill(1),
                beta2: Array(this.#hiddenSize).fill(0)
            })),
            outputWeights: this.#dynamicGeluInit(this.#hiddenSize, 1, this.#numLayers, this.#numLayers + 1, 2.1),
            outputBias: Array(1).fill(0)
        }));
    }

    #setGradientStructure () {
        return Array(this.#ensembleSize).fill().map(() => ({
            outputWeights: Array(this.#hiddenSize).fill().map(() => Array(1).fill(0)),
            outputBias: Array(1).fill(0),
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
            attentionBias: Array(this.#hiddenSize).fill(0),
            attentionWeightMatrix: Array(this.#hiddenSize).fill(0),
            specializationWeights: Array(this.#hiddenSize).fill().map(() => Array(this.#hiddenSize).fill(0))
        }));
    }

    #loadState () {
        const dbPath = path.join(this.#directoryPath, 'hivemind_state.db');
        let db;

        try {
            if (!fs.existsSync(dbPath)) {
                return {
                    status : false
                };
            }

            db = new Database(dbPath, { readonly: true });

            const metadataStmt = db.prepare('SELECT key, value FROM metadata WHERE key = ?');

            const trainingStepCount = metadataStmt.get('trainingStepCount');
            if (trainingStepCount && isValidNumber(Number(trainingStepCount.value))) {
                this.#trainingStepCount = Number(trainingStepCount.value);
            }

            const ensembleWeightsStmt = db.prepare('SELECT idx, weight FROM ensemble_weights');
            const ensembleWeights = ensembleWeightsStmt.all();
            ensembleWeights.forEach(({ idx, weight }) => {
                if (isValidNumber(weight) && idx >= 0 && idx < this.#ensembleSize) {
                    this.#ensembleWeights[idx] = weight;
                }
            });

            const performanceScoresStmt = db.prepare('SELECT idx, score FROM performance_scores');
            const performanceScores = performanceScoresStmt.all();
            performanceScores.forEach(({ idx, score }) => {
                if (isValidNumber(score) && idx >= 0 && idx < this.#ensembleSize) {
                    this.#performanceScores[idx] = score;
                }
            });

            const agreementScoresStmt = db.prepare('SELECT idx, score FROM agreement_scores');
            const agreementScores = agreementScoresStmt.all();
            agreementScores.forEach(({ idx, score }) => {
                if (isValidNumber(score) && idx >= 0 && idx < this.#ensembleSize) {
                    this.#agreementScores[idx] = score;
                }
            });

            const specializationScoresStmt = db.prepare('SELECT idx, score FROM specialization_scores');
            const specializationScores = specializationScoresStmt.all();
            specializationScores.forEach(({ idx, score }) => {
                if (isValidNumber(score) && idx >= 0 && idx < this.#ensembleSize) {
                    this.#specializationScores[idx] = score;
                }
            });

            const historicalPerformanceStmt = db.prepare('SELECT idx, step, score FROM historical_performance ORDER BY idx, step');
            const historicalPerformance = historicalPerformanceStmt.all();
            historicalPerformance.forEach(({ idx, step, score }) => {
                if (isValidNumber(score) && idx >= 0 && idx < this.#ensembleSize && Number.isInteger(step)) {
                    this.#historicalPerformance[idx][step] = score;
                }
            });

            const trustScoresHistoryStmt = db.prepare('SELECT idx, step, score FROM trust_scores_history ORDER BY idx, step');
            const trustScoresHistory = trustScoresHistoryStmt.all();
            trustScoresHistory.forEach(({ idx, step, score }) => {
                if (isValidNumber(score) && idx >= 0 && idx < this.#ensembleSize && Number.isInteger(step)) {
                    this.#trustScoresHistory[idx][step] = score;
                }
            });

            const adaptiveLearningRateStmt = db.prepare('SELECT idx, rate FROM adaptive_learning_rate');
            const adaptiveLearningRates = adaptiveLearningRateStmt.all();
            adaptiveLearningRates.forEach(({ idx, rate }) => {
                if (isValidNumber(rate) && idx >= 0 && idx < this.#ensembleSize) {
                    this.#adaptiveLearningRate[idx] = rate;
                }
            });

            const attentionWeightMatrixStmt = db.prepare('SELECT idx, row, value FROM attention_weight_matrix');
            const attentionWeightMatrix = attentionWeightMatrixStmt.all();
            attentionWeightMatrix.forEach(({ idx, row, value }) => {
                if (isValidNumber(value) && idx >= 0 && idx < this.#ensembleSize && row >= 0 && row < this.#hiddenSize) {
                    this.#attentionWeightMatrix[idx][row] = value;
                }
            });

            const attentionBiasStmt = db.prepare('SELECT idx, row, value FROM attention_bias');
            const attentionBias = attentionBiasStmt.all();
            attentionBias.forEach(({ idx, row, value }) => {
                if (isValidNumber(value) && idx >= 0 && idx < this.#ensembleSize && row >= 0 && row < this.#hiddenSize) {
                    this.#attentionBias[idx][row] = value;
                }
            });

            const specializationWeightsStmt = db.prepare('SELECT idx, row, col, value FROM specialization_weights');
            const specializationWeights = specializationWeightsStmt.all();
            specializationWeights.forEach(({ idx, row, col, value }) => {
                if (
                    isValidNumber(value) &&
                    idx >= 0 && idx < this.#ensembleSize &&
                    row >= 0 && row < this.#hiddenSize &&
                    col >= 0 && col < this.#hiddenSize
                ) {
                    this.#specializationWeights[idx][row][col] = value;
                }
            });

            const attentionMemoryStmt = db.prepare('SELECT idx, window, seq, dim, value FROM attention_memory');
            const attentionMemory = attentionMemoryStmt.all();
            attentionMemory.forEach(({ idx, window, seq, dim, value }) => {
                if (
                    isValidNumber(value) &&
                    idx >= 0 && idx < this.#ensembleSize &&
                    window >= 0 && window < this.#contextWindow &&
                    seq >= 0 && seq < this.#inputSize &&
                    dim >= 0 && dim < this.#hiddenSize
                ) {
                    this.#attentionMemory[idx][window][seq][dim] = value;
                }
            });

            const adaptiveContextStmt = db.prepare('SELECT idx, window, seq, dim, value FROM adaptive_context');
            const adaptiveContext = adaptiveContextStmt.all();
            adaptiveContext.forEach(({ idx, window, seq, dim, value }) => {
                if (
                    isValidNumber(value) &&
                    idx >= 0 && idx < this.#ensembleSize &&
                    window >= 0 && window < this.#adaptiveWindow &&
                    seq >= 0 && seq < this.#inputSize &&
                    dim >= 0 && dim < this.#hiddenSize
                ) {
                    this.#adaptiveContext[idx][window][seq][dim] = value;
                }
            });

            const transformerWeightsStmt = db.prepare('SELECT idx, layer, weight_type, row, col, value FROM transformers');
            const transformerBiasesStmt = db.prepare('SELECT idx, layer, bias_type, row, value FROM transformer_biases');
            const transformerLayerNormStmt = db.prepare('SELECT idx, layer, norm_type, row, value FROM transformer_layer_norm');
            const transformerWeights = transformerWeightsStmt.all();
            const transformerBiases = transformerBiasesStmt.all();
            const transformerLayerNorm = transformerLayerNormStmt.all();
            transformerWeights.forEach(({ idx, layer, weight_type, row, col, value }) => {
                if (
                    isValidNumber(value) &&
                    idx >= 0 && idx < this.#ensembleSize &&
                    row >= 0 && col >= 0
                ) {
                    if (weight_type === 'outputWeights' && layer === -1) {
                        if (row < this.#hiddenSize && col < 1) {
                            this.#transformers[idx].outputWeights[row][col] = value;
                        }
                    } else if (layer >= 0 && layer < this.#numLayers) {
                        if (weight_type === 'Wq' && row < this.#hiddenSize && col < this.#hiddenSize) {
                            this.#transformers[idx].attentionWeights[layer].Wq[row][col] = value;
                        } else if (weight_type === 'Wk' && row < this.#hiddenSize && col < this.#hiddenSize) {
                            this.#transformers[idx].attentionWeights[layer].Wk[row][col] = value;
                        } else if (weight_type === 'Wv' && row < this.#hiddenSize && col < this.#hiddenSize) {
                            this.#transformers[idx].attentionWeights[layer].Wv[row][col] = value;
                        } else if (weight_type === 'Wo' && row < this.#hiddenSize && col < this.#hiddenSize) {
                            this.#transformers[idx].attentionWeights[layer].Wo[row][col] = value;
                        } else if (weight_type === 'W1' && row < this.#hiddenSize && col < this.#feedForwardSize) {
                            this.#transformers[idx].ffnWeights[layer].W1[row][col] = value;
                        } else if (weight_type === 'W2' && row < this.#feedForwardSize && col < this.#hiddenSize) {
                            this.#transformers[idx].ffnWeights[layer].W2[row][col] = value;
                        }
                    } else if (layer === -2 && weight_type === 'positionalEncoding') {
                        if (row < this.#inputSize && col < this.#hiddenSize) {
                            this.#transformers[idx].positionalEncoding[row][col] = value;
                        }
                    }
                }
            });
            transformerBiases.forEach(({ idx, layer, bias_type, row, value }) => {
                if (
                    isValidNumber(value) &&
                    idx >= 0 && idx < this.#ensembleSize &&
                    row >= 0
                ) {
                    if (bias_type === 'outputBias' && layer === -1 && row < 1) {
                        this.#transformers[idx].outputBias[row] = value;
                    } else if (layer >= 0 && layer < this.#numLayers) {
                        if (bias_type === 'b1' && row < this.#feedForwardSize) {
                            this.#transformers[idx].ffnWeights[layer].b1[row] = value;
                        } else if (bias_type === 'b2' && row < this.#hiddenSize) {
                            this.#transformers[idx].ffnWeights[layer].b2[row] = value;
                        }
                    }
                }
            });
            transformerLayerNorm.forEach(({ idx, layer, norm_type, row, value }) => {
                if (
                    isValidNumber(value) &&
                    idx >= 0 && idx < this.#ensembleSize &&
                    layer >= 0 && layer < this.#numLayers &&
                    row >= 0 && row < this.#hiddenSize
                ) {
                    if (norm_type === 'gamma1') {
                        this.#transformers[idx].layerNormWeights[layer].gamma1[row] = value;
                    } else if (norm_type === 'beta1') {
                        this.#transformers[idx].layerNormWeights[layer].beta1[row] = value;
                    } else if (norm_type === 'gamma2') {
                        this.#transformers[idx].layerNormWeights[layer].gamma2[row] = value;
                    } else if (norm_type === 'beta2') {
                        this.#transformers[idx].layerNormWeights[layer].beta2[row] = value;
                    }
                }
            });

            const gradientAccumulationStmt = db.prepare('SELECT idx, layer, weight_type, row, col, value FROM gradient_accumulation');
            const gradientAccumulations = gradientAccumulationStmt.all();
            gradientAccumulations.forEach(({ idx, layer, weight_type, row, col, value }) => {
                if (
                    isValidNumber(value) &&
                    idx >= 0 && idx < this.#ensembleSize
                ) {
                    if (weight_type === 'outputWeights' && layer === -1 && row < this.#hiddenSize && col < 1) {
                        this.#gradientAccumulation[idx].outputWeights[row][col] = value;
                    } else if (weight_type === 'outputBias' && layer === -1 && row < 1) {
                        this.#gradientAccumulation[idx].outputBias[row] = value;
                    } else if (layer >= 0 && layer < this.#numLayers) {
                        if (weight_type === 'Wq' && row < this.#hiddenSize && col < this.#hiddenSize) {
                            this.#gradientAccumulation[idx].attentionWeights[layer].Wq[row][col] = value;
                        } else if (weight_type === 'Wk' && row < this.#hiddenSize && col < this.#hiddenSize) {
                            this.#gradientAccumulation[idx].attentionWeights[layer].Wk[row][col] = value;
                        } else if (weight_type === 'Wv' && row < this.#hiddenSize && col < this.#hiddenSize) {
                            this.#gradientAccumulation[idx].attentionWeights[layer].Wv[row][col] = value;
                        } else if (weight_type === 'Wo' && row < this.#hiddenSize && col < this.#hiddenSize) {
                            this.#gradientAccumulation[idx].attentionWeights[layer].Wo[row][col] = value;
                        } else if (weight_type === 'W1' && row < this.#hiddenSize && col < this.#feedForwardSize) {
                            this.#gradientAccumulation[idx].ffnWeights[layer].W1[row][col] = value;
                        } else if (weight_type === 'W2' && row < this.#feedForwardSize && col < this.#hiddenSize) {
                            this.#gradientAccumulation[idx].ffnWeights[layer].W2[row][col] = value;
                        } else if (weight_type === 'b1' && row < this.#feedForwardSize) {
                            this.#gradientAccumulation[idx].ffnWeights[layer].b1[row] = value;
                        } else if (weight_type === 'b2' && row < this.#hiddenSize) {
                            this.#gradientAccumulation[idx].ffnWeights[layer].b2[row] = value;
                        } else if (weight_type === 'gamma1' && row < this.#hiddenSize) {
                            this.#gradientAccumulation[idx].layerNormWeights[layer].gamma1[row] = value;
                        } else if (weight_type === 'beta1' && row < this.#hiddenSize) {
                            this.#gradientAccumulation[idx].layerNormWeights[layer].beta1[row] = value;
                        } else if (weight_type === 'gamma2' && row < this.#hiddenSize) {
                            this.#gradientAccumulation[idx].layerNormWeights[layer].gamma2[row] = value;
                        } else if (weight_type === 'beta2' && row < this.#hiddenSize) {
                            this.#gradientAccumulation[idx].layerNormWeights[layer].beta2[row] = value;
                        }
                    } else if (weight_type === 'attentionBias' && layer === -1 && row < this.#hiddenSize) {
                        this.#gradientAccumulation[idx].attentionBias[row] = value;
                    } else if (weight_type === 'attentionWeightMatrix' && layer === -1 && row < this.#hiddenSize) {
                        this.#gradientAccumulation[idx].attentionWeightMatrix[row] = value;
                    }
                }
            });

            return {
                status : true
            };
        } catch (error) {
            return {
                status : false,
                error : error.message,
                trace : error.stack
            };
        } finally {
            if (db) {
                db.close();
            }
        }
    }

    #saveState () {
        const dbPath = path.join(this.#directoryPath, 'hivemind_state.db');
        let db;

        try {
            db = new Database(dbPath, { fileMustExist: false });
            db.pragma('journal_mode = WAL');
            db.pragma('synchronous = NORMAL');

            db.exec('BEGIN TRANSACTION');

            db.exec(`
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );
                CREATE TABLE IF NOT EXISTS ensemble_weights (
                    idx INTEGER PRIMARY KEY,
                    weight REAL
                );
                CREATE TABLE IF NOT EXISTS performance_scores (
                    idx INTEGER PRIMARY KEY,
                    score REAL
                );
                CREATE TABLE IF NOT EXISTS agreement_scores (
                    idx INTEGER PRIMARY KEY,
                    score REAL
                );
                CREATE TABLE IF NOT EXISTS specialization_scores (
                    idx INTEGER PRIMARY KEY,
                    score REAL
                );
                CREATE TABLE IF NOT EXISTS historical_performance (
                    idx INTEGER,
                    step INTEGER,
                    score REAL,
                    PRIMARY KEY (idx, step)
                );
                CREATE TABLE IF NOT EXISTS trust_scores_history (
                    idx INTEGER,
                    step INTEGER,
                    score REAL,
                    PRIMARY KEY (idx, step)
                );
                CREATE TABLE IF NOT EXISTS adaptive_learning_rate (
                    idx INTEGER PRIMARY KEY,
                    rate REAL
                );
                CREATE TABLE IF NOT EXISTS attention_weight_matrix (
                    idx INTEGER,
                    row INTEGER,
                    value REAL,
                    PRIMARY KEY (idx, row)
                );
                CREATE TABLE IF NOT EXISTS attention_bias (
                    idx INTEGER,
                    row INTEGER,
                    value REAL,
                    PRIMARY KEY (idx, row)
                );
                CREATE TABLE IF NOT EXISTS specialization_weights (
                    idx INTEGER,
                    row INTEGER,
                    col INTEGER,
                    value REAL,
                    PRIMARY KEY (idx, row, col)
                );
                CREATE TABLE IF NOT EXISTS attention_memory (
                    idx INTEGER,
                    window INTEGER,
                    seq INTEGER,
                    dim INTEGER,
                    value REAL,
                    PRIMARY KEY (idx, window, seq, dim)
                );
                CREATE TABLE IF NOT EXISTS adaptive_context (
                    idx INTEGER,
                    window INTEGER,
                    seq INTEGER,
                    dim INTEGER,
                    value REAL,
                    PRIMARY KEY (idx, window, seq, dim)
                );
                CREATE TABLE IF NOT EXISTS transformers (
                    idx INTEGER,
                    layer INTEGER,
                    weight_type TEXT,
                    row INTEGER,
                    col INTEGER,
                    value REAL,
                    PRIMARY KEY (idx, layer, weight_type, row, col)
                );
                CREATE TABLE IF NOT EXISTS transformer_biases (
                    idx INTEGER,
                    layer INTEGER,
                    bias_type TEXT,
                    row INTEGER,
                    value REAL,
                    PRIMARY KEY (idx, layer, bias_type, row)
                );
                CREATE TABLE IF NOT EXISTS transformer_layer_norm (
                    idx INTEGER,
                    layer INTEGER,
                    norm_type TEXT,
                    row INTEGER,
                    value REAL,
                    PRIMARY KEY (idx, layer, norm_type, row)
                );
                CREATE TABLE IF NOT EXISTS gradient_accumulation (
                    idx INTEGER,
                    layer INTEGER,
                    weight_type TEXT,
                    row INTEGER,
                    col INTEGER,
                    value REAL,
                    PRIMARY KEY (idx, layer, weight_type, row, col)
                );
            `);

            const insertMetadata = db.prepare('INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)');
            insertMetadata.run('trainingStepCount', this.#trainingStepCount.toString());

            const insertEnsembleWeights = db.prepare('INSERT OR REPLACE INTO ensemble_weights (idx, weight) VALUES (?, ?)');
            this.#ensembleWeights.forEach((weight, idx) => {
                if (isValidNumber(weight)) {
                    insertEnsembleWeights.run(idx, weight);
                }
            });

            const insertPerformanceScores = db.prepare('INSERT OR REPLACE INTO performance_scores (idx, score) VALUES (?, ?)');
            this.#performanceScores.forEach((score, idx) => {
                if (isValidNumber(score)) {
                    insertPerformanceScores.run(idx, score);
                }
            });

            const insertAgreementScores = db.prepare('INSERT OR REPLACE INTO agreement_scores (idx, score) VALUES (?, ?)');
            this.#agreementScores.forEach((score, idx) => {
                if (isValidNumber(score)) {
                    insertAgreementScores.run(idx, score);
                }
            });

            const insertSpecializationScores = db.prepare('INSERT OR REPLACE INTO specialization_scores (idx, score) VALUES (?, ?)');
            this.#specializationScores.forEach((score, idx) => {
                if (isValidNumber(score)) {
                    insertSpecializationScores.run(idx, score);
                }
            });

            const insertHistoricalPerformance = db.prepare('INSERT OR REPLACE INTO historical_performance (idx, step, score) VALUES (?, ?, ?)');
            this.#historicalPerformance.forEach((history, idx) => {
                history.forEach((score, step) => {
                    if (isValidNumber(score)) {
                        insertHistoricalPerformance.run(idx, step, score);
                    }
                });
            });

            const insertTrustScoresHistory = db.prepare('INSERT OR REPLACE INTO trust_scores_history (idx, step, score) VALUES (?, ?, ?)');
            this.#trustScoresHistory.forEach((history, idx) => {
                history.forEach((score, step) => {
                    if (isValidNumber(score)) {
                        insertTrustScoresHistory.run(idx, step, score);
                    }
                });
            });

            const insertAdaptiveLearningRate = db.prepare('INSERT OR REPLACE INTO adaptive_learning_rate (idx, rate) VALUES (?, ?)');
            this.#adaptiveLearningRate.forEach((rate, idx) => {
                if (isValidNumber(rate)) {
                    insertAdaptiveLearningRate.run(idx, rate);
                }
            });

            const insertAttentionWeightMatrix = db.prepare('INSERT OR REPLACE INTO attention_weight_matrix (idx, row, value) VALUES (?, ?, ?)');
            this.#attentionWeightMatrix.forEach((weights, idx) => {
                weights.forEach((value, row) => {
                    if (isValidNumber(value)) {
                        insertAttentionWeightMatrix.run(idx, row, value);
                    }
                });
            });

            const insertAttentionBias = db.prepare('INSERT OR REPLACE INTO attention_bias (idx, row, value) VALUES (?, ?, ?)');
            this.#attentionBias.forEach((biases, idx) => {
                biases.forEach((value, row) => {
                    if (isValidNumber(value)) {
                        insertAttentionBias.run(idx, row, value);
                    }
                });
            });

            const insertSpecializationWeights = db.prepare('INSERT OR REPLACE INTO specialization_weights (idx, row, col, value) VALUES (?, ?, ?, ?)');
            this.#specializationWeights.forEach((matrix, idx) => {
                matrix.forEach((row, r) => {
                    row.forEach((value, c) => {
                        if (isValidNumber(value)) {
                            insertSpecializationWeights.run(idx, r, c, value);
                        }
                    });
                });
            });

            const insertAttentionMemory = db.prepare('INSERT OR REPLACE INTO attention_memory (idx, window, seq, dim, value) VALUES (?, ?, ?, ?, ?)');
            this.#attentionMemory.forEach((memory, idx) => {
                memory.forEach((window, w) => {
                    window.forEach((seq, s) => {
                        seq.forEach((value, d) => {
                            if (isValidNumber(value)) {
                                insertAttentionMemory.run(idx, w, s, d, value);
                            }
                        });
                    });
                });
            });

            const insertAdaptiveContext = db.prepare('INSERT OR REPLACE INTO adaptive_context (idx, window, seq, dim, value) VALUES (?, ?, ?, ?, ?)');
            this.#adaptiveContext.forEach((memory, idx) => {
                memory.forEach((window, w) => {
                    window.forEach((seq, s) => {
                        seq.forEach((value, d) => {
                            if (isValidNumber(value)) {
                                insertAdaptiveContext.run(idx, w, s, d, value);
                            }
                        });
                    });
                });
            });

            const insertTransformerWeights = db.prepare('INSERT OR REPLACE INTO transformers (idx, layer, weight_type, row, col, value) VALUES (?, ?, ?, ?, ?, ?)');
            const insertTransformerBiases = db.prepare('INSERT OR REPLACE INTO transformer_biases (idx, layer, bias_type, row, value) VALUES (?, ?, ?, ?, ?)');
            const insertTransformerLayerNorm = db.prepare('INSERT OR REPLACE INTO transformer_layer_norm (idx, layer, norm_type, row, value) VALUES (?, ?, ?, ?, ?)');
            this.#transformers.forEach((transformer, idx) => {
                transformer.attentionWeights.forEach((layer, l) => {
                    layer.Wq.forEach((row, r) => {
                        row.forEach((value, c) => {
                            if (isValidNumber(value)) {
                                insertTransformerWeights.run(idx, l, 'Wq', r, c, value);
                            }
                        });
                    });
                    layer.Wk.forEach((row, r) => {
                        row.forEach((value, c) => {
                            if (isValidNumber(value)) {
                                insertTransformerWeights.run(idx, l, 'Wk', r, c, value);
                            }
                        });
                    });
                    layer.Wv.forEach((row, r) => {
                        row.forEach((value, c) => {
                            if (isValidNumber(value)) {
                                insertTransformerWeights.run(idx, l, 'Wv', r, c, value);
                            }
                        });
                    });
                    layer.Wo.forEach((row, r) => {
                        row.forEach((value, c) => {
                            if (isValidNumber(value)) {
                                insertTransformerWeights.run(idx, l, 'Wo', r, c, value);
                            }
                        });
                    });
                });
                transformer.ffnWeights.forEach((layer, l) => {
                    layer.W1.forEach((row, r) => {
                        row.forEach((value, c) => {
                            if (isValidNumber(value)) {
                                insertTransformerWeights.run(idx, l, 'W1', r, c, value);
                            }
                        });
                    });
                    layer.W2.forEach((row, r) => {
                        row.forEach((value, c) => {
                            if (isValidNumber(value)) {
                                insertTransformerWeights.run(idx, l, 'W2', r, c, value);
                            }
                        });
                    });
                    layer.b1.forEach((value, r) => {
                        if (isValidNumber(value)) {
                            insertTransformerBiases.run(idx, l, 'b1', r, value);
                        }
                    });
                    layer.b2.forEach((value, r) => {
                        if (isValidNumber(value)) {
                            insertTransformerBiases.run(idx, l, 'b2', r, value);
                        }
                    });
                });
                transformer.layerNormWeights.forEach((layer, l) => {
                    layer.gamma1.forEach((value, r) => {
                        if (isValidNumber(value)) {
                            insertTransformerLayerNorm.run(idx, l, 'gamma1', r, value);
                        }
                    });
                    layer.beta1.forEach((value, r) => {
                        if (isValidNumber(value)) {
                            insertTransformerLayerNorm.run(idx, l, 'beta1', r, value);
                        }
                    });
                    layer.gamma2.forEach((value, r) => {
                        if (isValidNumber(value)) {
                            insertTransformerLayerNorm.run(idx, l, 'gamma2', r, value);
                        }
                    });
                    layer.beta2.forEach((value, r) => {
                        if (isValidNumber(value)) {
                            insertTransformerLayerNorm.run(idx, l, 'beta2', r, value);
                        }
                    });
                });
                transformer.outputWeights.forEach((row, r) => {
                    row.forEach((value, c) => {
                        if (isValidNumber(value)) {
                            insertTransformerWeights.run(idx, -1, 'outputWeights', r, c, value);
                        }
                    });
                });
                transformer.outputBias.forEach((value, r) => {
                    if (isValidNumber(value)) {
                        insertTransformerBiases.run(idx, -1, 'outputBias', r, value);
                    }
                });
                transformer.positionalEncoding.forEach((row, r) => {
                    row.forEach((value, c) => {
                        if (isValidNumber(value)) {
                            insertTransformerWeights.run(idx, -2, 'positionalEncoding', r, c, value);
                        }
                    });
                });
            });

            const insertGradientAccumulation = db.prepare('INSERT OR REPLACE INTO gradient_accumulation (idx, layer, weight_type, row, col, value) VALUES (?, ?, ?, ?, ?, ?)');
            this.#gradientAccumulation.forEach((grad, idx) => {
                grad.outputWeights.forEach((row, r) => {
                    row.forEach((value, c) => {
                        if (isValidNumber(value)) {
                            insertGradientAccumulation.run(idx, -1, 'outputWeights', r, c, value);
                        }
                    });
                });
                grad.outputBias.forEach((value, r) => {
                    if (isValidNumber(value)) {
                        insertGradientAccumulation.run(idx, -1, 'outputBias', r, 0, value);
                    }
                });
                grad.attentionWeights.forEach((layer, l) => {
                    layer.Wq.forEach((row, r) => {
                        row.forEach((value, c) => {
                            if (isValidNumber(value)) {
                                insertGradientAccumulation.run(idx, l, 'Wq', r, c, value);
                            }
                        });
                    });
                    layer.Wk.forEach((row, r) => {
                        row.forEach((value, c) => {
                            if (isValidNumber(value)) {
                                insertGradientAccumulation.run(idx, l, 'Wk', r, c, value);
                            }
                        });
                    });
                    layer.Wv.forEach((row, r) => {
                        row.forEach((value, c) => {
                            if (isValidNumber(value)) {
                                insertGradientAccumulation.run(idx, l, 'Wv', r, c, value);
                            }
                        });
                    });
                    layer.Wo.forEach((row, r) => {
                        row.forEach((value, c) => {
                            if (isValidNumber(value)) {
                                insertGradientAccumulation.run(idx, l, 'Wo', r, c, value);
                            }
                        });
                    });
                });
                grad.ffnWeights.forEach((layer, l) => {
                    layer.W1.forEach((row, r) => {
                        row.forEach((value, c) => {
                            if (isValidNumber(value)) {
                                insertGradientAccumulation.run(idx, l, 'W1', r, c, value);
                            }
                        });
                    });
                    layer.W2.forEach((row, r) => {
                        row.forEach((value, c) => {
                            if (isValidNumber(value)) {
                                insertGradientAccumulation.run(idx, l, 'W2', r, c, value);
                            }
                        });
                    });
                    layer.b1.forEach((value, r) => {
                        if (isValidNumber(value)) {
                            insertGradientAccumulation.run(idx, l, 'b1', r, 0, value);
                        }
                    });
                    layer.b2.forEach((value, r) => {
                        if (isValidNumber(value)) {
                            insertGradientAccumulation.run(idx, l, 'b2', r, 0, value);
                        }
                    });
                });
                grad.layerNormWeights.forEach((layer, l) => {
                    layer.gamma1.forEach((value, r) => {
                        if (isValidNumber(value)) {
                            insertGradientAccumulation.run(idx, l, 'gamma1', r, 0, value);
                        }
                    });
                    layer.beta1.forEach((value, r) => {
                        if (isValidNumber(value)) {
                            insertGradientAccumulation.run(idx, l, 'beta1', r, 0, value);
                        }
                    });
                    layer.gamma2.forEach((value, r) => {
                        if (isValidNumber(value)) {
                            insertGradientAccumulation.run(idx, l, 'gamma2', r, 0, value);
                        }
                    });
                    layer.beta2.forEach((value, r) => {
                        if (isValidNumber(value)) {
                            insertGradientAccumulation.run(idx, l, 'beta2', r, 0, value);
                        }
                    });
                });
                grad.attentionBias.forEach((value, r) => {
                    if (isValidNumber(value)) {
                        insertGradientAccumulation.run(idx, -1, 'attentionBias', r, 0, value);
                    }
                });
                grad.attentionWeightMatrix.forEach((value, r) => {
                    if (isValidNumber(value)) {
                        insertGradientAccumulation.run(idx, -1, 'attentionWeightMatrix', r, 0, value);
                    }
                });
            });

            db.exec('COMMIT');

            return {
                status : true
            }
        } catch (error) {
            db.exec('ROLLBACK');

            return {
                status : false,
                error : error.message,
                trace : error.stack
            }
        } finally {
            if (db) {
                db.close();
            }
        }
    }

    #gelu (x) {
        if (!isValidNumber(x)) return 0;
        return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));
    }

    #geluDerivative (x) {
        if (!isValidNumber(x)) return 0;

        const clampedX = Math.min(Math.max(x, -20), 20);

        const x3 = Math.pow(clampedX, 3);
        const poly = clampedX + 0.044715 * x3;
        const sqrtTerm = Math.sqrt(2 / Math.PI);
        const tanhArg = sqrtTerm * poly;

        const cdf = 0.5 * (1 + Math.tanh(tanhArg));

        const pdf = Math.exp(-0.5 * clampedX * clampedX) / Math.sqrt(2 * Math.PI);

        const derivative = cdf + clampedX * pdf;

        return isValidNumber(derivative) ? Math.min(Math.max(derivative, -20), 20) : 0;
    }

    #sigmoid (x) {
        return isValidNumber(x) ? 1 / (1 + Math.exp(-Math.min(Math.max(x, -100), 100))) : 0;
    }

    #softmax (arr) {
        if (!arr.every(isValidNumber)) return arr.map(() => 1 / arr.length);
        const max = Math.max(...arr);
        const exp = arr.map(x => Math.exp(x - max));
        const sum = exp.reduce((a, b) => a + b, 0) || 1;
        return exp.map(x => x / sum);
    }

    #layerNorm (x, gamma, beta, eps = 1e-6) {
        if (
            !Array.isArray(x) || x.length !== this.#hiddenSize ||
            !Array.isArray(gamma) || gamma.length !== this.#hiddenSize ||
            !Array.isArray(beta) || beta.length !== this.#hiddenSize ||
            !x.every(isValidNumber) || !gamma.every(isValidNumber) || !beta.every(isValidNumber)
        ) {
            return Array(this.#hiddenSize).fill(0);
        }

        const mean = x.reduce((sum, val) => sum + val, 0) / x.length;
        const variance = x.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / x.length;

        if (!isValidNumber(variance) || !isValidNumber(mean)) {
            return Array(this.#hiddenSize).fill(0);
        }

        const output = x.map((val, i) => {
            const normalized = (val - mean) / Math.sqrt(variance + eps);
            const gammaAdjusted = Math.abs(gamma[i] - 1.0) < 1e-6 ? 1.0 : gamma[i];
            return isValidNumber(normalized) ? gammaAdjusted * normalized + beta[i] : 0;
        });

        return output;
    }

    #dropout (x, rate, training = true) {
        if (
            !Array.isArray(x) || 
            !x.every(isValidNumber) || 
            !isValidNumber(rate) || 
            rate < 0 || 
            rate >= 1 ||
            !training
        ) {
            return x.slice();
        }

        return x.map(val => {
            if (!isValidNumber(val)) return 0;
            return Math.random() >= rate ? val / (1 - rate) : 0;
        });
    }

    #computeMemoryScore (memoryEntry, attentionScores, transformerIdx, entryIndex, ignoreRecency = false) {
        let valueSum = 0;
        let valueCount = 0;
        for (let i = 0; i < memoryEntry.length; i++) {
            for (let j = 0; j < memoryEntry[i].length; j++) {
                if (isValidNumber(memoryEntry[i][j])) {
                    valueSum += Math.abs(memoryEntry[i][j]);
                    valueCount++;
                }
            }
        }

        const dropoutScore = valueCount > 0 ? (valueSum / valueCount) * (1 - this.#dropoutRate) : 0;

        let attentionSum = 0;
        let headCount = 0;
        if (attentionScores && Array.isArray(attentionScores)) {
            for (let h = 0; h < attentionScores.length; h++) {
                let maxScore = 0;
                for (let i = 0; i < attentionScores[h].length; i++) {
                    for (let j = 0; j < attentionScores[h][i].length; j++) {
                        if (isValidNumber(attentionScores[h][i][j])) {
                            maxScore = Math.max(maxScore, Math.abs(attentionScores[h][i][j]));
                        }
                    }
                }
                if (maxScore > 0) {
                    attentionSum += maxScore;
                    headCount++;
                }
            }
        }
        const attentionScore = headCount > 0 ? attentionSum / headCount : 0;

        let normSum = 0;
        for (let i = 0; i < memoryEntry.length; i++) {
            for (let j = 0; j < memoryEntry[i].length; j++) {
                if (isValidNumber(memoryEntry[i][j]) && Math.abs(memoryEntry[i][j]) >= 1e-3) {
                    normSum += memoryEntry[i][j] ** 2;
                }
            }
        }
        const magnitudeScore = Math.sqrt(normSum);

        const specScore = isValidNumber(this.#specializationScores[transformerIdx])
            ? Math.min(Math.max(this.#specializationScores[transformerIdx], 0), 1)
            : 0.5;
        const dropoutWeight = 0.3;
        const attentionWeight = 0.4 * (1 - specScore) + 0.2;
        const magnitudeWeight = 0.2 * specScore + 0.1;

        const baseScore = dropoutScore * dropoutWeight + attentionScore * attentionWeight + magnitudeScore * magnitudeWeight;

        if (ignoreRecency || this.#attentionMemory[transformerIdx].length <= 1) {
            return baseScore;
        }

        const memoryLength = this.#attentionMemory[transformerIdx].length;
        const positionRatio = (memoryLength - entryIndex) / memoryLength;

        let recencyBoost;
        if (positionRatio >= 0.80) recencyBoost = 1.4;
        else if (positionRatio >= 0.50) recencyBoost = 1.25;
        else if (positionRatio >= 0.20) recencyBoost = 1.1;
        else recencyBoost = 0.95;

        const baseRecencyWeight = 0.2;
        const logRecency = baseRecencyWeight / (1 + Math.log10(Math.max(memoryLength, 1))) * 0.7;
        const baseFactor = 1 + logRecency / Math.max(memoryLength - entryIndex, 1);

        const recencyFactor = Math.min(baseFactor * recencyBoost, 2.0);

        return baseScore * recencyFactor;
    }

    #pruneMemory (transformerIdx, contextWindow, latestScores) {
        const memory = this.#attentionMemory[transformerIdx];
        if (memory.length <= contextWindow) {
            return;
        }

        const numToRemove = memory.length - contextWindow;

        const scoredEntries = memory.map((entry, index) => ({
            index,
            score: this.#computeMemoryScore(
                entry,
                index === memory.length - 1 ? latestScores : null,
                transformerIdx,
                index
            )
        }));

        scoredEntries.sort((a, b) => a.score - b.score);

        const indicesToRemove = scoredEntries
            .slice(0, numToRemove)
            .map(item => item.index)
            .sort((a, b) => b - a);

        for (const idx of indicesToRemove) {
            memory.splice(idx, 1);
        }
    }

    #normalizeToTarget = (vec, maxVal = 1.0) => {
        const targetNorm = maxVal * Math.sqrt(this.#hiddenSize);

        const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
        let scaled = vec;
        if (norm > targetNorm) {
            const scale = targetNorm / Math.max(norm, 1e-8);
            scaled = vec.map(v => v * scale);
        }

        const tanhScale = maxVal;
        const tanhNorm = Math.tanh(tanhScale);

        return scaled.map(v => Math.tanh(v * tanhScale) / tanhNorm * maxVal);
    }

    #multiHeadAttention (x, layer, transformerIdx, training = true) {
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
                preWoOutput: Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0)),
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
        if (this.#hiddenSize % this.#numHeads !== 0 || headSize <= 0) {
            console.log(
                `Critical configuration error: hiddenSize (${this.#hiddenSize}) ` +
                `must be evenly divisible by numHeads (${this.#numHeads}). ` +
                `Got headSize = ${headSize}`
            );
            process.exit();
        }

        const Q = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
        const K = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
        const V = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));

        for (let i = 0; i < this.#inputSize; i++) {
            for (let j = 0; j < this.#hiddenSize; j++) {
                for (let k = 0; k < this.#hiddenSize; k++) {
                    const specWeight = isValidNumber(this.#specializationWeights[transformerIdx][k % this.#hiddenSize][j])
                        ? Math.min(Math.max(1 + this.#specializationScores[transformerIdx] * this.#specializationWeights[transformerIdx][k % this.#hiddenSize][j], 0.5), 1.5)
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
                        ? sum / Math.sqrt(headSize)
                        : 0;
                }
                attentionProbs[h][i] = this.#softmax(attentionScores[h][i].map(score => isValidNumber(score) ? score : 0));
                attentionProbs[h][i] = this.#dropout(attentionProbs[h][i], this.#dropoutRate, training);
            }
        }

        const preWoOutput = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
        for (let h = 0; h < this.#numHeads; h++) {
            for (let i = 0; i < this.#inputSize; i++) {
                for (let j = 0; j < this.#inputSize; j++) {
                    for (let k = 0; k < headSize; k++) {
                        const idx = h * headSize + k;
                        preWoOutput[i][idx] += isValidNumber(attentionProbs[h][i][j]) && isValidNumber(V[j][idx])
                            ? attentionProbs[h][i][j] * V[j][idx]
                            : 0;
                    }
                }
            }
        }

        const finalOutput = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
        for (let i = 0; i < this.#inputSize; i++) {
            for (let j = 0; j < this.#hiddenSize; j++) {
                for (let k = 0; k < this.#hiddenSize; k++) {
                    const specWeight = isValidNumber(this.#specializationWeights[transformerIdx][k % this.#hiddenSize][j])
                        ? Math.min(Math.max(1 + this.#specializationScores[transformerIdx] * this.#specializationWeights[transformerIdx][k % this.#hiddenSize][j], 0.5), 1.5)
                        : 1;
                    finalOutput[i][j] += isValidNumber(preWoOutput[i][k]) && isValidNumber(layer.Wo[k][j])
                        ? preWoOutput[i][k] * layer.Wo[k][j] * specWeight
                        : 0;
                }
            }
            finalOutput[i] = this.#dropout(finalOutput[i], this.#dropoutRate, training);
        }

        const normalizedOutput = finalOutput.map(tokenVec => this.#normalizeToTarget(tokenVec));

        const noisedOutput = normalizedOutput.map(tokenVec =>
            tokenVec.map(val => {  
                const noiseSize = (Math.random() - 0.5) * 2;
                const noiseScale = 0.1 + Math.random() * 0.3;
                return val * (1 + noiseSize * noiseScale)
            })
        );

        this.#adaptiveContext[transformerIdx].push(noisedOutput);
        if (this.#adaptiveContext[transformerIdx].length > this.#adaptiveWindow) {
            this.#adaptiveContext[transformerIdx].shift();
        }

        if (training) {
            this.#attentionMemory[transformerIdx].push(normalizedOutput);
            this.#pruneMemory(transformerIdx, this.#contextWindow, attentionScores);
        }

        return {
            output: finalOutput,
            preWoOutput,
            Q,
            K,
            V,
            scores: attentionScores,
            probs: attentionProbs
        };
    }

    #feedForward (x, layer, transformerIdx, training = true) {
        if (
            !Array.isArray(x) || x.length !== this.#hiddenSize ||
            !x.every(isValidNumber) ||
            !Number.isInteger(transformerIdx) ||
            transformerIdx < 0 ||
            transformerIdx >= this.#ensembleSize ||
            !layer || !layer.W1 || !layer.W2 || !layer.b1 || !layer.b2 ||
            !layer.W1.every(row => Array.isArray(row) && row.length === this.#feedForwardSize && row.every(isValidNumber)) ||
            !layer.W2.every(row => Array.isArray(row) && row.length === this.#hiddenSize && row.every(isValidNumber)) ||
            !Array.isArray(layer.b1) || layer.b1.length !== this.#feedForwardSize || !layer.b1.every(isValidNumber) ||
            !Array.isArray(layer.b2) || layer.b2.length !== this.#hiddenSize || !layer.b2.every(isValidNumber)
        ) {
            return Array(this.#hiddenSize).fill(0);
        }

        const hidden = Array(this.#feedForwardSize).fill(0);
        for (let j = 0; j < this.#feedForwardSize; j++) {
            for (let i = 0; i < this.#hiddenSize; i++) {
                const specWeight = isValidNumber(this.#specializationWeights[transformerIdx][i % this.#hiddenSize][j % this.#hiddenSize])
                    ? Math.min(Math.max(1 + this.#specializationScores[transformerIdx] * this.#specializationWeights[transformerIdx][i % this.#hiddenSize][j % this.#hiddenSize], 0.5), 1.5)
                    : 1;
                hidden[j] += isValidNumber(x[i]) && isValidNumber(layer.W1[i][j])
                    ? x[i] * layer.W1[i][j] * specWeight
                    : 0;
            }
            hidden[j] = isValidNumber(hidden[j]) && isValidNumber(layer.b1[j])
                ? hidden[j] + layer.b1[j]
                : hidden[j];
        }

        let activated = hidden.map(val => this.#gelu(val));
        activated = this.#dropout(activated, this.#dropoutRate, training);

        const output = Array(this.#hiddenSize).fill(0);
        for (let j = 0; j < this.#hiddenSize; j++) {
            for (let i = 0; i < this.#feedForwardSize; i++) {
                const specWeight = isValidNumber(this.#specializationWeights[transformerIdx][j % this.#hiddenSize][i % this.#hiddenSize])
                    ? Math.min(Math.max(1 + this.#specializationScores[transformerIdx] * this.#specializationWeights[transformerIdx][j % this.#hiddenSize][i % this.#hiddenSize], 0.5), 1.5)
                    : 1;
                output[j] += isValidNumber(activated[i]) && isValidNumber(layer.W2[i][j])
                    ? activated[i] * layer.W2[i][j] * specWeight
                    : 0;
            }
            output[j] = isValidNumber(output[j]) && isValidNumber(layer.b2[j])
                ? output[j] + layer.b2[j]
                : output[j];
        }

        return this.#dropout(output, this.#dropoutRate, training);
    }

    #contextAwareAttention (inputs, transformerIdx, training = true) {
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

        const inputProjection = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
        for (let i = 0; i < this.#inputSize; i++) {
            for (let j = 0; j < this.#hiddenSize; j++) {
                const specWeight = isValidNumber(this.#specializationWeights[transformerIdx][i % this.#hiddenSize][j])
                    ? Math.min(Math.max(1 + this.#specializationScores[transformerIdx] * this.#specializationWeights[transformerIdx][i % this.#hiddenSize][j], 0.5), 1.5)
                    : 1;
                inputProjection[i][j] = isValidNumber(inputs[i])
                    ? inputs[i] * specWeight * (1 + this.#swarmIntelligenceFactor)
                    : 0;
            }
        }

        let rawComponent = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
        for (let i = 0; i < this.#inputSize; i++) {
            for (let j = 0; j < this.#hiddenSize; j++) {
                rawComponent[i][j] = isValidNumber(inputProjection[i][j]) && isValidNumber(transformer.positionalEncoding[i][j])
                    ? inputProjection[i][j] + transformer.positionalEncoding[i][j]
                    : inputProjection[i][j];
            }
        }

        let longTermComponent = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));

        const memory = this.#attentionMemory[transformerIdx];
        if (memory.length > 0) {
            const importanceScores = memory.map((entry, idx) =>
                this.#computeMemoryScore(entry, null, transformerIdx, idx, true)
            );

            const maxImp = Math.max(...importanceScores, 1);
            const minImp = Math.min(...importanceScores);
            const impRange = maxImp - minImp || 1;
            const normalizedImportance = importanceScores.map(s => (s - minImp) / impRange);

            const recencyWeights = [];
            let recencySum = 0;
            const decayBase = 0.98;
            const minRecency = 0.01;
            for (let age = 0; age < memory.length; age++) {
                let w = Math.pow(decayBase, age);
                w = Math.max(w, minRecency);
                recencyWeights[age] = w;
                recencySum += w;
            }
            const normalizedRecency = recencyWeights.map(w => w / recencySum);

            const hybridWeights = normalizedRecency.map((r, age) => {
                const memIdx = memory.length - 1 - age;
                return r * (0.6 + 0.4 * normalizedImportance[memIdx]);
            });

            const totalWeight = hybridWeights.reduce((a, b) => a + b, 0) || 1;
            const finalWeights = hybridWeights.map(w => w / totalWeight);

            for (let i = 0; i < this.#inputSize; i++) {
                const memoryToken = Array(this.#hiddenSize).fill(0);
                for (let m = 0; m < memory.length; m++) {
                    const age = memory.length - 1 - m;
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        const val = isValidNumber(memory[m][i][j]) ? memory[m][i][j] : 0;
                        memoryToken[j] += val * finalWeights[age];
                    }
                }
                longTermComponent[i] = this.#normalizeToTarget(memoryToken);
            }
        }

        let adaptiveComponent = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));

        const adaptive = this.#adaptiveContext[transformerIdx];
        if (adaptive.length > 0) {
            const weights = [];
            for (let i = 0; i < adaptive.length; i++) {
                weights[i] = adaptive.length - i;
            }
            const weightSum = weights.reduce((a, b) => a + b, 0);
            const normalizedWeights = weights.map(w => w / weightSum);

            for (let i = 0; i < this.#inputSize; i++) {
                const adaptiveToken = Array(this.#hiddenSize).fill(0);
                for (let m = 0; m < adaptive.length; m++) {
                    const weight = normalizedWeights[m];
                    const entry = adaptive[m];
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        const val = isValidNumber(entry[i][j]) ? entry[i][j] : 0;
                        adaptiveToken[j] += val * weight;
                    }
                }
                adaptiveComponent[i] = this.#normalizeToTarget(adaptiveToken);
            }
        }

        adaptiveComponent = adaptiveComponent.map(tokenVec => 
            this.#dropout(tokenVec, this.#dropoutRate, training)
        );

        const rawRatio       = 0.40;
        const longTermRatio  = 0.25;
        const adaptiveRatio  = 0.35;

        let output = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
        for (let i = 0; i < this.#inputSize; i++) {
            for (let j = 0; j < this.#hiddenSize; j++) {
                output[i][j] =
                    rawRatio       * rawComponent[i][j] +
                    longTermRatio  * longTermComponent[i][j] +
                    adaptiveRatio  * adaptiveComponent[i][j];
            }
        }

        output = output.map(tokenVec => this.#normalizeToTarget(tokenVec));

        for (let i = 0; i < this.#inputSize; i++) {
            output[i] = this.#layerNorm(
                output[i],
                transformer.layerNormWeights[0].gamma1,
                transformer.layerNormWeights[0].beta1
            );
        }

        return output;
    }

    #processTransformer (inputs, idx, computeIntermediates = false, training = true) {
        const transformer = this.#transformers[idx];

        let x = this.#contextAwareAttention(inputs, idx, training);
        const layerOutputs = computeIntermediates ? [x] : [];
        const activations = computeIntermediates ? [] : [];
        const attentionIntermediates = computeIntermediates ? [] : [];

        for (let layer = 0; layer < this.#numLayers; layer++) {
            const normX = x.map(row => this.#layerNorm(row, transformer.layerNormWeights[layer].gamma1, transformer.layerNormWeights[layer].beta1));
            const attentionResult = this.#multiHeadAttention(normX, transformer.attentionWeights[layer], idx, training);
            const attentionOutput = attentionResult.output;

            const attentionResidual = x.map((row, i) => row.map((val, j) => {
                const addVal = attentionOutput[i][j];
                if (!isValidNumber(addVal)) {
                    console.log(`CRITICAL: Invalid value detected in attention output at layer ${layer}, position [${i}][${j}].`);
                    process.exit();
                }
                return val + addVal;
            }));

            const normAttention = attentionResidual.map(row => this.#layerNorm(row, transformer.layerNormWeights[layer].gamma2, transformer.layerNormWeights[layer].beta2));
            const ffnOutputs = normAttention.map(row => this.#feedForward(row, transformer.ffnWeights[layer], idx, training));

            x = attentionResidual.map((row, i) => row.map((val, j) => {
                const addVal = ffnOutputs[i][j];
                if (!isValidNumber(addVal)) {
                    console.log(`CRITICAL: Invalid value detected in FFN output at layer ${layer}, position [${i}][${j}].`);
                    process.exit();
                }
                return val + addVal;
            }));

            if (computeIntermediates) {
                layerOutputs.push(x);
                activations.push({ normX, attentionOutput, normAttention });
                attentionIntermediates.push({
                    Q: attentionResult.Q,
                    K: attentionResult.K,
                    V: attentionResult.V,
                    attentionScores: attentionResult.scores,
                    attentionProbs: attentionResult.probs,
                    preWoOutput: attentionResult.preWoOutput
                });
            }
        }

        let output = Array(1).fill(0);
        for (let i = 0; i < this.#hiddenSize; i++) {
            output[0] += isValidNumber(x[0][i]) && isValidNumber(transformer.outputWeights[i][0])
                ? x[0][i] * transformer.outputWeights[i][0]
                : 0;
        }
        output[0] = isValidNumber(output[0]) && isValidNumber(transformer.outputBias[0])
            ? output[0] + transformer.outputBias[0]
            : output[0];

        if (computeIntermediates) {
            return { output: output[0], layerOutputs, activations, attentionIntermediates };
        }
        return output[0];
    }

    #computeAttentionWeights (inputs, outputs) {
        if (
            !Array.isArray(inputs) ||
            inputs.length !== this.#inputSize ||
            !inputs.every(isValidNumber) ||
            !Array.isArray(outputs) ||
            outputs.length !== this.#ensembleSize ||
            !outputs.every(isValidNumber)
        ) {
            return Array(this.#ensembleSize).fill(1 / this.#ensembleSize);
        }

        const attentionScores = Array(this.#ensembleSize).fill(0);
        for (let t = 0; t < this.#ensembleSize; t++) {
            const memory = this.#attentionMemory[t];
            const memLen = memory.length;

            const queries = inputs.map(x =>
                this.#attentionWeightMatrix[t].map(w => isValidNumber(x) && isValidNumber(w) ? x * w : 0)
            );

            const pooledMemory = Array(memLen).fill().map(() => Array(this.#hiddenSize).fill(0));
            for (let m = 0; m < memLen; m++) {
                for (let k = 0; k < this.#hiddenSize; k++) {
                    let sum = 0;
                    for (let i = 0; i < this.#inputSize; i++) {
                        sum += isValidNumber(memory[m][i][k]) ? memory[m][i][k] : 0;
                    }
                    pooledMemory[m][k] = sum / this.#inputSize;
                }
            }

            const keys = pooledMemory.map(pool =>
                pool.map((v, k) => isValidNumber(v) && isValidNumber(this.#attentionWeightMatrix[t][k]) ? v * this.#attentionWeightMatrix[t][k] : 0)
            );

            const values = keys;

            const biasSum = this.#attentionBias[t].reduce((sum, val) => sum + (isValidNumber(val) ? val : 0), 0);
            const avgBias = isValidNumber(biasSum) ? biasSum / this.#hiddenSize : 0;

            let score = 0;
            for (let i = 0; i < this.#inputSize; i++) {
                const rowScores = Array(memLen).fill(0);
                for (let j = 0; j < memLen; j++) {
                    let dotProduct = 0;
                    for (let k = 0; k < this.#hiddenSize; k++) {
                        dotProduct += isValidNumber(queries[i][k]) && isValidNumber(keys[j][k])
                            ? queries[i][k] * keys[j][k]
                            : 0;
                    }
                    rowScores[j] = isValidNumber(dotProduct)
                        ? dotProduct / Math.sqrt(this.#hiddenSize) + avgBias
                        : avgBias;
                }

                const attentionWeights = this.#softmax(rowScores.map(s => isValidNumber(s) ? s : 0));

                for (let j = 0; j < memLen; j++) {
                    for (let k = 0; k < this.#hiddenSize; k++) {
                        score += isValidNumber(attentionWeights[j]) && isValidNumber(values[j][k])
                            ? attentionWeights[j] * values[j][k]
                            : 0;
                    }
                }
            }

            score /= this.#inputSize;

            const performanceScore = isValidNumber(this.#performanceScores[t]) ? this.#performanceScores[t] : 0.5;
            const specializationBoost = 1 + (isValidNumber(this.#specializationScores[t])
                ? this.#specializationScores[t] * this.#swarmIntelligenceFactor
                : 0);

            attentionScores[t] = score * (0.7 + 0.2 * performanceScore + 0.1 * specializationBoost);
        }

        const weights = this.#softmax(attentionScores.map(score => isValidNumber(score) ? score : 0));

        const finalWeights = weights.map((w, idx) => {
            const performanceScore = isValidNumber(this.#performanceScores[idx]) ? this.#performanceScores[idx] : 0.5;
            const specializationFactor = 1 + (isValidNumber(this.#specializationScores[idx])
                ? this.#specializationScores[idx] * this.#swarmIntelligenceFactor
                : 0);
            const weight = 0.6 * w + 0.3 * performanceScore + 0.1 * specializationFactor;
            return isValidNumber(weight) && weight >= 0 ? weight : 1 / this.#ensembleSize;
        });

        const sum = finalWeights.reduce((s, w) => s + (isValidNumber(w) && w >= 0 ? w : 0), 0);
        if (sum <= 1e-6) {
            return Array(this.#ensembleSize).fill(1 / this.#ensembleSize);
        }

        this.#ensembleWeights = finalWeights.map(w => (isValidNumber(w) && w >= 0 ? w / sum : 1 / this.#ensembleSize));
        this.#normalizeEnsembleWeights();
    }

    #computeSpecializationScores (inputs, outputs) {
        if (
            !Array.isArray(inputs) ||
            inputs.length !== this.#inputSize ||
            !inputs.every(isValidNumber) ||
            !Array.isArray(outputs) ||
            outputs.length !== this.#ensembleSize ||
            !outputs.every(isValidNumber)
        ) {
            return;
        }

        const outputMean = outputs.reduce((sum, v) => sum + v, 0) / outputs.length;
        const outputStd = Math.sqrt(
            outputs.reduce((sum, v) => sum + (v - outputMean) ** 2, 0) / outputs.length
        ) || 1e-6;

        const zScores = outputs.map(out => (out - outputMean) / outputStd);

        this.#specializationScores = zScores.map((z, i) => {
            const performance = isValidNumber(this.#performanceScores[i]) ? this.#performanceScores[i] : 0.5;
            const magnitude = Math.abs(z);
            const bounded = 1 / (1 + Math.exp(-magnitude));
            return bounded * (0.5 + 0.5 * performance);
        });
    }

    #updateAdaptiveLearningRates () {
        const recent_performance = this.#historicalPerformance.map(history => {
            const len = history.length;
            if (len === 0) return 0.5;
            const sum = history.reduce((s, v) => s + (isValidNumber(v) ? v : 0), 0);
            return sum / len;
        });

        const recent_trust = this.#trustScoresHistory.map(history => {
            const len = history.length;
            if (len === 0) return 0.5;
            const sum = history.reduce((s, v) => s + (isValidNumber(v) ? v : 0), 0);
            return sum / len;
        });

        const sorted_specialization = this.#specializationScores.slice().sort((a, b) => b - a);
        const max_spec = sorted_specialization[Math.floor(this.#ensembleSize * 0.05)] || 1e-10;
        const min_spec = sorted_specialization[this.#ensembleSize - 1] || 0;
        const spec_range = Math.max(max_spec - min_spec, 1e-10);
        const normalized_specialization = this.#specializationScores.map(score => 
            Math.min(Math.max((score - min_spec) / spec_range, 0), 1)
        );

        const composite_scores = recent_performance.map((perf, idx) => {
            const trust = recent_trust[idx];
            const spec = normalized_specialization[idx];
            const weighted_sum = 0.4 * perf + 0.3 * trust + 0.3 * spec;
            return this.#sigmoid(5 * weighted_sum);
        });

        const sorted_composite = composite_scores.slice().sort((a, b) => b - a);
        const thresholdIdx = Math.max(1, Math.floor(composite_scores.length * 0.25));
        const acceptable_threshold = sorted_composite[thresholdIdx] || 0.5;

        const min_lr = this.#learningRate * 0.5;
        const max_lr = this.#learningRate * 1.5;

        this.#adaptiveLearningRate = this.#adaptiveLearningRate.map((lr, idx) => {
            let newLr = lr;
            const score = composite_scores[idx];
            const score_diff = score - acceptable_threshold;

            const adjustment_magnitude = Math.max(-1, Math.min(1, score_diff * 2));
            if (score >= acceptable_threshold) {
                newLr *= (1 - this.#learningRateDecay * this.#learningRate * Math.abs(adjustment_magnitude) * 0.8);
            } else {
                newLr *= (1 + this.#learningRateDecay * this.#learningRate * Math.abs(adjustment_magnitude) * 1.2);
            }

            newLr = 0.95 * newLr + 0.05 * lr;

            return Math.max(min_lr, Math.min(newLr, max_lr));
        });
    }

    #updatePerformanceScores (linearOutputs, target) {
        this.#performanceScores = this.#performanceScores.map((score, idx) => {
            const individualProbability = this.#sigmoid(linearOutputs[idx]);
            const brierScore = Math.pow(individualProbability - target, 2);
            const performance = 1 - brierScore;
            const newScore = 0.9 * score + 0.1 * (isValidNumber(performance) ? performance : 0);
            
            this.#historicalPerformance[idx].push(isValidNumber(newScore) ? newScore : 0);
            if (this.#historicalPerformance[idx].length > this.#maxPerformanceHistory) {
                this.#historicalPerformance[idx].shift();
            }
            
            return isValidNumber(newScore) ? newScore : 0;
        });
    }

    #updateAgreementScores (linearOutputs, finalProbability) {
        this.#agreementScores = this.#agreementScores.map((score, idx) => {
            const individualProbability = this.#sigmoid(linearOutputs[idx]);
            const absDiff = Math.abs(individualProbability - finalProbability);
            let agreement = 1 - absDiff;
            
            const avgHistoricalPerformance = this.#historicalPerformance[idx].length > 0
                ? this.#historicalPerformance[idx].reduce((sum, val) => sum + val, 0) / this.#historicalPerformance[idx].length
                : 0;
            const weightedAgreement = agreement * (0.5 + 0.5 * avgHistoricalPerformance);
            
            const newScore = 0.9 * score + 0.1 * (isValidNumber(weightedAgreement) ? weightedAgreement : 0);
            
            return isValidNumber(newScore) ? newScore : 0;
        });
    }

    #updateTrustScores () {
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
            const trustScore = this.#sigmoid(normalizedScore * (0.6 + 0.2 * agreementFactor + 0.1 * historicalTrend + 0.1 * specializationBoost));
            history.push(isValidNumber(trustScore) ? trustScore : 0.5);
            if (history.length > this.#maxTrustHistory) history.shift();
            return history;
        });
    }

    #updateEnsembleWeights () {
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

    #normalizeEnsembleWeights () {
        const sum = this.#ensembleWeights.reduce((s, w) => s + (isValidNumber(w) && w >= 0 ? w : 0), 0);

        if (sum <= 1e-6) {
            this.#ensembleWeights = Array(this.#ensembleSize).fill(1 / this.#ensembleSize);
            return;
        }

        this.#ensembleWeights = this.#ensembleWeights.map(w => 
            isValidNumber(w) && w >= 0 ? w / sum : 0
        );

        const finalSum = this.#ensembleWeights.reduce((s, w) => s + w, 0);
        if (Math.abs(finalSum - 1) > 1e-6) {
            const maxIndex = this.#ensembleWeights.indexOf(Math.max(...this.#ensembleWeights));
            this.#ensembleWeights[maxIndex] += 1 - finalSum;
        }
    }

    #adjustPerformanceScores () {
        const trustMomentum = 0.6;
        this.#performanceScores = this.#performanceScores.map((score, idx) => {
            const recentTrust = this.#trustScoresHistory[idx].slice(-5);
            const avgTrust = recentTrust.length > 0
                ? recentTrust.reduce((sum, val) => sum + (isValidNumber(val) ? val : 0), 0) / recentTrust.length
                : 0.5;
            const historicalWeight = this.#trustScoresHistory[idx].length / this.#maxTrustHistory;
            const specializationFactor = 1 + this.#specializationScores[idx] * this.#swarmIntelligenceFactor;
            return trustMomentum * (isValidNumber(score) ? score : 0) +
                (1 - trustMomentum) * avgTrust * (0.7 + 0.2 * historicalWeight + 0.1 * specializationFactor);
        });
    }

    #updateMetrics (inputs, outputs, target, prob) {
        this.#updatePerformanceScores(outputs, target);
        this.#updateAgreementScores(outputs, prob);
        this.#computeSpecializationScores(inputs, outputs);
        this.#updateTrustScores();
        this.#adjustPerformanceScores();
        this.#updateEnsembleWeights();
        this.#updateAdaptiveLearningRates();
    }

    #computeSpectralNorm (matrix) {
        if (!Array.isArray(matrix)) {
            const error = new Error('Input is not an array');
            console.error('Error:', error.message, '\nStack:', error.stack);
            throw error
            return 1;
        }
        if (matrix.length === 0) {
            const error = new Error('Matrix is empty');
            console.error('Error:', error.message, '\nStack:', error.stack);
            throw error
            return 1;
        }
        if (!matrix.every(row => Array.isArray(row))) {
            const error = new Error('Matrix rows are not arrays');
            console.error('Error:', error.message, '\nStack:', error.stack);
            throw error
            return 1;
        }
        if (!matrix.every(row => row.length === matrix[0].length)) {
            const error = new Error('Matrix rows have inconsistent lengths');
            console.error('Error:', error.message, '\nStack:', error.stack);
            throw error
            return 1;
        }
        if (!matrix.every(row => row.every(isValidNumber))) {
            const error = new Error('Matrix contains invalid numbers');
            console.error('Error:', error.message, '\nStack:', error.stack);
            throw error
            return 1;
        }
        if (matrix[0].length === 0) {
            const error = new Error('Matrix has no columns');
            console.error('Error:', error.message, '\nStack:', error.stack);
            throw error
            return 1;
        }

        const rows = matrix.length;
        const cols = matrix[0].length;
        let u = Array(cols).fill().map(() => Math.random());
        let v = Array(rows).fill(0);

        const uNormInit = Math.sqrt(u.reduce((sum, x) => sum + (isValidNumber(x) ? x * x : 0), 0)) || 1;
        if (!isValidNumber(uNormInit)) {
            const error = new Error('Invalid initial u norm');
            console.error('Error:', error.message, '\nStack:', error.stack);
            throw error;
        }
        u = u.map(x => isValidNumber(x) ? x / uNormInit : 0);

        const maxIter = 20;
        const tolerance = 1e-6;
        let prevNorm = 0;
        for (let iter = 0; iter < maxIter; iter++) {
            v = Array(rows).fill(0);
            for (let i = 0; i < rows; i++) {
                for (let j = 0; j < cols; j++) {
                    if (isValidNumber(matrix[i][j]) && isValidNumber(u[j])) {
                        v[i] += matrix[i][j] * u[j];
                    }
                }
            }
            const vNorm = Math.sqrt(v.reduce((sum, x) => sum + (isValidNumber(x) ? x * x : 0), 0)) || 1;
            if (!isValidNumber(vNorm)) {
                const error = new Error(`Invalid v norm in iteration ${iter + 1}`);
                console.error('Error:', error.message, '\nStack:', error.stack);
                throw error;
            }
            v = v.map(x => isValidNumber(x) ? x / vNorm : 0);

            u = Array(cols).fill(0);
            for (let j = 0; j < cols; j++) {
                for (let i = 0; i < rows; i++) {
                    if (isValidNumber(matrix[i][j]) && isValidNumber(v[i])) {
                        u[j] += matrix[i][j] * v[i];
                    }
                }
            }
            const uNorm = Math.sqrt(u.reduce((sum, x) => sum + (isValidNumber(x) ? x * x : 0), 0)) || 1;
            if (!isValidNumber(uNorm)) {
                const error = new Error(`Invalid u norm in iteration ${iter + 1}`);
                console.error('Error:', error.message, '\nStack:', error.stack);
                throw error;
            }
            u = u.map(x => isValidNumber(x) ? x / uNorm : 0);

            let currentNorm = 0;
            const temp = Array(rows).fill(0);
            for (let i = 0; i < rows; i++) {
                for (let j = 0; j < cols; j++) {
                    if (isValidNumber(matrix[i][j]) && isValidNumber(u[j])) {
                        temp[i] += matrix[i][j] * u[j];
                    }
                }
            }
            for (let i = 0; i < rows; i++) {
                if (isValidNumber(temp[i]) && isValidNumber(v[i])) {
                    currentNorm += temp[i] * v[i];
                }
            }
            currentNorm = Math.abs(currentNorm) || 1;
            if (!isValidNumber(currentNorm)) {
                const error = new Error(`Invalid norm in iteration ${iter + 1}`);
                console.error('Error:', error.message, '\nStack:', error.stack);
                throw error;
            }

            if (iter > 0 && Math.abs(currentNorm - prevNorm) < tolerance * Math.max(1, currentNorm)) {
                return currentNorm;
            }
            prevNorm = currentNorm;
        }

        let norm = 0;
        const temp = Array(rows).fill(0);
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                if (isValidNumber(matrix[i][j]) && isValidNumber(u[j])) {
                    temp[i] += matrix[i][j] * u[j];
                }
            }
        }
        for (let i = 0; i < rows; i++) {
            if (isValidNumber(temp[i]) && isValidNumber(v[i])) {
                norm += temp[i] * v[i];
            }
        }
        const finalNorm = Math.abs(norm) || 1;
        if (!isValidNumber(finalNorm)) {
            const error = new Error('Invalid final norm');
            console.error('Error:', error.message, '\nStack:', error.stack);
            throw error;
        }

        return finalNorm;
    }

    #computeGradientNorm (grad, isMatrix) {
        if (isMatrix) {
            return Math.sqrt(grad.reduce((sum, row) => 
                sum + row.reduce((s, val) => s + (isValidNumber(val) ? val * val : 0), 0), 0)) || 1;
        } else {
            return Math.sqrt(grad.reduce((sum, val) => 
                sum + (isValidNumber(val) ? val * val : 0), 0)) || 1;
        }
    }

    #computeDiversityScore (weightsList) {
        const norms = weightsList.map(weights => Math.sqrt(weights.reduce((sum, w) => sum + w * w, 0)));
        const sortedNorms = norms.sort((a, b) => a - b);
        const n = sortedNorms.length;
        const sum = sortedNorms.reduce((a, b) => a + b, 0);
        let gini = 0;
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                gini += Math.abs(sortedNorms[i] - sortedNorms[j]);
            }
        }
        const actualGini = sum > 0 ? gini / (2 * n * sum) : 0;
        return actualGini;
    }

    #computeWeightedSum (outputs, weights) {
        return outputs.reduce((sum, out, idx) => {
            if (isValidNumber(out) && isValidNumber(weights[idx])) {
                return sum + out * weights[idx];
            }
            return sum;
        }, 0);
    }

    #computeVariance (arr) {
        if (arr.length < 2) return 0;

        const median = arr.slice().sort((a, b) => a - b)[Math.floor(arr.length / 2)];
        const mad = arr.reduce((sum, val) => sum + Math.abs(val - median), 0) / Math.max(arr.length, 1);

        return Math.min(mad, 10);
    }

    #computeEMA (arr, beta) {
        if (arr.length === 0) return 0;

        let ema = arr[0];
        for (let i = 1; i < arr.length; i++) {
            ema = beta * ema + (1 - beta) * arr[i];
        }

        return ema;
    }

    #computeDualEMA (arr, betaShort, betaLong) {
        const shortEMA = this.#computeEMA(arr, betaShort);
        const longEMA = this.#computeEMA(arr, betaLong);
        const variance = this.#computeVariance(arr.slice(-10));
        const svrWeight = Math.min(0.8, Math.max(0.2, 0.5 * (1 - variance / (variance + 1))));

        return svrWeight * shortEMA + (1 - svrWeight) * longEMA;
    }

    #computeGradientConformity (norms) {
        if (norms.length < 2) return 1;

        const directions = norms.slice(1).map((val, i) => Math.sign(val - norms[i]));
        const consistency = directions.reduce((sum, val, i, arr) => sum + (i > 0 && val === arr[i - 1] ? 1 : 0), 0) / (directions.length || 1);

        return Math.min(1, Math.max(0.5, consistency));
    }

    #computeKernelRate (history) {
        const median = history.slice(-10).sort((a, b) => a - b)[Math.floor(history.length / 2)] || 0;
        const scaledHistory = history.slice(-10).map(val => (val - median) / (Math.abs(median) + 1e-10));
        const kernel = scaledHistory.map((val, i) => Math.exp(-0.1 * (scaledHistory.length - i - 1) ** 2));
        const kernelSum = kernel.reduce((sum, val) => sum + val, 0);

        return kernelSum > 0 ? kernel.reduce((sum, val, i) => sum + val * scaledHistory[i], 0) / kernelSum + 0.9 : 0.9;
    }

    #computeFractalDimension (arr) {
        if (arr.length < 2) return 1;

        const diffs = arr.slice(1).map((val, i) => Math.abs(val - arr[i]));
        const logDiffs = diffs.map(d => Math.log(Math.max(d, 1e-10)));
        const logScale = Math.log(arr.length);
        const fractalDim = logDiffs.length > 0 ? Math.abs(logDiffs.reduce((sum, val) => sum + val, 0) / logScale) : 1;

        return Math.min(2, Math.max(1, fractalDim));
    }

    #computePercentile (norms, percentile) {
        const sortedNorms = [...norms].sort((a, b) => a - b);
        const index = Math.floor(percentile * (sortedNorms.length - 1));

        return sortedNorms[index] || 1.0;
    }

    #computeNTKStability (norms, lossVariance) {
        if (norms.length < 2) return 1;

        const average = (arr) => arr.reduce((sum, val) => sum + val, 0) / Math.max(arr.length, 1);

        const medianNorm = norms.slice(-10).sort((a, b) => a - b)[Math.floor(norms.length / 2)] || 1;
        const bandwidth = Math.min(-0.01, Math.max(-0.1, -0.05 / (1 + lossVariance * medianNorm)));
        const kernel = norms.slice(-10).map((val, i) => Math.exp(bandwidth * (val - average(norms)) ** 2));
        const stability = kernel.reduce((sum, val) => sum + val, 0) / kernel.length;

        return Math.min(1.5, Math.max(0.5, stability));
    }

    #computeDynamicPercentile (norms, basePercentile) {
        const mad = this.#computeVariance(norms);
        const smoothness = this.#computeGradientConformity(norms);
        const adjustment = Math.min(0.05, (mad + 0.1 * smoothness) / (mad + smoothness + 1));

        return Math.min(0.99, Math.max(0.75, basePercentile - adjustment));
    }

    #computeSparseThreshold (norms) {
        const mad = this.#computeVariance(norms);

        return Math.max(1e-6, Math.min(1e-4, mad / 10));
    }

    #scaleGradientMatrix (gradMatrix, threshold, minScaleFactor, alpha, decay, sparseThreshold) {
        const spectralNorm = this.#computeSpectralNorm(gradMatrix);
        if (spectralNorm <= threshold) return;

        const scale = Math.max(minScaleFactor, Math.pow(threshold / spectralNorm, alpha) * decay);
        const quantScale = Math.ceil(scale * 255) / 255;
        for (let i = 0; i < gradMatrix.length; i++) {
            for (let j = 0; j < gradMatrix[i].length; j++) {
                if (Math.abs(gradMatrix[i][j]) > sparseThreshold) {
                    gradMatrix[i][j] *= quantScale;
                }
            }
        }
    }

    #scaleGradientVector (gradVector, threshold, minScaleFactor, alpha, decay, sparseThreshold) {
        const l2Norm = this.#computeGradientNorm(gradVector, false);
        if (l2Norm <= threshold) return;

        const scale = Math.max(minScaleFactor, Math.pow(threshold / l2Norm, alpha) * decay);
        const quantScale = Math.ceil(scale * 255) / 255;
        for (let i = 0; i < gradVector.length; i++) {
            if (Math.abs(gradVector[i]) > sparseThreshold) {
                gradVector[i] *= quantScale;
            }
        }
    }

    #applyAutoScaling (param, rows, cols, isVector = false, diversityScore = 0, layerIndex = 0, totalLayers = 1) {
        const diversityBoost = Math.min(diversityScore * 2.0, 0.5);
        const boost = 1 + diversityBoost;

        let currentNorm;

        if (isVector || cols === 1) {
            let flatParam = param;
            if (Array.isArray(param[0])) {
                flatParam = param.map(row => row[0]);
            }
            currentNorm = Math.sqrt(flatParam.reduce((s, v) => s + (isValidNumber(v) ? v * v : 0), 0));
        } else {
            currentNorm = this.#computeSpectralNorm(param);
        }

        if (currentNorm < 1e-6) return;

        let expectedNorm;
        if (isVector && (Array.isArray(param[0]) ? param.map(row => row[0]) : param).every(v => Math.abs(v - 1) < 1e-6)) {
            expectedNorm = Math.sqrt(rows) * 1.5;
        } else if (cols === 1 && rows > 1) {
            expectedNorm = Math.sqrt(2.2) * 2.0;
        } else if (isVector && currentNorm > 1 && currentNorm < 4) {
            expectedNorm = 3.0;
        } else {
            expectedNorm = 3.0;
        }

        const targetNorm = expectedNorm * boost;
        const explosionThreshold = targetNorm * 5.0;
        const scale = Math.max(targetNorm / currentNorm, 0.7);
        const shouldScale = currentNorm > explosionThreshold;

        if (isVector || cols === 1) {
            for (let i = 0; i < param.length; i++) {
                const isColumnVector = Array.isArray(param[i]);
                let val = isColumnVector ? param[i][0] : param[i];

                if (isValidNumber(val)) {
                    if (shouldScale) {
                        val *= scale;
                    }

                    if (Math.abs(val) <= 1e-15) {
                        val = this.#dynamicGeluInit(1, 1, layerIndex, totalLayers)[0][0];
                    }

                    if (isColumnVector) {
                        param[i][0] = val;
                    } else {
                        param[i] = val;
                    }
                }
            }
        } else {
            for (let i = 0; i < rows; i++) {
                for (let j = 0; j < cols; j++) {
                    if (isValidNumber(param[i][j])) {
                        let val = param[i][j];

                        if (shouldScale) {
                            val *= scale;
                        }

                        if (Math.abs(val) <= 1e-15) {
                            val = this.#dynamicGeluInit(1, 1, layerIndex, totalLayers)[0][0];
                        }

                        param[i][j] = val;
                    }
                }
            }
        }
    }

    #scaleGradients (idx) {
        const baseAttentionPercentile = 0.95;
        const baseFfnPercentile = 0.9;
        const baseVectorPercentile = 0.8;

        const trustVariance = this.#computeVariance(this.#trustScoresHistory[idx].slice(-10));
        const lossVariance = this.#computeVariance(this.#historicalPerformance[idx].slice(-10));
        const stabilityMetric = 0.5 * lossVariance + 0.5 * trustVariance;

        const minScaleFactor = Math.max(0.01, 0.1 / (Math.sqrt(this.#ensembleSize) * (1 + stabilityMetric)));

        const emaBetaShort = Math.min(0.95, Math.max(0.8, this.#computeKernelRate(this.#historicalPerformance[idx])));
        const emaBetaLong = Math.min(0.999, Math.max(0.95, this.#computeKernelRate(this.#trustScoresHistory[idx])));

        const fractalDim = this.#computeFractalDimension(this.#historicalPerformance[idx].slice(-10));
        const componentWeights = {
            ensemble: Math.min(0.4, Math.max(0.1, 0.2 * (1 + 0.1 * fractalDim))),
            specialization: Math.min(0.4, Math.max(0.1, 0.2 * (1 + 0.15 * fractalDim * Math.min(this.#specializationScores[idx], 10)))),
            trust: Math.min(0.4, Math.max(0.1, 0.2 * (1 - 0.1 * fractalDim))),
            historicalPerformance: Math.min(0.4, Math.max(0.1, 0.2 * (1 + 0.05 * fractalDim))),
            performance: Math.min(0.4, Math.max(0.1, 0.2 * (1 + 0.05 * fractalDim)))
        };
        const totalWeight = Object.values(componentWeights).reduce((sum, w) => sum + w, 0);
        Object.keys(componentWeights).forEach(key => {
            componentWeights[key] /= Math.max(totalWeight, 1e-8);
        });

        const trust_ema = this.#computeDualEMA(this.#trustScoresHistory[idx], emaBetaShort, emaBetaLong);
        const perf_ema = this.#computeDualEMA(this.#historicalPerformance[idx], emaBetaShort, emaBetaLong);

        const robustnessFactor = Math.min(0.9, Math.max(0.1, 0.5 + 0.5 * (trust_ema / (Math.abs(trust_ema) + Math.abs(perf_ema) + 1e-10))));
        const compositeScore = (
            this.#ensembleWeights[idx] * componentWeights.ensemble +
            this.#specializationScores[idx] * componentWeights.specialization +
            trust_ema * componentWeights.trust +
            perf_ema * componentWeights.historicalPerformance +
            this.#performanceScores[idx] * componentWeights.performance
        ) * robustnessFactor;

        const totalCompositeScore = this.#ensembleWeights.reduce((sum, _, i) => {
            const memberTrustEma = this.#computeDualEMA(this.#trustScoresHistory[i], emaBetaShort, emaBetaLong);
            const memberPerfEma = this.#computeDualEMA(this.#historicalPerformance[i], emaBetaShort, emaBetaLong);
            const memberRobustness = Math.min(0.9, Math.max(0.1, 0.5 + 0.5 * (memberTrustEma / (Math.abs(memberTrustEma) + Math.abs(memberPerfEma) + 1e-10))));
            return sum + (
                this.#ensembleWeights[i] * componentWeights.ensemble +
                this.#specializationScores[i] * componentWeights.specialization +
                memberTrustEma * componentWeights.trust +
                memberPerfEma * componentWeights.historicalPerformance +
                this.#performanceScores[i] * componentWeights.performance
            ) * memberRobustness;
        }, 0);

        const normalizedScore = Math.min(1, Math.max(0, 0.8 * (compositeScore / Math.max(totalCompositeScore, 1e-8)) + 0.2 / this.#ensembleSize));
        const ntkStability = this.#computeNTKStability(this.#historicalPerformance[idx].slice(-10), lossVariance);
        const weightFactorMin = 0.5 / (1 + lossVariance);
        const weightFactorMax = 1 + this.#ensembleSize / (10 * (1 + lossVariance));
        const weightFactor = Math.min(weightFactorMax, Math.max(weightFactorMin, normalizedScore * this.#ensembleSize * ntkStability));

        const sigmoidSlope = 5 / (1 + trustVariance);
        const alpha = Math.min(1, Math.max(0.1, 0.5 + 0.5 * (1 / (1 + Math.exp(sigmoidSlope * trust_ema)) - 0.5)));

        const dynamicK = 4 / (1 + trustVariance);
        const decay = Math.max(0.1, Math.min(1, 1 / (1 + dynamicK * trustVariance)));

        const attentionMatrixNorms = [];
        const ffnMatrixNorms = [];
        const vectorNorms = [];
        const specMatrixNorms = [];

        const outputWeightsGrad = this.#gradientAccumulation[idx].outputWeights;
        ffnMatrixNorms.push(this.#computeSpectralNorm(outputWeightsGrad));
        const outputBiasGrad = this.#gradientAccumulation[idx].outputBias;
        vectorNorms.push(this.#computeGradientNorm(outputBiasGrad, false));

        for (let layer = 0; layer < this.#numLayers; layer++) {
            ['Wq', 'Wk', 'Wv', 'Wo'].forEach(key => {
                const gradMatrix = this.#gradientAccumulation[idx].attentionWeights[layer][key];
                attentionMatrixNorms.push(this.#computeSpectralNorm(gradMatrix));
            });

            ['W1', 'W2'].forEach(key => {
                const gradMatrix = this.#gradientAccumulation[idx].ffnWeights[layer][key];
                ffnMatrixNorms.push(this.#computeSpectralNorm(gradMatrix));
            });

            ['b1', 'b2'].forEach(key => {
                const gradVector = this.#gradientAccumulation[idx].ffnWeights[layer][key];
                vectorNorms.push(this.#computeGradientNorm(gradVector, false));
            });

            ['gamma1', 'beta1', 'gamma2', 'beta2'].forEach(key => {
                const gradVector = this.#gradientAccumulation[idx].layerNormWeights[layer][key];
                vectorNorms.push(this.#computeGradientNorm(gradVector, false));
            });
        }

        const attentionBiasGrad = this.#gradientAccumulation[idx].attentionBias;
        vectorNorms.push(this.#computeGradientNorm(attentionBiasGrad, false));
        const attentionWeightMatrixGrad = this.#gradientAccumulation[idx].attentionWeightMatrix;
        vectorNorms.push(this.#computeGradientNorm(attentionWeightMatrixGrad, false));

        const specializationWeightsGrad = this.#gradientAccumulation[idx].specializationWeights;
        specMatrixNorms.push(this.#computeSpectralNorm(specializationWeightsGrad));

        const sparseThreshold = this.#computeSparseThreshold([...attentionMatrixNorms, ...ffnMatrixNorms, ...vectorNorms, ...specMatrixNorms]);

        const attentionPercentile = this.#computeDynamicPercentile(attentionMatrixNorms, baseAttentionPercentile);
        const ffnPercentile = this.#computeDynamicPercentile(ffnMatrixNorms, baseFfnPercentile);
        const vectorPercentile = this.#computeDynamicPercentile(vectorNorms, baseVectorPercentile);
        const specPercentile = this.#computeDynamicPercentile(specMatrixNorms, baseAttentionPercentile);
        const attentionMatrixThreshold = Math.min(this.#computePercentile(attentionMatrixNorms, attentionPercentile), 1.0) * weightFactor;
        const ffnMatrixThreshold = Math.min(this.#computePercentile(ffnMatrixNorms, ffnPercentile), 1.0) * weightFactor;
        const vectorThreshold = Math.min(this.#computePercentile(vectorNorms, vectorPercentile), 1.0) * weightFactor;
        const specMatrixThreshold = Math.min(this.#computePercentile(specMatrixNorms, specPercentile), 1.0) * weightFactor;

        this.#scaleGradientMatrix(outputWeightsGrad, ffnMatrixThreshold, minScaleFactor, alpha, decay, sparseThreshold);
        this.#scaleGradientVector(outputBiasGrad, vectorThreshold, minScaleFactor, alpha, decay, sparseThreshold);

        for (let layer = 0; layer < this.#numLayers; layer++) {
            ['Wq', 'Wk', 'Wv', 'Wo'].forEach(key => {
                this.#scaleGradientMatrix(this.#gradientAccumulation[idx].attentionWeights[layer][key], attentionMatrixThreshold, minScaleFactor, alpha, decay, sparseThreshold);
            });

            ['W1', 'W2'].forEach(key => {
                this.#scaleGradientMatrix(this.#gradientAccumulation[idx].ffnWeights[layer][key], ffnMatrixThreshold, minScaleFactor, alpha, decay, sparseThreshold);
            });

            ['b1', 'b2'].forEach(key => {
                this.#scaleGradientVector(this.#gradientAccumulation[idx].ffnWeights[layer][key], vectorThreshold, minScaleFactor, alpha, decay, sparseThreshold);
            });

            ['gamma1', 'beta1', 'gamma2', 'beta2'].forEach(key => {
                this.#scaleGradientVector(this.#gradientAccumulation[idx].layerNormWeights[layer][key], vectorThreshold, minScaleFactor, alpha, decay, sparseThreshold);
            });
        }

        this.#scaleGradientVector(attentionBiasGrad, vectorThreshold, minScaleFactor, alpha, decay, sparseThreshold);
        this.#scaleGradientVector(attentionWeightMatrixGrad, vectorThreshold, minScaleFactor, alpha, decay, sparseThreshold);

        this.#scaleGradientMatrix(specializationWeightsGrad, specMatrixThreshold, minScaleFactor, alpha, decay, sparseThreshold);
    }

    #regularizeWeights () {
        const diversityScore = this.#computeDiversityScore(this.#transformers.map(t => t.outputWeights.flat()));

        for (let idx = 0; idx < this.#ensembleSize; idx++) {
            const transformer = this.#transformers[idx];

            this.#applyAutoScaling(transformer.outputWeights, this.#hiddenSize, 1, false, diversityScore, this.#numLayers, this.#numLayers + 1);
            this.#applyAutoScaling(transformer.outputBias, 1, 1, true, diversityScore, 0, 1);

            for (let layer = 0; layer < this.#numLayers; layer++) {
                this.#applyAutoScaling(transformer.attentionWeights[layer].Wq, this.#hiddenSize, this.#hiddenSize, false, diversityScore, layer, this.#numLayers);
                this.#applyAutoScaling(transformer.attentionWeights[layer].Wk, this.#hiddenSize, this.#hiddenSize, false, diversityScore, layer, this.#numLayers);
                this.#applyAutoScaling(transformer.attentionWeights[layer].Wv, this.#hiddenSize, this.#hiddenSize, false, diversityScore, layer, this.#numLayers);
                this.#applyAutoScaling(transformer.attentionWeights[layer].Wo, this.#hiddenSize, this.#hiddenSize, false, diversityScore, layer, this.#numLayers);

                this.#applyAutoScaling(transformer.ffnWeights[layer].W1, this.#hiddenSize, this.#feedForwardSize, false, diversityScore, layer, this.#numLayers);
                this.#applyAutoScaling(transformer.ffnWeights[layer].W2, this.#feedForwardSize, this.#hiddenSize, false, diversityScore, layer, this.#numLayers);

                this.#applyAutoScaling(transformer.ffnWeights[layer].b1, this.#feedForwardSize, 1, true, diversityScore, 0, 1);
                this.#applyAutoScaling(transformer.ffnWeights[layer].b2, this.#hiddenSize, 1, true, diversityScore, 0, 1);
                this.#applyAutoScaling(transformer.layerNormWeights[layer].gamma1, this.#hiddenSize, 1, true, diversityScore, 0, 1);
                this.#applyAutoScaling(transformer.layerNormWeights[layer].beta1, this.#hiddenSize, 1, true, diversityScore, 0, 1);
                this.#applyAutoScaling(transformer.layerNormWeights[layer].gamma2, this.#hiddenSize, 1, true, diversityScore, 0, 1);
                this.#applyAutoScaling(transformer.layerNormWeights[layer].beta2, this.#hiddenSize, 1, true, diversityScore, 0, 1);
            }

            this.#applyAutoScaling(this.#attentionBias[idx], this.#hiddenSize, 1, true, diversityScore, 0, 1);
            this.#applyAutoScaling(this.#attentionWeightMatrix[idx], this.#hiddenSize, 1, true, diversityScore, 0, 1);
            this.#applyAutoScaling(this.#specializationWeights[idx], this.#hiddenSize, this.#hiddenSize, false, diversityScore, 0, 1);
        }
    }

    #accumulateGradients (inputs, layerOutputs, activations, attentionIntermediates, dL_d_output, dL_d_scores) {
        const inputFeatures = inputs.map(x =>
            Array(this.#hiddenSize).fill(isValidNumber(x) ? x : 0)
        );

        this.#transformers.forEach((transformer, idx) => {
            const adjustedLearningRate = this.#adaptiveLearningRate[idx];
            const delta = dL_d_output * adjustedLearningRate;

            const attentionGrad = dL_d_scores[idx] / Math.sqrt(this.#hiddenSize);

            for (let k = 0; k < this.#hiddenSize; k++) {
                this.#gradientAccumulation[idx].attentionBias[k] += attentionGrad * adjustedLearningRate;
                for (let i = 0; i < this.#inputSize; i++) {
                    if (isValidNumber(inputFeatures[i][k])) {
                        const specializationFactor = isValidNumber(this.#specializationScores[idx])
                            ? 1 + this.#specializationScores[idx] * this.#swarmIntelligenceFactor
                            : 1;
                        const matrixUpdate = attentionGrad * inputFeatures[i][k] * specializationFactor / (this.#inputSize + 1e-6);
                        this.#gradientAccumulation[idx].attentionWeightMatrix[k] += matrixUpdate * adjustedLearningRate;
                    }
                }
            }

            for (let j = 0; j < this.#hiddenSize; j++) {
                for (let k = 0; k < this.#hiddenSize; k++) {
                    const inputIdx = (j + k) % this.#inputSize;
                    const inputVal = isValidNumber(inputs[inputIdx]) ? inputs[inputIdx] : 0;
                    const specializationFactor = isValidNumber(this.#specializationScores[idx])
                        ? 1 + this.#specializationScores[idx] * this.#swarmIntelligenceFactor
                        : 1;
                    const gradUpdate = delta * inputVal * specializationFactor;
                    this.#gradientAccumulation[idx].specializationWeights[j][k] += isValidNumber(gradUpdate) ? gradUpdate : 0;
                }
            }

            let grad = Array(this.#hiddenSize).fill(0);

            for (let i = 0; i < this.#hiddenSize; i++) {
                for (let j = 0; j < 1; j++) {
                    const gradUpdate = isValidNumber(delta) && isValidNumber(this.#ensembleWeights[idx]) && isValidNumber(layerOutputs[idx][this.#numLayers][0][i])
                        ? delta * layerOutputs[idx][this.#numLayers][0][i]
                        : 0;
                    this.#gradientAccumulation[idx].outputWeights[i][j] += gradUpdate;
                    grad[i] += isValidNumber(delta) && isValidNumber(layerOutputs[idx][this.#numLayers][0][i])
                        ? delta * layerOutputs[idx][this.#numLayers][0][i]
                        : 0;
                }
            }
            this.#gradientAccumulation[idx].outputBias[0] += isValidNumber(delta) ? delta : 0;

            for (let layer = this.#numLayers - 1; layer >= 0; layer--) {
                const { normX, normAttention } = activations[idx][layer];
                const { Q, K, V, attentionProbs, preWoOutput } = attentionIntermediates[idx][layer];
                const headSize = this.#hiddenSize / this.#numHeads;

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
                        this.#gradientAccumulation[idx].ffnWeights[layer].W1[i][j] += isValidNumber(update) ? update : 0;
                    }
                }
                for (let i = 0; i < this.#feedForwardSize; i++) {
                    const update = adjustedLearningRate * ffnGrad[i];
                    this.#gradientAccumulation[idx].ffnWeights[layer].b1[i] += isValidNumber(update) ? update : 0;
                }
                for (let i = 0; i < this.#feedForwardSize; i++) {
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        const update = adjustedLearningRate * grad[j] * activated[i];
                        this.#gradientAccumulation[idx].ffnWeights[layer].W2[i][j] += isValidNumber(update) ? update : 0;
                    }
                }
                for (let i = 0; i < this.#hiddenSize; i++) {
                    const update = adjustedLearningRate * grad[i];
                    this.#gradientAccumulation[idx].ffnWeights[layer].b2[i] += isValidNumber(update) ? update : 0;
                }

                const normAttentionGrad = grad.slice();
                const gamma2Grad = Array(this.#hiddenSize).fill(0);
                const beta2Grad = Array(this.#hiddenSize).fill(0);
                const meanAttention = normAttention[0].reduce((sum, val) => sum + val, 0) / this.#hiddenSize;
                const varianceAttention = normAttention[0].reduce((sum, val) => sum + Math.pow(val - meanAttention, 2), 0) / this.#hiddenSize;
                const stdAttention = Math.sqrt(varianceAttention + 1e-6);
                for (let i = 0; i < this.#hiddenSize; i++) {
                    const normalized = (normAttention[0][i] - meanAttention) / stdAttention;
                    gamma2Grad[i] = isValidNumber(normAttentionGrad[i]) && isValidNumber(normalized)
                        ? normAttentionGrad[i] * normalized
                        : 0;
                    beta2Grad[i] = isValidNumber(normAttentionGrad[i]) ? normAttentionGrad[i] : 0;
                    const updateGamma = adjustedLearningRate * gamma2Grad[i];
                    const updateBeta = adjustedLearningRate * beta2Grad[i];
                    this.#gradientAccumulation[idx].layerNormWeights[layer].gamma2[i] += isValidNumber(updateGamma) ? updateGamma : 0;
                    this.#gradientAccumulation[idx].layerNormWeights[layer].beta2[i] += isValidNumber(updateBeta) ? updateBeta : 0;
                }

                const attentionGrad = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
                for (let i = 0; i < this.#inputSize; i++) {
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        attentionGrad[i][j] = grad[j];
                    }
                }
                const woGrad = Array(this.#hiddenSize).fill().map(() => Array(this.#hiddenSize).fill(0));
                for (let i = 0; i < this.#inputSize; i++) {
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        for (let k = 0; k < this.#hiddenSize; k++) {
                            woGrad[k][j] += isValidNumber(attentionGrad[i][j]) && isValidNumber(preWoOutput[i][k])
                                ? attentionGrad[i][j] * preWoOutput[i][k]
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
                                const idx = h * headSize + k;
                                vGrad[h][j][k] += isValidNumber(attentionGrad[i][idx]) && isValidNumber(attentionProbs[h][i][j])
                                    ? attentionGrad[i][idx] * attentionProbs[h][i][j]
                                    : 0;
                                scoreGrad[h][i][j] += isValidNumber(attentionGrad[i][idx]) && isValidNumber(V[j][idx])
                                    ? attentionGrad[i][idx] * V[j][idx]
                                    : 0;
                            }
                        }
                    }
                }

                const qGrad = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
                const kGrad = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));

                for (let h = 0; h < this.#numHeads; h++) {
                    for (let i = 0; i < this.#inputSize; i++) {
                        for (let j = 0; j < this.#inputSize; j++) {
                            for (let k = 0; k < headSize; k++) {
                                const idx = h * headSize + k;
                                const scaledScore = isValidNumber(scoreGrad[h][i][j])
                                    ? scoreGrad[h][i][j] / Math.sqrt(headSize)
                                    : 0;
                                qGrad[i][idx] += isValidNumber(scaledScore) && isValidNumber(K[j][idx])
                                    ? scaledScore * K[j][idx]
                                    : 0;
                                kGrad[j][idx] += isValidNumber(scaledScore) && isValidNumber(Q[i][idx])
                                    ? scaledScore * Q[i][idx]
                                    : 0;
                            }
                        }
                    }
                }

                for (let i = 0; i < this.#hiddenSize; i++) {
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        const wqUpdate = qGrad.reduce((sum, row, t) => sum + (isValidNumber(row[j]) && isValidNumber(normX[t][i]) ? row[j] * normX[t][i] : 0), 0);
                        const wkUpdate = kGrad.reduce((sum, row, t) => sum + (isValidNumber(row[j]) && isValidNumber(normX[t][i]) ? row[j] * normX[t][i] : 0), 0);
                        const wvUpdate = vGrad.reduce((sum, head, h) => 
                            sum + head.reduce((innerSum, keyTokenGrad, keyIdx) => 
                                innerSum + (isValidNumber(keyTokenGrad[j % headSize]) && isValidNumber(normX[keyIdx][i])
                                    ? keyTokenGrad[j % headSize] * normX[keyIdx][i]
                                    : 0), 0), 0);
                        const woUpdate = isValidNumber(woGrad[i][j]) ? woGrad[i][j] : 0;

                        const specializationFactor = isValidNumber(this.#specializationWeights[idx][i % this.#hiddenSize][j])
                            ? 1 + this.#specializationScores[idx] * this.#specializationWeights[idx][i % this.#hiddenSize][j]
                            : 1;

                        if (isValidNumber(wqUpdate)) {
                            const update = adjustedLearningRate * wqUpdate * specializationFactor;
                            this.#gradientAccumulation[idx].attentionWeights[layer].Wq[i][j] += isValidNumber(update) ? update : 0;
                        }
                        if (isValidNumber(wkUpdate)) {
                            const update = adjustedLearningRate * wkUpdate * specializationFactor;
                            this.#gradientAccumulation[idx].attentionWeights[layer].Wk[i][j] += isValidNumber(update) ? update : 0;
                        }
                        if (isValidNumber(wvUpdate)) {
                            const update = adjustedLearningRate * wvUpdate * specializationFactor;
                            this.#gradientAccumulation[idx].attentionWeights[layer].Wv[i][j] += isValidNumber(update) ? update : 0;
                        }
                        if (isValidNumber(woUpdate)) {
                            const update = adjustedLearningRate * woUpdate * specializationFactor;
                            this.#gradientAccumulation[idx].attentionWeights[layer].Wo[i][j] += isValidNumber(update) ? update : 0;
                        }
                    }
                }

                const normXGrad = qGrad;
                for (let i = 0; i < this.#inputSize; i++) {
                    const gamma1Grad = Array(this.#hiddenSize).fill(0);
                    const beta1Grad = Array(this.#hiddenSize).fill(0);
                    const meanX = normX[i].reduce((sum, val) => sum + val, 0) / this.#hiddenSize;
                    const varianceX = normX[i].reduce((sum, val) => sum + Math.pow(val - meanX, 2), 0) / this.#hiddenSize;
                    const stdX = Math.sqrt(varianceX + 1e-6);
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        const normalized = (normX[i][j] - meanX) / stdX;
                        gamma1Grad[j] = isValidNumber(normXGrad[i][j]) && isValidNumber(normalized)
                            ? normXGrad[i][j] * normalized
                            : 0;
                        beta1Grad[j] = isValidNumber(normXGrad[i][j]) ? normXGrad[i][j] : 0;
                        const updateGamma = adjustedLearningRate * gamma1Grad[j];
                        const updateBeta = adjustedLearningRate * beta1Grad[j];
                        this.#gradientAccumulation[idx].layerNormWeights[layer].gamma1[j] += isValidNumber(updateGamma) ? updateGamma : 0;
                        this.#gradientAccumulation[idx].layerNormWeights[layer].beta1[j] += isValidNumber(updateBeta) ? updateBeta : 0;
                    }
                }

                grad = qGrad.reduce((acc, vec) => acc.map((v, i) => v + vec[i]), Array(qGrad[0].length).fill(0)).map(v => v / this.#inputSize);
            }
        });
    }

    #applyGradients (shouldScale = true, shouldClone = false, steps = 1) {
        let gradClone;
        if (shouldClone) gradClone = this.#setGradientStructure();

        this.#transformers.forEach((transformer, idx) => {
            if (shouldScale) this.#scaleGradients(idx);

            for (let k = 0; k < this.#hiddenSize; k++) {
                const biasAcc = this.#gradientAccumulation[idx].attentionBias[k];
                if (isValidNumber(biasAcc)) {
                    const finalBiasAcc = biasAcc / steps;
                    this.#attentionBias[idx][k] -= finalBiasAcc;
                    if (shouldClone) gradClone[idx].attentionBias[k] = finalBiasAcc;
                }

                const matrixAcc = this.#gradientAccumulation[idx].attentionWeightMatrix[k];
                if (isValidNumber(matrixAcc)) {
                    const finalMatrixAcc = matrixAcc / steps;
                    this.#attentionWeightMatrix[idx][k] -= finalMatrixAcc;
                    if (shouldClone) gradClone[idx].attentionWeightMatrix[k] = finalMatrixAcc;
                }
            }

            for (let j = 0; j < this.#hiddenSize; j++) {
                for (let k = 0; k < this.#hiddenSize; k++) {
                    const specAcc = this.#gradientAccumulation[idx].specializationWeights[j][k];
                    if (isValidNumber(specAcc)) {
                        const finalSpecAcc = specAcc / steps;
                        this.#specializationWeights[idx][j][k] -= finalSpecAcc;
                        if (shouldClone) gradClone[idx].specializationWeights[j][k] = finalSpecAcc;
                    }
                }
            }

            for (let i = 0; i < this.#hiddenSize; i++) {
                for (let j = 0; j < 1; j++) {
                    const outWeightAcc = this.#gradientAccumulation[idx].outputWeights[i][j];
                    if (isValidNumber(outWeightAcc)) {
                        const finalOutWeightAcc = outWeightAcc / steps;
                        transformer.outputWeights[i][j] -= finalOutWeightAcc
                        if (shouldClone) gradClone[idx].outputWeights[i][j] = finalOutWeightAcc;
                    }
                }
            }

            for (let j = 0; j < 1; j++) {
                const outBiasAcc = this.#gradientAccumulation[idx].outputBias[j];
                if (isValidNumber(outBiasAcc)) {
                    const finalOutBiasAcc = outBiasAcc / steps;
                    transformer.outputBias[j] -= finalOutBiasAcc;
                    if (shouldClone) gradClone[idx].outputBias[j] = finalOutBiasAcc;
                }
            }

            for (let layer = 0; layer < this.#numLayers; layer++) {
                for (let i = 0; i < this.#hiddenSize; i++) {
                    for (let j = 0; j < this.#feedForwardSize; j++) {
                        const w1Acc = this.#gradientAccumulation[idx].ffnWeights[layer].W1[i][j];
                        if (isValidNumber(w1Acc)) {
                            const finalW1Acc = w1Acc / steps;
                            transformer.ffnWeights[layer].W1[i][j] -= finalW1Acc;
                            if (shouldClone) gradClone[idx].ffnWeights[layer].W1[i][j] = finalW1Acc;
                        }
                    }
                }

                for (let i = 0; i < this.#feedForwardSize; i++) {
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        const w2Acc = this.#gradientAccumulation[idx].ffnWeights[layer].W2[i][j];
                        if (isValidNumber(w2Acc)) {
                            const finalW2Acc = w2Acc / steps;
                            transformer.ffnWeights[layer].W2[i][j] -= finalW2Acc;
                            if (shouldClone) gradClone[idx].ffnWeights[layer].W2[i][j] = finalW2Acc;
                        }
                    }
                }

                for (let i = 0; i < this.#feedForwardSize; i++) {
                    const b1Acc = this.#gradientAccumulation[idx].ffnWeights[layer].b1[i];
                    if (isValidNumber(b1Acc)) {
                        const finalB1Acc = b1Acc / steps;
                        transformer.ffnWeights[layer].b1[i] -= finalB1Acc;
                        if (shouldClone) gradClone[idx].ffnWeights[layer].b1[i] = finalB1Acc;
                    }
                }

                for (let i = 0; i < this.#hiddenSize; i++) {
                    const b2Acc = this.#gradientAccumulation[idx].ffnWeights[layer].b2[i];
                    if (isValidNumber(b2Acc)) {
                        const finalB2Acc = b2Acc / steps;
                        transformer.ffnWeights[layer].b2[i] -= finalB2Acc;
                        if (shouldClone) gradClone[idx].ffnWeights[layer].b2[i] = finalB2Acc;
                    }
                }

                for (let i = 0; i < this.#hiddenSize; i++) {
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        const wqAcc = this.#gradientAccumulation[idx].attentionWeights[layer].Wq[i][j];
                        if (isValidNumber(wqAcc)) {
                            const finalWqAcc = wqAcc / steps;
                            transformer.attentionWeights[layer].Wq[i][j] -= finalWqAcc;
                            if (shouldClone) gradClone[idx].attentionWeights[layer].Wq[i][j] = finalWqAcc;
                        }

                        const wkAcc = this.#gradientAccumulation[idx].attentionWeights[layer].Wk[i][j];
                        if (isValidNumber(wkAcc)) {
                            const finalWkAcc = wkAcc / steps;
                            transformer.attentionWeights[layer].Wk[i][j] -= finalWkAcc;
                            if (shouldClone) gradClone[idx].attentionWeights[layer].Wk[i][j] = finalWkAcc;
                        }

                        const wvAcc = this.#gradientAccumulation[idx].attentionWeights[layer].Wv[i][j];
                        if (isValidNumber(wvAcc)) {
                            const finalWvAcc = wvAcc / steps;
                            transformer.attentionWeights[layer].Wv[i][j] -= finalWvAcc;
                            if (shouldClone) gradClone[idx].attentionWeights[layer].Wv[i][j] = finalWvAcc;
                        }

                        const woAcc = this.#gradientAccumulation[idx].attentionWeights[layer].Wo[i][j];
                        if (isValidNumber(woAcc)) {
                            const finalWoAcc = woAcc / steps;
                            transformer.attentionWeights[layer].Wo[i][j] -= finalWoAcc;
                            if (shouldClone) gradClone[idx].attentionWeights[layer].Wo[i][j] = finalWoAcc;
                        }
                    }
                }

                for (let i = 0; i < this.#hiddenSize; i++) {
                    const gamma1Acc = this.#gradientAccumulation[idx].layerNormWeights[layer].gamma1[i];
                    if (isValidNumber(gamma1Acc)) {
                        const finalGamma1Acc = gamma1Acc / steps;
                        transformer.layerNormWeights[layer].gamma1[i] -= finalGamma1Acc;
                        if (shouldClone) gradClone[idx].layerNormWeights[layer].gamma1[i] = finalGamma1Acc;
                    }

                    const gamma2Acc = this.#gradientAccumulation[idx].layerNormWeights[layer].gamma2[i];
                    if (isValidNumber(gamma2Acc)) {
                        const finalGamma2Acc = gamma2Acc / steps;
                        transformer.layerNormWeights[layer].gamma2[i] -= finalGamma2Acc;
                        if (shouldClone) gradClone[idx].layerNormWeights[layer].gamma2[i] = finalGamma2Acc;
                    }

                    const beta1Acc = this.#gradientAccumulation[idx].layerNormWeights[layer].beta1[i];
                    if (isValidNumber(beta1Acc)) {
                        const finalBeta1Acc = beta1Acc / steps;
                        transformer.layerNormWeights[layer].beta1[i] -= finalBeta1Acc;
                        if (shouldClone) gradClone[idx].layerNormWeights[layer].beta1[i] = finalBeta1Acc;
                    }

                    const beta2Acc = this.#gradientAccumulation[idx].layerNormWeights[layer].beta2[i];
                    if (isValidNumber(beta2Acc)) {
                        const finalBeta2Acc = beta2Acc / steps;
                        transformer.layerNormWeights[layer].beta2[i] -= finalBeta2Acc;
                        if (shouldClone) gradClone[idx].layerNormWeights[layer].beta2[i] = finalBeta2Acc;
                    }
                }
            }
        });

        if (shouldClone) return gradClone;
    }

    #rollbackGradients (gradClone) {
        this.#transformers.forEach((transformer, idx) => {
            for (let k = 0; k < this.#hiddenSize; k++) {
                const attBias = gradClone[idx].attentionBias[k];
                if (isValidNumber(attBias)) this.#attentionBias[idx][k] += attBias;

                const attMatrix = gradClone[idx].attentionWeightMatrix[k];
                if (isValidNumber(attMatrix)) this.#attentionWeightMatrix[idx][k] += attMatrix;
            }

            for (let j = 0; j < this.#hiddenSize; j++) {
                for (let k = 0; k < this.#hiddenSize; k++) {
                    const specWeight = gradClone[idx].specializationWeights[j][k];
                    if (isValidNumber(specWeight)) this.#specializationWeights[idx][j][k] += specWeight;
                }
            }

            for (let i = 0; i < this.#hiddenSize; i++) {
                for (let j = 0; j < 1; j++) {
                    const outputWeight = gradClone[idx].outputWeights[i][j];
                    if (isValidNumber(outputWeight)) transformer.outputWeights[i][j] += outputWeight;
                }
            }

            for (let j = 0; j < 1; j++) {
                const outputBias = gradClone[idx].outputBias[j];
                if (isValidNumber(outputBias)) transformer.outputBias[j] += outputBias;
            }

            for (let layer = 0; layer < this.#numLayers; layer++) {
                for (let i = 0; i < this.#hiddenSize; i++) {
                    for (let j = 0; j < this.#feedForwardSize; j++) {
                        const w1Weight = gradClone[idx].ffnWeights[layer].W1[i][j];
                        if (isValidNumber(w1Weight)) transformer.ffnWeights[layer].W1[i][j] += w1Weight;
                    }
                }

                for (let i = 0; i < this.#feedForwardSize; i++) {
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        const w2Weight = gradClone[idx].ffnWeights[layer].W2[i][j];
                        if (isValidNumber(w2Weight))transformer.ffnWeights[layer].W2[i][j] += w2Weight;
                    }
                }

                for (let i = 0; i < this.#feedForwardSize; i++) {
                    const b1Weight = gradClone[idx].ffnWeights[layer].b1[i];
                    if (isValidNumber(b1Weight)) transformer.ffnWeights[layer].b1[i] += b1Weight;
                }

                for (let i = 0; i < this.#hiddenSize; i++) {
                    const b2Weight = gradClone[idx].ffnWeights[layer].b2[i];
                    if (isValidNumber(b2Weight)) transformer.ffnWeights[layer].b2[i] += b2Weight;
                }

                for (let i = 0; i < this.#hiddenSize; i++) {
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        const wqWeight = gradClone[idx].attentionWeights[layer].Wq[i][j];
                        if (isValidNumber(wqWeight)) transformer.attentionWeights[layer].Wq[i][j] += wqWeight;

                        const wkWeight = gradClone[idx].attentionWeights[layer].Wk[i][j];
                        if (isValidNumber(wkWeight)) transformer.attentionWeights[layer].Wk[i][j] += wkWeight;

                        const wvWeight = gradClone[idx].attentionWeights[layer].Wv[i][j];
                        if (isValidNumber(wvWeight)) transformer.attentionWeights[layer].Wv[i][j] += wvWeight;

                        const woWeight = gradClone[idx].attentionWeights[layer].Wo[i][j];
                        if (isValidNumber(woWeight)) transformer.attentionWeights[layer].Wo[i][j] += woWeight;
                    }
                }

                for (let i = 0; i < this.#hiddenSize; i++) {
                    const gamma1Weight = gradClone[idx].layerNormWeights[layer].gamma1[i];
                    if (isValidNumber(gamma1Weight)) transformer.layerNormWeights[layer].gamma1[i] += gamma1Weight;

                    const gamma2Weight = gradClone[idx].layerNormWeights[layer].gamma2[i];
                    if (isValidNumber(gamma2Weight)) transformer.layerNormWeights[layer].gamma2[i] += gamma2Weight;

                    const beta1Weight = gradClone[idx].layerNormWeights[layer].beta1[i];
                    if (isValidNumber(beta1Weight)) transformer.layerNormWeights[layer].beta1[i] += beta1Weight;

                    const beta2Weight = gradClone[idx].layerNormWeights[layer].beta2[i];
                    if (isValidNumber(beta2Weight)) transformer.layerNormWeights[layer].beta2[i] += beta2Weight;
                }
            }
        });
    }

    #distillKnowledge (outputs, attentionIntermediatesAll, activationsAll, layerOutputsAll) {
        if (
            !Array.isArray(outputs) ||
            outputs.length !== this.#ensembleSize ||
            !outputs.every(isValidNumber) ||
            attentionIntermediatesAll.length !== this.#ensembleSize ||
            activationsAll.length !== this.#ensembleSize ||
            layerOutputsAll.length !== this.#ensembleSize
        ) {
            return;
        }

        const sortedIndices = this.#performanceScores
            .map((score, idx) => ({ score: isValidNumber(score) ? score : 0, idx }))
            .sort((a, b) => b.score - a.score);

        const topTierCount = Math.max(1, Math.floor(this.#ensembleSize * 0.30));
        const middleTierEnd = topTierCount + Math.floor(this.#ensembleSize * 0.40);
        const topPerformers = sortedIndices.slice(0, topTierCount).map(({ idx }) => idx);

        const topOutputs = topPerformers.map(idx => outputs[idx]);
        const topWeights = topPerformers.map(idx => this.#ensembleWeights[idx]);
        const weightSum = topWeights.reduce((sum, w) => sum + (isValidNumber(w) ? w : 0), 0) || 1;
        const teacherLogit = topOutputs.reduce((sum, output, i) =>
            sum + output * (topWeights[i] / weightSum), 0
        );

        const varianceTop = topOutputs.reduce((sum, out) => sum + Math.pow(out - teacherLogit, 2), 0) / topOutputs.length;
        const stdTopOutputs = Math.sqrt(varianceTop + 1e-8);

        const temperature = 2.0;
        const momentum = 0.9;

        this.#transformers.forEach((transformer, idx) => {
            if (topPerformers.includes(idx)) return;

            const rank = sortedIndices.findIndex(entry => entry.idx === idx);
            const isMiddleTier = rank >= topTierCount && rank < middleTierEnd;
            const isBottomTier = rank >= middleTierEnd;

            const kdGlobalScale = this.#adaptiveLearningRate[idx];
            const kdStrengthMultiplier = isMiddleTier ? 1.0 : 0.7;

            const diversityScore = Math.min(1, Math.abs(outputs[idx] - teacherLogit) / (stdTopOutputs || 1));
            const performanceGap = this.#performanceScores[topPerformers[0]] - this.#performanceScores[idx];

            let klWeight = Math.min(1.0, Math.max(0.1, performanceGap / (this.#performanceScores[topPerformers[0]] || 1)));
            klWeight *= (1 - diversityScore);
            klWeight *= kdStrengthMultiplier;
            klWeight = Math.min(1.0, klWeight);

            const studentLogit = outputs[idx];
            const studentProb = this.#sigmoid(studentLogit / temperature);
            const teacherProb = this.#sigmoid(teacherLogit / temperature);

            let baseGrad = (studentProb - teacherProb) / temperature;
            baseGrad = baseGrad * 0.5 * klWeight * kdGlobalScale;
            if (!isValidNumber(baseGrad)) baseGrad = 0;

            const adjustedLearningRate = this.#adaptiveLearningRate[idx];

            for (let i = 0; i < this.#hiddenSize; i++) {
                const specializationFactor = isValidNumber(this.#specializationWeights[idx][i % this.#hiddenSize][0])
                    ? Math.min(Math.max(1 + this.#specializationScores[idx] * this.#specializationWeights[idx][i % this.#hiddenSize][0], 0.5), 1.5)
                    : 1;
                const gradUpdate = baseGrad * (transformer.outputWeights[i][0] || 0) * specializationFactor;
                const update = 0.2 * adjustedLearningRate * gradUpdate;
                this.#gradientAccumulation[idx].outputWeights[i][0] =
                    momentum * (this.#gradientAccumulation[idx].outputWeights[i][0] || 0) + (1 - momentum) * update;
            }
            this.#gradientAccumulation[idx].outputBias[0] =
                momentum * (this.#gradientAccumulation[idx].outputBias[0] || 0) + (1 - momentum) * (0.2 * adjustedLearningRate * baseGrad);

            if (!isBottomTier) return;

            const deepBaseGrad = baseGrad * 1.8;
            const headSize = this.#hiddenSize / this.#numHeads;

            for (let layer = this.#numLayers - 1; layer >= 0; layer--) {
                const inter = attentionIntermediatesAll[idx][layer];
                const act = activationsAll[idx][layer];

                const attentionOutputGrad = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(deepBaseGrad));

                const woGrad = Array(this.#hiddenSize).fill().map(() => Array(this.#hiddenSize).fill(0));
                for (let i = 0; i < this.#inputSize; i++) {
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        for (let k = 0; k < this.#hiddenSize; k++) {
                            woGrad[k][j] += attentionOutputGrad[i][j] * (inter.preWoOutput[i][k] || 0);
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
                                const gradIndex = h * headSize + k;
                                const gradVal = attentionOutputGrad[i][gradIndex];

                                scoreGrad[h][i][j] += gradVal * (inter.V[j][gradIndex] || 0);
                                vGrad[h][j][k] += (inter.attentionProbs[h][i][j] || 0) * gradVal;
                            }
                        }
                    }
                }

                const qGrad = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
                const kGrad = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));

                for (let h = 0; h < this.#numHeads; h++) {
                    for (let i = 0; i < this.#inputSize; i++) {
                        for (let j = 0; j < this.#inputSize; j++) {
                            const scaledScore = scoreGrad[h][i][j] / Math.sqrt(headSize);
                            for (let k = 0; k < headSize; k++) {
                                const headIdx = h * headSize + k;
                                qGrad[i][headIdx] += scaledScore * (inter.K[j][headIdx] || 0);
                                kGrad[j][headIdx] += scaledScore * (inter.Q[i][headIdx] || 0);
                            }
                        }
                    }
                }

                const topPerformerAttention = topPerformers.map(pidx => this.#transformers[pidx].attentionWeights[layer]);

                for (let i = 0; i < this.#hiddenSize; i++) {
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        const avgDiff = (key) =>
                            topPerformerAttention.reduce((sum, w) =>
                                sum + Math.abs((transformer.attentionWeights[layer][key][i][j] || 0) - (w[key][i][j] || 0)), 0
                            ) / topPerformers.length;

                        const specializationFactor = isValidNumber(this.#specializationWeights[idx][i % this.#hiddenSize][j])
                            ? Math.min(Math.max(1 + this.#specializationScores[idx] * this.#specializationWeights[idx][i % this.#hiddenSize][j], 0.5), 1.5)
                            : 1;

                        let wqUpdate = 0;
                        for (let r = 0; r < this.#inputSize; r++) {
                            wqUpdate += qGrad[r][j] * (act.normX[r][i] || 0);
                        }
                        wqUpdate /= Math.sqrt(headSize);

                        let wkUpdate = 0;
                        for (let r = 0; r < this.#inputSize; r++) {
                            wkUpdate += kGrad[r][j] * (act.normX[r][i] || 0);
                        }
                        wkUpdate /= Math.sqrt(headSize);

                        let wvUpdate = 0;
                        for (let h = 0; h < this.#numHeads; h++) {
                            for (let keyIdx = 0; keyIdx < this.#inputSize; keyIdx++) {
                                const dim = j % headSize;
                                wvUpdate += vGrad[h][keyIdx][dim] * (act.normX[keyIdx][i] || 0);
                            }
                        }
                        wvUpdate /= (this.#numHeads * Math.sqrt(headSize));

                        const woUpdate = woGrad[i][j] / (this.#numHeads * this.#inputSize);

                        const applyUpdate = (key, gradVal) => {
                            if (isValidNumber(gradVal) && avgDiff(key) > 0.1) {
                                const update = adjustedLearningRate * gradVal * specializationFactor;
                                this.#gradientAccumulation[idx].attentionWeights[layer][key][i][j] =
                                    momentum * (this.#gradientAccumulation[idx].attentionWeights[layer][key][i][j] || 0) + (1 - momentum) * update;
                            }
                        };

                        applyUpdate('Wq', wqUpdate);
                        applyUpdate('Wk', wkUpdate);
                        applyUpdate('Wv', wvUpdate);
                        applyUpdate('Wo', woUpdate);
                    }
                }

                const ffnInput = act.normAttention[0];
                const hidden = Array(this.#feedForwardSize).fill(0);
                for (let i = 0; i < this.#hiddenSize; i++) {
                    for (let j = 0; j < this.#feedForwardSize; j++) {
                        hidden[j] += ffnInput[i] * (transformer.ffnWeights[layer].W1[i][j] || 0);
                    }
                }
                const activated = hidden.map((val, i) => this.#gelu(val + (transformer.ffnWeights[layer].b1[i] || 0)));

                let ffnGrad = Array(this.#feedForwardSize).fill(0);
                for (let i = 0; i < this.#feedForwardSize; i++) {
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        ffnGrad[i] += deepBaseGrad * (transformer.ffnWeights[layer].W2[i][j] || 0);
                    }
                    ffnGrad[i] *= this.#geluDerivative(hidden[i]);
                }

                for (let i = 0; i < this.#hiddenSize; i++) {
                    for (let j = 0; j < this.#feedForwardSize; j++) {
                        const spec = isValidNumber(this.#specializationWeights[idx][i % this.#hiddenSize][j % this.#hiddenSize])
                            ? Math.min(Math.max(1 + this.#specializationScores[idx] * this.#specializationWeights[idx][i % this.#hiddenSize][j % this.#hiddenSize], 0.5), 1.5)
                            : 1;
                        const update = adjustedLearningRate * ffnGrad[j] * ffnInput[i] * spec / Math.sqrt(this.#feedForwardSize);
                        this.#gradientAccumulation[idx].ffnWeights[layer].W1[i][j] =
                            momentum * (this.#gradientAccumulation[idx].ffnWeights[layer].W1[i][j] || 0) + (1 - momentum) * update;
                    }
                }
                for (let i = 0; i < this.#feedForwardSize; i++) {
                    const update = adjustedLearningRate * ffnGrad[i];
                    this.#gradientAccumulation[idx].ffnWeights[layer].b1[i] =
                        momentum * (this.#gradientAccumulation[idx].ffnWeights[layer].b1[i] || 0) + (1 - momentum) * update;
                }
                for (let i = 0; i < this.#feedForwardSize; i++) {
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        const spec = isValidNumber(this.#specializationWeights[idx][j % this.#hiddenSize][i % this.#hiddenSize])
                            ? Math.min(Math.max(1 + this.#specializationScores[idx] * this.#specializationWeights[idx][j % this.#hiddenSize][i % this.#hiddenSize], 0.5), 1.5)
                            : 1;
                        const update = adjustedLearningRate * deepBaseGrad * activated[i] * spec / Math.sqrt(this.#hiddenSize);
                        this.#gradientAccumulation[idx].ffnWeights[layer].W2[i][j] =
                            momentum * (this.#gradientAccumulation[idx].ffnWeights[layer].W2[i][j] || 0) + (1 - momentum) * update;
                    }
                }
                for (let i = 0; i < this.#hiddenSize; i++) {
                    const update = adjustedLearningRate * deepBaseGrad;
                    this.#gradientAccumulation[idx].ffnWeights[layer].b2[i] =
                        momentum * (this.#gradientAccumulation[idx].ffnWeights[layer].b2[i] || 0) + (1 - momentum) * update;
                }

                for (let i = 0; i < this.#inputSize; i++) {
                    const mean = act.normX[i].reduce((s, v) => s + v, 0) / this.#hiddenSize;
                    const variance = act.normX[i].reduce((s, v) => s + Math.pow(v - mean, 2), 0) / this.#hiddenSize;
                    const std = Math.sqrt(variance + 1e-6);
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        const normalized = (act.normX[i][j] - mean) / std;
                        const gamma1Grad = (qGrad[i][j] || 0) * normalized;
                        const beta1Grad = (qGrad[i][j] || 0);
                        const gamma2Grad = deepBaseGrad * normalized;
                        const beta2Grad = deepBaseGrad;

                        const addUpdate = (key, grad) => {
                            const update = adjustedLearningRate * grad;
                            this.#gradientAccumulation[idx].layerNormWeights[layer][key][j] =
                                momentum * (this.#gradientAccumulation[idx].layerNormWeights[layer][key][j] || 0) + (1 - momentum) * update;
                        };
                        addUpdate('gamma1', gamma1Grad);
                        addUpdate('beta1', beta1Grad);
                        addUpdate('gamma2', gamma2Grad);
                        addUpdate('beta2', beta2Grad);
                    }
                }
            }
        });
    }

    predict (inputs) {
        if ( !Array.isArray(inputs) || inputs.length !== this.#inputSize || !inputs.every(isValidNumber) ) { return 0 }

        const outputs = this.#transformers.map((_, idx) => this.#processTransformer(inputs, idx, false, false));

        this.#computeAttentionWeights(inputs, outputs);

        const finalOutput = this.#computeWeightedSum(outputs, this.#ensembleWeights);

        return this.#sigmoid(isValidNumber(finalOutput) ? finalOutput : 0);
    }

    train (inputs, target) {
        if ( !Array.isArray(inputs) || inputs.length !== this.#inputSize || !inputs.every(isValidNumber) || !isValidNumber(target) ) { return }

        this.#trainingStepCount++;

        const linearOutputs = [];
        const trainingLayerOutputs = [];
        const trainingActivations = [];
        const trainingAttentionIntermediates = [];

        this.#transformers.forEach((_, idx) => {
            const result = this.#processTransformer(inputs, idx, true, true);
            linearOutputs[idx] = result.output;
            trainingLayerOutputs[idx] = result.layerOutputs;
            trainingActivations[idx] = result.activations;
            trainingAttentionIntermediates[idx] = result.attentionIntermediates;
        });

        this.#computeAttentionWeights(inputs, linearOutputs);

        const finalLinearOutput = this.#computeWeightedSum(linearOutputs, this.#ensembleWeights);
        const finalProbability = this.#sigmoid(finalLinearOutput);

        const dL_d_output = finalProbability - target;
        const dL_d_w = linearOutputs.map(out => dL_d_output * out);

        const dL_d_scores = Array(this.#ensembleSize).fill(0);
        for (let t = 0; t < this.#ensembleSize; t++) {
            for (let s = 0; s < this.#ensembleSize; s++) {
                if (s === t) {
                    dL_d_scores[t] += dL_d_w[s] * this.#ensembleWeights[s] * (1 - this.#ensembleWeights[s]);
                } else {
                    dL_d_scores[t] -= dL_d_w[s] * this.#ensembleWeights[s] * this.#ensembleWeights[t];
                }
            }
        }

        this.#updateMetrics(inputs, linearOutputs, target, finalProbability);

        this.#accumulateGradients(inputs, trainingLayerOutputs, trainingActivations, trainingAttentionIntermediates, dL_d_output, dL_d_scores);

        if (this.#trainingStepCount % this.#gradientResetFrequency === 0) {
            const clonedGrads = this.#applyGradients(true, true, this.#gradientResetFrequency);

            this.#gradientAccumulation = structuredClone(clonedGrads);

            const freshOutputs = [];
            const cleanAttentionIntermediates = [];
            const cleanActivations = [];
            const cleanLayerOutputs = [];

            this.#transformers.forEach((_, idx) => {
                const result = this.#processTransformer(inputs, idx, true, false);
                freshOutputs[idx] = result.output;
                cleanAttentionIntermediates[idx] = result.attentionIntermediates;
                cleanActivations[idx] = result.activations;
                cleanLayerOutputs[idx] = result.layerOutputs;
            });

            this.#computeAttentionWeights(inputs, freshOutputs);
            const freshFinalOutput = this.#computeWeightedSum(freshOutputs, this.#ensembleWeights);
            const freshProbability = this.#sigmoid(freshFinalOutput);

            this.#updateMetrics(inputs, freshOutputs, target, freshProbability);

            this.#distillKnowledge(freshOutputs, cleanAttentionIntermediates, cleanActivations, cleanLayerOutputs);

            this.#rollbackGradients(clonedGrads);

            this.#applyGradients(false, false, 1);

            const endOutputs = this.#transformers.map((_, idx) => this.#processTransformer(inputs, idx, false, false));
            this.#computeAttentionWeights(inputs, endOutputs);
            const endFinalOutput = this.#computeWeightedSum(endOutputs, this.#ensembleWeights);
            const endProbability = this.#sigmoid(endFinalOutput);

            this.#updateMetrics(inputs, endOutputs, target, endProbability);

            this.#regularizeWeights();

            this.#gradientAccumulation = this.#setGradientStructure();
        }

        return this.#trainingStepCount;
    }

    dumpState () {
        return this.#saveState()
    }
}

export default HiveMind;