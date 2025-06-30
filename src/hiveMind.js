import fs from 'fs';
import path from 'path';
import Database from 'better-sqlite3';

import { isValidNumber } from './utils.js';

class HiveMind {
    #inputSize = 10;
    #ensembleSize = 7;
    #numLayers = 3;
    #numHeads = 4;
    #hiddenSize = this.#numHeads * 4;
    #feedForwardSize = this.#hiddenSize * 4;

    #learningRate = 1e-3;
    #learningRateDecay = 1e-4;
    #weightDecayRate = 1e-3;

    #dropoutRate = 0.1;
    #contextSensitivity = 0.5;
    #swarmIntelligenceFactor = 0.6;

    #distillationFrequency = 100;
    #regulationFrequency = 1000;

    #contextWindow = this.#hiddenSize * 2;
    #maxTrustHistory = this.#contextWindow * 2;
    #maxPerformanceHistory = this.#contextWindow * 4;
    #attentionScalingFactor = 1 / Math.sqrt(this.#hiddenSize / this.#numHeads);

    #adaptiveLearningRate = [];
    #transformers = [];
    #ensembleWeights = [];
    #gradientAccumulation = [];
    #attentionWeightMatrix = [];
    #attentionBias = [];
    #attentionMemory = [];
    #performanceScores = [];
    #agreementScores = [];
    #historicalPerformance = [];
    #trustScoresHistory = [];
    #specializationWeights = [];
    #specializationScores = [];

    #trainingStepCount = 0;
    #directoryPath;

    constructor(dp) {
        this.#directoryPath = dp;

        this.#performanceScores = Array(this.#ensembleSize).fill(0);
        this.#agreementScores = Array(this.#ensembleSize).fill(0);
        this.#specializationScores = Array(this.#ensembleSize).fill(0);
        this.#trustScoresHistory = Array(this.#ensembleSize).fill().map(() => [0]);
        this.#historicalPerformance = Array(this.#ensembleSize).fill().map(() => [0]);
        this.#adaptiveLearningRate = Array(this.#ensembleSize).fill(this.#learningRate);

        this.#ensembleWeights = Array(this.#ensembleSize).fill().map(() => 
            Math.max(0, 1 / this.#ensembleSize + (Math.random() - 0.5) * 0.1 / this.#ensembleSize)
        );

        this.#attentionMemory = Array(this.#ensembleSize).fill().map(() =>
            Array(this.#contextWindow).fill().map(() => Array(this.#hiddenSize).fill(0))
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

        this.#transformers = Array(this.#ensembleSize).fill().map(() => ({
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

        this.#gradientAccumulation = this.#createWeightStructure();

        this.#normalizeEnsembleWeights();

        const loadStatus = this.#loadState()

        if (!loadStatus.status && loadStatus.error) {
            console.log(`Load state failed! Error: ${loadStatus.error}. Trace : ${loadStatus.trace}`)
            process.exit()
        } else if (loadStatus.status) {
            console.log('State loaded successfully!')
        } else {
            console.log('Started with new state!')
        }
    }

    #saveState() {
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

    #loadState() {
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
            this.#normalizeEnsembleWeights();

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
            this.#historicalPerformance = Array(this.#ensembleSize).fill().map(() => []);
            historicalPerformance.forEach(({ idx, step, score }) => {
                if (isValidNumber(score) && idx >= 0 && idx < this.#ensembleSize && Number.isInteger(step)) {
                    this.#historicalPerformance[idx][step] = score;
                }
            });

            const trustScoresHistoryStmt = db.prepare('SELECT idx, step, score FROM trust_scores_history ORDER BY idx, step');
            const trustScoresHistory = trustScoresHistoryStmt.all();
            this.#trustScoresHistory = Array(this.#ensembleSize).fill().map(() => []);
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
            this.#attentionMemory = Array(this.#ensembleSize).fill().map(() =>
                Array(this.#contextWindow).fill().map(() => Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0)))
            );
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

    #dynamicGeluInit(rows, cols, layerIndex, totalLayers, customK = null) {
        let baseK = customK !== null ? customK : 2.2;
        let fanInScale = Math.min(1.0, 1000 / rows);
        let depthScale = Math.pow(totalLayers, -layerIndex / totalLayers);
        let k = baseK * fanInScale * depthScale;
        k = Math.max(1.5, Math.min(k, 3.0));
        
        return Array(rows).fill().map(() =>
            Array(cols).fill().map(() => (Math.random() - 0.5) * Math.sqrt(k / rows))
        );
    }

    #createPositionalEncoding() {
        const base = 10000 + (Math.random() - 0.5) * 2000;
        return Array(this.#inputSize).fill().map((_, pos) =>
            Array(this.#hiddenSize).fill().map((_, d) => {
                const exponent = 2 * Math.floor(d / 2) / this.#hiddenSize;
                const freq = 1 / (base ** exponent);
                return d % 2 === 0 ? Math.sin(pos * freq) : Math.cos(pos * freq);
            })
        );
    }

    #createWeightStructure() {
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
            attentionWeightMatrix: Array(this.#hiddenSize).fill(0)
        }));
    }

    #computeAttentionWeights(inputs, outputs) {
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

        const inputFeatures = inputs.map(x =>
            Array(this.#hiddenSize).fill(isValidNumber(x) ? x : 0)
        );

        const attentionScores = Array(this.#ensembleSize).fill(0);
        for (let t = 0; t < this.#ensembleSize; t++) {
            if (
                !Array.isArray(this.#attentionMemory[t]) ||
                this.#attentionMemory[t].length === 0 ||
                !Array.isArray(this.#attentionMemory[t][0]) ||
                this.#attentionMemory[t][0].length !== this.#inputSize ||
                !this.#attentionMemory[t][0].every(row => Array.isArray(row) && row.length === this.#hiddenSize && row.every(isValidNumber))
            ) {
                this.#attentionMemory[t] = Array(this.#contextWindow).fill().map(() =>
                    Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0))
                );
            }

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

            let score = 0;
            for (let i = 0; i < this.#inputSize; i++) {
                for (let j = 0; j < this.#inputSize; j++) {
                    let dotProduct = 0;
                    for (let k = 0; k < this.#hiddenSize; k++) {
                        dotProduct += isValidNumber(queries[i][k]) && isValidNumber(keys[j][k])
                            ? queries[i][k] * keys[j][k]
                            : 0;
                    }
                    const attentionWeight = this.#softmax(
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

            const biasSum = this.#attentionBias[t].reduce((sum, val) => sum + (isValidNumber(val) ? val : 0), 0);
            score = score / this.#inputSize + (isValidNumber(biasSum) ? biasSum / this.#hiddenSize : 0);

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
        const trustScore = this.#sigmoid(normalizedScore * (0.6 + 0.2 * agreementFactor + 0.1 * historicalTrend + 0.1 * specializationBoost));
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

    #computeSpecializationScores(inputs, outputs) {
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

        for (let i = 0; i < this.#ensembleSize; i++) {
            const specializationFactor = Math.min(
                Math.max(1 + this.#specializationScores[i] * this.#swarmIntelligenceFactor * 0.5, 0.5),
                1.5
            );

            for (let j = 0; j < this.#hiddenSize; j++) {
                for (let k = 0; k < this.#hiddenSize; k++) {
                    const inputIdx = (j + k) % this.#inputSize;
                    const inputVal = isValidNumber(inputs[inputIdx]) ? inputs[inputIdx] : 0;
                    const update = isValidNumber(this.#adaptiveLearningRate[i])
                        ? this.#adaptiveLearningRate[i] * inputVal * zScores[i] * specializationFactor * 0.1
                        : 0;
                    this.#specializationWeights[i][j][k] += update;
                }
            }
        }

        for (let i = 0; i < this.#ensembleSize; i++) {
            const specMatrix = this.#specializationWeights[i];
            const spectralNorm = this.#computeSpectralNorm(specMatrix);
            if (spectralNorm > 1.5) {
                const scale = 1.5 / spectralNorm;
                for (let j = 0; j < specMatrix.length; j++) {
                    const row = specMatrix[j];
                    for (let k = 0; k < row.length; k++) {
                        row[k] *= scale;
                    }
                }
            }
        }
    }

    #updateAdaptiveLearningRates() {
        const recent_performance = this.#historicalPerformance.map(history => {
            const len = history.length;
            if (len === 0) return 0.5;
            const sum = history.reduce((s, v) => s + (isValidNumber(v) ? v : 0), 0);
            return sum / len;
        });

        const sortedPerformances = recent_performance.slice().sort((a, b) => b - a);
        const thresholdIdx = Math.max(1, Math.floor(recent_performance.length * 0.25));
        const acceptable_threshold = sortedPerformances[thresholdIdx] || 0.5;

        const min_lr = this.#learningRate * 0.3;
        const max_lr = this.#learningRate * 3;
        const decay_rate = this.#learningRateDecay;
        const base_adjustment_factor = 0.01;
        const decay_constant = 0.001;

        const adjustment_factor = base_adjustment_factor * Math.exp(-decay_constant * this.#trainingStepCount);

        this.#adaptiveLearningRate = this.#adaptiveLearningRate.map((lr, idx) => {
            let newLr = lr;
            const perf = recent_performance[idx];
            const perf_diff = perf - acceptable_threshold;

            const adjustment_magnitude = this.#sigmoid(perf_diff) * 2 - 1;
            if (perf >= acceptable_threshold) {
                newLr *= (1 - decay_rate * adjustment_factor * Math.abs(adjustment_magnitude));
            } else {
                newLr *= (1 + decay_rate * adjustment_factor * Math.abs(adjustment_magnitude));
            }

            newLr = 0.9 * newLr + 0.1 * lr;

            return Math.max(min_lr, Math.min(newLr, max_lr));
        });
    }

    #normalizeEnsembleWeights() {
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

    #computeSpectralNorm(matrix) {
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

    #scaleGradients(idx) {
        const matrixPercentile = 0.95;
        const vectorPercentile = 0.95;
        const maxScaleFactor = 10.0;
        const minScaleFactor = 0.1;

        const computePercentile = (norms, percentile) => {
            const sortedNorms = [...norms].sort((a, b) => a - b);
            const index = Math.floor(percentile * (sortedNorms.length - 1));
            return sortedNorms[index] || 1.0;
        };

        const matrixNorms = [];
        const vectorNorms = [];

        const outputWeightsGrad = this.#gradientAccumulation[idx].outputWeights;
        matrixNorms.push(this.#computeSpectralNorm(outputWeightsGrad));

        const outputBiasGrad = this.#gradientAccumulation[idx].outputBias;
        vectorNorms.push(this.#computeGradientNorm(outputBiasGrad, false));

        for (let layer = 0; layer < this.#numLayers; layer++) {
            ['Wq', 'Wk', 'Wv', 'Wo'].forEach(key => {
                const gradMatrix = this.#gradientAccumulation[idx].attentionWeights[layer][key];
                matrixNorms.push(this.#computeSpectralNorm(gradMatrix));
            });

            ['W1', 'W2'].forEach(key => {
                const gradMatrix = this.#gradientAccumulation[idx].ffnWeights[layer][key];
                matrixNorms.push(this.#computeSpectralNorm(gradMatrix));
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

        const matrixThreshold = Math.min(computePercentile(matrixNorms, matrixPercentile), 1.0);
        const vectorThreshold = Math.min(computePercentile(vectorNorms, vectorPercentile), 1.0);

        let spectralNorm = this.#computeSpectralNorm(outputWeightsGrad);
        if (spectralNorm > matrixThreshold) {
            const scale = Math.max(minScaleFactor, Math.min(maxScaleFactor, matrixThreshold / spectralNorm));
            for (let i = 0; i < outputWeightsGrad.length; i++) {
                for (let j = 0; j < outputWeightsGrad[i].length; j++) {
                    outputWeightsGrad[i][j] *= scale;
                }
            }
        }

        let l2Norm = this.#computeGradientNorm(outputBiasGrad, false);
        if (l2Norm > vectorThreshold) {
            const scale = Math.max(minScaleFactor, Math.min(maxScaleFactor, vectorThreshold / l2Norm));
            for (let i = 0; i < outputBiasGrad.length; i++) {
                outputBiasGrad[i] *= scale;
            }
        }

        for (let layer = 0; layer < this.#numLayers; layer++) {
            ['Wq', 'Wk', 'Wv', 'Wo'].forEach(key => {
                const gradMatrix = this.#gradientAccumulation[idx].attentionWeights[layer][key];
                spectralNorm = this.#computeSpectralNorm(gradMatrix);
                if (spectralNorm > matrixThreshold) {
                    const scale = Math.max(minScaleFactor, Math.min(maxScaleFactor, matrixThreshold / spectralNorm));
                    for (let i = 0; i < gradMatrix.length; i++) {
                        for (let j = 0; j < gradMatrix[i].length; j++) {
                            gradMatrix[i][j] *= scale;
                        }
                    }
                }
            });

            ['W1', 'W2'].forEach(key => {
                const gradMatrix = this.#gradientAccumulation[idx].ffnWeights[layer][key];
                spectralNorm = this.#computeSpectralNorm(gradMatrix);
                if (spectralNorm > matrixThreshold) {
                    const scale = Math.max(minScaleFactor, Math.min(maxScaleFactor, matrixThreshold / spectralNorm));
                    for (let i = 0; i < gradMatrix.length; i++) {
                        for (let j = 0; j < gradMatrix[i].length; j++) {
                            gradMatrix[i][j] *= scale;
                        }
                    }
                }
            });

            ['b1', 'b2'].forEach(key => {
                const gradVector = this.#gradientAccumulation[idx].ffnWeights[layer][key];
                l2Norm = this.#computeGradientNorm(gradVector, false);
                if (l2Norm > vectorThreshold) {
                    const scale = Math.max(minScaleFactor, Math.min(maxScaleFactor, vectorThreshold / l2Norm));
                    for (let i = 0; i < gradVector.length; i++) {
                        gradVector[i] *= scale;
                    }
                }
            });

            ['gamma1', 'beta1', 'gamma2', 'beta2'].forEach(key => {
                const gradVector = this.#gradientAccumulation[idx].layerNormWeights[layer][key];
                l2Norm = this.#computeGradientNorm(gradVector, false);
                if (l2Norm > vectorThreshold) {
                    const scale = Math.max(minScaleFactor, Math.min(maxScaleFactor, vectorThreshold / l2Norm));
                    for (let i = 0; i < gradVector.length; i++) {
                        gradVector[i] *= scale;
                    }
                }
            });
        }

        l2Norm = this.#computeGradientNorm(attentionBiasGrad, false);
        if (l2Norm > vectorThreshold) {
            const scale = Math.max(minScaleFactor, Math.min(maxScaleFactor, vectorThreshold / l2Norm));
            for (let i = 0; i < attentionBiasGrad.length; i++) {
                attentionBiasGrad[i] *= scale;
            }
        }

        l2Norm = this.#computeGradientNorm(attentionWeightMatrixGrad, false);
        if (l2Norm > vectorThreshold) {
            const scale = Math.max(minScaleFactor, Math.min(maxScaleFactor, vectorThreshold / l2Norm));
            for (let i = 0; i < attentionWeightMatrixGrad.length; i++) {
                attentionWeightMatrixGrad[i] *= scale;
            }
        }
    }

    #computeGradientNorm(grad, isMatrix) {
        if (isMatrix) {
            return Math.sqrt(grad.reduce((sum, row) => 
                sum + row.reduce((s, val) => s + (isValidNumber(val) ? val * val : 0), 0), 0)) || 1;
        } else {
            return Math.sqrt(grad.reduce((sum, val) => 
                sum + (isValidNumber(val) ? val * val : 0), 0)) || 1;
        }
    }

    #computeWeightSimilarity(weights1, weights2) {
        let dotProduct = 0, norm1 = 0, norm2 = 0;
        for (let i = 0; i < weights1.length; i++) {
            for (let j = 0; j < weights1[i].length; j++) {
                if (isValidNumber(weights1[i][j]) && isValidNumber(weights2[i][j])) {
                    dotProduct += weights1[i][j] * weights2[i][j];
                    norm1 += weights1[i][j] * weights1[i][j];
                    norm2 += weights2[i][j] * weights2[i][j];
                }
            }
        }
        return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2) + 1e-6);
    }

    #gelu(x) {
        if (!isValidNumber(x)) return 0;
        return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));
    }

    #geluDerivative(x) {
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

    #layerNorm(x, gamma, beta, eps = 1e-6) {
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

    #contextAwareAttention(inputs, transformerIdx) {
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

        if (
            !Array.isArray(this.#attentionMemory[transformerIdx]) ||
            this.#attentionMemory[transformerIdx].length === 0 ||
            !this.#attentionMemory[transformerIdx].every(
                seq => Array.isArray(seq) && seq.length === this.#inputSize && seq.every(row => Array.isArray(row) && row.length === this.#hiddenSize && row.every(isValidNumber))
            )
        ) {
            this.#attentionMemory[transformerIdx] = Array(this.#contextWindow).fill().map(() =>
                Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0))
            );
        }

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

        const output = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
        for (let i = 0; i < this.#inputSize; i++) {
            for (let j = 0; j < this.#hiddenSize; j++) {
                output[i][j] = isValidNumber(inputProjection[i][j]) && isValidNumber(transformer.positionalEncoding[i][j])
                    ? inputProjection[i][j] + transformer.positionalEncoding[i][j]
                    : inputProjection[i][j];
            }
        }

        const contextWeight = 0.3;
        if (this.#attentionMemory[transformerIdx].length > 0) {
            const recentAttention = this.#attentionMemory[transformerIdx][this.#attentionMemory[transformerIdx].length - 1];
            for (let i = 0; i < this.#inputSize; i++) {
                for (let j = 0; j < this.#hiddenSize; j++) {
                    const historicalValue = isValidNumber(recentAttention[i][j]) ? recentAttention[i][j] : 0;
                    const specializationFactor = isValidNumber(this.#specializationWeights[transformerIdx][i % this.#hiddenSize][j])
                        ? Math.min(Math.max(1 + this.#specializationScores[transformerIdx] * this.#specializationWeights[transformerIdx][i % this.#hiddenSize][j], 0.5), 1.5)
                        : 1;
                    output[i][j] = isValidNumber(output[i][j])
                        ? (1 - contextWeight) * output[i][j] + contextWeight * historicalValue * specializationFactor
                        : historicalValue * specializationFactor;
                }
            }
        }

        for (let i = 0; i < this.#inputSize; i++) {
            output[i] = this.#layerNorm(
                output[i],
                transformer.layerNormWeights[0].gamma1,
                transformer.layerNormWeights[0].beta1
            );
        }

        return output;
    }

    #computeMemoryScore(memoryEntry, attentionScores, transformerIdx, entryIndex) {
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

        const memoryLength = this.#attentionMemory[transformerIdx].length;
        const baseRecencyWeight = 0.2;
        const recencyWeight = baseRecencyWeight / (1 + Math.log10(Math.max(memoryLength, 1))) * (1 - this.#contextSensitivity);
        const recencyFactor = Math.min(1 + recencyWeight / Math.max(memoryLength - entryIndex, 1), 1.5);

        return (dropoutScore * dropoutWeight + attentionScore * attentionWeight + magnitudeScore * magnitudeWeight) * recencyFactor;
    }

    #pruneMemory(transformerIdx, contextWindow, latestScores) {
        if (this.#attentionMemory[transformerIdx].length <= contextWindow) {
            return;
        }

        const scores = this.#attentionMemory[transformerIdx].map((entry, index) => ({
            index,
            score: this.#computeMemoryScore(entry, index === this.#attentionMemory[transformerIdx].length - 1 ? latestScores : null, transformerIdx, index)
        }));

        const minScoreEntry = scores.reduce((min, curr) => curr.score < min.score ? curr : min, scores[0]);

        this.#attentionMemory[transformerIdx].splice(minScoreEntry.index, 1);
    }

    #multiHeadAttention(x, layer, transformerIdx) {
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
                        ? sum / Math.sqrt(headSize) * this.#attentionScalingFactor * (1 + this.#specializationScores[transformerIdx])
                        : 0;
                }
                attentionProbs[h][i] = this.#softmax(attentionScores[h][i].map(score => isValidNumber(score) ? score : 0));
                attentionProbs[h][i] = this.#dropout(attentionProbs[h][i], this.#dropoutRate);
            }
        }

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

        const finalOutput = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
        for (let i = 0; i < this.#inputSize; i++) {
            for (let j = 0; j < this.#hiddenSize; j++) {
                for (let k = 0; k < this.#hiddenSize; k++) {
                    const specWeight = isValidNumber(this.#specializationWeights[transformerIdx][k % this.#hiddenSize][j])
                        ? Math.min(Math.max(1 + this.#specializationScores[transformerIdx] * this.#specializationWeights[transformerIdx][k % this.#hiddenSize][j], 0.5), 1.5)
                        : 1;
                    finalOutput[i][j] += isValidNumber(output[i][k]) && isValidNumber(layer.Wo[k][j])
                        ? output[i][k] * layer.Wo[k][j] * specWeight
                        : 0;
                }
            }
            finalOutput[i] = this.#dropout(finalOutput[i], this.#dropoutRate);
        }

        this.#attentionMemory[transformerIdx].push(finalOutput.map(row => row.slice()));

        this.#pruneMemory(transformerIdx, this.#contextWindow, attentionScores);

        return {
            output: finalOutput,
            Q,
            K,
            V,
            scores: attentionScores,
            probs: attentionProbs
        };
    }

    #feedForward(x, layer, transformerIdx) {
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
        activated = this.#dropout(activated, this.#dropoutRate);

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

        return this.#dropout(output, this.#dropoutRate);
    }

    #dropout(x, rate) {
        if (
            !Array.isArray(x) || 
            !x.every(isValidNumber) || 
            !isValidNumber(rate) || 
            rate < 0 || 
            rate >= 1
        ) {
            return x.slice();
        }

        return x.map(val => {
            if (!isValidNumber(val)) return 0;
            return Math.random() >= rate ? val / (1 - rate) : 0;
        });
    }

    #distillKnowledge(outputs, target) {
        if (
            !Array.isArray(outputs) ||
            outputs.length !== this.#ensembleSize ||
            !outputs.every(isValidNumber) ||
            !isValidNumber(target)
        ) {
            return;
        }

        const sortedIndices = this.#performanceScores
            .map((score, idx) => ({ score: isValidNumber(score) ? score : 0, idx }))
            .sort((a, b) => b.score - a.score);
        const topPerformers = sortedIndices.slice(0, Math.floor(this.#ensembleSize * 0.25)).map(({ idx }) => idx);

        const topOutputs = topPerformers.map(idx => outputs[idx]);
        const topWeights = topPerformers.map(idx => this.#ensembleWeights[idx]);
        const weightSum = topWeights.reduce((sum, w) => sum + (isValidNumber(w) ? w : 0), 0) || 1;
        const normalizedTopWeights = topWeights.map(w => (isValidNumber(w) ? w : 0) / weightSum);
        const targetOutput = topOutputs.reduce((sum, output, i) =>
            sum + (isValidNumber(output) && isValidNumber(normalizedTopWeights[i]) ? output * normalizedTopWeights[i] : 0), 0
        );

        const meanTop = targetOutput;
        const varianceTop = topOutputs.reduce((sum, out) => sum + Math.pow(out - meanTop, 2), 0) / topOutputs.length;
        const stdTopOutputs = Math.sqrt(varianceTop);

        const diversityLossBase = this.#computeGiniCoefficient(this.#transformers.map(t => t.outputWeights.flat()));

        this.#transformers.forEach((transformer, idx) => {
            if (topPerformers.includes(idx)) return;

            const rank = sortedIndices.findIndex(({ idx: i }) => i === idx);
            const diversityWeight = 0.5 + 0.5 * (rank / this.#ensembleSize);
            const diversityLoss = diversityLossBase * diversityWeight;

            const output = outputs[idx];
            const diversityScore = Math.min(1, Math.abs(output - meanTop) / (stdTopOutputs + 1e-6));
            const performanceGap = this.#performanceScores[topPerformers[0]] - this.#performanceScores[idx];
            const klWeight = Math.min(1.0, Math.max(0.1, performanceGap / this.#performanceScores[topPerformers[0]])) * (1 - diversityScore);

            const temperature = 2.0;
            const outputProb = this.#sigmoid(output / temperature);
            const targetProb = this.#sigmoid(targetOutput / temperature);
            const klLoss = isValidNumber(outputProb) && isValidNumber(targetProb)
                ? outputProb * Math.log((outputProb + 1e-6) / (targetProb + 1e-6)) + (1 - outputProb) * Math.log((1 - outputProb + 1e-6) / (1 - targetProb + 1e-6))
                : 0;

            let l2Reg = 0;
            for (let j = 0; j < this.#hiddenSize; j++) {
                if (isValidNumber(this.#attentionWeightMatrix[idx][j])) {
                    l2Reg += Math.pow(this.#attentionWeightMatrix[idx][j], 2);
                }
            }
            const l2Loss = 0.0005 * l2Reg;

            let specializationReg = 0;
            for (let i = 0; i < this.#hiddenSize; i++) {
                for (let j = 0; j < 1; j++) {
                    if (isValidNumber(this.#specializationWeights[idx][i][j])) {
                        specializationReg += Math.pow(this.#specializationWeights[idx][i][j], 2);
                    }
                }
            }
            const specializationLoss = 0.0001 * specializationReg;

            const totalLoss = 0.5 * klLoss * klWeight + diversityLoss + l2Loss + specializationLoss;

            const adjustedLearningRate = this.#adaptiveLearningRate[idx];
            const grad = isValidNumber(totalLoss) ? totalLoss : 0;
            const momentum = 0.9;

            for (let i = 0; i < this.#hiddenSize; i++) {
                for (let j = 0; j < 1; j++) {
                    const specializationFactor = isValidNumber(this.#specializationWeights[idx][i % this.#hiddenSize][j])
                        ? Math.min(Math.max(1 + this.#specializationScores[idx] * this.#specializationWeights[idx][i % this.#hiddenSize][j], 0.5), 1.5)
                        : 1;
                    const gradUpdate = grad * transformer.outputWeights[i][j] * specializationFactor;
                    const update = 0.2 * adjustedLearningRate * gradUpdate / Math.sqrt(1);
                    this.#gradientAccumulation[idx].outputWeights[i][j] = momentum * this.#gradientAccumulation[idx].outputWeights[i][j] + (1 - momentum) * update;
                    transformer.outputWeights[i][j] = isValidNumber(transformer.outputWeights[i][j])
                        ? transformer.outputWeights[i][j] - this.#gradientAccumulation[idx].outputWeights[i][j]
                        : 0;
                }
            }

            for (let i = 0; i < 1; i++) {
                const gradUpdate = grad;
                const update = 0.2 * adjustedLearningRate * gradUpdate;
                this.#gradientAccumulation[idx].outputBias[i] = momentum * this.#gradientAccumulation[idx].outputBias[i] + (1 - momentum) * update;
                transformer.outputBias[i] = isValidNumber(transformer.outputBias[i])
                    ? transformer.outputBias[i] - this.#gradientAccumulation[idx].outputBias[i]
                    : 0;
            }

            for (let layer = this.#numLayers - 1; layer >= 0; layer--) {
                if (
                    !Array.isArray(this.#attentionMemory[idx]) ||
                    this.#attentionMemory[idx].length === 0 ||
                    !this.#attentionMemory[idx].every(
                        seq => Array.isArray(seq) && seq.length === this.#inputSize && seq.every(row => Array.isArray(row) && row.length === this.#hiddenSize && row.every(isValidNumber))
                    )
                ) {
                    this.#attentionMemory[idx] = Array(this.#contextWindow).fill().map(() =>
                        Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0))
                    );
                }
                const attentionInput = this.#attentionMemory[idx][this.#attentionMemory[idx].length - 1];
                const headSize = this.#hiddenSize / this.#numHeads;

                const Q = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
                const K = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
                const V = Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0));
                for (let i = 0; i < this.#inputSize; i++) {
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        for (let k = 0; k < this.#hiddenSize; k++) {
                            const specWeight = isValidNumber(this.#specializationWeights[idx][k % this.#hiddenSize][j])
                                ? Math.min(Math.max(1 + this.#specializationScores[idx] * this.#specializationWeights[idx][k % this.#hiddenSize][j], 0.5), 1.5)
                                : 1;
                            Q[i][j] += isValidNumber(attentionInput[i][k]) && isValidNumber(transformer.attentionWeights[layer].Wq[k][j])
                                ? attentionInput[i][k] * transformer.attentionWeights[layer].Wq[k][j] * specWeight
                                : 0;
                            K[i][j] += isValidNumber(attentionInput[i][k]) && isValidNumber(transformer.attentionWeights[layer].Wk[k][j])
                                ? attentionInput[i][k] * transformer.attentionWeights[layer].Wk[k][j] * specWeight
                                : 0;
                            V[i][j] += isValidNumber(attentionInput[i][k]) && isValidNumber(transformer.attentionWeights[layer].Wv[k][j])
                                ? attentionInput[i][k] * transformer.attentionWeights[layer].Wv[k][j] * specWeight
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
                            attentionScores[h][i][j] = isValidNumber(sum)
                                ? sum / Math.sqrt(headSize) * (1 + this.#specializationScores[idx])
                                : 0;
                        }
                        attentionProbs[h][i] = this.#softmax(attentionScores[h][i].map(score => isValidNumber(score) ? score : 0));
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

                const attentionDiffThreshold = 0.1;
                const topPerformerAttention = topPerformers.map(idx => this.#transformers[idx].attentionWeights[layer]);
                for (let i = 0; i < this.#hiddenSize; i++) {
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        const specializationFactor = isValidNumber(this.#specializationWeights[idx][i % this.#hiddenSize][j])
                            ? Math.min(Math.max(1 + this.#specializationScores[idx] * this.#specializationWeights[idx][i % this.#hiddenSize][j], 0.5), 1.5)
                            : 1;
                        const wqUpdate = qGrad.reduce((sum, row, idx) => sum + (isValidNumber(row[j]) && isValidNumber(attentionInput[idx % this.#inputSize][i % this.#hiddenSize]) ? row[j] * attentionInput[idx % this.#inputSize][i % this.#hiddenSize] : 0), 0) / Math.sqrt(headSize);
                        const wkUpdate = kGrad.reduce((sum, row, idx) => sum + (isValidNumber(row[i]) && isValidNumber(attentionInput[idx % this.#inputSize][i % this.#hiddenSize]) ? row[i] * attentionInput[idx % this.#inputSize][i % this.#hiddenSize] : 0), 0) / Math.sqrt(headSize);
                        const wvUpdate = vGrad.reduce((sum, head, h) => sum + (isValidNumber(head[i % this.#inputSize][j % headSize]) && isValidNumber(attentionInput[i % this.#inputSize][j % headSize]) ? head[i % this.#inputSize][j % headSize] * attentionInput[i % this.#inputSize][j % headSize] : 0), 0) / (this.#numHeads * Math.sqrt(headSize));
                        const woUpdate = woGrad[i][j] / (this.#numHeads * this.#inputSize);

                        const avgAttentionDiffWq = topPerformerAttention.reduce((sum, weights) => {
                            return sum + Math.abs(transformer.attentionWeights[layer].Wq[i][j] - weights.Wq[i][j]);
                        }, 0) / topPerformers.length;
                        if (isValidNumber(wqUpdate) && avgAttentionDiffWq > attentionDiffThreshold) {
                            const update = adjustedLearningRate * wqUpdate * specializationFactor;
                            this.#gradientAccumulation[idx].attentionWeights[layer].Wq[i][j] = momentum * this.#gradientAccumulation[idx].attentionWeights[layer].Wq[i][j] + (1 - momentum) * update;
                            transformer.attentionWeights[layer].Wq[i][j] = isValidNumber(transformer.attentionWeights[layer].Wq[i][j])
                                ? transformer.attentionWeights[layer].Wq[i][j] - this.#gradientAccumulation[idx].attentionWeights[layer].Wq[i][j]
                                : 0;
                        }
                        const avgAttentionDiffWk = topPerformerAttention.reduce((sum, weights) => {
                            return sum + Math.abs(transformer.attentionWeights[layer].Wk[i][j] - weights.Wk[i][j]);
                        }, 0) / topPerformers.length;
                        if (isValidNumber(wkUpdate) && avgAttentionDiffWk > attentionDiffThreshold) {
                            const update = adjustedLearningRate * wkUpdate * specializationFactor;
                            this.#gradientAccumulation[idx].attentionWeights[layer].Wk[i][j] = momentum * this.#gradientAccumulation[idx].attentionWeights[layer].Wk[i][j] + (1 - momentum) * update;
                            transformer.attentionWeights[layer].Wk[i][j] = isValidNumber(transformer.attentionWeights[layer].Wk[i][j])
                                ? transformer.attentionWeights[layer].Wk[i][j] - this.#gradientAccumulation[idx].attentionWeights[layer].Wk[i][j]
                                : 0;
                        }
                        const avgAttentionDiffWv = topPerformerAttention.reduce((sum, weights) => {
                            return sum + Math.abs(transformer.attentionWeights[layer].Wv[i][j] - weights.Wv[i][j]);
                        }, 0) / topPerformers.length;
                        if (isValidNumber(wvUpdate) && avgAttentionDiffWv > attentionDiffThreshold) {
                            const update = adjustedLearningRate * wvUpdate * specializationFactor;
                            this.#gradientAccumulation[idx].attentionWeights[layer].Wv[i][j] = momentum * this.#gradientAccumulation[idx].attentionWeights[layer].Wv[i][j] + (1 - momentum) * update;
                            transformer.attentionWeights[layer].Wv[i][j] = isValidNumber(transformer.attentionWeights[layer].Wv[i][j])
                                ? transformer.attentionWeights[layer].Wv[i][j] - this.#gradientAccumulation[idx].attentionWeights[layer].Wv[i][j]
                                : 0;
                        }
                        const avgAttentionDiffWo = topPerformerAttention.reduce((sum, weights) => {
                            return sum + Math.abs(transformer.attentionWeights[layer].Wo[i][j] - weights.Wo[i][j]);
                        }, 0) / topPerformers.length;
                        if (isValidNumber(woUpdate) && avgAttentionDiffWo > attentionDiffThreshold) {
                            const update = adjustedLearningRate * woUpdate * specializationFactor;
                            this.#gradientAccumulation[idx].attentionWeights[layer].Wo[i][j] = momentum * this.#gradientAccumulation[idx].attentionWeights[layer].Wo[i][j] + (1 - momentum) * update;
                            transformer.attentionWeights[layer].Wo[i][j] = isValidNumber(transformer.attentionWeights[layer].Wo[i][j])
                                ? transformer.attentionWeights[layer].Wo[i][j] - this.#gradientAccumulation[idx].attentionWeights[layer].Wo[i][j]
                                : 0;
                        }
                    }
                }

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
                        const specializationFactor = isValidNumber(this.#specializationWeights[idx][i % this.#hiddenSize][j % this.#hiddenSize])
                            ? Math.min(Math.max(1 + this.#specializationScores[idx] * this.#specializationWeights[idx][i % this.#hiddenSize][j % this.#hiddenSize], 0.5), 1.5)
                            : 1;
                        const update = adjustedLearningRate * ffnGrad[j] * ffnInput[i] * specializationFactor / Math.sqrt(this.#feedForwardSize);
                        this.#gradientAccumulation[idx].ffnWeights[layer].W1[i][j] = momentum * this.#gradientAccumulation[idx].ffnWeights[layer].W1[i][j] + (1 - momentum) * update;
                        transformer.ffnWeights[layer].W1[i][j] = isValidNumber(transformer.ffnWeights[layer].W1[i][j])
                            ? transformer.ffnWeights[layer].W1[i][j] - this.#gradientAccumulation[idx].ffnWeights[layer].W1[i][j]
                            : 0;
                    }
                }
                for (let i = 0; i < this.#feedForwardSize; i++) {
                    const update = adjustedLearningRate * ffnGrad[i];
                    this.#gradientAccumulation[idx].ffnWeights[layer].b1[i] = momentum * this.#gradientAccumulation[idx].ffnWeights[layer].b1[i] + (1 - momentum) * update;
                    transformer.ffnWeights[layer].b1[i] = isValidNumber(transformer.ffnWeights[layer].b1[i])
                        ? transformer.ffnWeights[layer].b1[i] - this.#gradientAccumulation[idx].ffnWeights[layer].b1[i]
                        : 0;
                }
                for (let i = 0; i < this.#feedForwardSize; i++) {
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        const specializationFactor = isValidNumber(this.#specializationWeights[idx][j % this.#hiddenSize][i % this.#hiddenSize])
                            ? Math.min(Math.max(1 + this.#specializationScores[idx] * this.#specializationWeights[idx][j % this.#hiddenSize][i % this.#hiddenSize], 0.5), 1.5)
                            : 1;
                        const update = adjustedLearningRate * grad * activated[i] * specializationFactor / Math.sqrt(this.#hiddenSize);
                        this.#gradientAccumulation[idx].ffnWeights[layer].W2[i][j] = momentum * this.#gradientAccumulation[idx].ffnWeights[layer].W2[i][j] + (1 - momentum) * update;
                        transformer.ffnWeights[layer].W2[i][j] = isValidNumber(transformer.ffnWeights[layer].W2[i][j])
                            ? transformer.ffnWeights[layer].W2[i][j] - this.#gradientAccumulation[idx].ffnWeights[layer].W2[i][j]
                            : 0;
                    }
                }
                for (let i = 0; i < this.#hiddenSize; i++) {
                    const update = adjustedLearningRate * grad;
                    this.#gradientAccumulation[idx].ffnWeights[layer].b2[i] = momentum * this.#gradientAccumulation[idx].ffnWeights[layer].b2[i] + (1 - momentum) * update;
                    transformer.ffnWeights[layer].b2[i] = isValidNumber(transformer.ffnWeights[layer].b2[i])
                        ? transformer.ffnWeights[layer].b2[i] - this.#gradientAccumulation[idx].ffnWeights[layer].b2[i]
                        : 0;
                }

                const normInput = attentionInput;
                for (let i = 0; i < this.#inputSize; i++) {
                    const gamma1Grad = Array(this.#hiddenSize).fill(0);
                    const beta1Grad = Array(this.#hiddenSize).fill(0);
                    const gamma2Grad = Array(this.#hiddenSize).fill(0);
                    const beta2Grad = Array(this.#hiddenSize).fill(0);
                    const meanNorm = normInput[i].reduce((sum, val) => sum + val, 0) / this.#hiddenSize;
                    const varianceNorm = normInput[i].reduce((sum, val) => sum + Math.pow(val - meanNorm, 2), 0) / this.#hiddenSize;
                    const stdNorm = Math.sqrt(varianceNorm + 1e-6);
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        const normalized = (normInput[i][j] - meanNorm) / stdNorm;
                        gamma1Grad[j] = isValidNumber(qGrad[i][j]) && isValidNumber(normalized)
                            ? qGrad[i][j] * normalized
                            : 0;
                        beta1Grad[j] = isValidNumber(qGrad[i][j]) ? qGrad[i][j] : 0;
                        gamma2Grad[j] = isValidNumber(grad) && isValidNumber(normalized)
                            ? grad * normalized
                            : 0;
                        beta2Grad[j] = isValidNumber(grad) ? grad : 0;
                        const updateGamma1 = adjustedLearningRate * gamma1Grad[j];
                        const updateBeta1 = adjustedLearningRate * beta1Grad[j];
                        const updateGamma2 = adjustedLearningRate * gamma2Grad[j];
                        const updateBeta2 = adjustedLearningRate * beta2Grad[j];
                        this.#gradientAccumulation[idx].layerNormWeights[layer].gamma1[j] = momentum * this.#gradientAccumulation[idx].layerNormWeights[layer].gamma1[j] + (1 - momentum) * updateGamma1;
                        transformer.layerNormWeights[layer].gamma1[j] = isValidNumber(transformer.layerNormWeights[layer].gamma1[j])
                            ? transformer.layerNormWeights[layer].gamma1[j] - this.#gradientAccumulation[idx].layerNormWeights[layer].gamma1[j]
                            : 1;
                        this.#gradientAccumulation[idx].layerNormWeights[layer].beta1[j] = momentum * this.#gradientAccumulation[idx].layerNormWeights[layer].beta1[j] + (1 - momentum) * updateBeta1;
                        transformer.layerNormWeights[layer].beta1[j] = isValidNumber(transformer.layerNormWeights[layer].beta1[j])
                            ? transformer.layerNormWeights[layer].beta1[j] - this.#gradientAccumulation[idx].layerNormWeights[layer].beta1[j]
                            : 0;
                        this.#gradientAccumulation[idx].layerNormWeights[layer].gamma2[j] = momentum * this.#gradientAccumulation[idx].layerNormWeights[layer].gamma2[j] + (1 - momentum) * updateGamma2;
                        transformer.layerNormWeights[layer].gamma2[j] = isValidNumber(transformer.layerNormWeights[layer].gamma2[j])
                            ? transformer.layerNormWeights[layer].gamma2[j] - this.#gradientAccumulation[idx].layerNormWeights[layer].gamma2[j]
                            : 1;
                        this.#gradientAccumulation[idx].layerNormWeights[layer].beta2[j] = momentum * this.#gradientAccumulation[idx].layerNormWeights[layer].beta2[j] + (1 - momentum) * updateBeta2;
                        transformer.layerNormWeights[layer].beta2[j] = isValidNumber(transformer.layerNormWeights[layer].beta2[j])
                            ? transformer.layerNormWeights[layer].beta2[j] - this.#gradientAccumulation[idx].layerNormWeights[layer].beta2[j]
                            : 0;
                    }
                }
            }
        });
    }

    #applyMatrixScaling (matrix, baseThreshold, diversityLoss) {
        const threshold = baseThreshold * (1 + diversityLoss);
        const spectralNorm = this.#computeSpectralNorm(matrix);
        if (spectralNorm > threshold) {
            const scale = threshold / spectralNorm;
            for (let i = 0; i < matrix.length; i++) {
                for (let j = 0; j < matrix[i].length; j++) {
                    if (isValidNumber(matrix[i][j])) {
                        matrix[i][j] *= scale;
                    } else {
                        matrix[i][j] = 0;
                    }
                }
            }
        }
    }

    #applyVectorScaling (vector, baseThreshold, diversityLoss) {
        const threshold = baseThreshold * (1 + diversityLoss);
        const l2Norm = Math.sqrt(vector.reduce((sum, val) => sum + (isValidNumber(val) ? val * val : 0), 0));
        if (l2Norm > threshold) {
            const scale = threshold / l2Norm;
            for (let i = 0; i < vector.length; i++) {
                vector[i] = isValidNumber(vector[i]) ? vector[i] * scale : 0;
            }
        }
    }

    #computeGradientStats (grad, isMatrix) {
        const flat_grad_abs = isMatrix
            ? grad.map(row => row.map(Math.abs)).flat()
            : grad.map(Math.abs);
        const sum = flat_grad_abs.reduce((a, b) => a + b, 0);
        const mean = sum / flat_grad_abs.length;
        const sum_sq_diff = flat_grad_abs.reduce((a, b) => a + (b - mean) ** 2, 0);
        const variance = sum_sq_diff / flat_grad_abs.length;
        const std = Math.sqrt(variance);
        return { mean, std };
    }

    #computeHeadSimilarities (attentionWeightsLayer, numHeads, hiddenSize, includeWeights = ['Wq', 'Wk', 'Wv']) {
        const headSize = hiddenSize / numHeads;
        let totalSimilarities = Array(numHeads).fill(0);
        let weightCount = 0;

        for (const key of includeWeights) {
            if (!attentionWeightsLayer[key]) continue;
            const W = attentionWeightsLayer[key];
            const heads = Array(numHeads).fill().map((_, i) => {
                const startRow = i * headSize;
                const headRows = W.slice(startRow, startRow + headSize);
                return headRows.flat();
            });

            const layerSimilarities = Array(numHeads).fill(0);
            for (let i = 0; i < numHeads; i++) {
                let totalSimilarity = 0;
                let count = 0;
                for (let j = 0; j < numHeads; j++) {
                    if (i === j) continue;
                    const sim = this.#computeWeightSimilarity([heads[i]], [heads[j]]);
                    totalSimilarity += sim;
                    count++;
                }
                layerSimilarities[i] = count > 0 ? totalSimilarity / count : 0;
            }
            totalSimilarities = totalSimilarities.map((sum, i) => sum + layerSimilarities[i]);
            weightCount++;
        }

        return totalSimilarities.map(sim => weightCount > 0 ? sim / weightCount : 0);
    }

    #applyOrthogonalization (matrix, otherMatrix, penalty) {
        const flatMatrix = matrix.flat();
        const flatOther = otherMatrix.flat();
        const dotProduct = flatMatrix.reduce((sum, a, i) => sum + a * flatOther[i], 0);
        const normSquared = flatOther.reduce((sum, a) => sum + a * a, 0);
        if (normSquared < 1e-10) return;
        const scale = (dotProduct / normSquared) * penalty;
        for (let i = 0; i < matrix.length; i++) {
            for (let j = 0; j < matrix[i].length; j++) {
                if (isValidNumber(matrix[i][j]) && isValidNumber(otherMatrix[i][j])) {
                    matrix[i][j] -= scale * otherMatrix[i][j];
                }
            }
        }
    }

    #computeGiniCoefficient (weightsList) {
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
        gini = sum > 0 ? gini / (2 * n * sum) : 0;
        return 1 - gini;
    }

    #regulateWeightsAndMemory() {
        let diversityPenalty = 0.005;
        let similarityThreshold = 0.3;
        const alpha = 4;
        const beta = 0.5;

        const minWeightThreshold = 1e-6;
        const smallWeightThreshold = 1e-4;
        const noiseScale = 0.01;

        const maxDecay = 1.2;
        const minDecay = 0.5;
        const decayRate = 0.3;
        const layerDecaySchedule = Array.from({ length: this.#numLayers }, (_, i) =>
            minDecay + (maxDecay - minDecay) * Math.exp(-decayRate * i)
        );

        const baseThresholds = {
            'outputWeights': 4.0,
            'attentionWeights': 2.5,
            'ffnWeights': 3.0,
            'biases': 2.0,
            'layerNormWeights': 1.5,
            'attentionWeightMatrix': 2.5
        };

        const decayFactors = {
            'Wq': 0.8, 'Wk': 0.8, 'Wv': 0.8, 'Wo': 0.8,
            'W1': 1.2, 'W2': 1.2, 'b1': 0.5, 'b2': 0.5,
            'outputWeights': 1.0, 'outputBias': 0.5,
            'gamma1': 0.5, 'beta1': 0.5, 'gamma2': 0.5, 'beta2': 0.5,
            'attentionBias': 0.5, 'attentionWeightMatrix': 0.8,
            'specializationWeights': 0.6
        };

        const diversityLoss = this.#computeGiniCoefficient(this.#transformers.map(t => t.outputWeights.flat()));
        const avgPerformance = this.#performanceScores.reduce((sum, score) => sum + (isValidNumber(score) ? score : 0.5), 0) / this.#ensembleSize;
        
        diversityPenalty = 0.005 * (1 + diversityLoss);
        similarityThreshold = 0.3 * (1 - 0.5 * avgPerformance);

        for (let t = 0; t < this.#ensembleSize; t++) {
            if (
                !Array.isArray(this.#attentionMemory[t]) ||
                this.#attentionMemory[t].length !== this.#contextWindow ||
                !this.#attentionMemory[t].every(
                    seq => Array.isArray(seq) && seq.length === this.#inputSize && seq.every(row => Array.isArray(row) && row.length === this.#hiddenSize && row.every(isValidNumber))
                )
            ) {
                this.#attentionMemory[t] = Array(this.#contextWindow).fill().map(() =>
                    Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0))
                );
            }

            let varianceSum = 0;
            let count = 0;
            for (let i = 0; i < this.#contextWindow; i++) {
                for (let j = 0; j < this.#inputSize; j++) {
                    const mean = this.#attentionMemory[t][i][j].reduce((sum, x) => sum + (isValidNumber(x) ? x : 0), 0) / this.#hiddenSize || 0;
                    varianceSum += this.#attentionMemory[t][i][j].reduce((sum, x) => sum + (isValidNumber(x) ? Math.pow(x - mean, 2) : 0), 0);
                    count += this.#hiddenSize;
                }
            }
            const memoryVariance = count > 0 ? varianceSum / count : 1;
            const dynamicDecay = Math.min(0.95, Math.max(0.85, 1 - (isValidNumber(this.#swarmIntelligenceFactor) ? this.#swarmIntelligenceFactor : 0.1) * Math.sqrt(memoryVariance) * (1 + 0.5 * (1 - avgPerformance))));
            const dynamicCutoff = Math.max(1e-6, 1e-2 * memoryVariance);
            const performanceFactor = isValidNumber(this.#performanceScores[t]) ? this.#performanceScores[t] : 0.5;
            const specializationFactor = isValidNumber(this.#specializationScores[t]) ? 1 + this.#specializationScores[t] * (isValidNumber(this.#swarmIntelligenceFactor) ? this.#swarmIntelligenceFactor : 0.1) : 1;
            const adaptiveScale = Math.min(1.5, Math.max(0.5, performanceFactor * specializationFactor));

            for (let i = 0; i < this.#contextWindow; i++) {
                const sequenceMatrix = this.#attentionMemory[t][i];
                const spectralNorm = this.#computeSpectralNorm(sequenceMatrix);
                const normScale = spectralNorm > 1 ? 1 / spectralNorm : 1;

                for (let j = 0; j < this.#inputSize; j++) {
                    for (let k = 0; k < this.#hiddenSize; k++) {
                        if (isValidNumber(this.#attentionMemory[t][i][j][k]) && this.#attentionMemory[t][i][j][k] !== 0) {
                            let value = this.#attentionMemory[t][i][j][k] * normScale * adaptiveScale;
                            const prevValue = i > 0 ? this.#attentionMemory[t][i-1][j][k] : 0;
                            value = dynamicDecay * value + (1 - dynamicDecay) * prevValue;

                            if (Math.abs(value) > dynamicCutoff) {
                                this.#attentionMemory[t][i][j][k] = value + (Math.random() - 0.5) * noiseScale;
                            }
                        } else {
                            this.#attentionMemory[t][i][j][k] = 0;
                        }
                    }
                }
            }
        }

        for (let idx = 0; idx < this.#ensembleSize; idx++) {
            const transformer = this.#transformers[idx];
            const performanceFactor = isValidNumber(this.#performanceScores[idx]) ? this.#performanceScores[idx] : 0.5;
            const specializationFactor = isValidNumber(this.#specializationScores[idx]) ? this.#specializationScores[idx] : 0.5;
            const adaptiveDecayRate = this.#weightDecayRate * Math.exp(-2 * performanceFactor);

            const applyWeightDecay = (matrixOrVector, isMatrix = true, weightType = 'generic', layer = null) => {
                let lambda = adaptiveDecayRate;
                if (layer !== null && layer < layerDecaySchedule.length) {
                    lambda *= layerDecaySchedule[layer];
                }
                lambda *= decayFactors[weightType] || 1.0;

                const grad = this.#gradientAccumulation[idx][weightType];
                if (!grad) {
                    const decay = lambda;
                    if (isMatrix) {
                        for (let i = 0; i < matrixOrVector.length; i++) {
                            for (let j = 0; j < matrixOrVector[i].length; j++) {
                                if (isValidNumber(matrixOrVector[i][j])) {
                                    if (Math.abs(matrixOrVector[i][j]) > smallWeightThreshold) {
                                        matrixOrVector[i][j] *= (1 - decay);
                                    }
                                    if (Math.abs(matrixOrVector[i][j]) < minWeightThreshold) {
                                        matrixOrVector[i][j] = (Math.random() - 0.5) * noiseScale;
                                    }
                                } else {
                                    matrixOrVector[i][j] = 0;
                                }
                            }
                        }
                    } else {
                        for (let i = 0; i < matrixOrVector.length; i++) {
                            if (isValidNumber(matrixOrVector[i])) {
                                if (Math.abs(matrixOrVector[i]) > smallWeightThreshold) {
                                    matrixOrVector[i] *= (1 - decay);
                                }
                                if (Math.abs(matrixOrVector[i]) < minWeightThreshold) {
                                    matrixOrVector[i] = (Math.random() - 0.5) * noiseScale;
                                }
                            } else {
                                matrixOrVector[i] = 0;
                            }
                        }
                    }
                    return;
                }

                const { mean, std } = this.#computeGradientStats(grad, isMatrix);

                if (isMatrix && ['Wq', 'Wk', 'Wv'].includes(weightType) && layer !== null) {
                    const headSimilarities = this.#computeHeadSimilarities(
                        transformer.attentionWeights[layer],
                        this.#numHeads,
                        this.#hiddenSize,
                        ['Wq', 'Wk', 'Wv']
                    );
                    const headSize = this.#hiddenSize / this.#numHeads;

                    for (let i = 0; i < matrixOrVector.length; i++) {
                        const headIdx = Math.floor(i / headSize);
                        let adjustedLambda = lambda * (1 + beta * headSimilarities[headIdx]);
                        if (headSimilarities[headIdx] > similarityThreshold) {
                            adjustedLambda *= (1 + diversityPenalty * (headSimilarities[headIdx] - similarityThreshold));
                        }
                        for (let j = 0; j < matrixOrVector[i].length; j++) {
                            if (isValidNumber(matrixOrVector[i][j]) && isValidNumber(grad[i][j])) {
                                const abs_g = Math.abs(grad[i][j]);
                                const normalized_g = std > 0 ? (abs_g - mean) / std : 0;
                                const theta = 1 / (1 + Math.exp(-alpha * normalized_g));
                                const decay = adjustedLambda * theta;
                                if (Math.abs(matrixOrVector[i][j]) > smallWeightThreshold) {
                                    matrixOrVector[i][j] *= (1 - decay);
                                }
                                if (Math.abs(matrixOrVector[i][j]) < minWeightThreshold) {
                                    matrixOrVector[i][j] = (Math.random() - 0.5) * noiseScale;
                                }
                            } else {
                                matrixOrVector[i][j] = 0;
                            }
                        }
                    }
                } else if (isMatrix) {
                    for (let i = 0; i < matrixOrVector.length; i++) {
                        for (let j = 0; j < matrixOrVector[i].length; j++) {
                            if (isValidNumber(matrixOrVector[i][j]) && isValidNumber(grad[i][j])) {
                                const abs_g = Math.abs(grad[i][j]);
                                const normalized_g = std > 0 ? (abs_g - mean) / std : 0;
                                const theta = 1 / (1 + Math.exp(-alpha * normalized_g));
                                const decay = lambda * theta;
                                if (Math.abs(matrixOrVector[i][j]) > smallWeightThreshold) {
                                    matrixOrVector[i][j] *= (1 - decay);
                                }
                                if (Math.abs(matrixOrVector[i][j]) < minWeightThreshold) {
                                    matrixOrVector[i][j] = (Math.random() - 0.5) * noiseScale;
                                }
                            } else {
                                matrixOrVector[i][j] = 0;
                            }
                        }
                    }
                } else {
                    for (let i = 0; i < matrixOrVector.length; i++) {
                        if (isValidNumber(matrixOrVector[i]) && isValidNumber(grad[i])) {
                            const abs_g = Math.abs(grad[i]);
                            const normalized_g = std > 0 ? (abs_g - mean) / std : 0;
                            const theta = 1 / (1 + Math.exp(-alpha * normalized_g));
                            const decay = lambda * theta;
                            if (Math.abs(matrixOrVector[i]) > smallWeightThreshold) {
                                matrixOrVector[i] *= (1 - decay);
                            }
                            if (Math.abs(matrixOrVector[i]) < minWeightThreshold) {
                                matrixOrVector[i] = (Math.random() - 0.5) * noiseScale;
                            }
                        } else {
                            matrixOrVector[i] = 0;
                        }
                    }
                }
            };

            let norm = Math.sqrt(this.#attentionWeightMatrix[idx].reduce((sum, val) => 
                sum + (isValidNumber(val) ? val * val : 0), 0)) || 1;
            for (let j = 0; j < this.#hiddenSize; j++) {
                this.#attentionWeightMatrix[idx][j] = isValidNumber(this.#attentionWeightMatrix[idx][j]) 
                    ? this.#attentionWeightMatrix[idx][j] / norm 
                    : 0;
            }

            this.#applyMatrixScaling(transformer.outputWeights, baseThresholds['outputWeights'], diversityLoss);
            applyWeightDecay(transformer.outputWeights, true, 'outputWeights');

            this.#applyVectorScaling(transformer.outputBias, baseThresholds['biases'], diversityLoss);
            applyWeightDecay(transformer.outputBias, false, 'outputBias');

            for (let layer = 0; layer < this.#numLayers; layer++) {
                ['Wq', 'Wk', 'Wv', 'Wo'].forEach(key => {
                    const weightMatrix = transformer.attentionWeights[layer][key];
                    this.#applyMatrixScaling(weightMatrix, baseThresholds['attentionWeights'], diversityLoss);
                    applyWeightDecay(weightMatrix, true, key, layer);
                });

                ['W1', 'W2'].forEach(key => {
                    const weightMatrix = transformer.ffnWeights[layer][key];
                    this.#applyMatrixScaling(weightMatrix, baseThresholds['ffnWeights'], diversityLoss);
                    applyWeightDecay(weightMatrix, true, key, layer);
                });

                ['b1', 'b2'].forEach(key => {
                    const biasVector = transformer.ffnWeights[layer][key];
                    this.#applyVectorScaling(biasVector, baseThresholds['biases'], diversityLoss);
                    applyWeightDecay(biasVector, false, key, layer);
                });

                ['gamma1', 'beta1', 'gamma2', 'beta2'].forEach(key => {
                    const normVector = transformer.layerNormWeights[layer][key];
                    this.#applyVectorScaling(normVector, baseThresholds['layerNormWeights'], diversityLoss);
                    applyWeightDecay(normVector, false, key, layer);
                });
            }

            this.#applyVectorScaling(this.#attentionBias[idx], baseThresholds['biases'], diversityLoss);
            applyWeightDecay(this.#attentionBias[idx], false, 'attentionBias');

            this.#applyVectorScaling(this.#attentionWeightMatrix[idx], baseThresholds['attentionWeightMatrix'], diversityLoss);
            applyWeightDecay(this.#attentionWeightMatrix[idx], false, 'attentionWeightMatrix');

            const specDecayRate = this.#weightDecayRate * Math.exp(-2 * specializationFactor);
            for (let i = 0; i < this.#hiddenSize; i++) {
                for (let j = 0; j < this.#hiddenSize; j++) {
                    if (isValidNumber(this.#specializationWeights[idx][i][j])) {
                        const weight = this.#specializationWeights[idx][i][j];
                        if (Math.abs(weight) > smallWeightThreshold) {
                            this.#specializationWeights[idx][i][j] *= (1 - specDecayRate);
                        }
                        if (Math.abs(weight) < minWeightThreshold) {
                            this.#specializationWeights[idx][i][j] = (Math.random() - 0.5) * noiseScale;
                        }
                    } else {
                        this.#specializationWeights[idx][i][j] = 0;
                    }
                }
            }

            for (let otherIdx = 0; otherIdx < this.#ensembleSize; otherIdx++) {
                if (otherIdx === idx) continue;

                const similarityOutput = this.#computeWeightSimilarity(
                    transformer.outputWeights,
                    this.#transformers[otherIdx].outputWeights
                );
                if (similarityOutput > similarityThreshold) {
                    const penalty = diversityPenalty * (similarityOutput - similarityThreshold);
                    this.#applyOrthogonalization(
                        transformer.outputWeights,
                        this.#transformers[otherIdx].outputWeights,
                        penalty
                    );
                }

                for (let layer = 0; layer < this.#numLayers; layer++) {
                    ['Wq', 'Wk', 'Wv', 'Wo'].forEach(key => {
                        const weightMatrix = transformer.attentionWeights[layer][key];
                        const otherWeightMatrix = this.#transformers[otherIdx].attentionWeights[layer][key];
                        const similarity = this.#computeWeightSimilarity(weightMatrix, otherWeightMatrix);
                        if (similarity > similarityThreshold) {
                            const penalty = diversityPenalty * (similarity - similarityThreshold);
                            this.#applyOrthogonalization(weightMatrix, otherWeightMatrix, penalty);
                        }
                    });

                    ['W1', 'W2'].forEach(key => {
                        const weightMatrix = transformer.ffnWeights[layer][key];
                        const otherWeightMatrix = this.#transformers[otherIdx].ffnWeights[layer][key];
                        const similarity = this.#computeWeightSimilarity(weightMatrix, otherWeightMatrix);
                        if (similarity > similarityThreshold) {
                            const penalty = diversityPenalty * (similarity - similarityThreshold);
                            for (let i = 0; i < weightMatrix.length; i++) {
                                for (let j = 0; j < weightMatrix[i].length; j++) {
                                    if (isValidNumber(weightMatrix[i][j])) {
                                        weightMatrix[i][j] -= penalty * weightMatrix[i][j];
                                    }
                                }
                            }
                        }
                    });
                }
            }
        }
    }

    #processTransformer(inputs, idx, computeIntermediates = false) {
        const transformer = this.#transformers[idx];

        if (
            !Array.isArray(this.#attentionMemory[idx]) ||
            this.#attentionMemory[idx].length === 0 ||
            !this.#attentionMemory[idx][0].every(row => Array.isArray(row) && row.length === this.#hiddenSize)
        ) {
            this.#attentionMemory[idx] = Array(this.#contextWindow).fill().map(() =>
                Array(this.#inputSize).fill().map(() => Array(this.#hiddenSize).fill(0))
            );
        }

        let x = this.#contextAwareAttention(inputs, idx);
        const layerOutputs = computeIntermediates ? [x] : [];
        const activations = computeIntermediates ? [] : [];
        const attentionIntermediates = computeIntermediates ? [] : [];

        for (let layer = 0; layer < this.#numLayers; layer++) {
            const normX = x.map(row => this.#layerNorm(row, transformer.layerNormWeights[layer].gamma1, transformer.layerNormWeights[layer].beta1));
            const attentionResult = this.#multiHeadAttention(normX, transformer.attentionWeights[layer], idx);
            const attentionOutput = attentionResult.output;
            const attentionResidual = x.map((row, i) => row.map((val, j) =>
                isValidNumber(val) && isValidNumber(attentionOutput[i][j]) ? val + attentionOutput[i][j] : val
            ));
            const normAttention = attentionResidual.map(row => this.#layerNorm(row, transformer.layerNormWeights[layer].gamma2, transformer.layerNormWeights[layer].beta2));
            const ffnOutputs = normAttention.map(row => this.#feedForward(row, transformer.ffnWeights[layer], idx));
            x = attentionResidual.map((row, i) => row.map((val, j) =>
                isValidNumber(val) && isValidNumber(ffnOutputs[i][j]) ? val + ffnOutputs[i][j] : val
            ));

            if (computeIntermediates) {
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
        }

        let output = Array(1).fill(0);
        for (let i = 0; i < this.#hiddenSize; i++) {
            for (let j = 0; j < 1; j++) {
                output[j] += isValidNumber(x[0][i]) && isValidNumber(transformer.outputWeights[i][j])
                    ? x[0][i] * transformer.outputWeights[i][j]
                    : 0;
            }
        }
        output = output.map((val, i) => isValidNumber(val) && isValidNumber(transformer.outputBias[i])
            ? val + transformer.outputBias[i]
            : val
        );

        if (computeIntermediates) {
            return { output: output[0], layerOutputs, activations, attentionIntermediates };
        }
        return output[0];
    }

    #computeWeightedSum(outputs, weights) {
        return outputs.reduce((sum, out, idx) => {
            if (isValidNumber(out) && isValidNumber(weights[idx])) {
                return sum + out * weights[idx];
            }
            return sum;
        }, 0);
    }

    predict(inputs) {
        if ( !Array.isArray(inputs) || inputs.length !== this.#inputSize || !inputs.every(isValidNumber) ) { return 0 }

        const outputs = this.#transformers.map((_, idx) => this.#processTransformer(inputs, idx));

        this.#ensembleWeights = this.#computeAttentionWeights(inputs, outputs);
        this.#normalizeEnsembleWeights();

        const finalOutput = this.#computeWeightedSum(outputs, this.#ensembleWeights);

        return this.#sigmoid(isValidNumber(finalOutput) ? finalOutput : 0);
    }

    train(inputs, target) {
        if ( !Array.isArray(inputs) || inputs.length !== this.#inputSize || !inputs.every(isValidNumber) || !isValidNumber(target) ) { return }

        const linearOutputs = [];
        const layerOutputs = this.#transformers.map(() => []);
        const activations = this.#transformers.map(() => []);
        const attentionIntermediates = this.#transformers.map(() => []);

        this.#transformers.forEach((_, idx) => {
            const result = this.#processTransformer(inputs, idx, true);
            linearOutputs[idx] = result.output;
            layerOutputs[idx] = result.layerOutputs;
            activations[idx] = result.activations;
            attentionIntermediates[idx] = result.attentionIntermediates;
        });

        this.#ensembleWeights = this.#computeAttentionWeights(inputs, linearOutputs);
        this.#normalizeEnsembleWeights();

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

        for (let t = 0; t < this.#ensembleSize; t++) {
            const grad = dL_d_scores[t] / this.#hiddenSize;
            for (let k = 0; k < this.#hiddenSize; k++) {
                this.#gradientAccumulation[t].attentionBias[k] += grad;
                const inputFeatures = inputs.map(x =>
                    Array(this.#hiddenSize).fill(isValidNumber(x) ? x : 0)
                );
                for (let i = 0; i < this.#inputSize; i++) {
                    if (isValidNumber(inputFeatures[i][k])) {
                        this.#gradientAccumulation[t].attentionWeightMatrix[k] += grad * inputFeatures[i][k];
                    }
                }
            }
        }

        this.#performanceScores = this.#performanceScores.map((score, idx) => {
            const individualProbability = this.#sigmoid(linearOutputs[idx]);
            const individualError = Math.abs(target - individualProbability);
            const newScore = 0.9 * score + 0.1 * (1 - individualError);
            this.#historicalPerformance[idx].push(isValidNumber(newScore) ? newScore : 0);
            if (this.#historicalPerformance[idx].length > this.#maxPerformanceHistory) {
                this.#historicalPerformance[idx].shift();
            }
            return isValidNumber(newScore) ? newScore : 0;
        });

        this.#agreementScores = this.#agreementScores.map((score, idx) => {
            const individualProbability = this.#sigmoid(linearOutputs[idx]);
            const agreement = 1 - Math.abs(individualProbability - finalProbability);
            const newScore = 0.9 * score + 0.1 * (isValidNumber(agreement) ? agreement : 0);
            return isValidNumber(newScore) ? newScore : 0;
        });

        this.#updateTrustScores();
        this.#computeSpecializationScores(inputs, linearOutputs);
        this.#updateAdaptiveLearningRates();

        this.#transformers.forEach((transformer, idx) => {
            const adjustedLearningRate = this.#adaptiveLearningRate[idx];
            const delta = dL_d_output * adjustedLearningRate;
            let grad = Array(this.#hiddenSize).fill(0);

            for (let i = 0; i < this.#hiddenSize; i++) {
                for (let j = 0; j < 1; j++) {
                    const gradUpdate = isValidNumber(delta) && isValidNumber(this.#ensembleWeights[idx]) && isValidNumber(layerOutputs[idx][this.#numLayers][0][i])
                        ? delta * this.#ensembleWeights[idx] * layerOutputs[idx][this.#numLayers][0][i]
                        : 0;
                    this.#gradientAccumulation[idx].outputWeights[i][j] += gradUpdate;
                    grad[i] += isValidNumber(delta) && isValidNumber(transformer.outputWeights[i][j])
                        ? delta * transformer.outputWeights[i][j]
                        : 0;
                }
            }
            this.#gradientAccumulation[idx].outputBias[0] += (isValidNumber(delta) && isValidNumber(this.#ensembleWeights[idx]))
                ? delta * this.#ensembleWeights[idx]
                : 0;

            for (let layer = this.#numLayers - 1; layer >= 0; layer--) {
                const { normX, attentionOutput, normAttention } = activations[idx][layer];
                const { Q, K, V, attentionProbs } = attentionIntermediates[idx][layer];
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
                            woGrad[k][j] += isValidNumber(attentionGrad[i][j]) && isValidNumber(attentionOutput[i][k])
                                ? attentionGrad[i][j] * attentionOutput[i][k]
                                : 0;
                        }
                    }
                }
                const vGrad = Array(this.#numHeads).fill().map(() => Array(this.#inputSize).fill().map(() => Array(headSize).fill(0)));
                const scoreGrad = Array(this.#numHeads).fill().map(() => Array(this.#inputSize).fill().map(() => Array(this.#inputSize).fill(0)));
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
                        const wqUpdate = qGrad.reduce((sum, row) => sum + (isValidNumber(row[j]) && isValidNumber(normX[i % this.#inputSize][i]) ? row[j] * normX[i % this.#inputSize][i] : 0), 0);
                        const wkUpdate = kGrad.reduce((sum, row) => sum + (isValidNumber(row[j]) && isValidNumber(normX[i % this.#inputSize][i]) ? row[j] * normX[i % this.#inputSize][i] : 0), 0);
                        const wvUpdate = vGrad.reduce((sum, head) => sum + (isValidNumber(head[i % this.#inputSize][j % headSize]) ? head[i % this.#inputSize][j % headSize] : 0), 0);
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
                grad = qGrad[0];
            }
        });

        this.#transformers.forEach((transformer, idx) => {
            this.#scaleGradients(idx);

            for (let i = 0; i < this.#hiddenSize; i++) {
                for (let j = 0; j < 1; j++) {
                    transformer.outputWeights[i][j] -= isValidNumber(this.#gradientAccumulation[idx].outputWeights[i][j])
                        ? this.#gradientAccumulation[idx].outputWeights[i][j]
                        : 0;
                }
            }
            for (let j = 0; j < 1; j++) {
                transformer.outputBias[j] -= isValidNumber(this.#gradientAccumulation[idx].outputBias[j])
                    ? this.#gradientAccumulation[idx].outputBias[j]
                    : 0;
            }
            for (let layer = 0; layer < this.#numLayers; layer++) {
                for (let i = 0; i < this.#hiddenSize; i++) {
                    for (let j = 0; j < this.#feedForwardSize; j++) {
                        transformer.ffnWeights[layer].W1[i][j] -= isValidNumber(this.#gradientAccumulation[idx].ffnWeights[layer].W1[i][j])
                            ? this.#gradientAccumulation[idx].ffnWeights[layer].W1[i][j]
                            : 0;
                    }
                }
                for (let i = 0; i < this.#feedForwardSize; i++) {
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        transformer.ffnWeights[layer].W2[i][j] -= isValidNumber(this.#gradientAccumulation[idx].ffnWeights[layer].W2[i][j])
                            ? this.#gradientAccumulation[idx].ffnWeights[layer].W2[i][j]
                            : 0;
                    }
                }
                for (let i = 0; i < this.#feedForwardSize; i++) {
                    transformer.ffnWeights[layer].b1[i] -= isValidNumber(this.#gradientAccumulation[idx].ffnWeights[layer].b1[i])
                        ? this.#gradientAccumulation[idx].ffnWeights[layer].b1[i]
                        : 0;
                }
                for (let i = 0; i < this.#hiddenSize; i++) {
                    transformer.ffnWeights[layer].b2[i] -= isValidNumber(this.#gradientAccumulation[idx].ffnWeights[layer].b2[i])
                        ? this.#gradientAccumulation[idx].ffnWeights[layer].b2[i]
                        : 0;
                }
                for (let i = 0; i < this.#hiddenSize; i++) {
                    for (let j = 0; j < this.#hiddenSize; j++) {
                        transformer.attentionWeights[layer].Wq[i][j] -= isValidNumber(this.#gradientAccumulation[idx].attentionWeights[layer].Wq[i][j])
                            ? this.#gradientAccumulation[idx].attentionWeights[layer].Wq[i][j]
                            : 0;
                        transformer.attentionWeights[layer].Wk[i][j] -= isValidNumber(this.#gradientAccumulation[idx].attentionWeights[layer].Wk[i][j])
                            ? this.#gradientAccumulation[idx].attentionWeights[layer].Wk[i][j]
                            : 0;
                        transformer.attentionWeights[layer].Wv[i][j] -= isValidNumber(this.#gradientAccumulation[idx].attentionWeights[layer].Wv[i][j])
                            ? this.#gradientAccumulation[idx].attentionWeights[layer].Wv[i][j]
                            : 0;
                        transformer.attentionWeights[layer].Wo[i][j] -= isValidNumber(this.#gradientAccumulation[idx].attentionWeights[layer].Wo[i][j])
                            ? this.#gradientAccumulation[idx].attentionWeights[layer].Wo[i][j]
                            : 0;
                    }
                }
                for (let i = 0; i < this.#hiddenSize; i++) {
                    transformer.layerNormWeights[layer].gamma1[i] -= isValidNumber(this.#gradientAccumulation[idx].layerNormWeights[layer].gamma1[i])
                        ? this.#gradientAccumulation[idx].layerNormWeights[layer].gamma1[i]
                        : 0;
                    transformer.layerNormWeights[layer].beta1[i] -= isValidNumber(this.#gradientAccumulation[idx].layerNormWeights[layer].beta1[i])
                        ? this.#gradientAccumulation[idx].layerNormWeights[layer].beta1[i]
                        : 0;
                    transformer.layerNormWeights[layer].gamma2[i] -= isValidNumber(this.#gradientAccumulation[idx].layerNormWeights[layer].gamma2[i])
                        ? this.#gradientAccumulation[idx].layerNormWeights[layer].gamma2[i]
                        : 0;
                    transformer.layerNormWeights[layer].beta2[i] -= isValidNumber(this.#gradientAccumulation[idx].layerNormWeights[layer].beta2[i])
                        ? this.#gradientAccumulation[idx].layerNormWeights[layer].beta2[i]
                        : 0;
                }
            }

            for (let k = 0; k < this.#hiddenSize; k++) {
                if (isValidNumber(this.#gradientAccumulation[idx].attentionBias[k])) {
                    this.#attentionBias[idx][k] -= this.#gradientAccumulation[idx].attentionBias[k];
                }
                if (isValidNumber(this.#gradientAccumulation[idx].attentionWeightMatrix[k])) {
                    this.#attentionWeightMatrix[idx][k] -= this.#gradientAccumulation[idx].attentionWeightMatrix[k];
                }
            }
        });

        this.#trainingStepCount++;

        if (this.#trainingStepCount % this.#distillationFrequency === 0) {
            this.#distillKnowledge(linearOutputs, target);
        }

        if (this.#trainingStepCount % this.#regulationFrequency === 0) {
            this.#regulateWeightsAndMemory();
        }

        this.#gradientAccumulation = this.#createWeightStructure();

        return this.#trainingStepCount
    }

    dumpState() {
        return this.#saveState()
    }
}

export default HiveMind;