// frontend/src/components/AuraAnalysis.tsx
"use client";
import { useState } from 'react';
import axios from 'axios';
import styles from './ComponentStyles.module.css';

type PredictionResponse = {
  predicted_aura_stage: string;
  predicted_seizure_risk: string;
};

const AuraAnalysis = () => {
    const [symptomText, setSymptomText] = useState<string>('');
    const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [error, setError] = useState<string>('');

    const handlePredict = async () => {
        if (!symptomText.trim()) { setError('Please enter your symptoms.'); return; }
        setIsLoading(true);
        setError('');
        setPrediction(null);
        try {
            const response = await axios.post('http://127.0.0.1:5001/predict_aura', {
                symptom_text: symptomText,
            });
            setPrediction(response.data);
        } catch (err) {
            setError('Could not connect to the prediction service.');
        } finally {
            setIsLoading(false);
        }
    };

    const getResultStyle = () => {
        if (!prediction) return '';
        const risk = prediction.predicted_seizure_risk.toLowerCase();
        if (risk.includes('high')) return styles.highRisk;
        if (risk.includes('medium')) return styles.mediumRisk;
        return styles.lowRisk;
    };

    return (
        <div className={styles.animatedComponent}>
          <div className={styles.toolContainer}>
            <h1 className={styles.title}>Aura Symptom Analysis</h1>
            <p className={styles.description}>
                Describe your symptoms below to analyze the potential aura stage and seizure risk.
            </p>
            <div className={styles.inputGroup}>
                <textarea
                    className={styles.textarea}
                    value={symptomText}
                    onChange={(e) => setSymptomText(e.target.value)}
                    placeholder="e.g., I see shimmering lights..."
                    rows={5}
                />
                <button className={styles.button} onClick={handlePredict} disabled={isLoading}>
                    <div className={styles.buttonContent}>
                        {isLoading && <div className={styles.loader}></div>}
                        <span>{isLoading ? 'Analyzing...' : 'Analyze Symptoms'}</span>
                    </div>
                </button>
            </div>
            {error && <p className={styles.errorMessage}>{error}</p>}
            {prediction && (
                <div className={`${styles.resultsContainer} ${getResultStyle()}`}>
                    <h2>Analysis Results</h2>
                    <div className={styles.resultItem}>
                        <strong>Predicted Aura Stage:</strong> {prediction.predicted_aura_stage}
                    </div>
                    <div className={styles.resultItem}>
                        <strong>Predicted Seizure Risk:</strong> {prediction.predicted_seizure_risk}
                    </div>
                </div>
            )}
          </div>
        </div>
    );
};

export default AuraAnalysis;