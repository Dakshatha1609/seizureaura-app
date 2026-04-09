// frontend/src/components/EEGPrediction.tsx
"use client";
import { useState } from 'react';
import axios from 'axios';
import styles from './ComponentStyles.module.css';

const EEGPrediction = () => {
  const [file, setFile] = useState<File | null>(null);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      setFile(event.target.files[0]);
    }
  };

  const handlePredict = async () => {
    if (!file) { setError('Please upload an EEG file first.'); return; }
    setIsLoading(true);
    setError('');
    setPrediction(null);
    const formData = new FormData();
    formData.append('eeg_file', file);
    try {
      const response = await axios.post('http://127.0.0.1:5001/predict_eeg', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setPrediction(response.data.prediction);
    } catch (err) {
      setError('Could not process the EEG file.');
    } finally {
      setIsLoading(false);
    }
  };

  const getResultStyle = () => {
    if (!prediction) return '';
    const risk = prediction.toLowerCase();
    if (risk.includes('risk detected')) return styles.highRisk;
    return styles.lowRisk;
  };

  return (
    <div className={styles.animatedComponent}>
      <div className={styles.toolContainer}>
        <h1 className={styles.title}>EEG Seizure Prediction</h1>
        <p className={styles.description}>
          Upload an EEG file (.edf format) to predict the risk of a seizure.
        </p>

        <div className={styles.exampleContainer}>
          <span>No file? </span>
          <a href="/chb01_01.edf" download className={styles.exampleLink}>
            Download an example EEG file
          </a>
        </div>
        
        <div className={styles.inputGroup}>
          <div className={styles.fileUploader}>
            <input type="file" onChange={handleFileChange} accept=".edf" />
          </div>
          <button
            className={styles.button}
            onClick={handlePredict}
            disabled={isLoading || !file}
          >
            <div className={styles.buttonContent}>
              {isLoading && <div className={styles.loader}></div>}
              <span>{isLoading ? 'Processing...' : 'Predict Seizure Risk'}</span>
            </div>
          </button>
        </div>
        
        {error && <p className={styles.errorMessage}>{error}</p>}
        
        {prediction && (
          <div className={`${styles.resultsContainer} ${getResultStyle()}`}>
            <h2>Prediction Result</h2>
            <div className={styles.resultItem}>
              <strong>Result:</strong> {prediction}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default EEGPrediction;