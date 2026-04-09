// frontend/src/app/page.tsx (Simplified)
"use client";
import { useState } from 'react';
import Sidebar from '../components/Sidebar';
import Home from '../components/Home';
import AuraAnalysis from '../components/AuraAnalysis';
import EEGPrediction from '../components/EEGPrediction';
import Chatbot from '../components/Chatbot';
import styles from './page.module.css';

export default function App() {
  const [selectedPage, setSelectedPage] = useState('Home');

  const renderPage = () => {
    switch (selectedPage) {
      case 'Home':
        return <Home />;
      case 'Ask AI Chatbot':
        return <Chatbot />;
      case 'Aura Symptom Analysis':
        return <AuraAnalysis />;
      case 'EEG Seizure Prediction':
        return <EEGPrediction />;
      default:
        return <Home />;
    }
  };

  return (
    <div className={styles.layout}>
      <Sidebar activePage={selectedPage} onSelect={setSelectedPage} />
      <main className={styles.mainContent}>
        {renderPage()}
      </main>
    </div>
  );
}