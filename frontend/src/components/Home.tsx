// frontend/src/components/Home.tsx (New, cleaner version)
import styles from './ComponentStyles.module.css';
import { FaRegCommentDots, FaClipboardList, FaBrain } from 'react-icons/fa';

const Home = () => {
  return (
    <div className={styles.animatedComponent}>
      <div className={styles.homeContainer}>
        <div className={styles.homeContent}>
          <h1 className={styles.title} style={{ fontSize: '3.5rem' }}>
            Welcome to SeizureAura
          </h1>
          <p className={styles.description} style={{ fontSize: '1.4rem', maxWidth: '800px', margin: '1rem auto 2.5rem auto' }}>
            Your AI Health Companion for understanding and managing epilepsy-related symptoms.
          </p>
          
          <div className={`${styles.infoCard} ${styles.disclaimerCard}`}>
            <h3><span role="img" aria-label="warning">⚠️</span> Important Disclaimer</h3>
            <p>
              This tool is an experimental project and not a substitute for professional medical advice. Always consult a qualified health provider.
            </p>
          </div>

          {/* --- New Features Section --- */}
          <div className={styles.featureGrid}>
            <div className={styles.featureCard}>
              <FaRegCommentDots size={40} className={styles.featureIcon} />
              <h3>Ask AI Chatbot</h3>
              <p>Get instant, clear answers to your questions about epilepsy, auras, and seizure safety using our AI-powered knowledge base.</p>
            </div>

            <div className={styles.featureCard}>
              <FaClipboardList size={40} className={styles.featureIcon} />
              <h3>Aura Symptom Analysis</h3>
              <p>Describe your symptoms in your own words, and our NLP model will analyze the text to identify the potential aura stage and risk.</p>
            </div>
            
            <div className={styles.featureCard}>
              <FaBrain size={40} className={styles.featureIcon} />
              <h3>EEG Seizure Prediction</h3>
              <p>Upload a standard .edf EEG file, and our deep learning model will analyze the brainwave data to predict seizure risk.</p>
            </div>
          </div>
          {/* --- End of New Section --- */}
        </div>
      </div>
    </div>
  );
};

export default Home;