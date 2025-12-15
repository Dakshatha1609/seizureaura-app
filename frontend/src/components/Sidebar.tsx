// frontend/src/components/Sidebar.tsx
"use client";
import styles from './Sidebar.module.css';
// --- 1. Import styles from the page module ---
import pageStyles from '../app/page.module.css'; 
import { FaHome, FaRegCommentDots, FaClipboardList, FaBrain } from 'react-icons/fa';
import ThemeToggle from './ThemeToggle';

type SidebarProps = {
  activePage: string;
  onSelect: (option: string) => void;
};

const Sidebar = ({ activePage, onSelect }: SidebarProps) => {
  const handleSelect = (option: string) => { onSelect(option); };

  return (
    // --- 2. Add the className from page.module.css ---
    <nav className={`${styles.sidebar} ${pageStyles.sidebar}`}>
      <div>
        <h2 className={styles.title}>SeizureAura</h2>
        <ul className={styles.navList}>
          <li className={activePage === 'Home' ? styles.active : ''} onClick={() => handleSelect('Home')}>
            <FaHome /> <span>Home</span>
          </li>
          <li className={activePage === 'Ask AI Chatbot' ? styles.active : ''} onClick={() => handleSelect('Ask AI Chatbot')}>
            <FaRegCommentDots /> <span>Ask AI Chatbot</span>
          </li>
          <li className={activePage === 'Aura Symptom Analysis' ? styles.active : ''} onClick={() => handleSelect('Aura Symptom Analysis')}>
            <FaClipboardList /> <span>Aura Symptom Analysis</span>
          </li>
          <li className={activePage === 'EEG Seizure Prediction' ? styles.active : ''} onClick={() => handleSelect('EEG Seizure Prediction')}>
            <FaBrain /> <span>EEG Seizure Prediction</span>
          </li>
        </ul>
      </div>
      <div className={styles.sidebarFooter}>
        <ThemeToggle />
      </div>
    </nav>
  );
};

export default Sidebar;