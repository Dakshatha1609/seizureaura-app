// frontend/src/components/ThemeToggle.tsx
"use client";
import { useState, useEffect } from 'react';
import styles from './ThemeToggle.module.css';
import { FaSun, FaMoon } from 'react-icons/fa';

const ThemeToggle = () => {
  const [isDarkMode, setIsDarkMode] = useState(false);

  useEffect(() => {
    // Apply the theme when the component mounts
    if (localStorage.getItem('theme') === 'dark') {
      document.body.setAttribute('data-theme', 'dark');
      setIsDarkMode(true);
    } else {
      document.body.setAttribute('data-theme', 'light');
      setIsDarkMode(false);
    }
  }, []);

  const toggleTheme = () => {
    if (isDarkMode) {
      document.body.setAttribute('data-theme', 'light');
      localStorage.setItem('theme', 'light');
    } else {
      document.body.setAttribute('data-theme', 'dark');
      localStorage.setItem('theme', 'dark');
    }
    setIsDarkMode(!isDarkMode);
  };

  return (
    <div className={styles.toggleContainer} onClick={toggleTheme}>
      <FaSun className={`${styles.icon} ${!isDarkMode ? styles.active : ''}`} />
      <div className={styles.toggleSwitch}>
        <div className={`${styles.toggleKnob} ${isDarkMode ? styles.dark : ''}`}></div>
      </div>
      <FaMoon className={`${styles.icon} ${isDarkMode ? styles.active : ''}`} />
    </div>
  );
};

export default ThemeToggle;