// frontend/src/components/Chatbot.tsx
"use client";
import { useState, useRef, useEffect } from 'react';
import styles from './ComponentStyles.module.css';
import { FaUser, FaRobot, FaClipboard } from 'react-icons/fa';

type Message = {
  sender: 'user' | 'bot';
  text: string;
};

const Chatbot = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  const handleSendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = { sender: 'user', text: input };
    setMessages(prev => [...prev, userMessage, { sender: 'bot', text: '' }]);
    setInput('');
    setIsLoading(true);
    
    try {
      const response = await fetch('http://127.0.0.1:5001/ask_chatbot', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: input }),
      });

      if (!response.body) throw new Error("Response body is null");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let done = false;

      while (!done) {
        const { value, done: readerDone } = await reader.read();
        done = readerDone;
        const chunk = decoder.decode(value, { stream: true });
        
        setMessages(prevMessages => {
          const lastMessageIndex = prevMessages.length - 1;
          const updatedMessages = [...prevMessages];
          updatedMessages[lastMessageIndex] = {
            ...updatedMessages[lastMessageIndex],
            text: updatedMessages[lastMessageIndex].text + chunk,
          };
          return updatedMessages;
        });
      }
    } catch (error) {
        console.error("Streaming failed:", error);
        setMessages(prevMessages => {
            const lastMessageIndex = prevMessages.length - 1;
            const updatedMessages = [...prevMessages];
            updatedMessages[lastMessageIndex] = {
                ...updatedMessages[lastMessageIndex],
                text: 'Sorry, I ran into an error. Please try again.',
            };
            return updatedMessages;
          });
    } finally {
      setIsLoading(false);
    }
  };

  const handleCopy = (textToCopy: string, index: number) => {
    navigator.clipboard.writeText(textToCopy);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  return (
    <div className={styles.animatedComponent}>
      <div className={styles.toolContainer}>
        <h1 className={styles.title}>Ask AI Chatbot</h1>
        <p className={styles.description}>
          Ask questions about epilepsy, seizures, or auras.
        </p>
        
        <div className={styles.chatWindow}>
          {messages.map((msg, index) => (
            <div key={index} className={`${styles.chatMessage} ${styles[msg.sender]}`}>
              <div className={styles.avatar}>
                {msg.sender === 'user' ? <FaUser /> : <FaRobot />}
              </div>
              <div className={styles.messageText}>
                {msg.text}
                {msg.sender === 'bot' && msg.text && !isLoading && (
                  <button onClick={() => handleCopy(msg.text, index)} className={styles.copyButton}>
                    {copiedIndex === index ? 'Copied!' : <FaClipboard />}
                  </button>
                )}
                {isLoading && msg.sender === 'bot' && index === messages.length - 1 && (
                  <span className={styles.blinkingCursor}>▋</span>
                )}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
        
        <div className={styles.chatInputContainer}>
          <input
            type="text"
            className={styles.chatInput}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && !isLoading && handleSendMessage()}
            placeholder="Type your question here..."
            disabled={isLoading}
          />
          <button className={styles.sendButton} onClick={handleSendMessage} disabled={isLoading}>
            {isLoading ? '...' : 'Send'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default Chatbot;