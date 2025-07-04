import React, { useState, useRef } from 'react';
import axios from 'axios';
import './App.css';

// IMPORTANT: This relative URL works for both local dev and Vercel deployment
const API_BASE_URL = process.env.NODE_ENV === 'production' ? '/api' : 'http://127.0.0.1:5000';

function App() {
  const [question, setQuestion] = useState<string>('');
  const [answer, setAnswer] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [statusMessage, setStatusMessage] = useState<string>('');
  const [isDocProcessed, setIsDocProcessed] = useState<boolean>(false);
  const [fileName, setFileName] = useState<string>('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setFileName(file.name);
    setIsDocProcessed(false);
    setAnswer('');
    setStatusMessage('Reading file...');

    const reader = new FileReader();
    reader.onload = async (e) => {
      const text = e.target?.result;
      if (typeof text === 'string') {
        setIsLoading(true);
        setStatusMessage('Processing document with AI...');
        try {
          await axios.post(`${API_BASE_URL}/process`, { text });
          setIsDocProcessed(true);
          setStatusMessage('✅ Document is ready to be queried!');
        } catch (error) {
          console.error('Error processing document:', error);
          setStatusMessage('❌ Error processing document.');
        } finally {
          setIsLoading(false);
        }
      }
    };
    reader.readAsText(file);
  };

  const handleAskQuestion = async () => {
    if (!question.trim() || !isDocProcessed) return;

    setIsLoading(true);
    setAnswer('');
    setStatusMessage('AI is thinking...');
    try {
      const response = await axios.post(`${API_BASE_URL}/ask`, { question });
      setAnswer(response.data.answer);
      setStatusMessage(''); // Clear status on successful answer
    } catch (error) {
      console.error('Error asking question:', error);
      setStatusMessage('❌ Error getting answer.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
      <div className="App">
        <header className="App-header">
          <h1>AI Document Q&A</h1>
          <p>Upload a .txt file, and ask the AI questions about its contents.</p>
        </header>

        <div className="card">
          <h2>1. Upload Document</h2>
          <input type="file" accept=".txt" onChange={handleFileChange} ref={fileInputRef} style={{ display: 'none' }} />
          <button onClick={() => fileInputRef.current?.click()} disabled={isLoading}>Choose .txt file</button>
          {fileName && <p className="file-status">Selected: <strong>{fileName}</strong></p>}
        </div>

        <div className="status-container">
          {isLoading && <div className="loader"></div>}
          {statusMessage && <p className="status-message">{statusMessage}</p>}
        </div>

        {isDocProcessed && (
            <div className="card">
              <h2>2. Ask a Question</h2>
              <textarea
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="e.g., What was the main objective of the program?"
                  disabled={isLoading}
              />
              <button onClick={handleAskQuestion} disabled={isLoading || !question.trim()}>Get Answer</button>
            </div>
        )}

        {answer && (
            <div className="card answer-card">
              <h2>Answer</h2>
              <p>{answer}</p>
            </div>
        )}
      </div>
  );
}

export default App;