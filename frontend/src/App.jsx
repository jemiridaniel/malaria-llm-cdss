import React, { useState } from 'react';
import './App.css';
import DiagnosisForm from './components/DiagnosisForm';
import DiagnosisResult from './components/DiagnosisResult';
import { diagnosePatient } from './services/api';

export default function App() {
  const [result, setResult] = useState(null);
  const [storedDemographics, setStoredDemographics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (symptoms, demographics, patientName, patientId) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await diagnosePatient(symptoms, demographics, patientName, patientId);
      setResult(data);
      setStoredDemographics(demographics);
      setTimeout(() => {
        document.getElementById('result')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 100);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setStoredDemographics(null);
    setError(null);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <span className="header-icon">🦟</span>
          <div>
            <h1>Malaria Clinical Decision Support System</h1>
            <p>Rule-based classification · LLM explanation · Multi-provider fallback</p>
          </div>
        </div>
      </header>

      <main className="container">
        {error && (
          <div className="error-banner">
            <span>⚠</span>
            {error}
          </div>
        )}

        <DiagnosisForm onSubmit={handleSubmit} loading={loading} />

        {result && (
          <div id="result">
            <DiagnosisResult
              result={result}
              demographics={storedDemographics}
              onReset={handleReset}
            />
          </div>
        )}
      </main>
    </div>
  );
}
