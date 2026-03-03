import React, { useState } from 'react';
import { downloadReport } from '../services/api';

const STAGE_CONFIG = {
  No_Malaria: { cls: 'no-malaria', icon: '✓', label: 'No Malaria Detected' },
  Stage_I:    { cls: 'stage-1',   icon: '⚠', label: 'Stage I Malaria' },
  Stage_II:   { cls: 'stage-2',   icon: '⚡', label: 'Stage II Malaria' },
  Critical:   { cls: 'critical',  icon: '🚨', label: 'Critical — Severe Malaria' },
};

export default function DiagnosisResult({ result, demographics, onReset }) {
  const [pdfLoading, setPdfLoading] = useState(false);
  const [pdfError, setPdfError] = useState(null);

  const config = STAGE_CONFIG[result.severity_stage] || {
    cls: 'stage-1',
    icon: '?',
    label: result.severity_stage,
  };

  const patientLabel = result.patient_name
    ? result.patient_id
      ? `${result.patient_name} (ID: ${result.patient_id})`
      : result.patient_name
    : result.patient_id
      ? `ID: ${result.patient_id}`
      : null;

  const handleDownloadPdf = async () => {
    setPdfLoading(true);
    setPdfError(null);
    try {
      await downloadReport({
        patient_name:      result.patient_name,
        patient_id:        result.patient_id,
        timestamp:         result.timestamp,
        demographics:      demographics || {},
        positive_symptoms: result.positive_symptoms,
        severity_stage:    result.severity_stage,
        diagnosis_text:    result.diagnosis_text,
        confidence:        result.confidence,
        clinical_reasoning:result.clinical_reasoning,
        prescription:      result.prescription,
        model_used:        result.model_used,
      });
    } catch (err) {
      setPdfError(err.message);
    } finally {
      setPdfLoading(false);
    }
  };

  return (
    <div className="result-card">

      {/* ── Severity header ── */}
      <div className={`result-header ${config.cls}`}>
        <span className={`severity-badge ${config.cls}`}>
          <span>{config.icon}</span>
          {config.label}
        </span>
        <span className="confidence-badge">
          Confidence: {result.confidence}%
        </span>
      </div>

      {/* ── Patient + timestamp sub-header ── */}
      {(patientLabel || result.timestamp) && (
        <div className="result-patient-row">
          {patientLabel && (
            <span className="result-patient-name">Diagnosis for: {patientLabel}</span>
          )}
          {result.timestamp && (
            <span className="result-timestamp">{result.timestamp}</span>
          )}
        </div>
      )}

      <div className="result-body">

        {/* Diagnosis */}
        <div className="result-section">
          <div className="result-section-title">Diagnosis</div>
          <div className="diagnosis-text">{result.diagnosis_text}</div>
        </div>

        {/* Positive symptoms */}
        {result.positive_symptoms.length > 0 && (
          <div className="result-section">
            <div className="result-section-title">
              Reported Symptoms ({result.symptom_count})
            </div>
            <div className="symptom-chips">
              {result.positive_symptoms.map(s => (
                <span key={s} className="symptom-chip">{s}</span>
              ))}
            </div>
          </div>
        )}

        {/* AI Explanation */}
        {result.clinical_reasoning && (
          <div className="result-section">
            <div className="result-section-title">AI Clinical Explanation</div>
            <p className="reasoning-text">{result.clinical_reasoning}</p>
          </div>
        )}

        {/* Treatment */}
        <div className="result-section">
          <div className="result-section-title">Treatment Recommendation</div>
          <div className={`prescription-box ${config.cls === 'critical' ? 'critical' : ''}`}>
            {result.prescription}
          </div>
        </div>

        {/* Download PDF */}
        <div className="result-section">
          {pdfError && (
            <div className="error-banner" style={{ marginBottom: 10 }}>
              <span>⚠</span> {pdfError}
            </div>
          )}
          <button
            className="btn-download"
            onClick={handleDownloadPdf}
            disabled={pdfLoading}
          >
            {pdfLoading ? (
              <>
                <span className="spinner spinner--dark" />
                Generating PDF...
              </>
            ) : (
              <>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                  <polyline points="7 10 12 15 17 10"/>
                  <line x1="12" y1="15" x2="12" y2="3"/>
                </svg>
                Download PDF Report
              </>
            )}
          </button>
        </div>
      </div>

      {/* ── Footer ── */}
      <div className="result-footer">
        <span className="model-badge">
          <span className="model-dot" />
          {result.model_used}
        </span>
        <button
          className="btn-secondary"
          onClick={onReset}
          style={{ height: 36, padding: '0 18px', fontSize: '0.82rem' }}
        >
          New Assessment
        </button>
      </div>
    </div>
  );
}
