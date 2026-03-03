import React, { useState } from 'react';

// ── Symptom definitions ────────────────────────────────────────────────────

const SYMPTOM_GROUPS = [
  {
    label: 'Duration',
    symptoms: [
      { key: 'symptoms_less_1week', label: 'Symptoms < 1 week' },
      { key: 'symptoms_over_1week', label: 'Symptoms > 1 week' },
    ],
  },
  {
    label: 'General',
    symptoms: [
      { key: 'headache',      label: 'Headache' },
      { key: 'exhaustion',    label: 'Exhaustion / Fatigue' },
      { key: 'dizziness',     label: 'Dizziness' },
      { key: 'sweating',      label: 'Sweating / Chills' },
      { key: 'restlessness',  label: 'Restlessness' },
      { key: 'poor_appetite', label: 'Poor Appetite' },
    ],
  },
  {
    label: 'Physical',
    symptoms: [
      { key: 'body_pain',  label: 'Body Pain' },
      { key: 'back_pain',  label: 'Back Pain' },
      { key: 'chest_pain', label: 'Chest Pain' },
      { key: 'rash',       label: 'Skin Rash' },
      { key: 'cough',      label: 'Cough' },
    ],
  },
  {
    label: 'Digestive',
    symptoms: [
      { key: 'abdominal_pain', label: 'Abdominal Pain' },
      { key: 'diarrhea',       label: 'Diarrhea' },
      { key: 'constipation',   label: 'Constipation' },
    ],
  },
  {
    label: 'Neurological',
    symptoms: [
      { key: 'semi_closed_eyes', label: 'Eyes Stay Semi-Closed' },
      { key: 'blurry_vision',    label: 'Blurry Vision' },
      { key: 'hallucination',    label: 'Hallucination' },
    ],
  },
];

const DEFAULT_SYMPTOMS = Object.fromEntries(
  SYMPTOM_GROUPS.flatMap(g => g.symptoms.map(s => [s.key, 'no']))
);

const DEFAULT_DEMOGRAPHICS = {
  age: '',
  sex: 'Unknown',
  pregnant: 'no',
  genotype: 'AA',
  blood_type: 'O+',
  rdt_result: 'not_done',
  microscopy_result: 'not_done',
};

// ── Component ──────────────────────────────────────────────────────────────

export default function DiagnosisForm({ onSubmit, loading }) {
  const [patientName, setPatientName] = useState('');
  const [patientId, setPatientId] = useState('');
  const [symptoms, setSymptoms] = useState(DEFAULT_SYMPTOMS);
  const [demographics, setDemographics] = useState(DEFAULT_DEMOGRAPHICS);

  const toggleSymptom = (key) => {
    setSymptoms(prev => ({ ...prev, [key]: prev[key] === 'yes' ? 'no' : 'yes' }));
  };

  const updateDemo = (key, value) => {
    setDemographics(prev => ({ ...prev, [key]: value }));
  };

  const checkedCount = Object.values(symptoms).filter(v => v === 'yes').length;

  const handleSubmit = (e) => {
    e.preventDefault();
    const demo = { ...demographics, age: demographics.age || 'Unknown' };
    onSubmit(symptoms, demo, patientName.trim(), patientId.trim());
  };

  const handleReset = () => {
    setPatientName('');
    setPatientId('');
    setSymptoms(DEFAULT_SYMPTOMS);
    setDemographics(DEFAULT_DEMOGRAPHICS);
  };

  return (
    <form onSubmit={handleSubmit}>

      {/* ── Patient Identification ── */}
      <div className="card">
        <div className="card-header">
          <h2>Patient Identification <span className="optional-tag">optional</span></h2>
        </div>
        <div className="card-body">
          <div className="patient-id-grid">
            <div className="field field--large">
              <label>Patient Name</label>
              <input
                type="text"
                placeholder="e.g. John Doe"
                value={patientName}
                onChange={e => setPatientName(e.target.value)}
              />
            </div>
            <div className="field field--large">
              <label>Patient ID</label>
              <input
                type="text"
                placeholder="e.g. MLC-001"
                value={patientId}
                onChange={e => setPatientId(e.target.value)}
              />
            </div>
          </div>
        </div>
      </div>

      {/* ── Demographics ── */}
      <div className="card">
        <div className="card-header">
          <h2>Clinical Information</h2>
        </div>
        <div className="card-body">
          <div className="demographics-grid">
            <div className="field">
              <label>Age</label>
              <input
                type="number"
                min="0"
                max="120"
                placeholder="e.g. 28"
                value={demographics.age}
                onChange={e => updateDemo('age', e.target.value)}
              />
            </div>

            <div className="field">
              <label>Sex</label>
              <select value={demographics.sex} onChange={e => updateDemo('sex', e.target.value)}>
                <option value="Unknown">Unknown</option>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
              </select>
            </div>

            {demographics.sex === 'Female' && (
              <div className="field">
                <label>Pregnant</label>
                <select value={demographics.pregnant} onChange={e => updateDemo('pregnant', e.target.value)}>
                  <option value="no">No</option>
                  <option value="yes">Yes</option>
                </select>
              </div>
            )}

            <div className="field">
              <label>Genotype</label>
              <select value={demographics.genotype} onChange={e => updateDemo('genotype', e.target.value)}>
                <option value="AA">AA</option>
                <option value="AS">AS</option>
                <option value="SS">SS</option>
                <option value="AC">AC</option>
              </select>
            </div>

            <div className="field">
              <label>Blood Type</label>
              <select value={demographics.blood_type} onChange={e => updateDemo('blood_type', e.target.value)}>
                {['O+', 'O-', 'A+', 'A-', 'B+', 'B-', 'AB+', 'AB-'].map(bt => (
                  <option key={bt} value={bt}>{bt}</option>
                ))}
              </select>
            </div>

            <div className="field">
              <label>RDT Result</label>
              <select value={demographics.rdt_result} onChange={e => updateDemo('rdt_result', e.target.value)}>
                <option value="not_done">Not Done</option>
                <option value="positive">Positive</option>
                <option value="negative">Negative</option>
              </select>
            </div>

            <div className="field">
              <label>Microscopy</label>
              <select value={demographics.microscopy_result} onChange={e => updateDemo('microscopy_result', e.target.value)}>
                <option value="not_done">Not Done</option>
                <option value="positive">Positive</option>
                <option value="negative">Negative</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      {/* ── Symptoms ── */}
      <div className="card">
        <div className="card-header">
          <h2>
            Symptoms{' '}
            {checkedCount > 0 && (
              <span style={{ color: 'var(--primary)', fontWeight: 400 }}>
                ({checkedCount} selected)
              </span>
            )}
          </h2>
        </div>
        <div className="card-body">
          {SYMPTOM_GROUPS.map(group => (
            <div key={group.label} className="symptoms-group">
              <div className="symptoms-group-label">{group.label}</div>
              <div className="symptoms-grid">
                {group.symptoms.map(({ key, label }) => (
                  <label
                    key={key}
                    className={`symptom-checkbox ${symptoms[key] === 'yes' ? 'checked' : ''}`}
                  >
                    <input
                      type="checkbox"
                      checked={symptoms[key] === 'yes'}
                      onChange={() => toggleSymptom(key)}
                    />
                    <span>{label}</span>
                  </label>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Actions ── */}
      <div className="submit-row">
        <button type="submit" className="btn-primary" disabled={loading}>
          {loading ? (
            <>
              <span className="spinner" />
              Analyzing...
            </>
          ) : (
            'Analyze Symptoms'
          )}
        </button>
        <button type="button" className="btn-secondary" onClick={handleReset} disabled={loading}>
          Reset
        </button>
      </div>
    </form>
  );
}
