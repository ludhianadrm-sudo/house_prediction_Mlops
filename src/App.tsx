/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export default function App() {
  return (
    <div style={{ fontFamily: 'system-ui, -apple-system, sans-serif', padding: '30px', maxWidth: '900px', margin: '0 auto', color: '#1e293b', lineHeight: '1.6' }}>
      <header style={{ borderBottom: '1px solid #e2e8f0', paddingBottom: '20px', marginBottom: '24px' }}>
        <h1 style={{ fontSize: '24px', fontWeight: 'bold', color: '#0f172a', marginBottom: '8px' }}>
          House Price Prediction Using MLOps
        </h1>
        <p style={{ color: '#64748b', fontSize: '15px' }}>
          Practical Assignment: Beginner-Level Machine Learning & MLOps Pipeline
        </p>
      </header>

      <section style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '12px', padding: '24px', marginBottom: '24px' }}>
        <h2 style={{ fontSize: '18px', fontWeight: '600', color: '#0f172a', marginBottom: '12px' }}>
          🚀 Step-by-Step Terminal Commands
        </h2>
        <p style={{ fontSize: '14px', color: '#475569', marginBottom: '16px' }}>
          Run the following commands in your terminal from the project root directory:
        </p>
        
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <div>
            <strong style={{ fontSize: '13px', color: '#334155' }}>1. Install Python Dependencies:</strong>
            <pre style={{ background: '#0f172a', color: '#38bdf8', padding: '10px 14px', borderRadius: '6px', fontSize: '13px', margin: '4px 0 0 0', overflowX: 'auto' }}>pip install -r requirements.txt</pre>
          </div>

          <div>
            <strong style={{ fontSize: '13px', color: '#334155' }}>2. Generate Dataset (Task 1):</strong>
            <pre style={{ background: '#0f172a', color: '#38bdf8', padding: '10px 14px', borderRadius: '6px', fontSize: '13px', margin: '4px 0 0 0', overflowX: 'auto' }}>python generate_data.py</pre>
          </div>

          <div>
            <strong style={{ fontSize: '13px', color: '#334155' }}>3. Train Model, Evaluate & Track with MLflow (Tasks 2–5):</strong>
            <pre style={{ background: '#0f172a', color: '#38bdf8', padding: '10px 14px', borderRadius: '6px', fontSize: '13px', margin: '4px 0 0 0', overflowX: 'auto' }}>python train.py</pre>
          </div>

          <div>
            <strong style={{ fontSize: '13px', color: '#334155' }}>4. View MLflow Experiment Tracking Dashboard (Task 4):</strong>
            <pre style={{ background: '#0f172a', color: '#38bdf8', padding: '10px 14px', borderRadius: '6px', fontSize: '13px', margin: '4px 0 0 0', overflowX: 'auto' }}>mlflow ui</pre>
            <span style={{ fontSize: '12px', color: '#64748b' }}>Open browser at: http://127.0.0.1:5000</span>
          </div>

          <div>
            <strong style={{ fontSize: '13px', color: '#334155' }}>5. Start FastAPI Deployment Server (Task 6):</strong>
            <pre style={{ background: '#0f172a', color: '#38bdf8', padding: '10px 14px', borderRadius: '6px', fontSize: '13px', margin: '4px 0 0 0', overflowX: 'auto' }}>uvicorn api:app --reload --port 8000</pre>
          </div>

          <div>
            <strong style={{ fontSize: '13px', color: '#334155' }}>6. Test API using Swagger UI:</strong>
            <pre style={{ background: '#0f172a', color: '#38bdf8', padding: '10px 14px', borderRadius: '6px', fontSize: '13px', margin: '4px 0 0 0', overflowX: 'auto' }}>Open http://127.0.0.1:8000/docs in your browser</pre>
          </div>

          <div>
            <strong style={{ fontSize: '13px', color: '#334155' }}>7. Initialize Git & Commit:</strong>
            <pre style={{ background: '#0f172a', color: '#38bdf8', padding: '10px 14px', borderRadius: '6px', fontSize: '13px', margin: '4px 0 0 0', overflowX: 'auto' }}>git init && git add . && git commit -m "Initial commit"</pre>
          </div>
        </div>
      </section>

      <section style={{ background: '#ffffff', border: '1px solid #e2e8f0', borderRadius: '12px', padding: '24px' }}>
        <h2 style={{ fontSize: '18px', fontWeight: '600', color: '#0f172a', marginBottom: '12px' }}>
          📁 Project Structure & Files
        </h2>
        <ul style={{ paddingLeft: '20px', color: '#475569', fontSize: '14px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
          <li><code style={{ background: '#f1f5f9', padding: '2px 6px', borderRadius: '4px' }}>data/house_prices.csv</code> - Generated dataset</li>
          <li><code style={{ background: '#f1f5f9', padding: '2px 6px', borderRadius: '4px' }}>models/house_model.joblib</code> - Saved trained model</li>
          <li><code style={{ background: '#f1f5f9', padding: '2px 6px', borderRadius: '4px' }}>generate_data.py</code> - Dataset generation script</li>
          <li><code style={{ background: '#f1f5f9', padding: '2px 6px', borderRadius: '4px' }}>train.py</code> - Training, evaluation & MLflow tracking</li>
          <li><code style={{ background: '#f1f5f9', padding: '2px 6px', borderRadius: '4px' }}>api.py</code> - FastAPI prediction endpoint</li>
          <li><code style={{ background: '#f1f5f9', padding: '2px 6px', borderRadius: '4px' }}>requirements.txt</code> - Python package dependencies</li>
        </ul>
      </section>
    </div>
  );
}
