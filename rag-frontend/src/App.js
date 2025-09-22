 
import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResults([]);
     if (!query.trim()) {
      setLoading(false);
      return; // Stop the function here
    }

    try {
      const response = await axios.post('http://127.0.0.1:8000/ask', {
        q: query,
        k: 5,
        mode: 'hybrid'
      });
      // The fix is on the line below:
      setResults(response.data.contexts); 
    } catch (err) {
      setError('An error occurred while fetching data. Please check if the backend is running.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <header>
        <h1>Professional RAG Chatbot 🤖</h1>
      </header>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question about industrial safety..."
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Searching...' : 'Search'}
        </button>
      </form>

      {error && <div className="error">{error}</div>}

      <div className="results-container">
        {results.length > 0 ? (
          results.map((result, index) => (
            <div key={index} className="result-card">
              <h3>{result.source_title}</h3>
              <p>{result.text_snippet}</p>
              <a href={result.source_url} target="_blank" rel="noopener noreferrer">
                View Source
              </a>
            </div>
          ))
        ) : (
          !loading && <p className="no-results">No results found or waiting for a query.</p>
        )}
      </div>
    </div>
  );
}

export default App;