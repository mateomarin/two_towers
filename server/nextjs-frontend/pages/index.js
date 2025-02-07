// nextjs-frontend/pages/index.js
import { useState, useEffect } from 'react';
import styles from '../styles/Home.module.css';

export default function Home() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [searchHistory, setSearchHistory] = useState([]);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [selectedHistoryItem, setSelectedHistoryItem] = useState(null);

  // Load search history from localStorage on component mount
  useEffect(() => {
    const savedHistory = localStorage.getItem('searchHistory');
    if (savedHistory) {
      setSearchHistory(JSON.parse(savedHistory));
    }
  }, []);

  // Save to localStorage whenever history changes
  useEffect(() => {
    localStorage.setItem('searchHistory', JSON.stringify(searchHistory));
  }, [searchHistory]);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('http://localhost:8000/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });

      if (!response.ok) {
        throw new Error('Search failed');
      }

      const data = await response.json();
      setResults(data);
      
      // Add to search history
      setSearchHistory(prev => [{
        query: query,
        results: data.results,
        timestamp: new Date().toLocaleString()
      }, ...prev]);
      
    } catch (err) {
      console.error('Error searching:', err);
      setError('Failed to perform search');
    } finally {
      setLoading(false);
    }
  };

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  const showHistoryResults = (historyItem) => {
    setSelectedHistoryItem(historyItem);
  };

  const clearHistory = () => {
    setSearchHistory([]);
    localStorage.removeItem('searchHistory');
  };

  return (
    <div className={styles.container}>
      {/* Sidebar Toggle Button */}
      <button 
        className={styles.sidebarToggle}
        onClick={toggleSidebar}
      >
        {isSidebarOpen ? '×' : '☰'}
      </button>

      {/* Sidebar */}
      <div className={`${styles.sidebar} ${isSidebarOpen ? styles.sidebarOpen : ''}`}>
        <div className={styles.sidebarHeader}>
          <h2 className={styles.sidebarTitle}>Search History</h2>
          {searchHistory.length > 0 && (
            <button 
              className={styles.clearHistoryButton}
              onClick={clearHistory}
            >
              Clear History
            </button>
          )}
        </div>
        <div className={styles.historyList}>
          {searchHistory.length === 0 ? (
            <div className={styles.emptyHistory}>
              No search history
            </div>
          ) : (
            searchHistory.map((item, index) => (
              <div 
                key={index} 
                className={styles.historyItem}
                onClick={() => showHistoryResults(item)}
              >
                <span className={styles.historyQuery}>{item.query}</span>
                <span className={styles.historyTime}>{item.timestamp}</span>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Main Content */}
      <main className={styles.main}>
        <h1 className={styles.title}>
          Document Search
        </h1>

        <form onSubmit={handleSearch} className={styles.searchForm}>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter your search query..."
            className={styles.searchInput}
          />
          <button type="submit" className={styles.searchButton} disabled={loading}>
            {loading ? 'Searching...' : 'Search'}
          </button>
        </form>

        {error && (
          <div className={styles.error}>
            {error}
          </div>
        )}

        {results && (
          <div className={styles.results}>
            {results.results.map((result, index) => (
              <div key={index} className={styles.resultCard}>
                <div className={styles.resultHeader}>
                  <span className={styles.score}>
                    Score: {result.score.toFixed(4)}
                  </span>
                  {result.is_ground_truth && (
                    <span className={styles.checkmark}>✓</span>
                  )}
                </div>
                <p className={styles.resultText}>{result.text}</p>
              </div>
            ))}
          </div>
        )}
      </main>

      {/* History Modal */}
      {selectedHistoryItem && (
        <div className={styles.modal} onClick={() => setSelectedHistoryItem(null)}>
          <div className={styles.modalContent} onClick={e => e.stopPropagation()}>
            <button 
              className={styles.modalClose}
              onClick={() => setSelectedHistoryItem(null)}
            >
              ×
            </button>
            <h2 className={styles.modalTitle}>
              Results for: {selectedHistoryItem.query}
            </h2>
            <div className={styles.modalResults}>
              {selectedHistoryItem.results.map((result, index) => (
                <div key={index} className={styles.resultCard}>
                  <div className={styles.resultHeader}>
                    <span className={styles.score}>
                      Score: {result.score.toFixed(4)}
                    </span>
                    {result.is_ground_truth && (
                      <span className={styles.checkmark}>✓</span>
                    )}
                  </div>
                  <p className={styles.resultText}>{result.text}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}