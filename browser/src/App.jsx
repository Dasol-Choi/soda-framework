import { useState, useEffect } from 'react';

function App ({
  jsonUrl = './data/soda_grouped_images.json',
  initialPerRow = 5,
  initialImgHeight = 80,
}) {
  // Load data
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Layout state
  const [selectedCategory, setSelectedCategory] = useState('');
  const [perRow, setPerRow] = useState(initialPerRow);
  const [imgHeight, setImgHeight] = useState(initialImgHeight);

  useEffect(() => {
    fetch(jsonUrl)
      .then(res => {
        if (!res.ok) throw new Error(`Failed to load ${jsonUrl}`);
        return res.json();
      })
      .then(data => {
        setData(data);
        if (data.object_categories && data.object_categories.length) {
          setSelectedCategory(data.object_categories[data.object_categories.length-1]);
        }
      })
      .catch(err => setError(err.message))
      .finally(() => setLoading(false));
  }, [jsonUrl]);

  useEffect(() => {
    document.documentElement.style.setProperty('--per-row', perRow);
    document.documentElement.style.setProperty('--img-height', `${imgHeight}px`);
  }, [perRow, imgHeight]);

  if (loading) return <div className="loading">Loading...</div>;
  if (error) return <div>Error: {error}</div>;
  if (!data) return <div>No data available.</div>;

  const categories = data.object_categories || [];
  const models = data.models || [];
  const conditions = data.conditions || [];
  const groupedImages = data.images || {};

  const handlePerRowChange = e => {
    const v = Math.max(1, Math.min(20, parseInt(e.target.value, 10) || 1));
    setPerRow(v);
  };

  const handleImgHeightChange = e => {
    let h = Math.max(20, parseInt(e.target.value, 10) || 20);
    h = Math.round(h / 10) * 10;
    setImgHeight(h);
  };

  return (
    <div id="container">
      <div className="header">
        <div className="header-content">
          <div className="tabs">
            <span className="object-category-label">Select an object category:</span>
            {categories.map(cat => (
              <div key={cat}
                  className={`tab ${selectedCategory === cat ? 'active' : ''}`}
                  onClick={() => setSelectedCategory(cat)}>
                {cat}
              </div>
            ))}
          </div>

          <div className="controls">
            <label>
              Images per row:
              <input type="number" value={perRow} min={1} max={20}
                    onChange={handlePerRowChange} />
            </label>
            <label>
              Image height (px):
              <input type="number" value={imgHeight} min={20} step={10}
                    onChange={handleImgHeightChange} />
            </label>
          </div>
        </div>

        <a href="https://github.com/Dasol-Choi/soda-framework" className="github-link" target="_blank" rel="noopener noreferrer">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/>
          </svg>
          Code for SODA
        </a>
      </div>

      <div className="section">
        {conditions.map(cond => (
          <div key={cond.key}>
            <div className="condition-title">
              {cond.label === "null" ? 
                <span>Images from prompts with <strong>no demographic cue</strong></span> :
                <span>Images from prompts for a specific <strong>{cond.label}</strong></span>
              }
            </div>
            {(Array.isArray(cond.values) ? cond.values : [cond.values]).map(val => (
              <div key={val}>
                <div className="model-row">
                  <div className="model-column model-column-label">
                    {
                      val !== 'null' ?
                      <div className="value-title">{val}</div> :
                      ''
                    }
                  </div>
                  {models.map(m => {
                    const imgs = groupedImages[selectedCategory]?.[m]?.[cond.key]?.[val] || [];
                    return (
                      <div key={m} className="model-column">
                        <div className="model-title">{m}</div>
                        <div className="gallery">
                          {imgs.map((entry, i) => (
                            <img key={i} src={entry.image_base64}
                                alt={`${m}_${selectedCategory}_${cond.key}_${val}`} />
                          ))}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
};

export default App
