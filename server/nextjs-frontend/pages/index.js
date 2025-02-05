// nextjs-frontend/pages/index.js
import { useEffect, useState } from 'react';

export default function Home() {
  const [message, setMessage] = useState('Loading...');

  useEffect(() => {
    // Use the container's host IP or a relative URL if using a reverse proxy.
    fetch('http://localhost:5000/test')
      .then((res) => res.text())
      .then((data) => setMessage(data))
      .catch((err) => {
        console.error('Error fetching API:', err);
        setMessage('Failed to load data');
      });
  }, []);

  return (
    <div style={{ padding: '2rem' }}>
      <h1>Next.js Front-end</h1>
      <p>Message from API: {message}</p>
    </div>
  );
}