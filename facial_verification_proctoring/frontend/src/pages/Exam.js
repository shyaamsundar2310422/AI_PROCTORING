import React, { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import axios from 'axios';

const Exam = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const { user } = useAuth();
  const [exam, setExam] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [timeLeft, setTimeLeft] = useState(null);
  const [anomalies, setAnomalies] = useState([]);
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  useEffect(() => {
    // Get exam details
    axios.get(`/api/exam/${id}`)
      .then(response => {
        setExam(response.data);
        setTimeLeft(response.data.duration * 60); // Convert minutes to seconds
      })
      .catch(error => {
        console.error('Error fetching exam:', error);
        navigate('/dashboard');
      });

    // Start webcam
    navigator.mediaDevices.getUserMedia({ video: true, audio: true })
      .then(stream => {
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      })
      .catch(error => {
        console.error('Error accessing media devices:', error);
      });

    return () => {
      // Cleanup
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, [id, navigate]);

  useEffect(() => {
    if (timeLeft > 0) {
      const timer = setInterval(() => {
        setTimeLeft(prev => prev - 1);
      }, 1000);
      return () => clearInterval(timer);
    } else if (timeLeft === 0) {
      endExam();
    }
  }, [timeLeft]);

  const startExam = async () => {
    try {
      const response = await axios.post(`/api/exam/start/${id}`);
      setSessionId(response.data.session_id);
    } catch (error) {
      console.error('Error starting exam:', error);
    }
  };

  const endExam = async () => {
    try {
      await axios.post(`/api/exam/end/${sessionId}`);
      navigate('/dashboard');
    } catch (error) {
      console.error('Error ending exam:', error);
    }
  };

  const formatTime = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  if (!exam) {
    return <div>Loading...</div>;
  }

  return (
    <div className="exam-container">
      <div className="row">
        <div className="col-md-8">
          <div className="card mb-4">
            <div className="card-header d-flex justify-content-between align-items-center">
              <h4>{exam.title}</h4>
              <div className="timer">{formatTime(timeLeft)}</div>
            </div>
            <div className="card-body">
              {/* Exam content will go here */}
              <p>{exam.description}</p>
            </div>
          </div>
        </div>
        
        <div className="col-md-4">
          <div className="card">
            <div className="card-header">
              <h5>Proctoring</h5>
            </div>
            <div className="card-body">
              <video
                ref={videoRef}
                autoPlay
                muted
                className="w-100 mb-3"
                style={{ maxHeight: '200px' }}
              />
              {!sessionId ? (
                <button className="btn btn-primary w-100" onClick={startExam}>
                  Start Exam
                </button>
              ) : (
                <button className="btn btn-danger w-100" onClick={endExam}>
                  End Exam
                </button>
              )}
            </div>
          </div>
          
          {anomalies.length > 0 && (
            <div className="card mt-4">
              <div className="card-header">
                <h5>Detected Anomalies</h5>
              </div>
              <div className="card-body">
                <ul className="list-group">
                  {anomalies.map((anomaly, index) => (
                    <li key={index} className="list-group-item">
                      {anomaly}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Exam; 