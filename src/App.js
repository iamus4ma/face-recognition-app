import React, { useState, useRef, useEffect } from 'react';
import * as faceapi from 'face-api.js';
import './App.css';
import 'webrtc-adapter';

function App() {
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [loading, setLoading] = useState(false);
  const [referencePhoto, setReferencePhoto] = useState(null);
  console.log(referencePhoto, "referencePhoto");
  const [referenceDescriptor, setReferenceDescriptor] = useState(null);
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [match, setMatch] = useState(null);
  const [similarity, setSimilarity] = useState(0);

  const videoRef = useRef();
  const canvasRef = useRef();
  const imageUploadRef = useRef();
  const streamRef = useRef(null);

  // Load face-api.js models
  useEffect(() => {
    const loadModels = async () => {
      setLoading(true);
      try {
        const MODEL_URL = '/models';

        // Load all required models
        await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
        await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
        await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);

        setModelsLoaded(true);
        console.log('Models loaded successfully');
      } catch (error) {
        console.error('Failed to load models:', error);
      } finally {
        setLoading(false);
      }
    };

    loadModels();
  }, []);

  // Handle camera on/off
  useEffect(() => {
    if (isCameraOn) {
      startVideo();
    } else {
      stopVideo();
    }

    return () => {
      stopVideo();
    };
  }, [isCameraOn]);

  const startVideo = () => {
    // Add this check
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert('Camera API not available in your browser');
      setIsCameraOn(false);
      return;
    }

    navigator.mediaDevices.getUserMedia({
      video: {
        width: 640,
        height: 480,
        facingMode: 'user' // Front camera
      }
    })
      .then(stream => {
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      })
      .catch(err => {
        console.error('Camera error:', err);
        alert(`Cannot access camera: ${err.message}`);
        setIsCameraOn(false);
      });
  };
  const stopVideo = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  const handleImageUpload = async (e) => {
    if (!modelsLoaded) {
      alert('Models are still loading. Please wait.');
      return;
    }

    const file = e.target.files[0];
    if (!file) return;

    try {
      const image = await faceapi.bufferToImage(file);
      setReferencePhoto(image);

      // Get face descriptor from uploaded image
      const detection = await faceapi
        .detectSingleFace(image, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks()
        .withFaceDescriptor();


      if (detection) {
        setReferenceDescriptor(detection.descriptor);
      } else {
        alert('No face detected in the uploaded image');
      }
    } catch (error) {
      console.error('Error processing image:', error);
      alert('Error processing image. Please try another photo.');
    }
  };

  const capturePhoto = async () => {
    if (!videoRef.current || !modelsLoaded) {
      alert('Models not loaded or camera not ready');
      return;
    }

    try {
      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      canvas.getContext('2d').drawImage(videoRef.current, 0, 0);

      const image = new Image();
      image.src = canvas.toDataURL('image/png');
      await new Promise(resolve => image.onload = resolve);

      setReferencePhoto(image);

      // Get face descriptor from captured image
      const detection = await faceapi
        .detectSingleFace(videoRef.current, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks()
        .withFaceDescriptor();

      if (detection) {
        setReferenceDescriptor(detection.descriptor);
      } else {
        alert('No face detected in the captured image');
      }
    } catch (error) {
      console.error('Error capturing photo:', error);
      alert('Error capturing photo. Please try again.');
    }
  };

  const processVideo = async () => {
    if (!videoRef.current || !referenceDescriptor || !modelsLoaded) return;

    try {
      const canvas = canvasRef.current;
      const displaySize = {
        width: videoRef.current.videoWidth,
        height: videoRef.current.videoHeight
      };

      faceapi.matchDimensions(canvas, displaySize);

      const detections = await faceapi
        .detectAllFaces(videoRef.current, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks()
        .withFaceDescriptors();

      const resizedDetections = faceapi.resizeResults(detections, displaySize);

      // Clear canvas
      canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);

      let bestMatch = { distance: Infinity };

      resizedDetections.forEach(detection => {
        const distance = faceapi.euclideanDistance(referenceDescriptor, detection.descriptor);

        // Update best match
        if (distance < bestMatch.distance) {
          bestMatch = { distance, detection };
        }

        // Draw face box
        const box = detection.detection.box;
        const drawBox = new faceapi.draw.DrawBox(box, {
          label: `Distance: ${distance.toFixed(2)}`,
          boxColor: distance < 0.5 ? 'green' : distance < 0.6 ? 'orange' : 'red'
        });
        drawBox.draw(canvas);
      });

      // Update match state
      setMatch(bestMatch.distance < 0.6);
      setSimilarity(Math.max(0, 100 - (bestMatch.distance * 100)));

      // Continue processing
      if (isCameraOn) {
        requestAnimationFrame(processVideo);
      }
    } catch (error) {
      console.error('Error processing video frame:', error);
      if (isCameraOn) {
        requestAnimationFrame(processVideo);
      }
    }
  };

  useEffect(() => {
    if (isCameraOn && referenceDescriptor && modelsLoaded) {
      processVideo();
    }
  }, [isCameraOn, referenceDescriptor, modelsLoaded]);

  return (
    <div className="app">
      <h1>Face Recognition App</h1>
      <p>All processing happens in your browser - no images are sent to any server.</p>

      {loading && <div className="loading">Loading models... Please wait.</div>}

      <div className="controls">
        <div className="upload-section">
          <input
            type="file"
            accept="image/*"
            ref={imageUploadRef}
            onChange={handleImageUpload}
            style={{ display: 'none' }}
          />
          <button onClick={() => imageUploadRef.current.click()} disabled={!modelsLoaded || loading}>
            Upload Reference Photo
          </button>
          <button onClick={capturePhoto} disabled={!isCameraOn || !modelsLoaded || loading}>
            Take Reference Photo
          </button>
        </div>

        <button
          onClick={() => setIsCameraOn(!isCameraOn)}
          disabled={!modelsLoaded || loading || !referenceDescriptor}
        >
          {isCameraOn ? 'Stop Camera' : 'Start Camera'}
        </button>
      </div>

      <div className="content">
        <div className="reference">
          <h2>Reference Photo</h2>
          {referencePhoto ? (
            <img src={referencePhoto.src} alt="Reference" />
          ) : (
            <div className="placeholder">No reference photo selected</div>
          )}
        </div>

        <div className="camera">
          <h2>Live Camera</h2>
          <div className="video-container">
            <video
              ref={videoRef}
              autoPlay
              muted
              playsInline
              style={{ display: isCameraOn ? 'block' : 'none' }}
            />
            <canvas ref={canvasRef} />
            {!isCameraOn && <div className="placeholder">Camera is off</div>}
          </div>
        </div>
      </div>

      {match !== null && (
        <div className={`result ${match ? 'match' : 'no-match'}`}>
          {match ? (
            <p>✅ Match found! Similarity: {similarity.toFixed(1)}%</p>
          ) : (
            <p>❌ No match found. Similarity: {similarity.toFixed(1)}%</p>
          )}
        </div>
      )}
    </div>
  );
}

export default App;