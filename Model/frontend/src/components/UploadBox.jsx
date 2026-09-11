import { useState } from "react";
import axios from "axios";

import "../styles/UploadBox.css";
import ResultCard from "./ResultCard";

export default function UploadBox() {

    const [image, setImage] = useState(null);
    const [file, setFile] = useState(null);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);

    // Environmental readings for the fusion model.
    const [temperature, setTemperature] = useState("26.5");
    const [humidity, setHumidity] = useState("80");
    const [windSpeed, setWindSpeed] = useState("8");

    function handleChange(e) {

        const selectedFile = e.target.files[0];

        if (!selectedFile) return;

        setFile(selectedFile);
        setImage(URL.createObjectURL(selectedFile));
        setResult(null);
    }

    async function predict() {

        if (!file) {
            alert("Please upload an image.");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);
        formData.append("temperature_c", temperature);
        formData.append("humidity_percent", humidity);
        formData.append("wind_speed_kmh", windSpeed);

        try {

            setLoading(true);

            const response = await axios.post(
                "http://127.0.0.1:8000/predict-fusion",
                formData,
                {
                    headers: {
                        "Content-Type": "multipart/form-data",
                    },
                }
            );

            setResult(response.data);

        } catch (error) {

            console.error(error);

            const message = error.response?.data?.detail || "Prediction failed.";

            alert(message);

        } finally {

            setLoading(false);

        }

    }

    return (
    <div className="upload-container">

        <label className="upload-box">

            <input
                type="file"
                hidden
                accept="image/*"
                onChange={handleChange}
            />

            {!image ? (

                <>

                    <div className="upload-icon">☁️</div>

                    <h2>Upload Mango Leaf Image</h2>

                    <p>
                        Drag & Drop an image here or click to browse
                    </p>

                    <span>Supports JPG, PNG, JPEG</span>

                </>

            ) : (

                <>

                    <img
                        src={image}
                        alt="Preview"
                        className="preview-image"
                    />

                    <div className="image-status">

                        ✅ Image loaded successfully

                    </div>

                </>

            )}

        </label>

        <div className="env-inputs">

            <label>
                Temperature (°C)
                <input
                    type="number"
                    step="0.1"
                    value={temperature}
                    onChange={(e) => setTemperature(e.target.value)}
                />
            </label>

            <label>
                Humidity (%)
                <input
                    type="number"
                    step="0.1"
                    value={humidity}
                    onChange={(e) => setHumidity(e.target.value)}
                />
            </label>

            <label>
                Wind Speed (km/h)
                <input
                    type="number"
                    step="0.1"
                    value={windSpeed}
                    onChange={(e) => setWindSpeed(e.target.value)}
                />
            </label>

        </div>

        <button
            className="predict-btn"
            onClick={predict}
            disabled={loading}
        >
            {loading ? "Predicting..." : "🌿 Predict"}
        </button>

        <ResultCard result={result}/>

    </div>
);
}