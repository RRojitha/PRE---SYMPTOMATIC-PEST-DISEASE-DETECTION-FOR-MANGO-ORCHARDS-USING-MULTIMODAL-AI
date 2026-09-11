import "../styles/ResultCard.css";

export default function ResultCard({ result }) {

    if (!result) return null;

    return (

        <div className="result-card">

            <h2>Prediction Result</h2>

            <div className="row">
                <span>Disease</span>
                <strong>{result.disease}</strong>
            </div>

            <div className="row">
                <span>Status</span>
                <strong>{result.status}</strong>
            </div>

           

        </div>

    );

}