"""
Flask Application

REST API for retail demand forecasting.
"""

import os
import sys
import time
from datetime import datetime
from typing import Dict

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from src.exception import APIException, CustomException
from src.logger import get_logger
from src.pipeline.predict_pipeline import PredictionPipeline
from src.utils import load_config

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize logger
logger = get_logger("FlaskAPI")

# Load configuration
try:
    config = load_config()
    api_config = config.get("api", {})
except Exception as e:
    logger.warning(f"Could not load config: {e}")
    api_config = {}

# Initialize prediction pipeline (lazy loading)
prediction_pipeline = None


def get_pipeline() -> PredictionPipeline:
    """Get or create prediction pipeline (lazy loading)."""
    global prediction_pipeline
    if prediction_pipeline is None:
        prediction_pipeline = PredictionPipeline()
    return prediction_pipeline


@app.route("/")
def home():
    """Home page with forecast form."""
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    """Return empty response for favicon to prevent 404."""
    return "", 204


@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
    })


@app.route("/forecast", methods=["POST"])
def forecast():
    """
    Generate forecast from form submission.
    
    Form fields:
    - store_id: Store ID
    - product_id: Product ID
    - horizon: Forecast horizon (7, 14, or 30 days)
    """
    try:
        start_time = time.time()
        
        # Get form data
        store_id = int(request.form.get("store_id", 1))
        product_id = int(request.form.get("product_id", 1))
        horizon = int(request.form.get("horizon", 7))
        
        # Validate store_id (config has 50 stores)
        if store_id < 1 or store_id > 50:
            return render_template("forecast.html", error="Invalid store ID. Must be between 1 and 50.")
        
        # Validate product_id (config has 1000 products)
        if product_id < 1 or product_id > 1000:
            return render_template("forecast.html", error="Invalid product ID. Must be between 1 and 1000.")
        
        # Validate horizon
        if horizon not in [7, 14, 30]:
            horizon = 7
        
        # Generate prediction
        pipeline = get_pipeline()
        result = pipeline.predict(
            store_id=store_id,
            product_id=product_id,
            horizon=horizon,
        )
        
        # Add latency
        latency_ms = (time.time() - start_time) * 1000
        result["latency_ms"] = round(latency_ms, 2)
        
        logger.info(
            f"Forecast generated: store={store_id}, product={product_id}, "
            f"horizon={horizon}, latency={latency_ms:.2f}ms"
        )
        
        return render_template("forecast.html", result=result)
        
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return render_template("forecast.html", error=f"Invalid input: {e}")
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        return render_template("forecast.html", error=str(e))


@app.route("/api/forecast/<int:store_id>/<int:product_id>/<int:horizon>")
def api_forecast(store_id: int, product_id: int, horizon: int):
    """
    REST API endpoint for generating forecasts.
    
    Args:
        store_id: Store ID.
        product_id: Product ID.
        horizon: Forecast horizon (7, 14, or 30 days).
        
    Returns:
        JSON response with forecast.
    """
    try:
        start_time = time.time()
        
        # Validate store_id (config has 50 stores)
        if store_id < 1 or store_id > 50:
            return jsonify({
                "error": "Invalid store ID. Must be between 1 and 50.",
            }), 400
        
        # Validate product_id (config has 1000 products)
        if product_id < 1 or product_id > 1000:
            return jsonify({
                "error": "Invalid product ID. Must be between 1 and 1000.",
            }), 400
        
        # Validate horizon
        if horizon not in [7, 14, 30]:
            return jsonify({
                "error": "Invalid horizon. Must be 7, 14, or 30 days.",
                "valid_horizons": [7, 14, 30],
            }), 400
        
        # Generate prediction
        pipeline = get_pipeline()
        result = pipeline.predict(
            store_id=store_id,
            product_id=product_id,
            horizon=horizon,
        )
        
        # Add metadata
        latency_ms = (time.time() - start_time) * 1000
        result["latency_ms"] = round(latency_ms, 2)
        result["timestamp"] = datetime.now().isoformat()
        
        # Check latency target
        target_latency = api_config.get("target_latency_ms", 500)
        result["within_target"] = latency_ms <= target_latency
        
        logger.info(
            f"API forecast: store={store_id}, product={product_id}, "
            f"horizon={horizon}, latency={latency_ms:.2f}ms"
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/forecast/batch", methods=["POST"])
def api_forecast_batch():
    """
    Batch forecast endpoint.
    
    Request body (JSON):
    {
        "requests": [
            {"store_id": 1, "product_id": 1, "horizon": 7},
            {"store_id": 1, "product_id": 2, "horizon": 14},
            ...
        ]
    }
    
    Returns:
        JSON array of forecasts.
    """
    try:
        start_time = time.time()
        
        data = request.get_json()
        if not data or "requests" not in data:
            return jsonify({"error": "Missing 'requests' in request body"}), 400
        
        requests_list = data["requests"]
        if len(requests_list) > 100:
            return jsonify({"error": "Maximum 100 requests per batch"}), 400
        
        # Generate predictions
        pipeline = get_pipeline()
        results = pipeline.predict_batch(requests_list)
        
        latency_ms = (time.time() - start_time) * 1000
        
        logger.info(
            f"Batch forecast: {len(requests_list)} requests, "
            f"latency={latency_ms:.2f}ms"
        )
        
        return jsonify({
            "results": results,
            "count": len(results),
            "latency_ms": round(latency_ms, 2),
            "timestamp": datetime.now().isoformat(),
        })
        
    except Exception as e:
        logger.error(f"Batch API error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/models")
def api_models():
    """Get available models and their performance."""
    try:
        from pathlib import Path
        import json
        
        models_dir = Path("artifacts/models")
        models = []
        
        for model_file in models_dir.glob("*_model.joblib"):
            model_name = model_file.stem.replace("_model", "")
            models.append({
                "name": model_name,
                "path": str(model_file),
            })
        
        return jsonify({
            "models": models,
            "default_model": api_config.get("default_model", "lightgbm"),
        })
        
    except Exception as e:
        logger.error(f"Models API error: {e}")
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({"error": "Internal server error"}), 500


def create_app():
    """Application factory."""
    return app


if __name__ == "__main__":
    host = api_config.get("host", "0.0.0.0")
    port = api_config.get("port", 5000)
    debug = api_config.get("debug", False)  # Default to False for security
    
    logger.info(f"Starting Flask API on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
