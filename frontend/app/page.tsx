"use client";

import { ChangeEvent, useEffect, useState } from "react";

type ClassProbability = {
  class_index: number;
  class_name: string;
  probability: number;
};

type Prediction = {
  predicted_class: string;
  predicted_class_index: number;
  confidence: number;
  probabilities: ClassProbability[];
  model_version: string;
  request_id: string;
  filename?: string | null;
  content_type?: string | null;
};

type Explanation = {
  explained_class_index: number;
  explained_class: string;
  class_probability: number;
  method: string;
  target_layer: string;
  model_version: string;
  warning?: string | null;
  metadata: Record<string, unknown>;
  heatmap_png_base64: string;
  overlay_png_base64: string;
  request_id: string;
};

type Gate = {
  method: string;
  target_layer: string;
  model_version: string;
  warning?: string | null;
  metadata: Record<string, unknown>;
  gate_png_base64: string;
  request_id: string;
};

type Analysis = {
  prediction: Prediction;
  explanation: Explanation;
  asfr_gate: Gate;
  request_id: string;
};

const MAX_FILE_BYTES = 10 * 1024 * 1024;
const MAX_IMAGE_PIXELS = 20_000_000;

const ACCEPTED_TYPES = [
  "image/png",
  "image/jpeg",
  "image/bmp",
  "image/gif",
  "image/webp",
];

const API_BASE_URL = (
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000"
).replace(/\/+$/, "");

function percent(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

function imageData(base64: string): string {
  return `data:image/png;base64,${base64}`;
}

async function validateImage(file: File): Promise<string | null> {
  if (file.size === 0) return "The selected file is empty.";
  if (file.size > MAX_FILE_BYTES) return "The image exceeds the 10 MiB upload limit.";

  const extension = file.name.split(".").pop()?.toLowerCase();
  const allowedExtensions = ["png", "jpg", "jpeg", "bmp", "gif", "webp"];

  if (
    !ACCEPTED_TYPES.includes(file.type) &&
    !(allowedExtensions.includes(extension ?? "") && file.type === "")
  ) {
    return "Unsupported image format. Choose PNG, JPEG, BMP, GIF, or WebP.";
  }

  try {
    const bitmap = await createImageBitmap(file);
    const pixels = bitmap.width * bitmap.height;
    bitmap.close();

    if (pixels > MAX_IMAGE_PIXELS) {
      return "The image exceeds the 20-megapixel limit.";
    }
  } catch {
    return "The selected file could not be decoded as an image.";
  }

  return null;
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [showGate, setShowGate] = useState(true);
  const [status, setStatus] = useState("Ready for an image.");

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }

    const url = URL.createObjectURL(file);
    setPreviewUrl(url);

    return () => URL.revokeObjectURL(url);
  }, [file]);

  async function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0] ?? null;
    event.target.value = "";

    setAnalysis(null);
    setError(null);
    setFile(null);

    if (!selected) {
      setStatus("Ready for an image.");
      return;
    }

    setStatus("Validating imageâ€¦");
    const validationError = await validateImage(selected);

    if (validationError) {
      setError(validationError);
      setStatus("Image validation failed.");
      return;
    }

    setFile(selected);
    setStatus("Image validated. Ready to analyze.");
  }

  async function analyzeImage() {
    if (!file || busy) return;

    setBusy(true);
    setError(null);
    setAnalysis(null);
    setStatus("Sending image to the inference and explanation APIâ€¦");

    const formData = new FormData();
    formData.append("file", file);

    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 70_000);

    try {
      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });

      const payload = await response.json().catch(() => null);

      if (!response.ok) {
        const detail =
          typeof payload?.detail === "string"
            ? payload.detail
            : `The API returned HTTP ${response.status}.`;
        const requestId =
          typeof payload?.request_id === "string"
            ? ` Request ID: ${payload.request_id}`
            : "";
        throw new Error(`${detail}${requestId}`);
      }

      if (
        !payload?.prediction ||
        !payload?.explanation?.overlay_png_base64 ||
        !payload?.explanation?.heatmap_png_base64 ||
        !payload?.asfr_gate?.gate_png_base64 ||
        !Array.isArray(payload?.prediction?.probabilities)
      ) {
        throw new Error("The API response is missing required analysis fields.");
      }

      setAnalysis(payload as Analysis);
      setStatus("Analysis completed.");
    } catch (cause) {
      if (cause instanceof DOMException && cause.name === "AbortError") {
        setError(
          "The request exceeded the 70-second frontend limit. The backend may still be processing it; check backend health before retrying."
        );
      } else if (cause instanceof TypeError) {
        setError(
          `Could not reach the backend at ${API_BASE_URL}. Confirm that the API is running and that its CORS configuration permits this frontend origin.`
        );
      } else {
        setError(
          cause instanceof Error ? cause.message : "An unexpected error occurred."
        );
      }
      setStatus("Analysis failed.");
    } finally {
      window.clearTimeout(timeout);
      setBusy(false);
    }
  }

  const prediction = analysis?.prediction;
  const explanation = analysis?.explanation;
  const gate = analysis?.asfr_gate;

  return (
    <main className="site-shell">
      <header className="topbar">
        <div className="brand-mark" aria-hidden="true">ASFR</div>
        <div>
          <p className="eyebrow">RESEARCH PROTOTYPE</p>
          <p className="brand-name">Rice Leaf Analysis</p>
        </div>
        <span className="status-pill">
          <span className="status-dot" aria-hidden="true" />
          Image classification
        </span>
      </header>

      <section className="hero">
        <p className="eyebrow">ASYMMETRIC SPATIAL FEATURE ROUTING</p>
        <h1>Explore a rice leaf image.</h1>
        <p className="hero-copy">
          Submit a leaf image to obtain a nine-class prediction, a Grad-CAM
          explanation, and a visualization of the ASFR spatial gate.
        </p>
      </section>

      <section className="workspace" aria-label="Rice leaf analysis workspace">
        <div className="panel upload-panel">
          <div className="section-heading">
            <span className="step-number">01</span>
            <div>
              <h2>Input image</h2>
              <p>Upload a supported image file.</p>
            </div>
          </div>

          <label className="upload-zone" htmlFor="leaf-upload">
            {previewUrl ? (
              <img
                className="preview-image"
                src={previewUrl}
                alt="Preview of the selected rice leaf image"
              />
            ) : (
              <span className="upload-placeholder">
                <span className="upload-icon" aria-hidden="true">â†‘</span>
                <strong>Choose a leaf image</strong>
                <span>PNG, JPEG, BMP, GIF, or WebP</span>
                <span>Maximum 10 MiB Â· 20 megapixels</span>
              </span>
            )}
          </label>
          <input
            id="leaf-upload"
            className="file-input"
            type="file"
            accept="image/png,image/jpeg,image/bmp,image/gif,image/webp"
            onChange={handleFileChange}
            disabled={busy}
            aria-describedby="upload-help"
          />

          <p id="upload-help" className="file-details">
            {file
              ? `${file.name} Â· ${(file.size / (1024 * 1024)).toFixed(2)} MiB`
              : "Your image is sent to the backend for analysis."}
          </p>

          <button
            className="primary-button"
            type="button"
            onClick={analyzeImage}
            disabled={!file || busy}
          >
            {busy ? (
              <>
                <span className="spinner" aria-hidden="true" />
                Analyzing imageâ€¦
              </>
            ) : analysis ? (
              "Analyze again"
            ) : (
              "Analyze leaf"
            )}
          </button>

          <p className="status-message" role="status" aria-live="polite">
            {status}
          </p>

          {error && (
            <div className="error-panel" role="alert">
              <strong>Analysis unsuccessful</strong>
              <p>{error}</p>
              {file && (
                <button
                  className="secondary-button"
                  type="button"
                  onClick={analyzeImage}
                  disabled={busy}
                >
                  Retry analysis
                </button>
              )}
            </div>
          )}
        </div>

        <div className="panel result-panel" aria-live="polite">
          <div className="section-heading">
            <span className="step-number">02</span>
            <div>
              <h2>Prediction</h2>
              <p>Model output and class probabilities.</p>
            </div>
          </div>

          {prediction ? (
            <>
              <p className="result-label">Predicted class</p>
              <h3 className="disease-name">{prediction.predicted_class}</h3>

              <div className="confidence-row">
                <span>Predicted probability</span>
                <strong>{percent(prediction.confidence)}</strong>
              </div>
              <div
                className="confidence-track"
                role="progressbar"
                aria-label="Predicted class probability"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={prediction.confidence * 100}
              >
                <span style={{ width: `${prediction.confidence * 100}%` }} />
              </div>

              <div className="divider" />
              <h3 className="subheading">All nine classes</h3>

              <div className="probability-list">
                {[...prediction.probabilities]
                  .sort((a, b) => b.probability - a.probability)
                  .map((item) => (
                    <div className="probability-item" key={item.class_index}>
                      <div className="probability-label">
                        <span>{item.class_name}</span>
                        <strong>{percent(item.probability)}</strong>
                      </div>
                      <div className="probability-track">
                        <span
                          style={{ width: `${item.probability * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
              </div>

              <div className="metadata">
                <div>
                  <span>Model version</span>
                  <code>{prediction.model_version}</code>
                </div>
                <div>
                  <span>Request ID</span>
                  <code>{analysis?.request_id}</code>
                </div>
              </div>
            </>
          ) : (
            <div className="empty-state">
              <span className="empty-symbol" aria-hidden="true">â—Ž</span>
              <h3>Results will appear here</h3>
              <p>
                Select an image and run the analysis to view the predicted
                class and its probability distribution.
              </p>
            </div>
          )}
        </div>
      </section>

      {analysis && explanation && gate && (
        <section className="panel explain-panel">
          <div className="section-heading">
            <span className="step-number">03</span>
            <div>
              <h2>Visual explanation</h2>
              <p>
                Compare the input image with the model-generated visualizations.
              </p>
            </div>
          </div>

          <div className="visual-grid">
            <figure className="visual-card">
              <div className="visual-image-wrap">
                {previewUrl && (
                  <img src={previewUrl} alt="Original uploaded rice leaf" />
                )}
              </div>
              <figcaption>
                <strong>Original image</strong>
                <span>Uploaded input</span>
              </figcaption>
            </figure>

            <figure className="visual-card">
              <div className="visual-image-wrap">
                <img
                  src={imageData(explanation.overlay_png_base64)}
                  alt={`Grad-CAM overlay for ${explanation.explained_class}`}
                />
              </div>
              <figcaption>
                <strong>Grad-CAM overlay</strong>
                <span>Class: {explanation.explained_class}</span>
              </figcaption>
            </figure>

            <figure className="visual-card">
              <div className="visual-image-wrap">
                <img
                  src={imageData(explanation.heatmap_png_base64)}
                  alt="Grad-CAM heatmap"
                />
              </div>
              <figcaption>
                <strong>Grad-CAM heatmap</strong>
                <span>{explanation.method}</span>
              </figcaption>
            </figure>

            <figure className="visual-card">
              <div className="visual-image-wrap">
                <img
                  src={imageData(gate.gate_png_base64)}
                  alt="ASFR spatial gate visualization"
                />
              </div>
              <figcaption>
                <strong>ASFR spatial gate</strong>
                <span>{gate.target_layer}</span>
              </figcaption>
            </figure>
          </div>

          <div className="explanation-details">
            <div>
              <p className="result-label">Explanation method</p>
              <p><strong>{explanation.method}</strong></p>
              <p className="muted">
                Target layer: <code>{explanation.target_layer}</code>
              </p>
            </div>
            <label className="toggle-row">
              <input
                type="checkbox"
                checked={showGate}
                onChange={(event) => setShowGate(event.target.checked)}
              />
              Show ASFR gate details
            </label>
          </div>

          {showGate && (
            <div className="gate-details">
              <p>
                <strong>Gate method:</strong> {gate.method}
              </p>
              <p>
                <strong>Gate target layer:</strong>{" "}
                <code>{gate.target_layer}</code>
              </p>
              <details>
                <summary>Gate metadata</summary>
                <pre>{JSON.stringify(gate.metadata, null, 2)}</pre>
              </details>
            </div>
          )}

          {(explanation.warning || gate.warning) && (
            <div className="notice-panel">
              <strong>Interpretation notes</strong>
              {explanation.warning && <p>{explanation.warning}</p>}
              {gate.warning && <p>{gate.warning}</p>}
            </div>
          )}

          <div className="notice-panel">
            <strong>Research disclaimer</strong>
            <p>
              Grad-CAM highlights features associated with the selected
              prediction; it does not establish causal reasoning. The ASFR
              gate visualizes spatial gating and is not itself a
              class-specific explanation. This research prototype is not a
              substitute for expert plant-disease assessment. Benchmark
              performance does not establish robustness under field conditions.
            </p>
          </div>
        </section>
      )}

      <footer className="footer">
        <span>ASFR Â· Rice disease classification</span>
        <span>Research prototype Â· Not a diagnostic service</span>
      </footer>
    </main>
  );
}
