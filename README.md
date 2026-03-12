---
title: GeoVision Backend
emoji: 🌍
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# GeoVision Backend

Flask REST API backend for the GeoVision disaster risk prediction platform.

## Endpoints

- `GET /health` — service health check
- `GET /info` — API info & all route listing
- `POST /api/chat/message` — Gemini AI assistant
- `POST /api/hazardguard/predict` — disaster risk prediction
- `POST /api/weatherwise/forecast` — LSTM weather forecasting
- `POST /api/geovision/predict` — full fusion model prediction
- `POST /api/satellite/point` — GEE satellite imagery
- `GET /api/weather/data` — NASA POWER weather data

## Environment Variables

Set these in **Space Settings → Variables**:

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Google Gemini API key |
| `GEE_PROJECT_ID` | GCP project ID for Earth Engine |
| `GEE_SERVICE_ACCOUNT_KEY` | GEE service account JSON (single-line) |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service role key |
| `ALLOWED_ORIGINS` | Comma-separated CORS origins (your frontend URL) |
| `GCS_BUCKET_BASE_URL` | GCS public bucket URL for raster COG files |
