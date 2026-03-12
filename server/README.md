# GEO VISION Backend

Professional Flask backend with MVC architecture for satellite data analysis and disaster monitoring.

## 🏗️ Architecture

```
backend/
├── main.py                 # Main application entry point
├── requirements.txt        # Python dependencies
├── .env                   # Environment configuration
├── config/                # Configuration management
│   └── __init__.py        # Config classes and validation
├── services/              # External integrations
│   ├── __init__.py
│   ├── gee_service.py     # Google Earth Engine service
│   └── ai_service.py      # Gemini AI service
├── controllers/           # Business logic layer
│   ├── __init__.py
│   ├── chat_controller.py # Chat operations
│   └── satellite_controller.py # Satellite data operations
├── views/                 # API routes and endpoints
│   ├── __init__.py
│   ├── chat_routes.py     # Chat API endpoints
│   └── satellite_routes.py # Satellite API endpoints
├── models/                # Data models and schemas
│   └── __init__.py        # Data classes and models
└── utils/                 # Utility functions
    └── __init__.py        # Common utilities and helpers
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure Environment

Copy and edit the `.env` file with your credentials:

```bash
# Google Earth Engine
GEE_PROJECT_ID=your-gee-project-id

# Gemini AI
GEMINI_API_KEY=your-gemini-api-key

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
FLASK_HOST=127.0.0.1
FLASK_PORT=5000

# Application
APP_NAME=GEO VISION Backend
APP_VERSION=1.0.0
APP_USER=MEWTROS

# CORS
ALLOWED_ORIGINS=http://localhost:3000

# Logging
LOG_LEVEL=INFO
LOG_FILE=geovision.log
```

### 3. Run the Server

```bash
python main.py
```

The server will start at `http://127.0.0.1:5000`

## 📡 API Endpoints

### System Endpoints

- `GET /` - Welcome message and endpoint overview
- `GET /health` - Health check for all services
- `GET /info` - Application information and API documentation

### Chat API (`/api/chat/`)

- `POST /message` - Send chat message with optional context
- `POST /analyze` - Analyze location for disaster indicators
- `GET /disaster/<type>` - Get disaster-specific information
- `GET /health` - Chat service health check

### Satellite API (`/api/satellite/`)

- `GET|POST /point` - Get satellite data for specific coordinates
- `POST /region` - Get satellite data for a region (polygon)
- `GET|POST /availability` - Check data availability for a location
- `GET /status` - Satellite service status
- `GET /collections` - List available satellite collections

## 💬 Chat API Usage

### Send Message

```bash
curl -X POST http://localhost:5000/api/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What can you tell me about flood detection using satellites?",
    "context": {
      "location": {
        "latitude": 12.34,
        "longitude": 56.78
      }
    }
  }'
```

### Analyze Location

```bash
curl -X POST http://localhost:5000/api/chat/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 12.34,
    "longitude": 56.78,
    "days_back": 30,
    "query": "Check for flood indicators"
  }'
```

## 🛰️ Satellite API Usage

### Get Point Data

```bash
# GET version
curl "http://localhost:5000/api/satellite/point?latitude=12.34&longitude=56.78&start_date=2024-01-01&end_date=2024-01-31"

# POST version
curl -X POST http://localhost:5000/api/satellite/point \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 12.34,
    "longitude": 56.78,
    "start_date": "2024-01-01",
    "end_date": "2024-01-31",
    "cloud_filter": 20
  }'
```

### Get Region Data

```bash
curl -X POST http://localhost:5000/api/satellite/region \
  -H "Content-Type: application/json" \
  -d '{
    "bounds": [
      [-122.5, 37.7],
      [-122.4, 37.8],
      [-122.3, 37.7]
    ],
    "start_date": "2024-01-01",
    "end_date": "2024-01-31",
    "scale": 10
  }'
```

### Check Availability

```bash
curl "http://localhost:5000/api/satellite/availability?latitude=12.34&longitude=56.78&days_back=30"
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEE_PROJECT_ID` | Google Earth Engine project ID | Required |
| `GEMINI_API_KEY` | Gemini AI API key | Required |
| `FLASK_ENV` | Flask environment | `development` |
| `FLASK_DEBUG` | Enable debug mode | `True` |
| `FLASK_HOST` | Server host | `127.0.0.1` |
| `FLASK_PORT` | Server port | `5000` |
| `ALLOWED_ORIGINS` | CORS allowed origins | `http://localhost:3000` |
| `LOG_LEVEL` | Logging level | `INFO` |

### Satellite Collections

Supported satellite collections:
- `COPERNICUS/S2_SR` - Sentinel-2 Level-2A (default)
- `COPERNICUS/S2` - Sentinel-2 Level-1C
- `LANDSAT/LC08/C02/T1_L2` - Landsat 8 Collection 2

## 🏥 Health Monitoring

Check service health:

```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "healthy",
  "services": {
    "gee": "healthy",
    "ai": "healthy"
  },
  "version": "1.0.0",
  "environment": "development"
}
```

## 🔍 Error Handling

All endpoints return standardized error responses:

```json
{
  "error": "Error description",
  "status": "error",
  "timestamp": "2024-01-15T10:30:00",
  "details": {
    "additional": "context"
  }
}
```

## 🚀 Production Deployment

### Using Gunicorn

```bash
gunicorn -w 4 -b 0.0.0.0:5000 main:create_app()
```

### Using Waitress (Windows)

```bash
waitress-serve --host=0.0.0.0 --port=5000 main:create_app()
```

## 📝 Logging

Logs are configured based on `LOG_LEVEL` and optionally written to `LOG_FILE`. 

Available log levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

## 🔐 Security Notes

- Never commit `.env` files to version control
- Use environment-specific configuration for production
- Consider API rate limiting for production deployments
- Implement authentication for production use

## 📚 Development

### Adding New Endpoints

1. Create/modify controller in `controllers/`
2. Add routes in `views/`
3. Register blueprint in `main.py`
4. Update this documentation

### Adding New Services

1. Create service class in `services/`
2. Initialize in `main.py` `initialize_services()`
3. Inject into controllers as needed

## 🤝 Integration

This backend is designed to work with:
- Next.js frontend dashboard
- React/TypeScript components
- RESTful API clients
- Satellite data analysis tools

## 📞 Support

For issues or questions:
- Check the logs for detailed error information
- Verify environment configuration
- Ensure Google Earth Engine authentication
- Confirm Gemini AI API key validity