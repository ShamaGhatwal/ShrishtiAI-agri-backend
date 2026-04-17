"""State and Goa layer compatibility routes for the web dashboard."""
from __future__ import annotations

from typing import Any, Dict, List

from flask import Blueprint, jsonify

from utils import create_error_response, create_success_response

layers_bp = Blueprint("layers", __name__)


STATE_LAYERS: Dict[str, List[Dict[str, Any]]] = {
    "india": [
        {
            "id": "admin_boundaries",
            "title": "Administrative Boundaries",
            "description": "National and state boundaries for base reference.",
            "category": "boundaries",
            "icon": "map",
            "color": "#2563EB",
            "source": "GeoVision",
            "available": True,
            "file_size": 182304,
        },
        {
            "id": "major_rivers",
            "title": "Major Rivers",
            "description": "Primary river network across India.",
            "category": "hydrology",
            "icon": "waves",
            "color": "#0EA5E9",
            "source": "GeoVision",
            "available": True,
            "file_size": 96321,
        },
    ],
    "goa": [
        {
            "id": "water_bodies",
            "title": "Water Bodies",
            "description": "Key reservoirs, rivers, and wetlands in Goa.",
            "category": "environment",
            "icon": "droplets",
            "color": "#0284C7",
            "source": "AMCHE.IN / IndianOpenMaps",
            "available": True,
            "file_size": 45218,
        },
        {
            "id": "schools",
            "title": "Schools",
            "description": "Government and public school locations.",
            "category": "facilities",
            "icon": "school",
            "color": "#F59E0B",
            "source": "AMCHE.IN / IndianOpenMaps",
            "available": True,
            "file_size": 23810,
        },
    ],
    "karnataka": [
        {
            "id": "district_boundaries",
            "title": "District Boundaries",
            "description": "District-level boundaries for Karnataka.",
            "category": "boundaries",
            "icon": "map",
            "color": "#7C3AED",
            "source": "GeoVision",
            "available": True,
            "file_size": 112004,
        }
    ],
    "kerala": [
        {
            "id": "coastal_zones",
            "title": "Coastal Zones",
            "description": "Coastal influence zones for planning and alerts.",
            "category": "coastal",
            "icon": "waves",
            "color": "#059669",
            "source": "GeoVision",
            "available": True,
            "file_size": 79012,
        }
    ],
    "maharashtra": [
        {
            "id": "flood_prone_areas",
            "title": "Flood Prone Areas",
            "description": "Historical flood-prone polygons.",
            "category": "risk",
            "icon": "alert-triangle",
            "color": "#DC2626",
            "source": "GeoVision",
            "available": True,
            "file_size": 154009,
        }
    ],
    "tamilnadu": [
        {
            "id": "watershed_blocks",
            "title": "Watershed Blocks",
            "description": "Watershed management units.",
            "category": "hydrology",
            "icon": "grid-3x3",
            "color": "#0D9488",
            "source": "GeoVision",
            "available": True,
            "file_size": 120117,
        }
    ],
    "andhrapradesh": [
        {
            "id": "agri_zones",
            "title": "Agricultural Zones",
            "description": "Agriculture suitability zones.",
            "category": "agriculture",
            "icon": "landmark",
            "color": "#65A30D",
            "source": "GeoVision",
            "available": True,
            "file_size": 103884,
        }
    ],
}

GOA_CATEGORIES: List[Dict[str, Any]] = [
    {
        "id": "environment",
        "title": "Environment",
        "description": "Ecological and water resources.",
        "icon": "leaf",
        "color": "#22C55E",
    },
    {
        "id": "facilities",
        "title": "Facilities",
        "description": "Public facilities and institutions.",
        "icon": "school",
        "color": "#F59E0B",
    },
]


def _find_layer(state: str, layer_id: str) -> Dict[str, Any] | None:
    for layer in STATE_LAYERS.get(state, []):
        if layer.get("id") == layer_id:
            return layer
    return None


def _sample_feature_collection(state: str, layer: Dict[str, Any]) -> Dict[str, Any]:
    # Minimal valid GeoJSON so the map integration can render immediately.
    feature_collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [77.2090, 28.6139],
                },
                "properties": {
                    "state": state,
                    "layer_id": layer["id"],
                    "name": layer["title"],
                    "source": layer["source"],
                },
            }
        ],
        "_metadata": {
            "layer_id": layer["id"],
            "title": layer["title"],
            "description": layer["description"],
            "source": layer["source"],
            "color": layer["color"],
        },
        "_truncated": False,
    }
    return feature_collection


@layers_bp.route("/<state>/layers", methods=["GET"])
def get_layers(state: str):
    state_key = state.lower().strip()
    layers = STATE_LAYERS.get(state_key)
    if layers is None:
        return jsonify(create_error_response(f"Unknown state: {state}")), 404

    return jsonify(
        create_success_response(
            {
                "data": layers,
                "count": len(layers),
                "success": True,
            }
        )
    )


@layers_bp.route("/<state>/categories", methods=["GET"])
def get_categories(state: str):
    state_key = state.lower().strip()
    if state_key != "goa":
        return jsonify(create_error_response(f"Categories are only available for Goa, got: {state}")), 404

    layer_map: Dict[str, List[Dict[str, Any]]] = {}
    for layer in STATE_LAYERS["goa"]:
        layer_map.setdefault(layer["category"], []).append(layer)

    categories: List[Dict[str, Any]] = []
    for category in GOA_CATEGORIES:
        items = layer_map.get(category["id"], [])
        categories.append({**category, "layers": items, "layer_count": len(items)})

    return jsonify(
        create_success_response(
            {
                "data": categories,
                "count": len(categories),
                "success": True,
            }
        )
    )


@layers_bp.route("/<state>/layer/<layer_id>", methods=["GET"])
def get_layer_data(state: str, layer_id: str):
    state_key = state.lower().strip()
    layer = _find_layer(state_key, layer_id)
    if layer is None:
        return jsonify(create_error_response(f"Layer '{layer_id}' not found for state '{state}'")), 404

    data = _sample_feature_collection(state_key, layer)
    return jsonify(
        create_success_response(
            {
                "data": data,
                "success": True,
            }
        )
    )


@layers_bp.route("/<state>/layer/<layer_id>/stats", methods=["GET"])
def get_layer_stats(state: str, layer_id: str):
    state_key = state.lower().strip()
    layer = _find_layer(state_key, layer_id)
    if layer is None:
        return jsonify(create_error_response(f"Layer '{layer_id}' not found for state '{state}'")), 404

    stats = {
        "layer_id": layer["id"],
        "title": layer["title"],
        "feature_count": 1,
        "file_size_bytes": layer.get("file_size", 0),
        "file_size_mb": round(layer.get("file_size", 0) / (1024 * 1024), 3),
    }

    return jsonify(
        create_success_response(
            {
                "data": stats,
                "success": True,
            }
        )
    )
