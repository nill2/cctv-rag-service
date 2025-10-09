"""Response formatting utilities for MCP tools."""

from typing import Dict, Any, List


def format_face_data(face: Dict[str, Any]) -> Dict[str, Any]:
    """Format face detection data for MCP response."""
    return {
        "face_id": face.get("face_id"),
        "timestamp": face.get("timestamp"),
        "location": face.get("camera_location"),
        "confidence": round(face.get("confidence", 0), 2),
        "quality": (
            round(face.get("face_quality", 0), 2) if face.get("face_quality") else None
        ),
        "status": face.get("status"),
        "person_name": face.get("person_name"),
        "image_id": face.get("image_id"),
    }


def format_analytics_data(analytics: Dict[str, Any]) -> Dict[str, Any]:
    """Format analytics data for better readability."""
    formatted = {"summary": analytics.get("summary", {}), "insights": []}

    # Location insights
    location_dist = analytics.get("location_breakdown", [])
    if location_dist:
        busiest = location_dist[0]
        formatted["insights"].append(
            f"Busiest location: {busiest.get('_id')} with {busiest.get('face_count')} detections"
        )

    # Time insights
    daily_activity = analytics.get("daily_activity", [])
    if daily_activity:
        peak_day = max(daily_activity, key=lambda x: x.get("total_faces", 0))
        formatted["insights"].append(
            f"Peak activity on {peak_day.get('date')} with {peak_day.get('total_faces')} faces"
        )

    return formatted


def format_timeline_data(timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format timeline data for chronological display."""
    return [
        {
            "time": entry.get("timestamp"),
            "location": entry.get("location"),
            "confidence": round(entry.get("confidence", 0), 2),
            "event": f"Detected at {entry.get('location')} with {round(entry.get('confidence', 0) * 100)}% confidence",
        }
        for entry in timeline
    ]
