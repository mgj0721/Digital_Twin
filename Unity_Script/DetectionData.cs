using System;

[Serializable]
public class MessageTypeEnvelope { public string type; }

[Serializable]
public class DetectionFrame
{
    public string type;
    public int schemaVersion;
    public string source;
    public long sentAtUnixMs;
    public int seq;
    public CameraSize camera;
    public Detection[] detections;
}

[Serializable]
public class Detection
{
    public string id;
    public string className;
    public string @class;
    public float confidence;
    public BoundingBox bbox;
    public SurfaceCoordinate surface;
}

[Serializable]
public class BoundingBox
{
    public float x;
    public float y;
    public float width;
    public float height;
    public float cx;
    public float cy;
}

[Serializable]
public class SurfaceCoordinate
{
    public string id;
    public float u;
    public float v;
}

[Serializable]
public class CameraSize
{
    public int width;
    public int height;
}

[Serializable]
public class ArucoFrame
{
    public string type;
    public int schemaVersion;
    public string source;
    public long sentAtUnixMs;
    public int seq;
    public CameraSize camera;
    public ArucoMarker[] markers;
}

[Serializable]
public class ArucoMarker
{
    public int id;
    public float confidence;
    public float cx;
    public float cy;
    public float[] corners;
    public string surfaceId;
    public float u;
    public float v;
}
