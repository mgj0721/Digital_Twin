// SurfaceMarkerMapper.cs
//
// aruco_surface_mapper.py가 Jetson 쪽에서 붙여 보낸 detection.surface.id/u/v를
// 3D 평면(우드락 / 선박 패널) 위의 실제 월드 좌표로 변환해 균열 Prefab을 배치한다.
using System.Collections.Generic;
using UnityEngine;

public class SurfaceMarkerMapper : MonoBehaviour
{
    [System.Serializable]
    public class SurfacePlane
    {
        public string surfaceId;
        public Transform origin;
        public Transform uAxisEnd;
        public Transform vAxisEnd;
    }

    [Header("Data Source")]
    [SerializeField] private JetsonWebSocketReceiver receiver;

    [Header("Surface Planes (surface.id별로 등록, 표면 수만큼 추가)")]
    [SerializeField] private List<SurfacePlane> surfacePlanes = new List<SurfacePlane>();

    [Header("Marker")]
    [SerializeField] private Transform markerPrefab;
    [SerializeField] private float markerSurfaceOffset = 0.01f;
    [SerializeField, Range(0f, 1f)] private float minimumConfidence = 0.70f;

    private readonly Dictionary<string, Transform> activeMarkers = new Dictionary<string, Transform>();
    private readonly HashSet<string> visibleKeys = new HashSet<string>();
    private Dictionary<string, SurfacePlane> planeLookup;

    private void OnEnable()
    {
        if (receiver != null) receiver.DetectionFrameReceived += PlaceMarkers;
        RebuildPlaneLookup();
    }

    private void OnDisable()
    {
        if (receiver != null) receiver.DetectionFrameReceived -= PlaceMarkers;
    }

    private void RebuildPlaneLookup()
    {
        planeLookup = new Dictionary<string, SurfacePlane>();

        foreach (SurfacePlane plane in surfacePlanes)
        {
            if (plane == null || string.IsNullOrEmpty(plane.surfaceId))
            {
                Debug.LogWarning("[SurfaceMarkerMapper] surfaceId가 비어있는 표면 항목은 건너뜁니다.");
                continue;
            }

            if (plane.origin == null || plane.uAxisEnd == null || plane.vAxisEnd == null)
            {
                Debug.LogWarning($"[SurfaceMarkerMapper] '{plane.surfaceId}' 표면의 Transform이 모두 연결되지 않아 건너뜁니다.");
                continue;
            }

            planeLookup[plane.surfaceId] = plane;
        }
    }

    private void PlaceMarkers(DetectionFrame frame)
    {
        if (planeLookup == null) RebuildPlaneLookup();
        if (planeLookup.Count == 0 || markerPrefab == null) return;
        if (frame.detections == null) return;

        visibleKeys.Clear();

        for (int i = 0; i < frame.detections.Length; i++)
        {
            Detection detection = frame.detections[i];

            if (detection.surface == null || detection.confidence < minimumConfidence)
                continue;

            SurfacePlane plane;
            if (!planeLookup.TryGetValue(detection.surface.id, out plane))
                continue;

            string localKey = string.IsNullOrEmpty(detection.id) ? "surface-" + i : detection.id;
            string key = detection.surface.id + ":" + localKey;
            visibleKeys.Add(key);

            Transform marker = GetOrCreateMarker(key);

            marker.position = SurfaceToWorld(
                plane,
                detection.surface.u,
                detection.surface.v
            );

            marker.gameObject.SetActive(true);
        }

        foreach (KeyValuePair<string, Transform> item in activeMarkers)
        {
            if (!visibleKeys.Contains(item.Key))
                item.Value.gameObject.SetActive(false);
        }
    }

    private Transform GetOrCreateMarker(string key)
    {
        Transform marker;
        if (activeMarkers.TryGetValue(key, out marker))
            return marker;

        marker = Instantiate(markerPrefab, transform);
        marker.name = "SurfaceMarker_" + key;
        activeMarkers.Add(key, marker);

        return marker;
    }

    private Vector3 SurfaceToWorld(SurfacePlane plane, float u, float v)
    {
        Vector3 uAxis = plane.uAxisEnd.position - plane.origin.position;
        Vector3 vAxis = plane.vAxisEnd.position - plane.origin.position;
        Vector3 normal = Vector3.Cross(uAxis, vAxis).normalized;

        return plane.origin.position
             + uAxis * u
             + vAxis * v
             + normal * markerSurfaceOffset;
    }
}
