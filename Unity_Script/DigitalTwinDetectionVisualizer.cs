// Shows every YOLO bbox in a Unity UI image area.
// Attach to Canvas and assign Receiver, Camera Display Area, and Detection Box Prefab.
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class DigitalTwinDetectionVisualizer : MonoBehaviour
{
    [SerializeField] private JetsonWebSocketReceiver receiver;
    [SerializeField] private RectTransform cameraDisplayArea;
    [SerializeField] private RectTransform detectionBoxPrefab;
    [SerializeField, Range(0f, 1f)] private float minimumConfidence = 0.50f;

    private readonly Dictionary<string, RectTransform> activeBoxes = new Dictionary<string, RectTransform>();
    private readonly HashSet<string> visibleKeys = new HashSet<string>();

    private void OnEnable()
    {
        if (receiver != null) receiver.DetectionFrameReceived += ShowDetections;
    }

    private void OnDisable()
    {
        if (receiver != null) receiver.DetectionFrameReceived -= ShowDetections;
    }

    private void ShowDetections(DetectionFrame frame)
    {
        if (cameraDisplayArea == null || detectionBoxPrefab == null || frame.camera.width <= 0 || frame.camera.height <= 0) return;

        visibleKeys.Clear();
        for (int index = 0; index < frame.detections.Length; index++)
        {
            Detection detection = frame.detections[index];
            if (detection.bbox == null || detection.confidence < minimumConfidence) continue;

            string key = string.IsNullOrEmpty(detection.id) ? "detection-" + index : detection.id;
            visibleKeys.Add(key);
            RectTransform box = GetOrCreateBox(key);
            PositionBox(box, detection, frame.camera);
            box.gameObject.SetActive(true);
        }

        foreach (KeyValuePair<string, RectTransform> item in activeBoxes)
            if (!visibleKeys.Contains(item.Key)) item.Value.gameObject.SetActive(false);
    }

    private RectTransform GetOrCreateBox(string key)
    {
        RectTransform box;
        if (activeBoxes.TryGetValue(key, out box)) return box;

        box = Instantiate(detectionBoxPrefab, cameraDisplayArea);
        box.name = "DetectionBox_" + key;
        box.anchorMin = Vector2.zero;
        box.anchorMax = Vector2.zero;
        box.pivot = new Vector2(0f, 1f);
        activeBoxes.Add(key, box);
        return box;
    }

    private void PositionBox(RectTransform box, Detection detection, CameraSize camera)
    {
        BoundingBox bbox = detection.bbox;
        float displayWidth = cameraDisplayArea.rect.width;
        float displayHeight = cameraDisplayArea.rect.height;
        float x = bbox.x / (float)camera.width * displayWidth;
        float width = bbox.width / (float)camera.width * displayWidth;
        float height = bbox.height / (float)camera.height * displayHeight;
        float yTop = displayHeight - bbox.y / (float)camera.height * displayHeight;

        box.anchoredPosition = new Vector2(x, yTop);
        box.sizeDelta = new Vector2(width, height);

        Image image = box.GetComponent<Image>();
        if (image != null) image.color = detection.confidence >= 0.80f ? Color.red : new Color(1f, 0.86f, 0.08f, 0.95f);
    }
}
