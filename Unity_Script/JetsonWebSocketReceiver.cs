using System;
using System.Collections;
using System.IO;
using NativeWebSocket;
using UnityEngine;

public class JetsonWebSocketReceiver : MonoBehaviour
{
    [Header("Network")]
    // TODO: 노트북(중계 서버)이 접속된 실제 LAN IP 주소로 교체하거나, Inspector에서 값을 설정하세요. (예: "192.168.0.25")
    [SerializeField] private string laptopIp = "<LAPTOP_LAN_IP>";
    [SerializeField] private int port = 5000;
    [SerializeField] private bool autoReconnect = true;
    [SerializeField] private float reconnectDelaySeconds = 3f;

    [Header("Logging")]
    [SerializeField] private bool saveReceivedJson = true;

    private NativeWebSocket.WebSocket websocket;
    private int lastSequence = -1;
    private string logPath;
    private Coroutine reconnectCoroutine;
    private bool shuttingDown = false;

    public event Action<DetectionFrame> DetectionFrameReceived;

    private void Start()
    {
        logPath = Path.Combine(Application.persistentDataPath, "day9_received.jsonl");
        shuttingDown = false;
        Connect();
    }

    public async void Connect()
    {
        if (shuttingDown) return;
        if (websocket != null) return;

        string url = $"ws://{laptopIp}:{port}";
        Debug.Log($"[DAY9] Connecting to {url}");

        NativeWebSocket.WebSocket newSocket = new NativeWebSocket.WebSocket(url);
        websocket = newSocket;

        newSocket.OnOpen += () =>
        {
            if (shuttingDown) return;
            Debug.Log($"[DAY9] Connected: {url}");
            newSocket.SendText("UNITY");
        };

        newSocket.OnError += error =>
        {
            if (shuttingDown) return;
            Debug.LogError($"[DAY9] WebSocket error: {error}");
        };

        newSocket.OnClose += code =>
        {
            Debug.LogWarning($"[DAY9] Closed: {code}");
            if (websocket == newSocket) websocket = null;

            if (autoReconnect && !shuttingDown && isActiveAndEnabled && reconnectCoroutine == null)
                reconnectCoroutine = StartCoroutine(ReconnectAfterDelay());
        };

        newSocket.OnMessage += bytes =>
        {
            if (shuttingDown) return;
            try
            {
                string json = System.Text.Encoding.UTF8.GetString(bytes);
                HandleMessage(json);
            }
            catch (Exception exception)
            {
                Debug.LogError($"[DAY9] Message handling error: {exception.Message}");
            }
        };

        try
        {
            await newSocket.Connect();
        }
        catch (Exception exception)
        {
            Debug.LogError($"[DAY9] Cannot connect to {url}: {exception.Message}");
            if (websocket == newSocket) websocket = null;

            if (autoReconnect && !shuttingDown && isActiveAndEnabled && reconnectCoroutine == null)
                reconnectCoroutine = StartCoroutine(ReconnectAfterDelay());
        }
    }

    private IEnumerator ReconnectAfterDelay()
    {
        yield return new WaitForSeconds(reconnectDelaySeconds);
        reconnectCoroutine = null;

        if (!shuttingDown && isActiveAndEnabled && websocket == null)
            Connect();
    }

    private void HandleMessage(string json)
    {
        if (string.IsNullOrWhiteSpace(json)) return;

        MessageTypeEnvelope envelope;
        try
        {
            envelope = JsonUtility.FromJson<MessageTypeEnvelope>(json);
        }
        catch (Exception exception)
        {
            Debug.LogError($"[DAY9] Invalid JSON: {exception.Message}\n{json}");
            return;
        }

        if (envelope == null || string.IsNullOrEmpty(envelope.type))
        {
            Debug.LogError($"[DAY9] Message type missing:\n{json}");
            return;
        }

        if (envelope.type == "aruco_frame") { HandleArucoFrame(json); return; }
        if (envelope.type == "detection_frame") { HandleDetectionFrame(json); return; }

        Debug.LogWarning($"[DAY9] Unknown message type: {envelope.type}");
    }

    private void HandleArucoFrame(string json)
    {
        ArucoFrame arucoFrame;
        try { arucoFrame = JsonUtility.FromJson<ArucoFrame>(json); }
        catch (Exception exception)
        {
            Debug.LogError($"[DAY9] Invalid aruco_frame JSON: {exception.Message}\n{json}");
            return;
        }

        if (arucoFrame == null || arucoFrame.camera == null || arucoFrame.markers == null)
        {
            Debug.LogError($"[DAY9] Invalid aruco_frame schema:\n{json}");
            return;
        }

        long arucoLatencyMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - arucoFrame.sentAtUnixMs;
        Debug.Log($"[DAY9] ArUco frame: markers={arucoFrame.markers.Length}, latency≈{arucoLatencyMs}ms");

        if (saveReceivedJson) SaveJson(json);
    }

    private void HandleDetectionFrame(string json)
    {
        DetectionFrame frame;
        try { frame = JsonUtility.FromJson<DetectionFrame>(json); }
        catch (Exception exception)
        {
            Debug.LogError($"[DAY9] Invalid detection_frame JSON: {exception.Message}\n{json}");
            return;
        }

        if (frame == null || frame.camera == null || frame.detections == null)
        {
            Debug.LogError($"[DAY9] Invalid detection_frame schema:\n{json}");
            return;
        }

        if (frame.seq <= lastSequence)
            Debug.LogWarning($"[DAY9] Old/duplicate frame: seq={frame.seq}, last={lastSequence}");

        lastSequence = frame.seq;

        long latencyMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - frame.sentAtUnixMs;
        Debug.Log($"[DAY9] seq={frame.seq}, source={frame.source}, detections={frame.detections.Length}, latency≈{latencyMs}ms");

        if (saveReceivedJson) SaveJson(json);
        DetectionFrameReceived?.Invoke(frame);
    }

    private void SaveJson(string json)
    {
        try { File.AppendAllText(logPath, json + Environment.NewLine); }
        catch (Exception exception) { Debug.LogError($"[DAY9] Failed to save JSON: {exception.Message}"); }
    }

    private void Update()
    {
#if !UNITY_WEBGL || UNITY_EDITOR
        if (websocket != null) websocket.DispatchMessageQueue();
#endif
    }

    private async void OnApplicationQuit()
    {
        shuttingDown = true;

        if (reconnectCoroutine != null)
        {
            StopCoroutine(reconnectCoroutine);
            reconnectCoroutine = null;
        }

        if (websocket != null)
        {
            NativeWebSocket.WebSocket socketToClose = websocket;
            websocket = null;

            try { await socketToClose.Close(); }
            catch (Exception exception) { Debug.LogWarning($"[DAY9] Close error: {exception.Message}"); }
        }
    }

    private void OnDestroy()
    {
        shuttingDown = true;

        if (reconnectCoroutine != null)
        {
            StopCoroutine(reconnectCoroutine);
            reconnectCoroutine = null;
        }
    }
}
