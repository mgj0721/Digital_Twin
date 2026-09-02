import asyncio
import logging
import websockets

# ==========================================
# 노트북 WebSocket 서버
# 역할:
#   Jetson이 보낸 JSON을 받아서
#   연결된 Unity 클라이언트에게 그대로 전달한다.
# ==========================================

PORT = 5000

jetson_clients = set()
unity_clients = set()


async def handle_client(websocket):
    peer = websocket.remote_address
    logging.info("클라이언트 연결: %s", peer)

    try:
        # 첫 메시지로 클라이언트 종류 확인
        first_message = await websocket.recv()

        if first_message == "JETSON":
            jetson_clients.add(websocket)
            logging.info("Jetson 등록: %s", peer)

        elif first_message == "UNITY":
            unity_clients.add(websocket)
            logging.info("Unity 등록: %s", peer)

        else:
            logging.warning("알 수 없는 클라이언트: %s", peer)
            await websocket.close()
            return

        # Jetson -> Server -> Unity
        if websocket in jetson_clients:
            async for message in websocket:
                logging.info("Jetson -> Server: %d bytes", len(message))

                disconnected = set()

                for unity in unity_clients:
                    try:
                        await unity.send(message)
                    except websockets.ConnectionClosed:
                        disconnected.add(unity)

                for client in disconnected:
                    unity_clients.discard(client)

        # 현재 Unity -> Server 데이터는 테스트하지 않는다.
        elif websocket in unity_clients:
            async for message in websocket:
                logging.info("Unity -> Server: %s", message)

    except websockets.ConnectionClosed:
        logging.info("클라이언트 연결 종료: %s", peer)

    finally:
        jetson_clients.discard(websocket)
        unity_clients.discard(websocket)


async def main():
    logging.info("==============================")
    logging.info(" Digital Twin DAY9 Server")
    logging.info("==============================")
    logging.info("Port: %s", PORT)
    logging.info("Jetson / Unity 접속 대기 중...")

    # 0.0.0.0은 노트북의 모든 네트워크 인터페이스에서
    # 연결을 받을 수 있도록 하는 서버용 주소다.
    async with websockets.serve(
        handle_client,
        "0.0.0.0",
        PORT,
        ping_interval=20,
        ping_timeout=20
    ):
        # 서버가 종료되지 않고 계속 실행되도록 유지
        await asyncio.Future()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    asyncio.run(main())
