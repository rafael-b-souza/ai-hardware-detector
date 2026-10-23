"""
AI Inference Client — SDK Python para o servidor Android de inferência de IA

Conecta a qualquer smartphone rodando o AI Hardware Detector e envia
solicitações de inferência via REST API.

Uso:
    from ai_client import InferenceClient, DevicePool

    # Único dispositivo
    client = InferenceClient("192.168.1.50", port=8080)
    print(client.status())
    print(client.hardware())
    detections = client.detect("foto.jpg")
    response = client.llm_generate("Explique IA em 3 frases")

    # Pool de dispositivos (S22 + S21 + Note 8)
    pool = DevicePool([
        ("192.168.1.50", 8080),  # S22
        ("192.168.1.51", 8080),  # S21
        ("192.168.1.52", 8080),  # Note 8
    ])
    pool.auto_discover()  # ou descoberta automática na rede
    pool.detect_batch(["img1.jpg", "img2.jpg", "img3.jpg"])  # distribui entre devices
"""

import requests
import base64
import json
import time
import socket
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: Dict[str, float]

@dataclass
class Classification:
    label: str
    confidence: float

@dataclass
class LLMResponse:
    text: str
    tokens_generated: int
    tokens_per_second: float
    total_ms: float
    accelerator: str

@dataclass
class BenchmarkResult:
    accelerator: str
    delegate: str
    success: bool
    avg_ms: float
    fps: float
    memory_mb: float
    error: Optional[str] = None

@dataclass
class DeviceInfo:
    address: str
    model: str
    soc: str
    accelerators: List[Dict]
    best_accelerator: str
    loaded_models: int
    latency_ms: float = 0.0


class InferenceClient:
    """Cliente para servidor de inferência em dispositivo Android."""

    def __init__(
        self,
        host: str,
        port: int = 8080,
        api_key: Optional[str] = None,
        timeout: int = 30
    ):
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.session = requests.Session()

        if api_key:
            self.session.headers["X-API-Key"] = api_key

        self.session.headers["Content-Type"] = "application/json"

    def status(self) -> Dict:
        """Retorna status do servidor (uptime, requests, modelos carregados)."""
        return self._get("/api/v1/status")

    def hardware(self) -> Dict:
        """Retorna detalhes do hardware (aceleradores, RAM, CPU, NPU)."""
        return self._get("/api/v1/hardware")

    def models(self) -> Dict:
        """Lista modelos carregados."""
        return self._get("/api/v1/models")

    def queue(self) -> Dict:
        """Status da fila de processamento."""
        return self._get("/api/v1/queue")

    def ping(self) -> float:
        """Mede latência em ms."""
        start = time.time()
        try:
            self.status()
            return (time.time() - start) * 1000
        except Exception:
            return -1

    def detect(
        self,
        image: Union[str, bytes],
        model: str = "ssd_mobilenet_v2.tflite",
        confidence: float = 0.5,
        accelerator: Optional[str] = None
    ) -> List[Detection]:
        """Detecção de objetos em imagem."""
        payload = {
            "image": self._encode_image(image),
            "model": model,
            "confidence": confidence,
        }
        if accelerator:
            payload["accelerator"] = accelerator

        result = self._post("/api/v1/detect", payload)
        return [Detection(
            label=d["label"],
            confidence=d["confidence"],
            bbox=d["bbox"]
        ) for d in result.get("detections", [])]

    def classify(
        self,
        image: Union[str, bytes],
        model: str = "efficientnet_lite0.tflite",
        top_k: int = 5
    ) -> List[Classification]:
        """Classificação de imagem. Retorna top-K classes."""
        payload = {
            "image": self._encode_image(image),
            "model": model,
            "top_k": top_k,
        }
        result = self._post("/api/v1/classify", payload)
        return [Classification(
            label=c["label"],
            confidence=c["confidence"]
        ) for c in result.get("classifications", [])]

    def llm_generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        model: str = "gemma_2b_int8.tflite"
    ) -> LLMResponse:
        """Geração de texto com LLM local."""
        result = self._post("/api/v1/llm/generate", {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "model": model,
        })
        return LLMResponse(
            text=result["text"],
            tokens_generated=result["tokens_generated"],
            tokens_per_second=result["tokens_per_second"],
            total_ms=result["total_ms"],
            accelerator=result["accelerator"],
        )

    def llm_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        model: str = "gemma_2b_int8.tflite"
    ) -> Dict:
        """Chat multi-turno com LLM local."""
        return self._post("/api/v1/llm/chat", {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "model": model,
        })

    def remove_background(
        self,
        image: Union[str, bytes],
        model: str = "rmbg2.tflite",
        timeout: int = 120
    ) -> bytes:
        """Remove o fundo da imagem retornando PNG em bytes (RGBA)."""
        payload = {
            "image": self._encode_image(image),
            "model": model,
        }
        resp = self.session.post(
            f"{self.base_url}/api/v1/rembg",
            json=payload,
            timeout=timeout
        )
        self._raise(resp)
        result = resp.json()
        return base64.b64decode(result["result"])

    def raw_inference(
        self,
        model: str,
        input_data: List[float],
        input_shape: List[int],
        model_type: str = "IMAGE_CLASSIFICATION",
        accelerator: Optional[str] = None
    ) -> Dict:
        """Inferência genérica com modelo TFLite."""
        payload = {
            "model": model,
            "input": input_data,
            "input_shape": input_shape,
            "model_type": model_type,
        }
        if accelerator:
            payload["accelerator"] = accelerator
        return self._post("/api/v1/inference/raw", payload)

    def load_model(self, name: str, path: str, model_type: str) -> Dict:
        """Carrega um modelo no servidor."""
        return self._post("/api/v1/models/load", {
            "name": name,
            "path": path,
            "type": model_type,
        })

    def unload_model(self, name: str) -> Dict:
        """Descarrega um modelo."""
        return self._delete(f"/api/v1/models/{name}")

    def upload_model(self, file_path: str, name: Optional[str] = None) -> Dict:
        """Faz upload de um modelo .tflite para o dispositivo."""
        name = name or Path(file_path).name
        with open(file_path, "rb") as f:
            resp = self.session.post(
                f"{self.base_url}/api/v1/models/upload",
                files={"file": (name, f, "application/octet-stream")},
                data={"name": name},
                timeout=self.timeout * 3,
                headers={"Content-Type": None},  # deixar requests definir multipart
            )
        resp.raise_for_status()
        return resp.json()

    def benchmark_rembg(self, model: str = "rmbg2.tflite") -> List[BenchmarkResult]:
        """Executa benchmark de remoção de fundo nos aceleradores."""
        return self.benchmark(
            model=model,
            model_type="SEGMENTATION",
            input_shape=[1, 512, 512, 3]
        )

    def benchmark(
        self,
        model: str = "model.tflite",
        model_type: str = "IMAGE_CLASSIFICATION",
        input_shape: List[int] = None
    ) -> List[BenchmarkResult]:
        """Roda benchmark em todos os aceleradores."""
        payload = {"model": model, "type": model_type}
        if input_shape:
            payload["input_shape"] = input_shape

        result = self._post("/api/v1/benchmark", payload)
        return [BenchmarkResult(
            accelerator=r["accelerator"],
            delegate=r["delegate"],
            success=r["success"],
            avg_ms=r.get("avg_ms", 0),
            fps=r.get("fps", 0),
            memory_mb=r.get("memory_mb", 0),
            error=r.get("error"),
        ) for r in result.get("results", [])]

    def _get(self, path: str) -> Dict:
        resp = self.session.get(f"{self.base_url}{path}", timeout=self.timeout)
        self._raise(resp)
        return resp.json()

    def _post(self, path: str, data: Dict) -> Dict:
        resp = self.session.post(
            f"{self.base_url}{path}",
            json=data,
            timeout=self.timeout
        )
        self._raise(resp)
        return resp.json()

    def _delete(self, path: str) -> Dict:
        resp = self.session.delete(f"{self.base_url}{path}", timeout=self.timeout)
        self._raise(resp)
        return resp.json()

    def _raise(self, resp) -> None:
        if not resp.ok:
            try:
                body = resp.json()
                msg = body.get('message') or body.get('error') or resp.text
            except Exception:
                msg = resp.text or resp.reason
            raise Exception(f"HTTP {resp.status_code} from device: {msg}")

    def _encode_image(self, image: Union[str, bytes]) -> str:
        if isinstance(image, bytes):
            return base64.b64encode(image).decode()
        if isinstance(image, str):
            path = Path(image)
            if path.exists():
                return base64.b64encode(path.read_bytes()).decode()
        raise ValueError(f"Imagem inválida: {image}")


class DevicePool:
    """Gerencia múltiplos smartphones distribuindo carga de inferência por capacidade de hardware."""

    def __init__(
        self,
        devices: Optional[List[tuple]] = None,
        api_key: Optional[str] = None
    ):
        self.api_key = api_key
        self.clients: Dict[str, InferenceClient] = {}
        self.device_info: Dict[str, DeviceInfo] = {}
        self.device_weights: Dict[str, float] = {}

        if devices:
            for host, port in devices:
                self.add_device(host, port)

    def add_device(self, host: str, port: int = 8080) -> bool:
        """Adiciona um dispositivo ao pool."""
        try:
            client = InferenceClient(host, port, api_key=self.api_key, timeout=320)
            status = client.status()
            hw = client.hardware()
            latency = client.ping()

            key = f"{host}:{port}"
            self.clients[key] = client
            self.device_info[key] = DeviceInfo(
                address=key,
                model=status.get("device", "unknown"),
                soc=status.get("soc", "unknown"),
                accelerators=hw.get("accelerators", []),
                best_accelerator=status.get("best_accelerator", "CPU"),
                loaded_models=status.get("loaded_models", 0),
                latency_ms=latency,
            )

            max_tops = max(
                (a.get("estimated_tops", 0) for a in hw.get("accelerators", [])),
                default=0.1
            )
            self.device_weights[key] = max(max_tops, 0.1)

            print(f"[{key}] {status['device']} ({status['soc']}) — {latency:.0f}ms — {max_tops} TOPS")
            return True
        except Exception as e:
            print(f"[{host}:{port}] Erro: {e}")
            return False

    def auto_discover(
        self,
        subnet: str = "192.168.1",
        port: int = 8080,
        timeout: float = 0.5
    ) -> int:
        """Varre a subnet na porta indicada para localizar servidores."""
        print(f"Escaneando {subnet}.1-254 na porta {port}...")
        found = 0

        def try_connect(ip):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(timeout)
                result = sock.connect_ex((ip, port))
                sock.close()
                if result == 0:
                    if self.add_device(ip, port):
                        return True
            except Exception:
                pass
            return False

        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = {
                executor.submit(try_connect, f"{subnet}.{i}"): i
                for i in range(1, 255)
            }
            for future in as_completed(futures):
                if future.result():
                    found += 1

        print(f"Encontrados: {found} dispositivos")
        return found

    def pool_status(self) -> List[DeviceInfo]:
        """Status de todos os dispositivos no pool."""
        return list(self.device_info.values())

    def detect_batch(
        self,
        images: List[str],
        model: str = "ssd_mobilenet_v2.tflite",
        confidence: float = 0.5
    ) -> List[Dict]:
        """Distribui detecção de múltiplas imagens entre os dispositivos do pool."""
        assignments = self._distribute_work(len(images))
        results = [None] * len(images)

        with ThreadPoolExecutor(max_workers=len(self.clients)) as executor:
            futures = {}
            img_idx = 0

            for device_key, count in assignments.items():
                client = self.clients[device_key]
                for _ in range(count):
                    if img_idx < len(images):
                        idx = img_idx
                        future = executor.submit(
                            client.detect, images[idx], model, confidence
                        )
                        futures[future] = idx
                        img_idx += 1

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = {
                        "index": idx,
                        "image": images[idx],
                        "detections": [
                            {"label": d.label, "confidence": d.confidence, "bbox": d.bbox}
                            for d in future.result()
                        ]
                    }
                except Exception as e:
                    results[idx] = {"index": idx, "image": images[idx], "error": str(e)}

        return [r for r in results if r is not None]

    def classify_batch(self, images: List[str], **kwargs) -> List[Dict]:
        """Distribui classificação entre dispositivos."""
        assignments = self._distribute_work(len(images))
        results = [None] * len(images)

        with ThreadPoolExecutor(max_workers=len(self.clients)) as executor:
            futures = {}
            img_idx = 0

            for device_key, count in assignments.items():
                client = self.clients[device_key]
                for _ in range(count):
                    if img_idx < len(images):
                        idx = img_idx
                        future = executor.submit(client.classify, images[idx], **kwargs)
                        futures[future] = idx
                        img_idx += 1

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = {
                        "index": idx,
                        "classifications": [
                            {"label": c.label, "confidence": c.confidence}
                            for c in future.result()
                        ]
                    }
                except Exception as e:
                    results[idx] = {"index": idx, "error": str(e)}

        return [r for r in results if r is not None]

    def llm_generate_first(self, prompt: str, **kwargs) -> LLMResponse:
        """Envia para o dispositivo mais rápido (maior TOPS)."""
        best_key = max(self.device_weights, key=self.device_weights.get)
        return self.clients[best_key].llm_generate(prompt, **kwargs)

    def benchmark_all(self, **kwargs) -> Dict[str, List[BenchmarkResult]]:
        """Roda benchmark em todos os dispositivos simultaneamente."""
        results = {}
        with ThreadPoolExecutor(max_workers=len(self.clients)) as executor:
            futures = {
                executor.submit(client.benchmark, **kwargs): key
                for key, client in self.clients.items()
            }
            for future in as_completed(futures):
                key = futures[future]
                try:
                    results[key] = future.result()
                except Exception as e:
                    results[key] = [BenchmarkResult(
                        accelerator="ERROR", delegate="", success=False,
                        avg_ms=0, fps=0, memory_mb=0, error=str(e)
                    )]
        return results

    def _distribute_work(self, total: int) -> Dict[str, int]:
        total_weight = sum(self.device_weights.values())
        assignments = {}
        assigned = 0

        for key, weight in sorted(self.device_weights.items(), key=lambda x: -x[1]):
            count = round(total * (weight / total_weight))
            count = min(count, total - assigned)
            if count > 0:
                assignments[key] = count
                assigned += count

        if assigned < total:
            best_key = max(self.device_weights, key=self.device_weights.get)
            assignments[best_key] = assignments.get(best_key, 0) + (total - assigned)

        return assignments


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("""
AI Inference Client — Uso:

    python ai_client.py 192.168.1.50
    python ai_client.py 192.168.1.50 --port 8080 --key minha_key
    python ai_client.py discover --subnet 192.168.1
    python ai_client.py 192.168.1.50 --detect foto.jpg
    python ai_client.py 192.168.1.50 --llm "Explique IA"
        """)
        sys.exit(0)

    host = sys.argv[1]

    if host == "discover":
        subnet = "192.168.1"
        port = 8080
        for i, arg in enumerate(sys.argv):
            if arg == "--subnet" and i + 1 < len(sys.argv):
                subnet = sys.argv[i + 1]
            if arg == "--port" and i + 1 < len(sys.argv):
                port = int(sys.argv[i + 1])

        pool = DevicePool()
        pool.auto_discover(subnet=subnet, port=port)

        print("\nDispositivos encontrados:")
        for info in pool.pool_status():
            print(f"  {info.address} — {info.model} ({info.soc}) — "
                  f"{info.best_accelerator} — {info.latency_ms:.0f}ms")
        sys.exit(0)

    port = 8080
    api_key = None
    for i, arg in enumerate(sys.argv):
        if arg == "--port" and i + 1 < len(sys.argv):
            port = int(sys.argv[i + 1])
        if arg == "--key" and i + 1 < len(sys.argv):
            api_key = sys.argv[i + 1]

    client = InferenceClient(host, port, api_key=api_key)

    try:
        status = client.status()
        hw = client.hardware()
        ping = client.ping()
        print(f"\nConectado: {status['device']} ({status['soc']})")
        print(f"   Uptime: {status['uptime_seconds']}s | Requests: {status['total_requests']}")
        print(f"   Latência: {ping:.0f}ms")
        print(f"   Aceleradores: {len(hw['accelerators'])}")
        for a in hw["accelerators"]:
            state = "[OK]" if a["available"] else "[FAIL]"
            print(f"     {state} {a['type']}: {a['name']} ({a['estimated_tops']} TOPS)")
    except Exception as e:
        print(f"Erro conectando a {host}:{port} — {e}")
        sys.exit(1)

    for i, arg in enumerate(sys.argv):
        if arg == "--detect" and i + 1 < len(sys.argv):
            image_path = sys.argv[i + 1]
            print(f"\nDetectando objetos em {image_path}...")
            detections = client.detect(image_path)
            for d in detections:
                print(f"  {d.label}: {d.confidence:.1%} @ {d.bbox}")

        if arg == "--llm" and i + 1 < len(sys.argv):
            prompt = sys.argv[i + 1]
            print(f"\nGerando texto: '{prompt}'")
            resp = client.llm_generate(prompt)
            print(f"  Resposta: {resp.text}")
            print(f"  {resp.tokens_generated} tokens @ {resp.tokens_per_second:.1f} tok/s")
            print(f"  Acelerador: {resp.accelerator}")
