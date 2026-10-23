import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'client'))

from flask import Flask, render_template, request, jsonify
from ai_client import InferenceClient, DevicePool
import requests as req_lib

app = Flask(__name__)

clients: dict[str, InferenceClient] = {}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/devices', methods=['GET'])
def list_devices():
    result = []
    for key, client in clients.items():
        try:
            status = client.status()
            hw = client.hardware()
            ping = client.ping()
            result.append({'key': key, 'status': status, 'hardware': hw, 'ping': round(ping), 'online': True})
        except Exception as e:
            result.append({'key': key, 'online': False, 'error': str(e)})
    return jsonify(result)


@app.route('/api/devices', methods=['POST'])
def add_device():
    data = request.json
    host = data.get('host', '').strip()
    port = int(data.get('port', 8080))
    api_key = data.get('api_key') or None
    key = f"{host}:{port}"
    try:
        client = InferenceClient(host, port, api_key=api_key, timeout=320)
        status = client.status()
        clients[key] = client
        return jsonify({'success': True, 'key': key, 'device': status.get('device', key)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/devices/<key>', methods=['DELETE'])
def remove_device(key):
    if key in clients:
        del clients[key]
        return jsonify({'success': True})
    return jsonify({'error': 'Device not found'}), 404


@app.route('/api/devices/<key>/status', methods=['GET'])
def device_status(key):
    if key not in clients:
        return jsonify({'error': 'Device not found'}), 404
    try:
        return jsonify({
            'status': clients[key].status(),
            'hardware': clients[key].hardware(),
            'ping': round(clients[key].ping()),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/discover', methods=['POST'])
def discover():
    data = request.json or {}
    subnet = data.get('subnet', '192.168.1')
    port = int(data.get('port', 8080))
    pool = DevicePool()
    found = pool.auto_discover(subnet=subnet, port=port)
    for key, client in pool.clients.items():
        clients[key] = client
    return jsonify({'found': found, 'devices': list(pool.clients.keys())})


@app.route('/api/models', methods=['GET'])
def list_models():
    key = request.args.get('device')
    if key not in clients:
        return jsonify({'error': 'Device not found'}), 404
    try:
        return jsonify(clients[key].models())
    except req_lib.exceptions.ConnectionError:
        return jsonify({'error': 'Device disconnected'}), 502


@app.route('/api/models/upload', methods=['POST'])
def upload_model():
    key = request.form.get('device')
    if key not in clients:
        return jsonify({'error': 'Device not found'}), 404
    f = request.files.get('model')
    if not f:
        return jsonify({'error': 'No file provided'}), 400
    model_type = request.form.get('model_type', 'OBJECT_DETECTION')
    tmp = f'/tmp/{f.filename}'
    f.save(tmp)
    try:
        upload_result = clients[key].upload_model(tmp, name=f.filename)
        full_path = upload_result.get('path', f.filename)
        name = upload_result.get('name', f.filename)
        load_result = clients[key].load_model(name, full_path, model_type)
        return jsonify({**upload_result, 'loaded': True, 'load': load_result})
    except req_lib.exceptions.ConnectionError:
        return jsonify({'error': 'Device disconnected during upload/load'}), 502
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


@app.route('/api/detect', methods=['POST'])
def detect():
    key = request.form.get('device')
    if key not in clients:
        return jsonify({'error': 'Device not found'}), 404
    img = request.files.get('image')
    if not img:
        return jsonify({'error': 'No image provided'}), 400
    confidence = float(request.form.get('confidence', 0.5))
    model = request.form.get('model', 'ssd_mobilenet_v2.tflite')
    accelerator = request.form.get('accelerator') or None
    try:
        detections = clients[key].detect(img.read(), model=model, confidence=confidence, accelerator=accelerator)
        return jsonify({'detections': [{'label': d.label, 'confidence': round(d.confidence, 4), 'bbox': d.bbox} for d in detections]})
    except req_lib.exceptions.ConnectionError:
        return jsonify({'error': 'Device disconnected — the phone may have crashed during inference. Check that the model is compatible and the app is still running.'}), 502
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/classify', methods=['POST'])
def classify():
    key = request.form.get('device')
    if key not in clients:
        return jsonify({'error': 'Device not found'}), 404
    img = request.files.get('image')
    if not img:
        return jsonify({'error': 'No image provided'}), 400
    top_k = int(request.form.get('top_k', 5))
    model = request.form.get('model', 'efficientnet_lite0.tflite')
    try:
        results = clients[key].classify(img.read(), model=model, top_k=top_k)
        return jsonify({'classifications': [{'label': c.label, 'confidence': round(c.confidence, 4)} for c in results]})
    except req_lib.exceptions.ConnectionError:
        return jsonify({'error': 'Device disconnected — the phone may have crashed during inference.'}), 502
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/llm/generate', methods=['POST'])
def llm_generate():
    data = request.json or {}
    key = data.get('device')
    if key not in clients:
        return jsonify({'error': 'Device not found'}), 404
    try:
        resp = clients[key].llm_generate(
            prompt=data['prompt'],
            max_tokens=int(data.get('max_tokens', 128)),
            temperature=float(data.get('temperature', 0.7)),
            model=data.get('model', 'gemma_2b_int8.tflite'),
        )
        return jsonify({'text': resp.text, 'tokens_generated': resp.tokens_generated,
                        'tokens_per_second': round(resp.tokens_per_second, 2),
                        'total_ms': round(resp.total_ms), 'accelerator': resp.accelerator})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/llm/chat', methods=['POST'])
def llm_chat():
    data = request.json or {}
    key = data.get('device')
    if key not in clients:
        return jsonify({'error': 'Device not found'}), 404
    try:
        result = clients[key].llm_chat(
            messages=data['messages'],
            max_tokens=int(data.get('max_tokens', 256)),
            temperature=float(data.get('temperature', 0.7)),
            model=data.get('model', 'gemma_2b_int8.tflite'),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/rembg', methods=['POST'])
def rembg():
    key = request.form.get('device')
    if key not in clients:
        return jsonify({'error': 'Device not found'}), 404
    img = request.files.get('image')
    if not img:
        return jsonify({'error': 'No image provided'}), 400
    model = request.form.get('model', 'rmbg2.tflite')
    try:
        import base64
        img_b64 = base64.b64encode(img.read()).decode()
        result = clients[key]._post('/api/v1/rembg', {'image': img_b64, 'model': model})
        return jsonify(result)
    except req_lib.exceptions.ConnectionError:
        return jsonify({'error': 'Device disconnected during processing'}), 502
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/rembg/benchmark', methods=['POST'])
def rembg_benchmark():
    data = request.json or {}
    key = data.get('device')
    if key not in clients:
        return jsonify({'error': 'Device not found'}), 404
    try:
        results = clients[key].benchmark_rembg(model=data.get('model', 'rmbg2.tflite'))
        return jsonify({'results': [{'accelerator': r.accelerator, 'delegate': r.delegate,
                                     'success': r.success, 'avg_ms': round(r.avg_ms, 2),
                                     'fps': round(r.fps, 2), 'memory_mb': round(r.memory_mb, 1),
                                     'error': r.error} for r in results]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/benchmark', methods=['POST'])
def benchmark():
    data = request.json or {}
    key = data.get('device')
    if key not in clients:
        return jsonify({'error': 'Device not found'}), 404
    try:
        results = clients[key].benchmark(
            model=data.get('model', 'model.tflite'),
            model_type=data.get('model_type', 'IMAGE_CLASSIFICATION'),
        )
        return jsonify({'results': [{'accelerator': r.accelerator, 'delegate': r.delegate,
                                     'success': r.success, 'avg_ms': round(r.avg_ms, 2),
                                     'fps': round(r.fps, 2), 'memory_mb': round(r.memory_mb, 1),
                                     'error': r.error} for r in results]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
