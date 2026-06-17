from flask import Flask, request, jsonify
import json

app = Flask(__name__)

@app.route('/gps', methods=['POST'])
def receive_gps():
    try:
        # 接收来自 FiPy 的 JSON 数据
        data = request.get_json()
        print(f"收到 FiPy 数据: {json.dumps(data)}")
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    # 监听树莓派所有网络接口的 5000 端口
    app.run(host='0.0.0.0', port=5000)
