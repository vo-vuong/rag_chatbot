from flask import Flask, jsonify, request
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)

# Route mặc định
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to Flask API!"})

# API nhận dữ liệu từ Postman (POST)
@app.route('/post-data', methods=['POST'])
def post_data():
    data = request.json  # Nhận JSON từ request
    if not data or 'name' not in data:
        return jsonify({"error": "Missing 'name' in request body"}), 400
    
    name = data['name']
    return jsonify({"message": f"Hello, {name}!"})

# Chạy server
if __name__ == '__main__':
    app.run(debug=True)
