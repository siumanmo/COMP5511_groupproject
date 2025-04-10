from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import socket

app = Flask(__name__)

def get_local_ip():
    """Get the local network IP address"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't need to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

try:
    model_pkg = joblib.load('models/vitamin_recommender_nb.pkl')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading failed: {str(e)}")
    raise

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    local_ip = get_local_ip()
    print(f"\n🌐 Access URLs:")
    print(f"Local: http://localhost:5000")
    print(f"Network: http://{local_ip}:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=True)