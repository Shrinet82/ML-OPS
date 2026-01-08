
from http.server import HTTPServer, BaseHTTPRequestHandler
import random
import time

# Simple Prometheus metrics server
# No heavy dependencies - just expose drift metrics

# Simulated drift values (in real deployment, these would come from Evidently)
drift_score = 0.15
drift_detected = 0

class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global drift_score, drift_detected
        
        if self.path == '/metrics':
            # Simulate some drift variation
            drift_score = 0.1 + random.random() * 0.3  # 0.1 - 0.4
            drift_detected = 1 if drift_score > 0.3 else 0
            
            metrics = f"""# HELP credit_risk_drift_score Data drift score from Evidently
# TYPE credit_risk_drift_score gauge
credit_risk_drift_score {drift_score:.4f}

# HELP credit_risk_drift_detected Whether drift was detected (1=yes, 0=no)
# TYPE credit_risk_drift_detected gauge
credit_risk_drift_detected {drift_detected}

# HELP credit_risk_predictions_total Total predictions logged
# TYPE credit_risk_predictions_total counter
credit_risk_predictions_total {int(time.time()) % 1000}
"""
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; version=0.0.4')
            self.end_headers()
            self.wfile.write(metrics.encode())
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'OK')
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass  # Suppress logging

if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', 8000), MetricsHandler)
    print("Metrics server running on :8000")
    server.serve_forever()
