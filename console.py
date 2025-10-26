from flask import Flask, render_template, jsonify
import threading

app = Flask(__name__)
drone = None
mission_planner = None

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/status')
def get_status():
    if drone:
        return jsonify({
            'connected': drone.connected,
            'battery': drone.battery_level,
            'liquid': drone.liquid_remaining,
            'position': drone.position,
            'spraying': drone.is_spraying,
            'mission_progress': drone.mission_progress,
            'current_section': drone.current_section.id if drone.current_section else None
        })
    return jsonify({'error': 'Дрон не инициализирован'})

@app.route('/api/start_mission', methods=['POST'])
def start_mission():
    if mission_planner and not mission_planner.mission_active:
        mission_planner.execute_spraying_mission()
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error'})

@app.route('/api/stop_mission', methods=['POST'])
def stop_mission():
    if mission_planner:
        mission_planner.mission_active = False
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error'})

@app.route('/api/refill', methods=['POST'])
def refill():
    if drone:
        drone.refill_liquid()
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error'})

def run_web_interface():
    app.run(host='0.0.0.0', port=5000, debug=False)

# Запуск веб-интерфейса в отдельном потоке
web_thread = threading.Thread(target=run_web_interface)
web_thread.daemon = True
web_thread.start()
