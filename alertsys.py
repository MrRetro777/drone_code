class AlertSystem:
    """Система оповещений о критических ситуациях"""
    
    def __init__(self):
        self.alerts = []
    
    def check_alerts(self, drone):
        """Проверка условий для оповещений"""
        alerts = []
        
        if drone.battery_level < 20:
            alerts.append({
                'level': 'warning',
                'message': f'Низкий уровень заряда: {drone.battery_level}%',
                'action': 'RETURN_HOME'
            })
        
        if drone.liquid_remaining < 1.0:
            alerts.append({
                'level': 'warning', 
                'message': 'Мало жидкости для опрыскивания',
                'action': 'REFILL'
            })
        
        if drone.battery_level < 10:
            alerts.append({
                'level': 'critical',
                'message': 'КРИТИЧЕСКИ НИЗКИЙ ЗАРЯД!',
                'action': 'EMERGENCY_LAND'
            })
        
        return alerts
    
    def send_alert(self, alert):
        """Отправка оповещения"""
        self.alerts.append({
            'timestamp': datetime.now().isoformat(),
            **alert
        })
        
        # Здесь может быть интеграция с SMS/email/telegram
        logging.warning(f"ОПОВЕЩЕНИЕ [{alert['level']}]: {alert['message']}")
