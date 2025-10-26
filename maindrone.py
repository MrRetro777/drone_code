#!/usr/bin/env python3
"""
Автономная система опрыскивания полей от вредителей с использованием дрона
Система включает: планирование маршрута, мониторинг вредителей, автоматическое опрыскивание
"""

import time
import json
import logging
import threading
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
import math
from math import radians, cos, sin, sqrt
import cv2
import numpy as np
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('spraying_mission.log'),
        logging.StreamHandler()
    ]
)

class SprayerStatus(Enum):
    OFF = 0
    SPRAYING = 1
    PAUSED = 2
    EMPTY = 3

@dataclass
class FieldSection:
    id: int
    coordinates: List[Tuple[float, float]]  # GPS координаты границ участка
    area: float  # площадь в гектарах
    crop_type: str
    pest_level: float  # уровень зараженности 0-1
    sprayed: bool = False

@dataclass
class SprayingParameters:
    liquid_flow: float  # л/мин
    spray_width: float  # м
    flight_speed: float  # м/с
    liquid_density: float  # л/га

class AgriculturalSprayingDrone:
    """
    Класс для управления дроном-опрыскивателем
    """
    
    def __init__(self, drone_id: str = "AGDRONE-001"):
        self.drone_id = drone_id
        self.connected = False
        self.position = (0.0, 0.0, 0.0)  # (lat, lon, alt)
        self.home_position = (0.0, 0.0, 0.0)
        self.battery_level = 100
        self.sprayer_status = SprayerStatus.OFF
        self.liquid_remaining = 10.0  # литры
        self.liquid_capacity = 10.0  # литры
        self.is_spraying = False
        self.current_section = None
        self.mission_progress = 0
        
        # Параметры опрыскивания
        self.spraying_params = SprayingParameters(
            liquid_flow=1.0,
            spray_width=4.0,
            flight_speed=5.0,
            liquid_density=200.0  # л/га
        )
        
        # Система мониторинга
        self.camera = None
        self.pest_detector = PestDetector()
        
        # Данные миссии
        self.field_sections: List[FieldSection] = []
        self.spraying_log = []
        
        logging.info(f"Инициализирован дрон-опрыскиватель {drone_id}")

    def connect(self) -> bool:
        """Подключение к дрону"""
        try:
            # Имитация подключения к реальному дрону
            # В реальной системе здесь будет код для DJI SDK или другого API
            self.connected = True
            self.battery_level = 95
            self.liquid_remaining = self.liquid_capacity
            
            # Инициализация камеры для мониторинга
            self._initialize_camera()
            
            logging.info("Дрон подключен и готов к работе")
            return True
        except Exception as e:
            logging.error(f"Ошибка подключения: {e}")
            return False

    def _initialize_camera(self):
        """Инициализация камеры для мониторинга"""
        try:
            # Для реального дрона здесь будет инициализация камеры DJI
            self.camera = cv2.VideoCapture(0)  # Имитация камеры
            logging.info("Камера для мониторинга инициализирована")
        except Exception as e:
            logging.warning(f"Не удалось инициализировать камеру: {e}")

    def takeoff(self, altitude: float = 10.0) -> bool:
        """Взлет на заданную высоту"""
        if not self.connected:
            logging.error("Дрон не подключен")
            return False
            
        if self.battery_level < 20:
            logging.error("Недостаточный уровень заряда для взлета")
            return False
            
        try:
            # Имитация взлета
            logging.info(f"Взлет на высоту {altitude} метров")
            self.position = (self.position[0], self.position[1], altitude)
            self.home_position = self.position
            time.sleep(2)  # Имитация времени взлета
            logging.info("Взлет завершен")
            return True
        except Exception as e:
            logging.error(f"Ошибка при взлете: {e}")
            return False

    def land(self) -> bool:
        """Посадка дрона"""
        try:
            logging.info("Начало посадки")
            # Выключение опрыскивателя при посадке
            self.stop_spraying()
            
            # Имитация посадки
            self.position = (self.position[0], self.position[1], 0.0)
            time.sleep(2)
            logging.info("Посадка завершена")
            return True
        except Exception as e:
            logging.error(f"Ошибка при посадке: {e}")
            return False

    def goto_position(self, lat: float, lon: float, alt: float = 15.0) -> bool:
        """Перелет в указанную позицию"""
        try:
            logging.info(f"Перелет к позиции: ({lat:.6f}, {lon:.6f}, {alt}m)")
            
            # Имитация перелета
            distance = self._calculate_distance(self.position, (lat, lon, alt))
            flight_time = distance / 8.0  # Предполагаемая скорость 8 м/с
            
            # Обновление позиции
            self.position = (lat, lon, alt)
            
            # Расход батареи во время полета
            battery_consumption = distance * 0.1  # 0.1% батареи на метр
            self.battery_level = max(0, self.battery_level - battery_consumption)
            
            logging.info(f"Перелет завершен. Пройдено: {distance:.1f}м")
            return True
        except Exception as e:
            logging.error(f"Ошибка при перелете: {e}")
            return False

    def start_spraying(self) -> bool:
        """Запуск системы опрыскивания"""
        if self.liquid_remaining <= 0:
            logging.error("Жидкость для опрыскивания закончилась")
            self.sprayer_status = SprayerStatus.EMPTY
            return False
            
        try:
            self.sprayer_status = SprayerStatus.SPRAYING
            self.is_spraying = True
            logging.info("Система опрыскивания запущена")
            
            # Запуск мониторинга расхода жидкости
            self._start_liquid_monitoring()
            
            return True
        except Exception as e:
            logging.error(f"Ошибка запуска опрыскивания: {e}")
            return False

    def stop_spraying(self) -> bool:
        """Остановка системы опрыскивания"""
        try:
            self.sprayer_status = SprayerStatus.OFF
            self.is_spraying = False
            logging.info("Система опрыскивания остановлена")
            return True
        except Exception as e:
            logging.error(f"Ошибка остановки опрыскивания: {e}")
            return False

    def _start_liquid_monitoring(self):
        """Мониторинг расхода жидкости"""
        def monitor():
            while self.is_spraying and self.liquid_remaining > 0:
                # Расход жидкости в зависимости от параметров опрыскивания
                liquid_used = (self.spraying_params.liquid_flow / 60) * 1  # за 1 секунду
                self.liquid_remaining = max(0, self.liquid_remaining - liquid_used)
                
                if self.liquid_remaining <= 0:
                    self.stop_spraying()
                    self.sprayer_status = SprayerStatus.EMPTY
                    logging.warning("Жидкость для опрыскивания закончилась")
                    break
                    
                time.sleep(1)
        
        thread = threading.Thread(target=monitor)
        thread.daemon = True
        thread.start()

    def analyze_pest_level(self, image: np.ndarray = None) -> float:
        """Анализ уровня зараженности вредителями"""
        try:
            if image is None and self.camera:
                ret, frame = self.camera.read()
                if ret:
                    image = frame
            
            if image is not None:
                pest_level = self.pest_detector.analyze_image(image)
                logging.info(f"Уровень зараженности: {pest_level:.2f}")
                return pest_level
            else:
                # Если камера недоступна, возвращаем случайное значение для демонстрации
                return 0.3 + 0.4 * (hash(str(self.position)) % 100) / 100
                
        except Exception as e:
            logging.warning(f"Ошибка анализа зараженности: {e}")
            return 0.5

    def _calculate_distance(self, pos1: tuple, pos2: tuple) -> float:
        """Расчет расстояния между двумя точками (упрощенный)"""
        lat1, lon1, alt1 = pos1
        lat2, lon2, alt2 = pos2
        
        # Упрощенный расчет расстояния (для демонстрации)
        lat_diff = (lat2 - lat1) * 111000  # метров на градус широты
        lon_diff = (lon2 - lon1) * 111000 * cos(radians((lat1 + lat2) / 2))
        alt_diff = alt2 - alt1
        
        return sqrt(lat_diff**2 + lon_diff**2 + alt_diff**2)

    def refill_liquid(self, amount: float = None) -> bool:
        """Заправка жидкости для опрыскивания"""
        try:
            if amount is None:
                amount = self.liquid_capacity - self.liquid_remaining
            
            self.liquid_remaining = min(self.liquid_capacity, self.liquid_remaining + amount)
            self.sprayer_status = SprayerStatus.OFF
            logging.info(f"Заправлено {amount:.1f}л жидкости. Всего: {self.liquid_remaining:.1f}л")
            return True
        except Exception as e:
            logging.error(f"Ошибка заправки: {e}")
            return False

class PestDetector:
    """Класс для обнаружения вредителей на изображениях"""
    
    def __init__(self):
        # В реальной системе здесь будет загружена ML модель
        self.model_loaded = False
        self._load_model()
    
    def _load_model(self):
        """Загрузка модели для обнаружения вредителей"""
        try:
            # Имитация загрузки ML модели
            # В реальности: tensorflow, pytorch, или другая ML библиотека
            self.model_loaded = True
            logging.info("Модель обнаружения вредителей загружена")
        except Exception as e:
            logging.warning(f"Не удалось загрузить модель: {e}")
    
    def analyze_image(self, image: np.ndarray) -> float:
        """Анализ изображения на наличие вредителей"""
        try:
            if not self.model_loaded:
                # Базовая имитация анализа для демонстрации
                return self._simulate_pest_detection(image)
            
            # Здесь будет код реального ML анализа
            # Например, использование YOLO для обнаружения насекомых
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Простая логика для демонстрации
            pest_level = np.random.random() * 0.3 + 0.1  # Имитация обнаружения
            
            return min(1.0, pest_level)
            
        except Exception as e:
            logging.error(f"Ошибка анализа изображения: {e}")
            return 0.5
    
    def _simulate_pest_detection(self, image: np.ndarray) -> float:
        """Имитация обнаружения вредителей для демонстрации"""
        # Анализ цвета и текстуры для определения здоровья растений
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Цветовые диапазоны для здоровых/больных растений
        healthy_green_low = np.array([35, 40, 40])
        healthy_green_high = np.array([85, 255, 255])
        
        # Создание маски здоровых областей
        healthy_mask = cv2.inRange(hsv, healthy_green_low, healthy_green_high)
        
        # Процент здоровых пикселей
        healthy_ratio = np.sum(healthy_mask > 0) / (image.shape[0] * image.shape[1])
        
        # Уровень зараженности обратно пропорционален здоровым участкам
        pest_level = max(0.1, 1.0 - healthy_ratio)
        
        return pest_level

class FieldMissionPlanner:
    """Планировщик миссий для опрыскивания поля"""
    
    def __init__(self, drone: AgriculturalSprayingDrone):
        self.drone = drone
        self.mission_active = False
        
    def load_field_data(self, field_data: Dict):
        """Загрузка данных о поле"""
        try:
            self.field_sections = []
            
            for section_data in field_data.get('sections', []):
                section = FieldSection(
                    id=section_data['id'],
                    coordinates=section_data['coordinates'],
                    area=section_data['area'],
                    crop_type=section_data['crop_type'],
                    pest_level=section_data.get('pest_level', 0.5)
                )
                self.field_sections.append(section)
            
            logging.info(f"Загружено {len(self.field_sections)} участков поля")
            return True
        except Exception as e:
            logging.error(f"Ошибка загрузки данных поля: {e}")
            return False
    
    def plan_spraying_route(self) -> List[FieldSection]:
        """Планирование маршрута опрыскивания"""
        # Сортировка участков по уровню зараженности (сначала наиболее зараженные)
        sorted_sections = sorted(self.field_sections, 
                               key=lambda x: x.pest_level, 
                               reverse=True)
        
        # Расчет необходимого количества жидкости
        total_area = sum(section.area for section in sorted_sections)
        required_liquid = total_area * self.drone.spraying_params.liquid_density / 1000  # в литрах
        
        logging.info(f"Общая площадь: {total_area:.2f} га")
        logging.info(f"Требуется жидкости: {required_liquid:.1f} л")
        
        if required_liquid > self.drone.liquid_capacity:
            logging.warning("Недостаточно жидкости для полного опрыскивания")
            # Выбираем наиболее зараженные участки в пределах емкости
            available_liquid = self.drone.liquid_remaining
            selected_sections = []
            
            for section in sorted_sections:
                section_liquid = section.area * self.drone.spraying_params.liquid_density / 1000
                if section_liquid <= available_liquid:
                    selected_sections.append(section)
                    available_liquid -= section_liquid
                else:
                    break
                    
            return selected_sections
        
        return sorted_sections
    
    def execute_spraying_mission(self) -> bool:
        """Выполнение миссии опрыскивания"""
        if self.mission_active:
            logging.warning("Миссия уже выполняется")
            return False
            
        def mission_thread():
            self.mission_active = True
            total_sections = len(self.drone.field_sections)
            completed_sections = 0
            
            try:
                # Планирование маршрута
                route = self.plan_spraying_route()
                
                for i, section in enumerate(route):
                    if not self.mission_active:
                        break
                        
                    logging.info(f"Обработка участка {section.id} ({section.crop_type})")
                    
                    # Перелет к участку
                    center = self._calculate_section_center(section)
                    self.drone.goto_position(center[0], center[1], 15.0)
                    
                    # Анализ текущего уровня зараженности
                    current_pest_level = self.drone.analyze_pest_level()
                    logging.info(f"Текущий уровень зараженности: {current_pest_level:.2f}")
                    
                    # Определение необходимости опрыскивания
                    if current_pest_level > 0.2:  # Порог для опрыскивания
                        # Опрыскивание участка
                        if self._spray_section(section):
                            section.sprayed = True
                            completed_sections += 1
                            logging.info(f"Участок {section.id} обработан")
                        else:
                            logging.warning(f"Не удалось обработать участок {section.id}")
                    else:
                        logging.info(f"Участок {section.id} не требует обработки")
                    
                    # Обновление прогресса
                    self.drone.mission_progress = (i + 1) / total_sections * 100
                    self.drone.current_section = section
                    
                    # Проверка уровня жидкости и батареи
                    if self.drone.liquid_remaining <= 0:
                        logging.warning("Жидкость закончилась, завершение миссии")
                        break
                        
                    if self.drone.battery_level < 20:
                        logging.warning("Низкий уровень заряда, возврат на базу")
                        break
                
                # Возврат на домашнюю позицию
                logging.info("Возврат на домашнюю позицию")
                self.drone.goto_position(*self.drone.home_position)
                
                # Посадка
                self.drone.land()
                
                # Логирование результатов миссии
                self._log_mission_results(completed_sections, total_sections)
                
            except Exception as e:
                logging.error(f"Ошибка во время миссии: {e}")
            finally:
                self.mission_active = False
                self.drone.mission_progress = 100
        
        # Запуск миссии в отдельном потоке
        thread = threading.Thread(target=mission_thread)
        thread.daemon = True
        thread.start()
        
        return True
    
    def _spray_section(self, section: FieldSection) -> bool:
        """Опрыскивание конкретного участка"""
        try:
            # Расчет времени опрыскивания для участка
            area = section.area * 10000  # перевод в м²
            spray_width = self.drone.spraying_params.spray_width
            path_length = area / spray_width
            
            flight_time = path_length / self.drone.spraying_params.flight_speed
            required_liquid = area * self.drone.spraying_params.liquid_density / 10000 / 1000  # в литрах
            
            logging.info(f"Опрыскивание участка {section.id}: {flight_time:.1f} сек, {required_liquid:.1f} л")
            
            # Запуск опрыскивания
            if not self.drone.start_spraying():
                return False
            
            # Имитация времени опрыскивания
            time.sleep(min(flight_time, 10))  # Максимум 10 секунд для демонстрации
            
            # Остановка опрыскивания
            self.drone.stop_spraying()
            
            # Запись в лог
            spray_log = {
                'timestamp': datetime.now().isoformat(),
                'section_id': section.id,
                'area': section.area,
                'pest_level': section.pest_level,
                'liquid_used': required_liquid,
                'position': self.drone.position
            }
            self.drone.spraying_log.append(spray_log)
            
            return True
            
        except Exception as e:
            logging.error(f"Ошибка опрыскивания участка: {e}")
            self.drone.stop_spraying()
            return False
    
    def _calculate_section_center(self, section: FieldSection) -> Tuple[float, float]:
        """Расчет центра участка"""
        lats = [coord[0] for coord in section.coordinates]
        lons = [coord[1] for coord in section.coordinates]
        
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
        
        return (center_lat, center_lon)
    
    def _log_mission_results(self, completed: int, total: int):
        """Логирование результатов миссии"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'drone_id': self.drone.drone_id,
            'completed_sections': completed,
            'total_sections': total,
            'success_rate': (completed / total) * 100 if total > 0 else 0,
            'liquid_remaining': self.drone.liquid_remaining,
            'battery_remaining': self.drone.battery_level,
            'spraying_log': self.drone.spraying_log
        }
        
        # Сохранение в файл
        with open(f'mission_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"Миссия завершена. Обработано: {completed}/{total} участков")

def main():
    """Основная функция для демонстрации работы системы"""
    
    # Создание дрона
    drone = AgriculturalSprayingDrone("AGDRONE-001")
    
    # Подключение к дрону
    if not drone.connect():
        logging.error("Не удалось подключиться к дрону")
        return
    
    # Заправка жидкости
    drone.refill_liquid(10.0)
    
    # Создание планировщика миссий
    mission_planner = FieldMissionPlanner(drone)
    
    # Пример данных поля (в реальности загружаются из GIS системы)
    field_data = {
        'sections': [
            {
                'id': 1,
                'coordinates': [(55.7558, 37.6173), (55.7559, 37.6174), (55.7557, 37.6175)],
                'area': 2.5,
                'crop_type': 'пшеница',
                'pest_level': 0.8
            },
            {
                'id': 2,
                'coordinates': [(55.7560, 37.6176), (55.7561, 37.6177), (55.7559, 37.6178)],
                'area': 3.0,
                'crop_type': 'ячмень',
                'pest_level': 0.6
            },
            {
                'id': 3,
                'coordinates': [(55.7562, 37.6179), (55.7563, 37.6180), (55.7561, 37.6181)],
                'area': 1.5,
                'crop_type': 'овёс',
                'pest_level': 0.3
            }
        ]
    }
    
    # Загрузка данных поля
    if not mission_planner.load_field_data(field_data):
        logging.error("Не удалось загрузить данные поля")
        return
    
    # Установка домашней позиции (база)
    drone.home_position = (55.7558, 37.6173, 0.0)
    drone.position = drone.home_position
    
    # Взлет
    if not drone.takeoff(15.0):
        logging.error("Не удалось взлететь")
        return
    
    # Запуск миссии опрыскивания
    logging.info("Запуск миссии опрыскивания")
    mission_planner.execute_spraying_mission()
    
    # Мониторинг прогресса
    try:
        while mission_planner.mission_active:
            print(f"Прогресс миссии: {drone.mission_progress:.1f}% | "
                  f"Батарея: {drone.battery_level:.1f}% | "
                  f"Жидкость: {drone.liquid_remaining:.1f}л")
            time.sleep(5)
    except KeyboardInterrupt:
        logging.info("Получен сигнал прерывания")
        mission_planner.mission_active = False
    
    logging.info("Программа завершена")

if __name__ == "__main__":
    main()
