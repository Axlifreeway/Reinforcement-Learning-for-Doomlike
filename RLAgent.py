# RLAgent
# Импорт необходимых библиотек

import cv2
import numpy as np
import pyautogui
import time
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from mss import mss
import random
import matplotlib.pyplot as plt
import tkinter as tk

pyautogui.FAILSAFE = False

# Определение класса среды для RL-агента
class DoomGameEnvironment(gym.Env):
    """
    Класс среды для обучения RL-агента в игре Doom.
    """

    def __init__(self, monitor_index=0, capture_mode="fullscreen"):
        """
        Инициализация среды.
        """
        super(DoomGameEnvironment, self).__init__()

        self.monitor_index = monitor_index
        self.capture_mode = capture_mode
        self.monitors = self._get_monitors_params()

        self.game_region = self._setup_capture_region()
        self.action_space = spaces.MultiDiscrete([5, 360, 2])
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(84, 84, 1), 
            dtype=np.uint8
        )
        
        self.previous_health = 100      
        self.previous_items = 0         
        self.previous_enemies = 0       
        self.step_count = 0            
        self.last_screenshot = None    
        
        self.mouse_sensitivity = 2     
        self.movement_duration = 0.1   

    def _get_monitors_params(self):
        """
        Получение списка доступных мониторов.
        """

        monitors = []

        try:
            
            # Использование mss для получения информации о мониторе
            with mss() as sct:
                for i, monitor in enumerate(sct.monitors):
                    if i == 0:
                        continue
                    monitors.append({
                        'index': i - 1,
                        'top': monitor['top'],
                        'left': monitor['left'],
                        'width': monitor['width'],
                        'height': monitor['height']
                    })
                    print(f"🖥️  Монитор {i-1}: {monitor['width']}x{monitor['height']} в позиции ({monitor['left']}, {monitor['top']})")
                    
        except Exception as e:

            # Если не удалось получить информацию о мониторах, добавляем монитор по умолчанию
            monitors.append({
                'index': 0,
                'top': 0,
                'left': 0,
                'width': 1920,
                'height': 1080
            })
        
        return monitors
    
    def _setup_capture_region(self):
        """
        Настраивает область экрана для захвата в зависимости от режима
        
        Возвращает: словарь с координатами области захвата
        """
        # Проверка корректности индекса монитора
        if self.monitor_index >= len(self.monitors):
            self.monitor_index = 0
            
        monitor = self.monitors[self.monitor_index]
        
        if self.capture_mode == "fullscreen":
            region = {
                "top": monitor['top'],
                "left": monitor['left'],
                "width": monitor['width'],
                "height": monitor['height']
            }
            
        elif self.capture_mode == "gaming_area":
            margin_x = int(monitor['width'] * 0.05)
            margin_y = int(monitor['height'] * 0.05)
            
            region = {
                "top": monitor['top'] + margin_y,
                "left": monitor['left'] + margin_x,
                "width": monitor['width'] - (margin_x * 2),
                "height": monitor['height'] - (margin_y * 2)
            }                
            
        return region
    
    def capture_screen(self):
        """
        Захватывает скриншот заданной области экрана
        
        Возвращает: numpy array с изображением
        """
        try:
            with mss() as sct:
                screenshot = sct.grab(self.game_region)
                img = np.array(screenshot)[:, :, :3]  
                return img
        except Exception as e:
            return np.zeros((self.game_region['height'], self.game_region['width'], 3), dtype=np.uint8)
        
    def preprocess_image(self, image):
        """
        Предобработка изображения для нейросети
        
        image: исходное изображение

        Возвращает: обработанное изображение 84x84 в градациях серого
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        resized = cv2.resize(gray, (84, 84))
        
        resized = np.expand_dims(resized, axis=-1)
        
        return resized
    
    def detect_health(self, screenshot):
        height, width = screenshot.shape[:2]
        health_area = screenshot[int(height * 0.85):, int(width * 0.3):int(width * 0.7)]
        
        hsv = cv2.cvtColor(health_area, cv2.COLOR_RGB2HSV)
        
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        red_mask = mask1 + mask2
        
        kernel_open = np.ones((2, 2), np.uint8)
        cleaned_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel_open)
        kernel_close = np.ones((3, 3), np.uint8)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_close)
        
        # Находим контуры, как и раньше
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        heart_contours = []
        
        # --- УЛУЧШЕННАЯ ЛОГИКА ФИЛЬТРАЦИИ ---
        min_heart_area = 60  # Немного снижаем минимальный порог
        max_heart_area = 500 # Добавляем максимальный порог на всякий случай

        for contour in contours:
            area = cv2.contourArea(contour)
            
            # 1. Фильтруем по диапазону площадей
            if min_heart_area < area < max_heart_area:
                # 2. Добавляем проверку на форму (соотношение сторон)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                
                # Сердца должны быть примерно квадратной формы
                if 0.7 < aspect_ratio < 1.3:
                    heart_contours.append(contour)
        
        total_hearts = len(heart_contours)
        max_hearts = 10  # Предполагаемое максимальное количество сердец
        
        health_percentage = int((total_hearts / max_hearts) * 100)
        
        return  min(100, max(0, health_percentage))
    
    # Метод для обнаружения врагов на основе движения между кадрами
    def detect_enemies(self, screenshot):
        return null
    
    # Метод для определения количества предметов в инвентаре
    def detect_items(self, screenshot):
        return null
    
    # Метод для выполнения игровых действий
    def execute_action(self, action):
        """
        Выполняет действие в игре
        
        action: массив действий
        """
        movement, mouse_turn, shoot = action
        
        pyautogui.keyUp('w')
        pyautogui.keyUp('a')
        pyautogui.keyUp('s')
        pyautogui.keyUp('d')
        
        if movement == 1:
            pyautogui.keyDown('w')                    
            time.sleep(self.movement_duration)        
            pyautogui.keyUp('w')                      
        elif movement == 2:
            pyautogui.keyDown('s')                    
            time.sleep(self.movement_duration)       
            pyautogui.keyUp('s')                     
        elif movement == 3:
            pyautogui.keyDown('a')                    
            time.sleep(self.movement_duration)     
            pyautogui.keyUp('a')                  
        elif movement == 4:
            pyautogui.keyDown('d')                    
            time.sleep(self.movement_duration)
            pyautogui.keyUp('d')
        
        if mouse_turn != 180:  
            turn_angle = mouse_turn - 180
            
            screen_width = self.game_region['width']
            sensitivity_modifier = screen_width / 1920
            adjusted_sensitivity = self.mouse_sensitivity * sensitivity_modifier
            
            mouse_delta = turn_angle * adjusted_sensitivity
            
            pyautogui.moveRel(int(mouse_delta), 0)
        
        # ========== СТРЕЛЬБА ==========
        if shoot == 1:  # Если команда стрелять
            pyautogui.click()  # Эмуляция клика левой кнопкой мыши

    def calculate_reward(self, screenshot):
        """
        Вычисляет награду для обучения AI
        
        screenshot: текущий скриншот
        Возвращает: значение награды
        """

        reward = 0.0
        
        current_health = self.detect_health(screenshot)
        health_change = current_health - self.previous_health
        
        if health_change < 0:
            reward += health_change * 0.5  # Будет отрицательным числом
        elif health_change > 0:
            reward += health_change * 0.3
        
        self.previous_health = current_health
        
        if current_health > 0:
            reward += 1.0
        else:
            reward -= 100.0
        reward += 0.1  
        
        return reward
    
    def reset(self, seed=None, options=None):
        """
        Сбрасывает среду в начальное состояние
        
        Возвращает: начальное наблюдение
        """
        self.step_count = 0
        self.previous_health = 100
        self.last_screenshot = None
        
        time.sleep(0.5)
        screenshot = self.capture_screen()
        observation = self.preprocess_image(screenshot)
        info = {}
        
        return observation, info
    
    def step(self, action):
        """
        Выполняет один шаг в среде
        """
        self.step_count += 1
        
        self.execute_action(action)
        
        time.sleep(0.1)
        screenshot = self.capture_screen()
        observation = self.preprocess_image(screenshot)
        
        reward = self.calculate_reward(screenshot)
        
        current_health = self.detect_health(screenshot)
        
        terminated = (
            current_health <= 0 or self.step_count >= 1000
        )
        
        truncated = False
        
        info = {
            'health': current_health,         
            'step': self.step_count      
        }
        
        return observation, reward, terminated, truncated, info
    

if __name__ == "__main__":
    env = DoomGameEnvironment(monitor_index=0, capture_mode="fullscreen")
    env.detect_health()