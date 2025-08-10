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
        
        mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
        red_mask = mask1 + mask2
        
        kernel_open = np.ones((2, 2), np.uint8)
        cleaned_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel_open)
        kernel_close = np.ones((3, 3), np.uint8)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_close)
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        heart_contours = []
        min_heart_area = 100
        max_heart_area = 200

        for contour in contours:
            area = cv2.contourArea(contour)
            if min_heart_area < area < max_heart_area:
                heart_contours.append(contour)
        
        total_hearts = len(heart_contours)
        max_hearts = 10
        
        health_percentage = int((total_hearts / max_hearts) * 100)
        
        return  min(100, max(0, health_percentage))
    
    # Метод для определения количества предметов в инвентаре
    def detect_items(self, screenshot):
        """
        Детекция предметов по белой обводке
        """

        height, width = screenshot.shape[:2]
        search_area = screenshot[int(height*0.07):int(height*0.14), int(width*0.3):int(width*0.7)]
        if search_area.size == 0:
            return 0
        
        hsv = cv2.cvtColor(search_area, cv2.COLOR_RGB2HSV)
        
        white_lower, white_upper = (np.array([0, 0, 180]), np.array([180, 30, 255]))
        mask = cv2.inRange(hsv, white_lower, white_upper)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)
        valid_items = []
        h, w = cleaned_mask.shape
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
            width_comp, height_comp = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            
            if (area >= 5 and 
                2 < x < w-2 and 2 < y < h-2 and 
                x + width_comp < w-2 and y + height_comp < h-2 and
                20 < width_comp < 120 and 20 < height_comp < 120):
                
                valid_items.append({
                    'centroid': centroids[i]
                })
        
        grouped_items = self._simple_grouping(valid_items)
        
        return min(len(grouped_items), 20)
    
    def _simple_grouping(self, items):
        """Упрощенная группировка близких элементов"""
        if not items:
            return []
        
        groups = []
        used = [False] * len(items)
        merge_distance = 30
        
        for i, item in enumerate(items):
            if used[i]:
                continue
                
            group = [item]
            used[i] = True
            cx1, cy1 = item['centroid']
            
            for j in range(i + 1, len(items)):
                if used[j]:
                    continue
                    
                cx2, cy2 = items[j]['centroid']
                if np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2) <= merge_distance:
                    group.append(items[j])
                    used[j] = True
            
            groups.append(group)
        
        return groups
    
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