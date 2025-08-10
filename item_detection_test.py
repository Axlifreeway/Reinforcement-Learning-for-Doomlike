import cv2
import numpy as np

class ColorBorderItemDetector:
    def __init__(self):
        # HSV диапазон для белой обводки (показал лучшие результаты)
        self.white_range = (np.array([0, 0, 180]), np.array([180, 30, 255]))
    
    def detect_items(self, screenshot):
        """Детекция предметов по белой обводке"""
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
        
        for i in range(1, num_labels):  # Пропускаем фон (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
            width_comp, height_comp = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            
            # Фильтры: минимальная площадь, не на границе, разумный размер
            if (area >= 5 and 
                2 < x < w-2 and 2 < y < h-2 and 
                x + width_comp < w-2 and y + height_comp < h-2 and
                20 < width_comp < 120 and 20 < height_comp < 120):
                
                valid_items.append({
                    'centroid': centroids[i]
                })
        
        # Группируем близкие элементы
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
                
            # Создаем новую группу
            group = [item]
            used[i] = True
            cx1, cy1 = item['centroid']
            
            # Добавляем близкие элементы
            for j in range(i + 1, len(items)):
                if used[j]:
                    continue
                    
                cx2, cy2 = items[j]['centroid']
                if np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2) <= merge_distance:
                    group.append(items[j])
                    used[j] = True
            
            groups.append(group)
        
        return groups

def main():
    image_path = "screenitems.png"
    
    screenshot = cv2.imread(image_path)
    screenshot_rgb = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
    
    detector = ColorBorderItemDetector()
    items_count = detector.detect_items(screenshot_rgb)
    
    print(f"Найдено предметов: {items_count}")

if __name__ == "__main__":
    main()