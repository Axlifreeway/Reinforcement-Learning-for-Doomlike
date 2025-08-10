import cv2
import numpy as np
import matplotlib.pyplot as plt

class SimpleHealthDetector:
    def __init__(self):
        self.lower_red1 = np.array([0, 100, 100])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([160, 100, 100])
        self.upper_red2 = np.array([180, 255, 255])
    
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
        
        return {
            'health_percentage': min(100, max(0, health_percentage)),
            'hearts_found': total_hearts,
            'mask': cleaned_mask,
            'health_area': health_area
        }
    
    def visualize_detection(self, screenshot, result):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        axes[0].imshow(screenshot)
        axes[0].set_title("Исходное изображение")
        axes[0].axis('off')
        
        axes[1].imshow(result['health_area'])
        axes[1].set_title("Область поиска")
        axes[1].axis('off')
        
        axes[2].imshow(result['mask'], cmap='gray')
        axes[2].set_title("Маска красного цвета")
        axes[2].axis('off')
        
        overlay = result['health_area'].copy()
        overlay[result['mask'] > 0] = [0, 255, 0]
        axes[3].imshow(cv2.addWeighted(result['health_area'], 0.7, overlay, 0.3, 0))
        axes[3].set_title(f"Здоровье: {result['health_percentage']}%\nСердец: {result['hearts_found']}")
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    image_path = "screen_health.png"
    
    screenshot = cv2.imread(image_path)
    screenshot_rgb = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
    detector = SimpleHealthDetector()
    
    result = detector.detect_health(screenshot_rgb)
    detector.visualize_detection(screenshot_rgb, result)

if __name__ == "__main__":
    main()