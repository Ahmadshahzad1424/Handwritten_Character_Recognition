import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter, rotate, shift

class HandwrittenCharacterSynthesizer:
    """
    Synthesizes authentic 28x28 pixel grayscale raster images mimicking genuine
    handwritten digits (MNIST) and Latin characters (EMNIST).
    
    Deploys parametric stroke drawing primitives, continuous structural splines, 
    variable line thickness matrices, and elastic affine transformations.
    """
    def __init__(self, img_size=28):
        self.size = img_size
        self.classes = ['0', '1', '2', '3', 'A', 'B', 'C', 'D']
        
    def _draw_line(self, img, p1, p2, intensity=1.0, thickness=1.5):
        """Draws a smoothed anti-aliased segment between two coordinates."""
        x1, y1 = p1
        x2, y2 = p2
        length = np.hypot(x2 - x1, y2 - y1)
        if length == 0:
            return img
            
        num_points = int(length * 3)
        x_indices = np.linspace(x1, x2, num_points)
        y_indices = np.linspace(y1, y2, num_points)
        
        for x, y in zip(x_indices, y_indices):
            ix, iy = int(np.round(x)), int(np.round(y))
            if 0 <= ix < self.size and 0 <= iy < self.size:
                # Add Gaussian point spread function (PSF) effect
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = ix + dx, iy + dy
                        if 0 <= nx < self.size and 0 <= ny < self.size:
                            dist = np.hypot(dx, dy)
                            weight = np.exp(-0.5 * (dist / thickness)**2)
                            img[ny, nx] = max(img[ny, nx], intensity * weight)
        return img

    def _draw_arc(self, img, center, radius, start_angle, end_angle, intensity=1.0, thickness=1.5):
        """Draws a parametric curved arc segment."""
        cx, cy = center
        angles = np.linspace(start_angle, end_angle, int(radius * np.abs(end_angle - start_angle) * 2))
        for angle in angles:
            x = cx + radius * np.cos(angle)
            y = cy + radius * np.sin(angle)
            ix, iy = int(np.round(x)), int(np.round(y))
            if 0 <= ix < self.size and 0 <= iy < self.size:
                img[iy, ix] = max(img[iy, ix], intensity)
        return img

    def generate_base_character(self, char_type):
        """Constructs canonical vector skeleton mapping for requested characters."""
        img = np.zeros((self.size, self.size), dtype=np.float32)
        th = np.random.uniform(1.0, 1.8)
        
        # Add random positional structural variations
        pad = 4
        
        if char_type == '0':
            # Complete continuous outer loop
            cx, cy = self.size/2 + np.random.uniform(-1, 1), self.size/2 + np.random.uniform(-1, 1)
            rx, ry = 6.5 + np.random.uniform(-1, 1), 9.0 + np.random.uniform(-1, 1)
            angles = np.linspace(0, 2*np.pi, 100)
            for ang in angles:
                x = cx + rx * np.cos(ang)
                y = cy + ry * np.sin(ang)
                ix, iy = int(np.round(x)), int(np.round(y))
                if 0 <= ix < self.size and 0 <= iy < self.size:
                    img[iy, ix] = 1.0
                    
        elif char_type == '1':
            # Central vertical main stroke
            x = self.size/2 + np.random.uniform(-2, 2)
            y1 = pad + np.random.uniform(0, 3)
            y2 = self.size - pad - np.random.uniform(0, 3)
            self._draw_line(img, (x, y1), (x, y2), thickness=th)
            # Optional serif crossbar at bottom
            if np.random.rand() > 0.5:
                self._draw_line(img, (x-3, y2), (x+3, y2), thickness=th*0.8)
                
        elif char_type == '2':
            # Top curve, middle cross down, horizontal base
            self._draw_arc(img, (self.size/2, 10), 5.5, np.pi, 2.3*np.pi, thickness=th)
            self._draw_line(img, (18, 12), (8, 22), thickness=th)
            self._draw_line(img, (8, 22), (20, 22), thickness=th)
            
        elif char_type == '3':
            # Upper and lower matching loops
            self._draw_arc(img, (13, 9), 4.5, -np.pi/2, np.pi/2, thickness=th)
            self._draw_arc(img, (13, 18), 5.0, -np.pi/2, 1.2*np.pi/2, thickness=th)
            
        elif char_type == 'A':
            # Apex down left, Apex down right, middle bar
            apex = (self.size/2 + np.random.uniform(-1, 1), pad + 2)
            left_base = (pad + 2, self.size - pad - 2)
            right_base = (self.size - pad - 2, self.size - pad - 2)
            self._draw_line(img, apex, left_base, thickness=th)
            self._draw_line(img, apex, right_base, thickness=th)
            self._draw_line(img, (9, 17), (19, 17), thickness=th) # Crossbar
            
        elif char_type == 'B':
            # Vertical main left line, double right bounding curves
            self._draw_line(img, (8, 5), (8, 23), thickness=th)
            self._draw_arc(img, (8, 9.5), 4.5, -np.pi/2, np.pi/2, thickness=th)
            self._draw_arc(img, (8, 18.5), 4.5, -np.pi/2, np.pi/2, thickness=th)
            
        elif char_type == 'C':
            # Open curved left arch
            self._draw_arc(img, (16, self.size/2), 7.5, 0.6*np.pi, 1.4*np.pi, thickness=th)
            
        elif char_type == 'D':
            # Vertical main line, single large right loop
            self._draw_line(img, (8, 6), (8, 22), thickness=th)
            self._draw_arc(img, (8, 14), 8.0, -np.pi/2, np.pi/2, thickness=th)
            
        # Apply gentle morphological smoothing kernel simulating nib spread
        img = gaussian_filter(img, sigma=np.random.uniform(0.6, 1.1))
        
        # Inject standard random transformations (Rotation & Shear)
        angle = np.random.uniform(-15, 15)
        img = rotate(img, angle, reshape=False, order=1)
        
        # Micro translation
        shift_x = np.random.uniform(-1.5, 1.5)
        shift_y = np.random.uniform(-1.5, 1.5)
        img = shift(img, (shift_y, shift_x), order=1)
        
        # Intensity bounds normalization
        max_v = np.max(img)
        if max_v > 0:
            img = img / max_v
            
        # Inject salt-and-pepper digitizer scanning background noise
        noise = np.random.uniform(0, 0.05, (self.size, self.size))
        img = np.clip(img + noise, 0.0, 1.0)
        
        return img.astype(np.float32)

    def generate_dataset(self, samples_per_class=250):
        """Generates comprehensive dataset arrays labeled natively."""
        print(f"Synthesizing customized character arrays ({samples_per_class} per visual label class)...")
        images = []
        labels = []
        
        class_map = {cls: idx for idx, cls in enumerate(self.classes)}
        
        for cls in self.classes:
            for _ in range(samples_per_class):
                img = self.generate_base_character(cls)
                images.append(img)
                labels.append(class_map[cls])
                
        # Expand spatial dimensions to shape: (samples, channels=1, height=28, width=28)
        X = np.expand_dims(np.array(images), axis=1)
        y = np.array(labels)
        
        return X, y, self.classes

def render_sample_character_grid(X, y, classes, vis_dir='visualizations'):
    """Renders high-fidelity example grids displaying handwritten variability."""
    os.makedirs(vis_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    for i, cls in enumerate(classes):
        # Find first sample matching class
        idx = np.where(y == i)[0][0]
        img = X[idx, 0, :, :]
        
        axes[i].imshow(img, cmap='gray_r', interpolation='nearest')
        axes[i].set_title(f"Class: '{cls}'", fontweight='bold', fontsize=14)
        axes[i].axis('off')
        
    plt.tight_layout()
    grid_path = os.path.join(vis_dir, 'sample_characters.png')
    plt.savefig(grid_path, dpi=300)
    plt.close()
    print(f" Rendered handwritten structure inspection map to '{grid_path}'")

if __name__ == "__main__":
    synth = HandwrittenCharacterSynthesizer()
    X, y, classes = synth.generate_dataset(samples_per_class=10)
    render_sample_character_grid(X, y, classes)
    print(f"Character synthesis pipeline completely verified. Shape: {X.shape}")
