import torch
import numpy as np
from PIL import Image
import cv2
import time

class A2V_Multi_Image_Composite:
    """A2V Multi Image Composite Node with Interactive Preview"""
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite_images"
    CATEGORY = "A2V/Image"
    OUTPUT_NODE = True

    def __init__(self):
        self.last_update_time = 0
        self.update_interval = 0.016  # ~60 FPS
        
        self.preview_window = "A2V Image Adjustment"
        self.control_window = "Control Panel"
        self.layers_window = "Layers Panel"
        
        self.dragging = False
        self.selected_image = 0
        self.current_images = []
        self.current_bg = None
        self.image_params = []
        self.layer_order = []
        self.has_alpha = False
        
        self.scale_mode = False
        self.rotate_mode = False
        self.show_grid = False
        
        self.cached_images = []
        self.cached_params = []
        self.drag_start_pos = None
        self.drag_start_window = None
        
        cv2.setNumThreads(8)

    @classmethod
    def INPUT_TYPES(cls):
        default_input = {
            "default": 0,
            "min": -4096,
            "max": 4096,
            "step": 1
        }
        default_scale = {
            "default": 1.0,
            "min": 0.1,
            "max": 10.0,
            "step": 0.1
        }
        default_rotation = {
            "default": 0.0,
            "min": -180.0,
            "max": 180.0,
            "step": 1.0
        }
        default_opacity = {
            "default": 1.0,
            "min": 0.0,
            "max": 1.0,
            "step": 0.1
        }

        required = {
            "background": ("IMAGE",),
            "image1": ("IMAGE",),
            "x_pos1": ("INT", default_input),
            "y_pos1": ("INT", default_input),
            "scale1": ("FLOAT", default_scale),
            "rotation1": ("FLOAT", default_rotation),
            "opacity1": ("FLOAT", default_opacity),
            "blend_mode1": (["normal", "multiply", "screen", "overlay"],),
            "flip_h1": ("BOOLEAN", {"default": False}),
            "flip_v1": ("BOOLEAN", {"default": False}),
            "enable_preview": ("BOOLEAN", {"default": True}),
        }

        optional = {}
        for i in range(2, 6):  # Add inputs for images 2-5
            optional.update({
                f"image{i}": ("IMAGE",),
                f"x_pos{i}": ("INT", default_input),
                f"y_pos{i}": ("INT", default_input),
                f"scale{i}": ("FLOAT", default_scale),
                f"rotation{i}": ("FLOAT", default_rotation),
                f"opacity{i}": ("FLOAT", default_opacity),
                f"blend_mode{i}": (["normal", "multiply", "screen", "overlay"],),
                f"flip_h{i}": ("BOOLEAN", {"default": False}),
                f"flip_v{i}": ("BOOLEAN", {"default": False}),
            })

        return {
            "required": required,
            "optional": optional
        }

    def create_windows(self):
        try:
            cv2.namedWindow(self.preview_window, cv2.WINDOW_NORMAL)
            cv2.namedWindow(self.control_window, cv2.WINDOW_NORMAL)
            cv2.namedWindow(self.layers_window, cv2.WINDOW_NORMAL)

            bg_h, bg_w = self.current_bg.shape[:2]
            win_w = min(1024, bg_w)
            win_h = int(win_w * (bg_h / bg_w))
            
            cv2.resizeWindow(self.preview_window, win_w, win_h)
            cv2.resizeWindow(self.control_window, 400, 200)
            cv2.resizeWindow(self.layers_window, 200, 300)
            
            cv2.moveWindow(self.preview_window, 100, 100)
            cv2.moveWindow(self.control_window, 100 + win_w + 20, 100)
            cv2.moveWindow(self.layers_window, 100 + win_w + 20, 320)

            # Create trackbars
            params = self.image_params[self.selected_image]
            
            cv2.createTrackbar('Scale %', self.control_window, 
                             int(params['scale']*100), 1000,
                             lambda x: self.update_param('scale', x/100))
            cv2.createTrackbar('Rotate°', self.control_window, 
                             int(params['rotation']+180), 360,
                             lambda x: self.update_param('rotation', x-180))
            cv2.createTrackbar('Opacity %', self.control_window,
                             int(params['opacity']*100), 100,
                             lambda x: self.update_param('opacity', x/100))
            cv2.createTrackbar('Grid', self.control_window,
                             int(self.show_grid), 1,
                             lambda x: self.toggle_grid(x))
            cv2.createTrackbar('Flip H', self.control_window,
                             int(params.get('flip_h', False)), 1,
                             lambda x: self.update_param('flip_h', bool(x)))
            cv2.createTrackbar('Flip V', self.control_window,
                             int(params.get('flip_v', False)), 1,
                             lambda x: self.update_param('flip_v', bool(x)))

            cv2.setMouseCallback(self.preview_window, self.mouse_callback)
            cv2.setMouseCallback(self.layers_window, self.layers_mouse_callback)
            
        except Exception as e:
            print(f"Warning: Control panel creation failed: {str(e)}")

    def convert_to_display(self, img):
        """Convert image to correct color space for display"""
        if img.shape[2] == 4:  # RGBA
            if not self.has_alpha:
                self.has_alpha = True
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        else:  # RGB
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def draw_grid(self, img):
        h, w = img.shape[:2]
        overlay = np.zeros_like(img)
        
        grid_color = (128, 128, 128)
        center_color = (0, 255, 0)
        
        for x in range(0, w, 50):
            cv2.line(overlay, (x, 0), (x, h), grid_color, 1)
        for y in range(0, h, 50):
            cv2.line(overlay, (0, y), (w, y), grid_color, 1)

        cv2.line(overlay, (w//2, 0), (w//2, h), center_color, 1)
        cv2.line(overlay, (0, h//2), (w, h//2), center_color, 1)

        return cv2.addWeighted(overlay, 0.3, img, 1.0, 0)

    def create_status_overlay(self, img):
        params = self.image_params[self.selected_image]
        layer_idx = self.layer_order.index(self.selected_image)

        # Create flip status
        flip_status = []
        if params.get('flip_h', False):
            flip_status.append('H')
        if params.get('flip_v', False):
            flip_status.append('V')
        flip_str = ''.join(flip_status) if flip_status else 'None'

        status = f"Image {self.selected_image+1}/{len(self.current_images)} | " \
                f"Layer {layer_idx+1}/{len(self.layer_order)} | " \
                f"Pos: ({params['x_pos']}, {params['y_pos']}) | " \
                f"Scale: {params['scale']:.2f}x | " \
                f"Rot: {params['rotation']}° | " \
                f"Flip: {flip_str} | " \
                f"Opacity: {params['opacity']:.2f} | " \
                f"Blend: {params['blend_mode']} | " \
                f"{'Scale' if self.scale_mode else 'Rotate' if self.rotate_mode else 'Move'} Mode"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        
        (text_w, text_h), baseline = cv2.getTextSize(status, font, font_scale, thickness)
        
        cv2.rectangle(img, (8, 8), (text_w + 12, text_h + 12), (0, 0, 0), -1)
        cv2.putText(img, status, (10, text_h + 10), font, font_scale, (255, 255, 255), thickness)
        
        return img

    def update_layer_window(self):
        height = len(self.layer_order) * 30 + 10
        panel = np.ones((max(height, 100), 190, 3), dtype=np.uint8) * 240

        for i, layer_idx in enumerate(self.layer_order):
            y = i * 30 + 5
            is_selected = layer_idx == self.selected_image
            color = (200, 220, 255) if is_selected else (255, 255, 255)
            cv2.rectangle(panel, (5, y), (185, y+25), color, -1)
            cv2.rectangle(panel, (5, y), (185, y+25), (180, 180, 180), 1)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"Image {layer_idx+1}"
            cv2.putText(panel, text, (10, y+17), font, 0.5, (0, 0, 0), 1)
            
            if i > 0:  # Can move up
                cv2.putText(panel, "▲", (160, y+17), font, 0.5, (0, 0, 0), 1)
            if i < len(self.layer_order)-1:  # Can move down
                cv2.putText(panel, "▼", (175, y+17), font, 0.5, (0, 0, 0), 1)

        cv2.imshow(self.layers_window, panel)

    def layers_mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            layer_idx = y // 30
            if 0 <= layer_idx < len(self.layer_order):
                if x > 155:  # Arrow controls area
                    if x < 170 and layer_idx > 0:  # Move up
                        self.move_layer_up(layer_idx)
                    elif x >= 170 and layer_idx < len(self.layer_order)-1:  # Move down
                        self.move_layer_down(layer_idx)
                else:  # Select layer
                    self.selected_image = self.layer_order[layer_idx]
                    self.update_control_panel()
                self.update_preview()

    def move_layer_up(self, idx):
        if idx > 0:
            self.layer_order[idx], self.layer_order[idx-1] = \
                self.layer_order[idx-1], self.layer_order[idx]
            self.update_preview()

    def move_layer_down(self, idx):
        if idx < len(self.layer_order)-1:
            self.layer_order[idx], self.layer_order[idx+1] = \
                self.layer_order[idx+1], self.layer_order[idx]
            self.update_preview()

    def apply_blend_mode(self, background, foreground, mode, opacity):
        """Apply blend mode between background and foreground with proper alpha handling"""
        background = background.astype(np.float32) / 255.0
        foreground = foreground.astype(np.float32) / 255.0
        
        # Expand opacity to 3 channels to match background shape
        opacity_3ch = np.dstack([opacity] * 3)

        if mode == "normal":
            result = background * (1 - opacity_3ch) + foreground * opacity_3ch
        elif mode == "multiply":
            result = background * foreground * opacity_3ch + background * (1 - opacity_3ch)
        elif mode == "screen":
            result = 1 - (1 - background) * (1 - foreground * opacity_3ch)
        elif mode == "overlay":
            mask = background >= 0.5
            result = np.zeros_like(background)
            result[mask] = 1 - 2 * (1 - background[mask]) * (1 - foreground[mask])
            result[~mask] = 2 * background[~mask] * foreground[~mask]
            result = result * opacity_3ch + background * (1 - opacity_3ch)
        
        return (np.clip(result, 0, 1) * 255).astype(np.uint8)

    def transform_image(self, img, params):
        """Transform image with scale, rotation, and flips"""
        h, w = img.shape[:2]
        center = (w/2, h/2)
        
        M = cv2.getRotationMatrix2D(center, params['rotation'], params['scale'])
        
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        if img.shape[2] == 4:
            rgb_img = img[:, :, :3]
            alpha_img = img[:, :, 3]
            
            rgb_transformed = cv2.warpAffine(
                rgb_img, M, (new_w, new_h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )
            
            alpha_transformed = cv2.warpAffine(
                alpha_img, M, (new_w, new_h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            
            result = np.dstack((rgb_transformed, alpha_transformed))
        else:
            result = cv2.warpAffine(
                img, M, (new_w, new_h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )

        # Apply flips if enabled
        if params.get('flip_h', False):
            result = cv2.flip(result, 1)  # Horizontal flip
        if params.get('flip_v', False):
            result = cv2.flip(result, 0)  # Vertical flip
            
        return result

    def select_layer_at_point(self, x, y):
        """Find the topmost layer at clicked point"""
        for layer_idx in self.layer_order:
            params = self.image_params[layer_idx]
            img = self.current_images[layer_idx]
            transformed = self.transform_image(img, params)
            
            h_img, w_img = transformed.shape[:2]
            x_pos = params['x_pos']
            y_pos = params['y_pos']
            
            if (x_pos <= x < x_pos + w_img and 
                y_pos <= y < y_pos + h_img):
                local_x = x - x_pos
                local_y = y - y_pos
                if transformed.shape[2] == 4:
                    alpha = transformed[local_y, local_x, 3]
                    if alpha > 0:  # Point is on non-transparent area
                        return layer_idx
                else:
                    return layer_idx
        return None

    def update_preview(self):
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return

        if not self.current_images or self.current_bg is None:
            return

        try:
            result = self.current_bg.copy()
            
            # Composite all images
            for layer_idx in self.layer_order:
                img = self.current_images[layer_idx]
                params = self.image_params[layer_idx]
                
                transformed = self.transform_image(img, params)
                
                x_pos = params['x_pos']
                y_pos = params['y_pos']
                h_img, w_img = transformed.shape[:2]
                h_bg, w_bg = result.shape[:2]
                
                x1 = max(0, x_pos)
                y1 = max(0, y_pos)
                x2 = min(w_bg, x_pos + w_img)
                y2 = min(h_bg, y_pos + h_img)
                x1_img = max(0, -x_pos)
                y1_img = max(0, -y_pos)
                
                if x1 < w_bg and y1 < h_bg and x2 > 0 and y2 > 0:
                    img_roi = transformed[y1_img:y1_img + (y2-y1), x1_img:x1_img + (x2-x1)]
                    bg_region = result[y1:y2, x1:x2]
                    
                    if img_roi.shape[2] == 4:
                        # Extract alpha and ensure it's a 2D array
                        alpha = img_roi[:, :, 3] / 255.0
                        img_roi_rgb = img_roi[:, :, :3]
                        
                        # Apply alpha to base opacity
                        combined_opacity = params['opacity'] * alpha
                        
                        blended = self.apply_blend_mode(
                            bg_region,
                            img_roi_rgb,
                            params['blend_mode'],
                            combined_opacity
                        )
                        result[y1:y2, x1:x2] = blended
                    else:
                        blended = self.apply_blend_mode(
                            bg_region,
                            img_roi,
                            params['blend_mode'],
                            params['opacity']
                        )
                        result[y1:y2, x1:x2] = blended

            if self.show_grid:
                result = self.draw_grid(result)
            result = self.create_status_overlay(result)
            
            # Convert to correct color space for display
            display_result = self.convert_to_display(result)
            cv2.imshow(self.preview_window, display_result)
            self.update_layer_window()
            self.last_update_time = current_time
            
        except Exception as e:
            print(f"Warning: Preview update error: {str(e)}")
            import traceback
            traceback.print_exc()  # Print detailed error for debugging

    def update_control_panel(self):
        """Update control panel values for selected image"""
        params = self.image_params[self.selected_image]
        cv2.setTrackbarPos('Scale %', self.control_window, int(params['scale']*100))
        cv2.setTrackbarPos('Rotate°', self.control_window, int(params['rotation']+180))
        cv2.setTrackbarPos('Opacity %', self.control_window, int(params['opacity']*100))
        cv2.setTrackbarPos('Flip H', self.control_window, int(params.get('flip_h', False)))
        cv2.setTrackbarPos('Flip V', self.control_window, int(params.get('flip_v', False)))

    def mouse_callback(self, event, x, y, flags, param):
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return

        shift_pressed = flags & cv2.EVENT_FLAG_SHIFTKEY
        ctrl_pressed = flags & cv2.EVENT_FLAG_CTRLKEY

        if event == cv2.EVENT_LBUTTONDOWN:
            layer_idx = self.select_layer_at_point(x, y)
            if layer_idx is not None:
                self.selected_image = layer_idx
                self.update_control_panel()
            
            self.dragging = True
            params = self.image_params[self.selected_image]
            self.drag_start_pos = (params['x_pos'], params['y_pos'])
            self.drag_start_window = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            dx = x - self.drag_start_window[0]
            dy = y - self.drag_start_window[1]
            
            if shift_pressed:
                dx = dx // 5
                dy = dy // 5
            elif ctrl_pressed:
                dx = dx // 10
                dy = dy // 10
            
            params = self.image_params[self.selected_image]
            params['x_pos'] = self.drag_start_pos[0] + dx
            params['y_pos'] = self.drag_start_pos[1] + dy
            self.update_preview()
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
        
        elif event == cv2.EVENT_MOUSEWHEEL:
            if self.rotate_mode:
                delta = 1 if flags > 0 else -1
                if shift_pressed:
                    delta /= 5
                elif ctrl_pressed:
                    delta /= 10
                    
                params = self.image_params[self.selected_image]
                new_rotation = (params['rotation'] + delta) % 360
                params['rotation'] = new_rotation
                cv2.setTrackbarPos('Rotate°', self.control_window, int(new_rotation + 180))
            else:
                scale_factor = 1.1 if flags > 0 else 0.9
                if ctrl_pressed:
                    scale_factor = 1.01 if flags > 0 else 0.99
                elif shift_pressed:
                    scale_factor = 1.05 if flags > 0 else 0.95
                
                params = self.image_params[self.selected_image]
                new_scale = params['scale'] * scale_factor
                new_scale = max(0.1, min(10.0, new_scale))
                
                params['scale'] = new_scale
                cv2.setTrackbarPos('Scale %', self.control_window, int(new_scale * 100))
            
            self.update_preview()

    def toggle_grid(self, value):
        """Toggle grid overlay"""
        self.show_grid = bool(value)
        self.update_preview()

    def update_param(self, name, value):
        """Update parameter for selected image"""
        params = self.image_params[self.selected_image]
        old_value = params.get(name)
        params[name] = value
        self.update_preview()

    def process_image(self, img, x_pos=0, y_pos=0, scale=1.0, rotation=0.0, 
                     opacity=1.0, blend_mode="normal", flip_h=False, flip_v=False):
        """Helper function to process image parameters"""
        if img is not None:
            img_np = (img[0].cpu().numpy() * 255).astype(np.uint8)
            return img_np, {
                'x_pos': x_pos,
                'y_pos': y_pos,
                'scale': max(0.1, float(scale)),
                'rotation': float(rotation),
                'opacity': opacity,
                'blend_mode': blend_mode if blend_mode in ["normal", "multiply", "screen", "overlay"] else "normal",
                'flip_h': flip_h,
                'flip_v': flip_v
            }
        return None, None

    def composite_images(self, background, image1, x_pos1, y_pos1, scale1, rotation1, opacity1, blend_mode1,
                        flip_h1, flip_v1, enable_preview, 
                        image2=None, x_pos2=None, y_pos2=None, scale2=None, rotation2=None, opacity2=None, blend_mode2=None, flip_h2=False, flip_v2=False,
                        image3=None, x_pos3=None, y_pos3=None, scale3=None, rotation3=None, opacity3=None, blend_mode3=None, flip_h3=False, flip_v3=False,
                        image4=None, x_pos4=None, y_pos4=None, scale4=None, rotation4=None, opacity4=None, blend_mode4=None, flip_h4=False, flip_v4=False,
                        image5=None, x_pos5=None, y_pos5=None, scale5=None, rotation5=None, opacity5=None, blend_mode5=None, flip_h5=False, flip_v5=False):
        try:
            # Convert background to numpy array
            bg = (background[0].cpu().numpy() * 255).astype(np.uint8)
            
            # Initialize lists
            images = []
            params_list = []
            
            # Process all images
            img_data = [
                (image1, x_pos1, y_pos1, scale1, rotation1, opacity1, blend_mode1, flip_h1, flip_v1),
                (image2, x_pos2, y_pos2, scale2, rotation2, opacity2, blend_mode2, flip_h2, flip_v2),
                (image3, x_pos3, y_pos3, scale3, rotation3, opacity3, blend_mode3, flip_h3, flip_v3),
                (image4, x_pos4, y_pos4, scale4, rotation4, opacity4, blend_mode4, flip_h4, flip_v4),
                (image5, x_pos5, y_pos5, scale5, rotation5, opacity5, blend_mode5, flip_h5, flip_v5)
            ]

            for img_params in img_data:
                img, params = self.process_image(*img_params)
                if img is not None and params is not None:
                    images.append(img)
                    params_list.append(params)

            # Store current state
            self.current_bg = bg.copy()
            self.current_images = images
            self.image_params = params_list
            self.layer_order = list(range(len(images)))
            self.selected_image = 0
            self.has_alpha = bg.shape[2] == 4 or any(img.shape[2] == 4 for img in images)
            
            if enable_preview:
                try:
                    self.create_windows()
                    self.update_preview()
                    
                    while True:
                        key = cv2.waitKey(1) & 0xFF
                        
                        if key == 27 or key == 13 or key == ord('q'):  # ESC, ENTER or Q
                            break
                        elif key == ord('b'):  # Cycle blend modes
                            params = self.image_params[self.selected_image]
                            blend_modes = ["normal", "multiply", "screen", "overlay"]
                            current_idx = blend_modes.index(params['blend_mode'])
                            next_idx = (current_idx + 1) % len(blend_modes)
                            params['blend_mode'] = blend_modes[next_idx]
                            self.update_preview()
                        elif key == ord('g'):  # Toggle grid
                            self.show_grid = not self.show_grid
                            cv2.setTrackbarPos('Grid', self.control_window, int(self.show_grid))
                            self.update_preview()
                        elif key == ord('r'):  # Reset current image
                            params = self.image_params[self.selected_image]
                            params.update({
                                'x_pos': 0, 'y_pos': 0,
                                'scale': 1.0, 'rotation': 0.0,
                                'opacity': 1.0, 'blend_mode': 'normal',
                                'flip_h': False, 'flip_v': False
                            })
                            self.update_control_panel()
                            self.update_preview()
                        elif key == ord('s'):  # Toggle scale mode
                            self.scale_mode = not self.scale_mode
                            self.rotate_mode = False
                            self.update_preview()
                        elif key == ord('t'):  # Toggle rotate mode
                            self.rotate_mode = not self.rotate_mode
                            self.scale_mode = False
                            self.update_preview()
                        elif key == ord('c'):  # Center current image
                            img = self.current_images[self.selected_image]
                            params = self.image_params[self.selected_image]
                            img_h, img_w = img.shape[:2]
                            bg_h, bg_w = self.current_bg.shape[:2]
                            params['x_pos'] = (bg_w - img_w) // 2
                            params['y_pos'] = (bg_h - img_h) // 2
                            self.update_preview()
                        elif key == ord('h'):  # Toggle horizontal flip
                            params = self.image_params[self.selected_image]
                            flip_h = not params.get('flip_h', False)
                            params['flip_h'] = flip_h
                            cv2.setTrackbarPos('Flip H', self.control_window, int(flip_h))
                            self.update_preview()
                        elif key == ord('v'):  # Toggle vertical flip
                            params = self.image_params[self.selected_image]
                            flip_v = not params.get('flip_v', False)
                            params['flip_v'] = flip_v
                            cv2.setTrackbarPos('Flip V', self.control_window, int(flip_v))
                            self.update_preview()
                        elif key == ord('w'):  # Move layer up
                            idx = self.layer_order.index(self.selected_image)
                            self.move_layer_up(idx)
                        elif key == ord('x'):  # Move layer down
                            idx = self.layer_order.index(self.selected_image)
                            self.move_layer_down(idx)
                        elif key == ord('n'):  # Select next image
                            current_idx = self.layer_order.index(self.selected_image)
                            next_idx = (current_idx + 1) % len(self.layer_order)
                            self.selected_image = self.layer_order[next_idx]
                            self.update_control_panel()
                            self.update_preview()
                        elif key == ord('p'):  # Select previous image
                            current_idx = self.layer_order.index(self.selected_image)
                            prev_idx = (current_idx - 1) % len(self.layer_order)
                            self.selected_image = self.layer_order[prev_idx]
                            self.update_control_panel()
                            self.update_preview()
                            
                finally:
                    cv2.destroyAllWindows()
                    for _ in range(4):
                        cv2.waitKey(1)

            # Create final composite
            result = self.current_bg.copy()
            
            # Composite in layer order
            for layer_idx in self.layer_order:
                img = self.current_images[layer_idx]
                params = self.image_params[layer_idx]
                
                transformed = self.transform_image(img, params)
                
                # Convert to PIL for composition
                if transformed.shape[2] == 4:
                    img_pil = Image.fromarray(transformed, mode='RGBA')
                else:
                    img_pil = Image.fromarray(transformed, mode='RGB')
                
                if result.shape[2] == 4:
                    result_pil = Image.fromarray(result, mode='RGBA')
                else:
                    result_pil = Image.fromarray(result, mode='RGB')
                
                # Composite with proper blending
                if transformed.shape[2] == 4:
                    mask = img_pil.split()[3]
                    result_pil.paste(img_pil, (params['x_pos'], params['y_pos']), mask)
                else:
                    result_pil.paste(img_pil, (params['x_pos'], params['y_pos']))
                
                result = np.array(result_pil)

            # Convert to tensor
            result_array = result.astype(np.float32) / 255.0
            result_tensor = torch.from_numpy(result_array)[None,]
            
            # Clear resources
            self.current_images = []
            self.current_bg = None
            self.image_params = []
            self.layer_order = []
            self.has_alpha = False
            
            return (result_tensor,)
            
        except Exception as e:
            print(f"Error in composite_images: {str(e)}")
            cv2.destroyAllWindows()
            return (background,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "A2V_Multi_Image_Composite": A2V_Multi_Image_Composite
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "A2V_Multi_Image_Composite": "A2V Multi Image Composite"
}
