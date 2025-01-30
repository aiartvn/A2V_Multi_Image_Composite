# A2V Multi Image Composite for ComfyUI

An advanced image composition node for ComfyUI that allows you to combine multiple images with interactive preview and layer management.

## Features

- Support up to 5 input images
- Interactive preview window
- Layer management panel
- Multiple blend modes
- Transform controls (scale, rotation, position, flip)
- Accurate color handling (RGB/RGBA)

### Preview Controls

- **Mouse Controls:**
  - Click to select layer
  - Drag to move
  - Mouse wheel to scale/rotate
  - Shift/Ctrl + mouse for fine adjustments

- **Keyboard Shortcuts:**
  - W/X: Move layer up/down
  - N/P: Next/Previous layer
  - B: Cycle blend modes
  - G: Toggle grid
  - R: Reset current layer
  - S: Toggle scale mode
  - T: Toggle rotate mode
  - C: Center current layer
  - H: Flip horizontal
  - V: Flip vertical
  - ESC/ENTER/Q: Close preview

### Transform Options
- Scale
- Rotation
- Position
- Horizontal Flip
- Vertical Flip

### Blend Modes
- Normal
- Multiply
- Screen
- Overlay

### Layer Controls
- Layer ordering (Above/Below)
- Layer visibility
- Layer selection
- Layer transformations

## Installation

1. Create a folder named `a2v_multi_image_composite` in your ComfyUI custom_nodes directory
2. Copy all files from this repository into that folder
3. Restart ComfyUI

The node will appear as "A2V Multi Image Composite" in the A2V/Image category.

## Dependencies

- numpy
- opencv-python
- Pillow
- torch

## Usage

1. Connect background image and up to 5 source images
2. Set initial parameters for each image (position, scale, rotation, flip, etc)
3. Enable preview for interactive adjustments
4. Use mouse and keyboard controls to compose
5. Click ESC/ENTER when done

## Tips
- Use H/V keys to quickly flip images
- Use mouse wheel in rotate mode (T) to rotate
- Hold Shift/Ctrl for finer adjustments
- Use layer panel for quick layer management

## License

MIT License (see LICENSE file)

## Credits

Created by Aỉatvn Team
