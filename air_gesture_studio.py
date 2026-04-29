import cv2
import mediapipe as mp
import numpy as np
import math
import random
import time
import os

# ---------------------------------------------------------
# AIR GESTURE STUDIO - PYTHON DESKTOP APPLICATION
# ---------------------------------------------------------

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Colors (BGR for OpenCV)
NEON_CYAN = (255, 243, 0)      # #00f3ff in BGR
NEON_MAGENTA = (255, 0, 255)   # #ff00ff in BGR
NEON_BLUE = (255, 100, 0)
WHITE = (255, 255, 255)

class Particle:
    """Handles visual particle effects for trails and explosions."""
    def __init__(self, x, y, color=NEON_CYAN):
        self.x = float(x)
        self.y = float(y)
        # Random outward velocity
        self.vx = (random.random() - 0.5) * 15
        self.vy = (random.random() - 0.5) * 15
        self.life = 1.0
        self.color = color
        
    def update(self):
        # Physics update
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.5  # slight gravity
        self.life -= 0.04 # Fade out
        
    def draw(self, overlay):
        if self.life > 0:
            # Scale color intensity by life for fade effect
            c = (int(self.color[0] * self.life), int(self.color[1] * self.life), int(self.color[2] * self.life))
            radius = max(1, int(5 * self.life))
            cv2.circle(overlay, (int(self.x), int(self.y)), radius, c, -1)

class DrawingObject:
    """Represents a drawn stroke or recognized shape as a manipulatable layer."""
    def __init__(self, color=NEON_CYAN):
        self.points = []
        self.color = color
        
        # Transform properties
        self.cx = 0.0
        self.cy = 0.0
        self.scale = 1.0
        self.angle = 0.0  # Radians
        
        # State
        self.selected = False
        self.is_shape = False
        self.shape_type = None  # 'circle', 'rectangle', 'line'
        
        # Bounding info
        self.width = 0
        self.height = 0
        self.radius = 0
        
        # Physics (Inertia)
        self.vx = 0.0
        self.vy = 0.0

    def add_point(self, x, y):
        self.points.append((x, y))

    def finalize(self):
        """Called when user releases pinch. Computes center and normalizes points."""
        if len(self.points) < 2:
            return
            
        pts = np.array(self.points, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        
        self.cx = x + w / 2.0
        self.cy = y + h / 2.0
        self.width = max(w, 20)
        self.height = max(h, 20)
        
        # Convert absolute points to relative-to-center points
        self.points = [(p[0] - self.cx, p[1] - self.cy) for p in self.points]
        
        # Attempt to recognize geometric shapes
        self.recognize_shape()
        
    def recognize_shape(self):
        """Heuristics to detect lines, circles, and rectangles."""
        if len(self.points) < 10:
            return
            
        pts = np.array(self.points, dtype=np.int32)
        
        # Simplify path using Douglas-Peucker algorithm
        epsilon = 0.05 * cv2.arcLength(pts, False)
        approx = cv2.approxPolyDP(pts, epsilon, True)
        
        p1 = self.points[0]
        p2 = self.points[-1]
        dist_ends = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
        
        is_closed = dist_ends < max(self.width, self.height) * 0.3
        
        if is_closed:
            aspect_ratio = self.width / float(self.height if self.height > 0 else 1)
            r = (self.width + self.height) / 4.0
            
            # Check variance from perfect circle
            is_circle = True
            for p in self.points:
                if abs(math.hypot(p[0], p[1]) - r) > r * 0.35:
                    is_circle = False
                    break
            
            if is_circle and 0.6 < aspect_ratio < 1.4:
                self.is_shape = True
                self.shape_type = 'circle'
                self.radius = int(r)
            else:
                self.is_shape = True
                self.shape_type = 'rectangle'
        else:
            if len(approx) <= 3:  # Simplified to a straight line
                self.is_shape = True
                self.shape_type = 'line'
                
    def apply_physics(self):
        """Applies inertia and damping."""
        self.cx += self.vx
        self.cy += self.vy
        self.vx *= 0.85  # Friction
        self.vy *= 0.85
        if abs(self.vx) < 0.1: self.vx = 0
        if abs(self.vy) < 0.1: self.vy = 0

    def get_transformed_points(self):
        """Returns points with scale, rotation, and translation applied."""
        if not self.points: return []
        pts = np.array(self.points, dtype=np.float32)
        
        # 1. Scale
        pts *= self.scale
        # 2. Rotate
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        pts = np.dot(pts, rot_mat.T)
        # 3. Translate
        pts[:, 0] += self.cx
        pts[:, 1] += self.cy
        
        return pts.astype(np.int32)

    def draw(self, img, overlay_img):
        """Renders the object onto the overlay for glowing effects."""
        if not self.points: return
        
        color = WHITE if self.selected else self.color
        
        if self.is_shape:
            if self.shape_type == 'circle':
                # Thick glow
                cv2.circle(overlay_img, (int(self.cx), int(self.cy)), int(self.radius * self.scale), color, 12, cv2.LINE_AA)
                # Thin bright core
                cv2.circle(overlay_img, (int(self.cx), int(self.cy)), int(self.radius * self.scale), WHITE, 3, cv2.LINE_AA)
            
            elif self.shape_type == 'rectangle':
                w = int(self.width * self.scale / 2)
                h = int(self.height * self.scale / 2)
                rect_pts = np.array([[-w, -h], [w, -h], [w, h], [-w, h]], dtype=np.float32)
                # Rotate
                cos_a, sin_a = math.cos(self.angle), math.sin(self.angle)
                rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                rect_pts = np.dot(rect_pts, rot_mat.T)
                # Translate
                rect_pts[:, 0] += self.cx
                rect_pts[:, 1] += self.cy
                rect_pts = rect_pts.astype(np.int32)
                
                cv2.polylines(overlay_img, [rect_pts], True, color, 12, cv2.LINE_AA)
                cv2.polylines(overlay_img, [rect_pts], True, WHITE, 3, cv2.LINE_AA)
            
            elif self.shape_type == 'line':
                pts = self.get_transformed_points()
                cv2.line(overlay_img, tuple(pts[0]), tuple(pts[-1]), color, 12, cv2.LINE_AA)
                cv2.line(overlay_img, tuple(pts[0]), tuple(pts[-1]), WHITE, 3, cv2.LINE_AA)
        else:
            # Freehand path
            pts = self.get_transformed_points()
            cv2.polylines(overlay_img, [pts], False, color, 12, cv2.LINE_AA)
            cv2.polylines(overlay_img, [pts], False, WHITE, 3, cv2.LINE_AA)
            
        if self.selected:
            # Render bounding box directly on frame
            w = int(self.width * self.scale / 2) + 20
            h = int(self.height * self.scale / 2) + 20
            box_pts = np.array([[-w, -h], [w, -h], [w, h], [-w, h]], dtype=np.float32)
            cos_a, sin_a = math.cos(self.angle), math.sin(self.angle)
            rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            box_pts = np.dot(box_pts, rot_mat.T)
            box_pts[:, 0] += self.cx
            box_pts[:, 1] += self.cy
            box_pts = box_pts.astype(np.int32)
            cv2.polylines(img, [box_pts], True, WHITE, 1, cv2.LINE_AA)

def is_open_palm(landmarks, w, h):
    """Detects if hand is open by checking finger extensions."""
    wrist_x, wrist_y = landmarks[0].x * w, landmarks[0].y * h
    is_open = True
    for tip_idx in [8, 12, 16, 20]: # Index, Middle, Ring, Pinky
        tip_d = math.hypot(landmarks[tip_idx].x * w - wrist_x, landmarks[tip_idx].y * h - wrist_y)
        mcp_d = math.hypot(landmarks[tip_idx-3].x * w - wrist_x, landmarks[tip_idx-3].y * h - wrist_y)
        if tip_d < mcp_d * 1.5:  # Tip is closer to wrist than MCP (finger curled)
            is_open = False
            break
    return is_open

def main():
    cap = cv2.VideoCapture(0)
    # Request high resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    cv2.namedWindow("Air Gesture Studio", cv2.WINDOW_NORMAL)
    
    # App State
    mode = "DRAW"
    drawings = []
    current_drawing = None
    particles = []
    
    # Manipulation State
    active_object = None
    is_dragging = False
    last_grab_pos = (0, 0)
    grab_offset = (0, 0)
    
    # Two-hand transformation state
    initial_pinch_dist = None
    initial_scale = 1.0
    initial_angle = 0.0
    initial_obj_angle = 0.0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Mirror the frame
        frame = cv2.flip(frame, 1)
        H, W, _ = frame.shape
        
        # Process Hand Tracking
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Darken the camera feed for better neon visibility (HUD style)
        frame = cv2.convertScaleAbs(frame, alpha=0.4, beta=0)
        overlay = np.zeros_like(frame) # Blank layer for neon drawing
        
        # 1. Physics & Render Existing Drawings
        for d in drawings:
            d.apply_physics()
            d.draw(frame, overlay)
            
        # 2. Render Active Drawing
        if current_drawing and len(current_drawing.points) > 1:
            pts = np.array(current_drawing.points, dtype=np.int32)
            cv2.polylines(overlay, [pts], False, current_drawing.color, 12, cv2.LINE_AA)
            cv2.polylines(overlay, [pts], False, WHITE, 3, cv2.LINE_AA)
            
        # 3. Render Particles
        for p in particles[:]:
            p.update()
            p.draw(overlay)
            if p.life <= 0: particles.remove(p)

        # 4. Handle Gestures
        if results.multi_hand_landmarks:
            h1 = results.multi_hand_landmarks[0].landmark
            
            # Primary hand coordinates
            idx_x, idx_y = int(h1[8].x * W), int(h1[8].y * H)
            thb_x, thb_y = int(h1[4].x * W), int(h1[4].y * H)
            
            pinch_dist = math.hypot(idx_x - thb_x, idx_y - thb_y)
            is_pinching = pinch_dist < 40
            is_open = is_open_palm(h1, W, H)
            
            # Draw cursor
            cursor_color = NEON_MAGENTA if is_pinching else NEON_CYAN
            cv2.circle(overlay, (idx_x, idx_y), 15 if is_pinching else 10, cursor_color, -1)
            cv2.circle(frame, (idx_x, idx_y), 20, WHITE, 2, cv2.LINE_AA)
            
            # --- DRAW MODE ---
            if mode == "DRAW":
                if is_pinching:
                    if not current_drawing:
                        current_drawing = DrawingObject(color=NEON_CYAN)
                    current_drawing.add_point(idx_x, idx_y)
                    # Emit particles
                    particles.append(Particle(idx_x, idx_y, NEON_CYAN))
                else:
                    if current_drawing:
                        current_drawing.finalize()
                        drawings.append(current_drawing)
                        current_drawing = None
                        
            # --- MANIPULATE MODE ---
            elif mode == "MANIPULATE":
                if is_pinching:
                    if not is_dragging:
                        # Find object under cursor
                        for d in reversed(drawings):
                            # Hitbox check
                            if math.hypot(idx_x - d.cx, idx_y - d.cy) < (max(d.width, d.height) * d.scale / 2) + 30:
                                active_object = d
                                is_dragging = True
                                d.selected = True
                                d.vx, d.vy = 0, 0 # Stop physics
                                grab_offset = (d.cx - idx_x, d.cy - idx_y)
                                last_grab_pos = (idx_x, idx_y)
                                break
                    elif active_object:
                        # Drag logic
                        active_object.cx = idx_x + grab_offset[0]
                        active_object.cy = idx_y + grab_offset[1]
                        
                        # Calculate velocity for throw physics
                        active_object.vx = (idx_x - last_grab_pos[0]) * 0.4
                        active_object.vy = (idx_y - last_grab_pos[1]) * 0.4
                        last_grab_pos = (idx_x, idx_y)
                        
                        # --- Two Hand Interaction (Scale & Rotate) ---
                        if len(results.multi_hand_landmarks) >= 2:
                            h2 = results.multi_hand_landmarks[1].landmark
                            idx2_x, idx2_y = int(h2[8].x * W), int(h2[8].y * H)
                            thb2_x, thb2_y = int(h2[4].x * W), int(h2[4].y * H)
                            
                            # Draw second cursor
                            cv2.circle(overlay, (idx2_x, idx2_y), 10, NEON_BLUE, -1)
                            
                            if math.hypot(idx2_x - thb2_x, idx2_y - thb2_y) < 40: # Second pinch active
                                dist2 = math.hypot(idx_x - idx2_x, idx_y - idx2_y)
                                angle2 = math.atan2(idx2_y - idx_y, idx2_x - idx_x)
                                
                                if initial_pinch_dist is None:
                                    initial_pinch_dist = dist2
                                    initial_scale = active_object.scale
                                    initial_angle = angle2
                                    initial_obj_angle = active_object.angle
                                    
                                # Apply Scale
                                active_object.scale = initial_scale * (dist2 / max(initial_pinch_dist, 1))
                                # Apply Rotation
                                active_object.angle = initial_obj_angle + (angle2 - initial_angle)
                            else:
                                initial_pinch_dist = None
                        else:
                            initial_pinch_dist = None
                else:
                    # Release
                    if is_dragging and active_object:
                        # Throw gesture (high velocity release = delete)
                        if math.hypot(active_object.vx, active_object.vy) > 25:
                            if active_object in drawings:
                                drawings.remove(active_object)
                            # Explosion particles
                            for _ in range(30): particles.append(Particle(active_object.cx, active_object.cy, NEON_MAGENTA))
                        
                        active_object.selected = False
                        is_dragging = False
                        active_object = None
                        initial_pinch_dist = None
                        
            # --- ERASE GESTURE (Global) ---
            if is_open and mode != "DRAW":
                for d in reversed(drawings):
                    if math.hypot(idx_x - d.cx, idx_y - d.cy) < (max(d.width, d.height) * d.scale / 2) + 20:
                        drawings.remove(d)
                        for _ in range(20): particles.append(Particle(d.cx, d.cy, NEON_MAGENTA))
                        break

        else:
            # No hands detected
            if current_drawing:
                current_drawing.finalize()
                drawings.append(current_drawing)
                current_drawing = None
            if active_object:
                active_object.selected = False
                is_dragging = False
                active_object = None
                initial_pinch_dist = None
                
        # 5. Composite Overlay via Alpha Blending
        # Using addWeighted creates the glowing "light" effect over the dark frame
        cv2.addWeighted(overlay, 1.0, frame, 1.0, 0, frame)
        
        # 6. Draw HUD
        cv2.putText(frame, "AIR GESTURE STUDIO", (30, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, NEON_CYAN, 2, cv2.LINE_AA)
        
        mode_color = NEON_MAGENTA if mode == 'MANIPULATE' else NEON_CYAN
        cv2.putText(frame, f"MODE: {mode}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2, cv2.LINE_AA)
        
        # Instructions
        cv2.putText(frame, "[D] Draw Mode   |  [M] Manipulate Mode", (30, H - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, "[S] Save Image  |  [C] Clear Canvas  |  [ESC] Quit", (30, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, "Throw object to delete | Open palm to erase", (W - 400, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, NEON_MAGENTA, 1, cv2.LINE_AA)
        
        cv2.imshow("Air Gesture Studio", frame)
        
        # Input Handling
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC
            break
        elif key == ord('d') or key == ord('D'):
            mode = "DRAW"
        elif key == ord('m') or key == ord('M'):
            mode = "MANIPULATE"
        elif key == ord('c') or key == ord('C'):
            drawings.clear()
        elif key == ord('s') or key == ord('S'):
            # Render cleanly onto a black background for saving
            save_img = np.zeros((H, W, 3), dtype=np.uint8)
            save_overlay = np.zeros((H, W, 3), dtype=np.uint8)
            for d in drawings:
                d.selected = False
                d.draw(save_img, save_overlay)
            cv2.addWeighted(save_overlay, 1.0, save_img, 1.0, 0, save_img)
            
            filename = f"air_art_{int(time.time())}.png"
            filepath = os.path.join(os.getcwd(), filename)
            cv2.imwrite(filepath, save_img)
            print(f"Artwork saved to: {filepath}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
