# """
# Narration service for generating natural language descriptions
# """

class NarrationService:
    def __init__(self):
        """Initialize the narration service."""
        self.last_narration = None
        self.narration_count = 0
        self.min_repeat_frames = 3
        self.last_objects = {}
        
    def _get_spatial_position(self, rel_x, rel_y):
        """Convert relative coordinates to natural language position."""
        x_pos = "on the left" if rel_x < 0.33 else "in the center" if rel_x < 0.66 else "on the right"
        y_pos = "at the top" if rel_y < 0.33 else "in the middle" if rel_y < 0.66 else "at the bottom"
        return x_pos, y_pos

    def _get_relative_position(self, obj1, obj2):
        """Get the relative position between two objects with improved spatial logic."""
        x1, y1 = obj1['position']['center']
        x2, y2 = obj2['position']['center']
        
        # Check if this is a valid object relationship
        if obj1['class'] not in self.valid_relationships or \
           obj2['class'] not in self.valid_relationships.get(obj1['class'], []):
            return None
            
        dx = x2 - x1
        dy = y2 - y1
        
        # Calculate relative positions with thresholds
        if obj1['class'] == 'desk':
            if abs(dy) < obj1['depth_score'] * 100:  # If object is at similar height
                return "on" if y2 < y1 else "in front of"
            else:
                return "above" if y2 < y1 else "below"
        else:
            if abs(dx) > abs(dy):
                return "to the left of" if dx < 0 else "to the right of"
            else:
                return "above" if dy < 0 else "below"

    def generate(self, objects, texts, caption=None):
        """Generate natural language description combining caption and detected objects."""
        if not objects and not caption:
            return None
            
        # Track stable objects across frames
        current_objects = {obj['class']: obj for obj in objects}
        
        # Only use objects that have been stable for multiple frames
        stable_objects = {}
        for cls, obj in current_objects.items():
            if cls in self.last_objects:
                stable_objects[cls] = obj
                
        self.last_objects = current_objects
        
        # Start with the caption if available
        if caption and caption != self.last_narration:
            base_narration = caption
        else:
            # Fall back to object-based narration
            main_objects = []
            for obj in objects[:3]:  # Limit to 3 most confident objects
                pos_x, pos_y = self._get_spatial_position(obj['position']['x'], obj['position']['y'])
                main_objects.append(f"a {obj['class']} {pos_x}")
                
            if main_objects:
                base_narration = "I can see " + ", and ".join(main_objects)
            else:
                return None
                
        # Avoid repeating the same narration
        if base_narration == self.last_narration:
            return None
            
        self.last_narration = base_narration
        return base_narration