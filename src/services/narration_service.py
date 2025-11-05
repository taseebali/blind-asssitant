# """
# Narration service for generating natural language descriptions
# """

class NarrationService:
    def __init__(self):
        """Initialize the narration service."""
        self.last_objects = set()
        self.last_text = set()
        self.depth_threshold = 0.05  # Minimum size relative to frame to be considered "close"
        
    def _get_spatial_position(self, rel_x, rel_y):
        """Convert relative coordinates to natural language position."""
        x_pos = "on the left" if rel_x < 0.33 else "in the center" if rel_x < 0.66 else "on the right"
        y_pos = "at the top" if rel_y < 0.33 else "in the middle" if rel_y < 0.66 else "at the bottom"
        return x_pos, y_pos

    def _get_relative_position(self, obj1, obj2):
        """Get the relative position between two objects."""
        x1, y1 = obj1['position']['center']
        x2, y2 = obj2['position']['center']
        
        dx = x2 - x1
        dy = y2 - y1
        
        if abs(dx) > abs(dy):
            return "to the left of" if dx < 0 else "to the right of"
        else:
            return "above" if dy < 0 else "below"

    def generate(self, objects, texts, caption=None):
        """Generate a natural language description of the scene."""
        if not objects and not texts:
            return None
        
        # Sort objects by depth (closest first)
        close_objects = [obj for obj in objects if obj['depth_score'] > self.depth_threshold]
        close_objects.sort(key=lambda x: x['depth_score'], reverse=True)
        
        # Group objects by class to avoid duplicate descriptions
        obj_groups = {}
        for obj in close_objects:
            class_name = obj['class']
            if class_name not in obj_groups:
                obj_groups[class_name] = []
            obj_groups[class_name].append(obj)
        
        narration = []
        
        # Describe close objects with spatial relationships
        if obj_groups:
            # Describe the main (closest/largest) object first
            if close_objects:
                main_obj = close_objects[0]
                x_pos, y_pos = self._get_spatial_position(main_obj['position']['x'], main_obj['position']['y'])
                
                count = len(obj_groups[main_obj['class']])
                if count > 1:
                    desc = f"I can see {count} {main_obj['class']}s {x_pos}"
                else:
                    desc = f"I can see a {main_obj['class']} {x_pos}"
                
                # Find OTHER types of objects nearby
                other_classes = [cls for cls in obj_groups.keys() if cls != main_obj['class']]
                if other_classes and len(close_objects) > count:
                    related_objs = []
                    for other_class in other_classes[:2]:  # Limit to 2 other types
                        other_obj = obj_groups[other_class][0]
                        rel_pos = self._get_relative_position(main_obj, other_obj)
                        obj_count = len(obj_groups[other_class])
                        if obj_count > 1:
                            related_objs.append(f"{obj_count} {other_class}s {rel_pos} it")
                        else:
                            related_objs.append(f"a {other_class} {rel_pos} it")
                    
                    if related_objs:
                        desc += f", with {' and '.join(related_objs)}"
                
                narration.append(desc)
        
        # Add text descriptions for nearby text
        if texts:
            # Calculate text box sizes relative to image size
            close_texts = []
            for t in texts:
                if 'box' in t:
                    box_width = t['box'][2][0] - t['box'][0][0]
                    box_height = t['box'][2][1] - t['box'][0][1]
                    relative_size = (box_width * box_height) / (640 * 480)  # Using standard frame size
                    if relative_size > self.depth_threshold:
                        close_texts.append(t)
            
            if close_texts:
                text_content = [t['text'] for t in close_texts]
                text_narration = "I can read the following nearby text: " + ", ".join(text_content)
                narration.append(text_narration)
        
        final_narration = ". ".join(narration) if narration else None
        return final_narration