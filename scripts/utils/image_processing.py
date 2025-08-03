from os import path
import random

from PIL import ImageDraw, ImageFont, ImageOps, ImageEnhance
font_path = path.join(path.dirname(ImageFont.__file__), "DejaVuSans.ttf")
font = ImageFont.truetype(font_path, size=28)  # Increase size as needed

def draw_with_boxes(img, boxes_to_draw, width, color, labels, text_height):
    draw = ImageDraw.Draw(img)
    for i, box in enumerate(boxes_to_draw):
        draw.rectangle(box, outline=color, width = width)
        if not labels:
            continue
        x1, y1, x2, y2 = box
        textbox = [x1, y1 - text_height, x2, y1]
        draw.rectangle(textbox, fill=(255, 255, 255, 50))
        draw.text(textbox, f"# {i}", fill="black", font=None)
    return img

class SidewalkBalletImage:
    def __init__(self, image, boxes):
        self.image = image
        self._boxes = sorted(boxes, key=lambda box: (box[0], box[1])) # sort by x1, y1
        # self._boxes = sorted(boxes, key=lambda box: (box[0], box[1], box[2], box[3]))
        self._boxes = [[int(box[0]), int(box[1]), int(box[2]), int(box[3])] for box in self._boxes]
    def show(self, all_boxes=False, box_ids = [], width = 2, color="red", labels = True, text_height = 10):
        img = self.image.copy()
        if all_boxes:
            boxes_to_draw = [item for item in self._boxes]
        else:
            boxes_to_draw = [self._boxes[id] for id in box_ids]
        return draw_with_boxes(img, boxes_to_draw, width, color, labels, text_height)
    
    def crop(self, box_ids = [], padding = 30):
        img = self.image.copy()
        boxes = self._boxes
        if len(boxes) == 0:
            return SidewalkBalletImage(img, [])
        x1 = max(min([boxes[i][0] for i in box_ids]) - padding, 0)
        y1 = max(min([boxes[i][1] for i in box_ids]) - padding, 0)
        x2 = min(max([boxes[i][2] for i in box_ids]) + padding, img.width)
        y2 = min(max([boxes[i][3] for i in box_ids]) + padding, img.height)
        relative_boxes = [[boxes[i][0] - x1, boxes[i][1] - y1, boxes[i][2] - x1, boxes[i][3] - y1] for i in box_ids]
        relative_boxes
        return SidewalkBalletImage(img.crop((x1, y1, x2, y2)), relative_boxes)
    
    def focus(self, box_ids = [], padding_percent=0.7, minimum_padding=60):
        cropped_no_padding = self.crop(box_ids=box_ids, padding=0)
        w_small, h_small = cropped_no_padding.show().size
        padding = max(max(w_small, h_small) * padding_percent, minimum_padding)
        return self.crop(box_ids=box_ids, padding=padding)
    
    def random_flip(self, prob = 0.5, seed = 42):
        random.seed(seed)
        img = self.image.copy()
        if random.random() < prob:
            boxes_flipped = [[img.width - box[2], box[1], img.width - box[0], box[3]] for box in self._boxes]
            return SidewalkBalletImage(ImageOps.mirror(img), boxes_flipped)
        else:
            return SidewalkBalletImage(img, self._boxes)

    def random_enhance(self, brightness = (0.8, 1.2), contrast = (0.8, 1.2), seed = 42):
        random.seed(seed)
        img = self.image.copy()
        
        brightness_factor = random.uniform(brightness[0], brightness[1])
        enhancer = ImageEnhance.Brightness(img)
        image = enhancer.enhance(brightness_factor)
        
        contrast_factor = random.uniform(contrast[0], contrast[1])
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)
        return SidewalkBalletImage(image, self._boxes)
        
    def save(self, *args, **kwargs):
        self.image.save(*args, **kwargs)
        self.image_path = args[0]
    
    @property
    def boxes(self):
        """Return the stored bounding boxes."""
        return self._boxes
    
    @property
    def path(self):
        if not hasattr(self, 'image_path'):
            return None
        """Return the image_path."""
        return self.image_path