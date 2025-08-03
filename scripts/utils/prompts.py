##########################################
### Prompt 7: Focuesd and Depth bboxed ###
##########################################

def prompt7(bboxes, depth1, depth2):
    return (
        "In the first image<image>, two individuals are highlighted with bounding boxes (each given as [x1, y1, x2, y2]):\n\n"
        "Box1: <|box_start|> " + ",".join([str(item) for item in bboxes[0]]) + " <|box_end|>\n"
        "Box2: <|box_start|> " + ",".join([str(item) for item in bboxes[1]]) + " <|box_end|>\n\n"
        "In the second image<image>, which is the depth view of the same scene, the same two individuals are highlighted again:\n\n"
        "Box1': <|box_start|> " + ",".join([str(item) for item in bboxes[0]]) + " <|box_end|>\n"
        "Box2': <|box_start|> " + ",".join([str(item) for item in bboxes[1]]) + " <|box_end|>\n\n"
        "Note: Box1 and Box1' represent the same person, and Box2 and Box2' represent the same person.\n\n"
        f"Depth Values (from 0-255, where 0 means far and 255 means close; these values are critical for determining proximity): "
        f"Box1 and Box1' = {depth1}, Box2 and Box2' = {depth2}, so the Depth Difference is {abs(depth1 - depth2)}\n\n"
        "Carefully analyze both images by considering all visual cues. In particular, pay attention to the following cues:\n"
        "- Body orientation\n"
        "- Facial expressions\n"
        "- Gestures\n"
        "- Depth distance\n"
        "- Relative positioning\n\n"
        "Based on these details, determine whether the individuals are actively interacting (e.g., engaged in conversation or displaying clear interactive behavior) "
        "or if they are merely near each other without meaningful interaction.\n\n"
        "Your output must contain exactly one choice: 'Yes', 'No', or 'Not sure' with no additional commentary."
    )
    
def prompt7_question(image_focused_bboxed, image_focused_depth_bboxed, text_value):
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_focused_bboxed},
            {"type": "image", "image": image_focused_depth_bboxed},
            {"type": "text", "text": text_value.replace("<image>", "")}, # from prompt7
        ],
    }]

__all__ = [
    "prompt7"
    "prompt7_question",
]