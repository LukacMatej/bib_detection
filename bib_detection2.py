import cv2
import numpy as np
import time
from ultralytics import YOLO
from picamera2 import Picamera2

def main():
    # Load your YOLO model with NCNN backend
    model_path = "./best_ncnn_model"  # Update with your model path
    
    # Initialize the model with NCNN backend
    # The Ultralytics library should auto-detect the NCNN model from file extension
    model = YOLO(model_path)
    
    # Initialize the Raspberry Pi camera
    picam2 = Picamera2()
    
    # Configure the camera
    config = picam2.create_preview_configuration(main={"size": (1280, 720), "format": "RGB888"})
    picam2.configure(config)
    
    # Start the camera
    picam2.start()
    print("Camera started. Press 'q' to quit")
    
    try:
        while True:
            # Capture a frame
            frame = picam2.capture_array()
            
            # Run inference on the frame
            start_time = time.time()
            results = model(frame, verbose=False)  # Use the YOLO model to perform inference
            inference_time = time.time() - start_time
            
            # Process results (first result as we only process one image)
            result = results[0]
            
            # Visualize the results on the frame
            annotated_frame = result.plot()
            
            # Add FPS information
            fps = 1.0 / inference_time
            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the frame with detections
            cv2.imshow("YOLO NCNN Bib Detection", annotated_frame)
            
            # Process detection data if needed
            if result.boxes is not None:
                for box in result.boxes:
                    # Get coordinates in (x1, y1, x2, y2) format
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Get confidence
                    confidence = box.conf[0].item()
                    
                    # Get class (if classification task)
                    if hasattr(box, 'cls'):
                        class_id = int(box.cls[0].item())
                        class_name = model.names[class_id]
                        print(f"Detected bib - Class: {class_name}, Confidence: {confidence:.2f}")
            
            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        # Clean up
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
