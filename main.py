from geoclip import GeoCLIP
model = GeoCLIP()
print("loaded")
from PIL import Image
import tempfile
from pathlib import Path
import gradio as gr
def predict(image):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmppath = Path(tmp_dir) / "tmp.jpg"
        image.save(str(tmppath))
        top_pred_gps, top_pred_prob = model.predict(str(tmppath), top_k=50)
        
    predictions = []
    for i in range(5):
        lat, lon = top_pred_gps[i]
        probpercent = top_pred_prob[i] * 100
        prediction = f"{i+1}: ({lat:.6f}, {lon:.6f}) - probability: {probpercent:.2f}%"
        predictions.append(prediction)
    return "\n".join(predictions)
app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="upload iamge"),
    outputs=gr.Textbox(label="predictions"),
    title="web interface for geolocation project @inputoutputcontrol",
    description="upload image to predict location",
)
app.launch(share=True)
