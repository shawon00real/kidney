"""
Kidney Stone Detection Web Application - Flask Version
======================================================

Professional web application for kidney stone detection with XAI visualization.

Features:
- Modern, responsive UI
- File upload for kidney images
- AI prediction with confidence scores
- Grad-CAM heatmap visualization (shows where stones are)
- Downloadable PDF report
- Patient information form

Installation:
    pip install flask torch torchvision pillow opencv-python grad-cam reportlab

Usage:
    python app.py
    
Then open: http://localhost:5000
"""

import os
import io
import base64
from datetime import datetime
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename

# Grad-CAM imports
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# PDF Report generation
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib import colors

# =============================================================================
# FLASK APP CONFIGURATION
# =============================================================================

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create necessary folders
os.makedirs('uploads', exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('static/results', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

MODEL_PATH = 'best_kidney_model.pth'  # Your model file
CLASS_NAMES = ['Normal', 'Stone']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    print(f"⚠️  WARNING: Model file not found at: {MODEL_PATH}")
    print(f"   Current directory: {os.getcwd()}")
    print(f"   Please ensure 'best_robust_model.pth' is in the same directory as app.py")
else:
    print(f"✓ Model file found at: {MODEL_PATH}")

# Image preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# For Grad-CAM visualization
transform_no_norm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model():
    """Load the trained FlexMatch model with EfficientNet backbone."""
    print("="*60)
    print("Loading model...")
    print(f"Model path: {MODEL_PATH}")
    print(f"File exists: {os.path.exists(MODEL_PATH)}")
    print(f"Device: {DEVICE}")
    print("="*60)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Model file not found at: {MODEL_PATH}")
        print(f"   Make sure 'best_robust_model.pth' is in the same directory as app.py")
        return None
    
    try:
        # Load checkpoint first to inspect structure
        print("Loading checkpoint...")
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        print("✓ Checkpoint loaded")
        
        # Extract state dict
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print("✓ Found 'model_state_dict' in checkpoint")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("✓ Found 'state_dict' in checkpoint")
            else:
                state_dict = checkpoint
                print("✓ Using checkpoint as state_dict")
        else:
            state_dict = checkpoint
            print("✓ Checkpoint is state_dict")
        
        # Check what classifier structure was saved
        has_nested_classifier = any('classifier.1.1' in key for key in state_dict.keys())
        print(f"Nested classifier structure: {has_nested_classifier}")
        
        # EfficientNetV2-S architecture
        print("Creating EfficientNetV2-S model...")
        model = models.efficientnet_v2_s(weights=None)
        num_features = model.classifier[1].in_features
        
        # Match the saved classifier structure
        if has_nested_classifier:
            # Original FlexMatch structure with nested Sequential
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Sequential(
                    nn.Linear(num_features, 2)
                )
            )
        else:
            # Simple structure
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(num_features, 2)
            )
        
        # Load weights
        print("Loading weights into model...")
        model.load_state_dict(state_dict, strict=False)
        
        model = model.to(DEVICE)
        model.eval()
        print("="*60)
        print(f"✅ Model loaded successfully on {DEVICE}")
        print("="*60)
        return model
    
    except FileNotFoundError:
        print(f"❌ ERROR: File not found - {MODEL_PATH}")
        print("   Please ensure 'best_robust_model.pth' is in the same directory")
        return None
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

# Load model at startup
MODEL = load_model()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_image(model, image_path):
    """
    Make prediction on kidney image.
    
    Returns:
        prediction: 'Normal' or 'Stone'
        confidence: confidence percentage
        probabilities: [normal_prob, stone_prob]
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        probs = probabilities[0].cpu().numpy()
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probs[predicted_class] * 100
    
    prediction = CLASS_NAMES[predicted_class]
    
    return prediction, confidence, probs

def generate_gradcam(model, image_path, save_path):
    """
    Generate Grad-CAM heatmap showing where the model detects stones.
    
    This creates a visualization with red/yellow areas showing suspicious regions.
    """
    # Load image
    pil_image = Image.open(image_path).convert('RGB')
    
    # Prepare for Grad-CAM
    input_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)
    
    # For visualization, we need the original image normalized to [0, 1]
    rgb_img = np.array(pil_image.resize((224, 224))) / 255.0
    
    # Target layer for EfficientNetV2-S (last convolutional layer)
    target_layers = [model.features[-1]]
    
    # Create Grad-CAM object
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Generate CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]
    
    # Overlay CAM on image
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    # Save visualization
    cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    
    return save_path

def generate_pdf_report(patient_name, patient_id, age, gender, image_path, 
                        prediction, confidence, probs, gradcam_path, 
                        doctor_name="", report_path="report.pdf"):
    """
    Generate professional PDF report with all results.
    """
    doc = SimpleDocTemplate(report_path, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2C3E50'),
        spaceAfter=30,
        alignment=1  # Center
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#34495E'),
        spaceAfter=12
    )
    
    # Title
    title = Paragraph("KIDNEY STONE DETECTION REPORT", title_style)
    story.append(title)
    story.append(Spacer(1, 0.3*inch))
    
    # Report metadata
    report_date = datetime.now().strftime("%B %d, %Y - %H:%M")
    metadata = Paragraph(f"<b>Report Generated:</b> {report_date}", styles['Normal'])
    story.append(metadata)
    story.append(Spacer(1, 0.3*inch))
    
    # Patient Information
    story.append(Paragraph("PATIENT INFORMATION", heading_style))
    
    patient_data = [
        ['Patient Name:', patient_name],
        ['Patient ID:', patient_id],
        ['Age:', f"{age} years"],
        ['Gender:', gender],
    ]
    
    patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ECF0F1')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 0.3*inch))
    
    # AI Analysis Results
    story.append(Paragraph("AI ANALYSIS RESULTS", heading_style))
    
    result_color = colors.red if prediction == "Stone" else colors.green
    result_icon = "⚠️" if prediction == "Stone" else "✓"
    
    result_text = f"<font color='{result_color.hexval()}'><b>{result_icon} {prediction.upper()}</b></font>"
    story.append(Paragraph(result_text, styles['Normal']))
    story.append(Spacer(1, 0.1*inch))
    
    confidence_text = f"<b>Confidence Level:</b> {confidence:.2f}%"
    story.append(Paragraph(confidence_text, styles['Normal']))
    story.append(Spacer(1, 0.1*inch))
    
    prob_data = [
        ['Classification', 'Probability'],
        ['Normal', f"{probs[0]*100:.2f}%"],
        ['Stone', f"{probs[1]*100:.2f}%"],
    ]
    
    prob_table = Table(prob_data, colWidths=[2*inch, 2*inch])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498DB')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Images
    story.append(Paragraph("VISUAL ANALYSIS", heading_style))
    
    # Original and Grad-CAM side by side
    try:
        img1 = RLImage(image_path, width=2.5*inch, height=2.5*inch)
        img2 = RLImage(gradcam_path, width=2.5*inch, height=2.5*inch)
        
        image_data = [
            ['Original Image', 'AI Heatmap (Red = Stone Region)'],
            [img1, img2]
        ]
        
        image_table = Table(image_data, colWidths=[3*inch, 3*inch])
        image_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ]))
        story.append(image_table)
    except:
        pass
    
    story.append(Spacer(1, 0.3*inch))
    
    # Interpretation
    story.append(Paragraph("INTERPRETATION", heading_style))
    
    if prediction == "Stone":
        interpretation = """
        <b>Stone Detected:</b> The AI model has identified patterns consistent with kidney stones.
        The heatmap shows the suspicious regions in red/yellow. 
        
        <b>Recommended Actions:</b>
        • Consult with a urologist for treatment options
        • Consider additional imaging (CT scan) for detailed assessment
        • Discuss stone size, location, and treatment approaches
        • Monitor symptoms and hydration levels
        """
    else:
        interpretation = """
        <b>Normal Kidney:</b> The AI model has not detected patterns consistent with kidney stones.
        The heatmap shows uniform distribution without suspicious regions.
        
        <b>Recommended Actions:</b>
        • Continue routine monitoring
        • Maintain adequate hydration
        • Follow preventive measures
        • Schedule regular checkups as recommended
        """
    
    story.append(Paragraph(interpretation, styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Confidence interpretation
    if confidence >= 95:
        conf_text = "<b>High Confidence:</b> The model is very certain about this prediction."
    elif confidence >= 85:
        conf_text = "<b>Moderate Confidence:</b> The prediction is likely correct, but radiologist review is recommended."
    else:
        conf_text = "<b>Low Confidence:</b> Manual review by radiologist is strongly recommended."
    
    story.append(Paragraph(conf_text, styles['Normal']))
    story.append(Spacer(1, 0.4*inch))
    
    # Disclaimer
    disclaimer = """
    <b>MEDICAL DISCLAIMER:</b> This report is generated by an AI system for decision support purposes only. 
    It should not be used as the sole basis for medical diagnosis or treatment decisions. 
    All results must be verified by qualified medical professionals. This system is not FDA approved 
    and is intended for research and educational purposes.
    """
    
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey,
        borderPadding=10
    )
    story.append(Paragraph(disclaimer, disclaimer_style))
    
    # Build PDF
    doc.build(story)
    return report_path

# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction."""
    
    if MODEL is None:
        return jsonify({'error': 'Model not loaded. Please check MODEL_PATH.'}), 500
    
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use JPG, JPEG, or PNG.'}), 400
    
    # Get patient information
    patient_name = request.form.get('patient_name', 'N/A')
    patient_id = request.form.get('patient_id', 'N/A')
    age = request.form.get('age', 'N/A')
    gender = request.form.get('gender', 'N/A')
    doctor_name = request.form.get('doctor_name', '')
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_filename = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)
    
    try:
        # Make prediction
        prediction, confidence, probs = predict_image(MODEL, filepath)
        
        # Generate Grad-CAM heatmap
        gradcam_filename = f"gradcam_{unique_filename}"
        gradcam_path = os.path.join('static/results', gradcam_filename)
        generate_gradcam(MODEL, filepath, gradcam_path)
        
        # Generate PDF report
        report_filename = f"report_{timestamp}.pdf"
        report_path = os.path.join('reports', report_filename)
        generate_pdf_report(
            patient_name, patient_id, age, gender, 
            filepath, prediction, confidence, probs,
            gradcam_path, doctor_name, report_path
        )
        
        # Prepare response
        result = {
            'success': True,
            'prediction': prediction,
            'confidence': round(confidence, 2),
            'probabilities': {
                'normal': round(probs[0] * 100, 2),
                'stone': round(probs[1] * 100, 2)
            },
            'original_image': f'/static/results/{unique_filename}',
            'gradcam_image': f'/results/{gradcam_filename}',
            'report_url': f'/download_report/{report_filename}'
        }
        
        # Copy original image to static for display
        import shutil
        shutil.copy(filepath, os.path.join('static/results', unique_filename))
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/download_report/<filename>')
def download_report(filename):
    """Download PDF report."""
    report_path = os.path.join('reports', filename)
    
    if os.path.exists(report_path):
        return send_file(report_path, as_attachment=True, download_name=filename)
    else:
        return "Report not found", 404

@app.route('/results/<filename>')
def serve_result(filename):
    """Serve result images."""
    return send_file(os.path.join('static/results', filename))

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("="*80)
    print("KIDNEY STONE DETECTION WEB APPLICATION")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Model loaded: {MODEL is not None}")
    print(f"\nStarting server...")
    print(f"Open your browser and go to: http://localhost:5000")
    print("="*80)
    
    app.run(debug=True, host='0.0.0.0', port=5000)