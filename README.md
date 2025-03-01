# Automated Detection of Plus Disease from Infant Retinal Images  

## 📌 Overview  
**Plus Disease** is a critical indicator in **Retinopathy of Prematurity (ROP)** that determines the urgency of treatment. **This project presents a deep learning-based pipeline for detecting Plus disease in premature infants’ retinal images**, classifying them into **No-Plus, Pre-Plus, or Plus** categories.  

Our method enhances fundus images, segments blood vessels using **U-Net**, and classifies disease severity using a **ConvNeXT model**, achieving **94.44% accuracy**. This system provides **objective and repeatable** screening results, assisting ophthalmologists in clinical decision-making.  

## 🔬 Features  
- **Contrast Enhancement**: Green channel extraction + Contrast Limited Adaptive Histogram Equalization (CLAHE).  
- **Blood Vessel Segmentation**: U-Net trained on manually labeled vessel masks.  
- **Feature Superimposition**: Vessel masks overlaid on enhanced images for better feature extraction.  
- **Plus Disease Classification**: ConvNeXT-based classifier detects **No-Plus, Pre-Plus, and Plus Disease**.  

## 🗂 Dataset  
- Fundus images obtained from **Narayana Nethralaya, Bengaluru** using **3nethra Neo** fundus cameras.  
- Dataset consists of **300 high-quality images per category**: No-Plus, Pre-Plus, and Plus.  
- Images resized to **512x512 pixels** for computational efficiency.  

## 🚀 Installation  
To set up the project, follow these steps:  

```bash
# Clone the repository
git clone https://github.com/your-username/Plus-Disease-Detection.git
cd Plus-Disease-Detection

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
## 🖼 Model Pipeline  
The project follows a structured pipeline to detect Plus disease:  

### **Preprocessing**  
- Extracts the green channel from fundus images to emphasize blood vessels.  
- Applies Contrast-Limited Adaptive Histogram Equalization (CLAHE) to enhance vessel visibility.  

### **Blood Vessel Segmentation**  
- U-Net is trained on manually labeled vessel masks.  
- Generates binary masks highlighting vascular structures.  

### **Superimposition**  
- The segmented blood vessel mask is overlaid on the enhanced image.  
- Helps the classifier focus on vessel morphology and tortuosity.

  ## 📊 Performance  
### **Plus Disease Classification (ConvNeXT)**  
| Model         | Accuracy | Precision | Recall | F1 Score |  
|--------------|---------|-----------|--------|---------|  
| **ConvNeXT** | 94.44%  | 94.53%    | 94.44% | 94.37%  |  
| ResNet50     | 87.78%  | 88.01%    | 87.78% | 87.83%  |  
| EfficientNetV2S | 86.67% | 86.84%  | 86.67% | 86.55%  |  
| AlexNet      | 85.56%  | 85.45%    | 85.56% | 85.50%  |  
| RegNetY-8GF  | 89.44%  | 89.60%    | 89.44% | 89.28%  |  

## 🏗 Future Work  
- Improve segmentation with advanced attention-based architectures.  
- Expand dataset with more diverse images for better generalization.  
- Address misclassification due to vessel overlap by refining preprocessing.  
- Integrate temporal analysis to track disease progression.  


### **Plus Disease Classification**  
- Uses ConvNeXT, a state-of-the-art deep learning model.  
- Classifies images into No-Plus, Pre-Plus, or Plus Disease categories.  

