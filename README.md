# Underwater Species Image Classification with Explainable AI (XAI)

## Project Overview
**✅ Successfully Implemented & Functional**

This project successfully develops a robust deep learning model for the classification of various underwater marine animal species from images. The key achievement is the successful integration of Explainable AI (XAI) techniques that provide transparency and interpretability into the model's decision-making process. 

**Status**: All components are working and producing results, including trained models and XAI visualizations.

## ## Purpose and Motivation

The primary motivation behind this project is to contribute to the field of **automated marine life monitoring and conservation**. Manual identification of underwater species is time-consuming, expensive, and requires specialized expertise. An automated system can significantly accelerate this process.

Furthermore, for research and deployment in critical environments (like those relevant to DRDO), simply having a high-accuracy model is often insufficient. Understanding the model's reasoning is vital for:

- 🎯 **Trust and Reliability**: Ensuring the model is making decisions based on relevant visual features, not spurious correlations
- 🔍 **Debugging**: Identifying potential biases or errors in the model's learning
- 🧬 **Scientific Discovery**: Gaining insights into what visual cues are most important for distinguishing species, which can inform marine biologists
- 📋 **Compliance**: Meeting regulatory requirements for explainable systems

This project aims to demonstrate proficiency in **deep learning**, **computer vision**, and the critical skill of **model interpretability**, aligning directly with the requirements for research-oriented positions in related fields.

## Features

- 🐠 **Multi-class Image Classification**: Accurately classifies 9 distinct marine animal species
- 🔄 **Transfer Learning**: Utilizes pre-trained Convolutional Neural Networks (CNNs) for efficient and high-performance model development
- 🎨 **Data Augmentation**: Implements various augmentation techniques to improve model generalization and robustness to image variations
- 🔍 **Explainable AI (XAI) Integration**: Applies techniques like Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize the regions of an image that are most influential in the model's prediction
- 📊 **Comprehensive Evaluation**: Reports standard classification metrics (accuracy, precision, recall, F1-score) and visualizes results with confusion matrices
- 🛠️ **Modular Codebase**: Designed for clarity and easy extension

## Dataset

This project utilizes the **"Marine Animal Images"** dataset from Kaggle, provided by `mikoajfish99`.

### Dataset Details:

- 📂 **Source**: [Kaggle - Marine Animal Images](https://www.kaggle.com/datasets/mikoajfish99/marine-animal-images)
- 🏷️ **Classes**: 9 distinct marine animal species:

  | Species | Icon | Description |
  |---------|------|-------------|
  | 🐟 Fish | 🐟 | General fish species |
  | 🐠 Goldfish | 🐠 | Ornamental goldfish |
  | 🦭 Harbor seal | 🦭 | Marine mammal seals |
  | 🪼 Jellyfish | 🪼 | Transparent marine cnidarians |
  | 🦞 Lobster | 🦞 | Crustacean arthropods |
  | 🦪 Oyster | 🦪 | Bivalve mollusks |
  | 🐢 Sea turtle | 🐢 | Marine reptiles |
  | 🦑 Squid | 🦑 | Cephalopod mollusks |
  | ⭐ Starfish | ⭐ | Echinoderms (sea stars) |

- 📁 **Structure**: The dataset is pre-organized into train and test directories, with subfolders for each species. A validation set is created from the training data during preprocessing

## Technologies Used

### 🐍 Programming Language
- **Python (3.8+)**

### 🧠 Deep Learning Framework
- **PyTorch** (Recommended for XAI libraries like Captum and pytorch-gradcam)
- *Alternative: TensorFlow/Keras*

### 📚 Core Libraries
- **`numpy`**: Numerical operations
- **`pandas`**: Data manipulation (if needed for metadata)
- **`Pillow (PIL)`**: Image handling
- **`torchvision`** (for PyTorch): Dataset loading, transformations, pre-trained models
- **`scikit-learn`**: Evaluation metrics, train-test split
- **`matplotlib`, `seaborn`**: Data visualization, plotting XAI heatmaps
- **`opencv-python`**: Image processing (optional, for advanced augmentation/visualization)

### 🔍 XAI Libraries
- **`pytorch-grad-cam`**: For easy implementation of Grad-CAM and its variants
- **`grad-cam`**: Alternative Grad-CAM implementation
- **`captum`** (PyTorch): A comprehensive library for model interpretability (includes Grad-CAM, LIME, SHAP, etc.)
- *Alternative for TensorFlow/Keras: tf-keras-vis, eli5, shap*

## Project Structure

```
📁 marine-species-classification/
├── 📊 data/
│   └── 🖼️ images/
│       ├── 🏋️ train/
│       │   ├── 🐟 Fish/
│       │   ├── 🐠 Goldfish/
│       │   ├── 🦭 Harbor seal/
│       │   ├── 🪼 Jellyfish/
│       │   ├── 🦞 Lobster/
│       │   ├── 🦪 Oyster/
│       │   ├── 🐢 Sea turtle/
│       │   ├── 🦑 Squid/
│       │   └── ⭐ Starfish/
│       └── 🧪 test/
│           ├── 🐟 Fish/
│           └── ... (other species folders)
├── 📓 notebooks/
│   ├── 1️⃣ 1_Data_Exploration_Preprocessing.ipynb
│   ├── 2️⃣ 2_Model_Training_Evaluation.ipynb
│   └── 3️⃣ 3_XAI_Analysis.ipynb
├── 🛠️ src/
│   ├── 📦 dataset.py          # Custom dataset and data loading utilities
│   ├── 🧠 model.py            # Model definition and loading pre-trained weights
│   ├── 🏋️ train.py            # Training loop and evaluation functions
│   ├── 🔍 xai_utils.py        # Functions for applying and visualizing XAI methods
│   └── ⚙️ config.py           # Configuration parameters (paths, hyperparameters)
├── 📈 outputs/
│   ├── 💾 models/             # Saved model checkpoints
│   ├── 📊 predictions/        # Prediction results (e.g., CSV)
│   ├── 🎨 visualizations/     # XAI heatmaps, confusion matrices
│   └── 📝 logs/               # Training logs
├── 📄 README.md               # This file
└── 📋 requirements.txt        # List of Python dependencies
```

## Installation

### 1️⃣ Clone the repository:
```bash
git clone https://github.com/atharva262005/marine-species-classification.git
cd marine-species-classification
```

### 2️⃣ Create a virtual environment (recommended):
```bash
python -m venv venv

# On Windows:
.\venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```

> **Note**: If you choose TensorFlow/Keras, you'll need to adjust `requirements.txt` accordingly.

### 4️⃣ Download the Dataset:
1. 🌐 Go to the [Kaggle Marine Animal Images dataset page](https://www.kaggle.com/datasets/mikoajfish99/marine-animal-images)
2. 📥 Download the dataset as a ZIP file
3. 📂 Unzip the contents into the `data/` directory within your project structure
4. ✅ Ensure the `images` folder structure matches the project layout

## Project Workflow - Successfully Implemented

### 1. Data Preparation ✅
- **Notebook**: `1_Data_Exploration_Preprocessing.ipynb`
- **Achievement**: Successfully loaded and preprocessed marine animal images
- **Output**: Configured data loaders with proper augmentation for 9 species classes

### 2. Model Training ✅  
- **Notebook**: `2_Model_Training_Evaluation.ipynb`
- **Achievement**: Trained ResNet50 model with transfer learning
- **Output**: Saved best model checkpoint (`resnet50_best_model.pth`)

### 3. XAI Analysis ✅
- **Notebook**: `3_XAI_Analysis.ipynb` 
- **Achievement**: Generated Grad-CAM visualizations for model interpretability
- **Output**: Multiple species-specific heatmap visualizations

### Technical Challenges Solved:
- ✅ **Import Path Issues**: Resolved module import problems across notebooks
- ✅ **Gradient Flow**: Fixed gradient computation for fine-tuned models in XAI analysis
- ✅ **Target Layer Selection**: Correctly identified optimal layers for Grad-CAM in ResNet50
- ✅ **Library Compatibility**: Resolved pytorch-grad-cam API compatibility issues

## Usage (How to Run) - Verified Working

### 1️⃣ Data Preparation
- 📓 **Run the notebook**: `1_Data_Exploration_Preprocessing.ipynb` in the `notebooks/` directory
- 🎯 **Purpose**: This notebook will guide you through loading the dataset, performing initial exploration, and setting up data loaders with augmentation
- ⚙️ **Configuration**: Ensure your `config.py` in `src/` points to the correct dataset path

### 2️⃣ Model Training
You can train the model using either approach:

#### 📜 **Option A: Using Python Script**
```bash
python src/train.py
```
- 🚀 This will start the training process, save model checkpoints, and log training progress

#### 📓 **Option B: Using Jupyter Notebook** (Recommended)
- 📂 Open `2_Model_Training_Evaluation.ipynb` and run all cells
- 📊 Provides a step-by-step walkthrough of the training process and initial evaluation

### 3️⃣ Explainable AI Analysis
- 📓 **After training**: Use the `3_XAI_Analysis.ipynb` notebook to apply XAI techniques
- 🔍 **Functionality**: Load your trained model, select specific images, and generate Grad-CAM heatmaps
- 💾 **Output**: Generated visualizations will be saved to the `outputs/visualizations/` directory

## Model Training Details

### 🏗️ Architecture
- **Base Model**: Pre-trained ResNet50 (or EfficientNetB0) used as the backbone
- **Fine-tuning**: Fine-tuned on the marine animal dataset
- **Classification Layer**: Final layer replaced to match the 9 output classes

### ⚙️ Training Configuration
- 📉 **Loss Function**: CrossEntropyLoss
- 🎯 **Optimizer**: Adam 
- 📊 **Learning Rate**: Initial learning rate of 0.001 (adjustable)
- 🔄 **Epochs**: Trained for 20-50 epochs (adjust based on validation performance)
- 📦 **Batch Size**: 32 (adjust based on GPU memory)

### 🎨 Data Augmentation
- **Training Data**:
  - 🖼️ Random resized crop
  - 🔄 Horizontal flip
  - 🌈 Color jitter
  - 📊 Normalization
- **Validation/Test Data**:
  - 📏 Resize and normalization only

## Explainable AI (XAI) Implementation & Analysis

This project successfully implements Grad-CAM to generate visual explanations for model predictions.

### How Grad-CAM Works:
Grad-CAM uses the gradients of the target concept (e.g., the predicted class score) flowing into the final convolutional layer to produce a coarse localization map highlighting the important regions in the image for predicting the concept. This heatmap is then overlaid on the original image.

### Successful XAI Implementation:

The project successfully generates Grad-CAM visualizations for multiple marine species:

**✅ Working Visualizations:**
- **Fish Classification**: `gradcam_Fish_93a6d5a722daadd3.png`
- **Lobster Classification**: `gradcam_Lobster_1475e042f8d9255f.png`, `gradcam_Lobster_418f44ae9827c21e.png`
- **Oyster Classification**: `gradcam_Oyster_9a4e16ebca5aa320.png`
- **Squid Classification**: `gradcam_Squid_602cf8e3204d82e8.png`

**Key Technical Achievements:**
- Successfully resolved gradient flow issues in fine-tuned models
- Implemented proper target layer selection for ResNet50 architecture
- Generated high-quality heatmap overlays showing model attention patterns
- Provides interpretable explanations for each classification decision

### XAI Insights:

The Grad-CAM analysis reveals:
- **Feature Focus**: Model correctly identifies species-specific anatomical features
- **Decision Patterns**: Clear visualization of which image regions influence predictions
- **Model Validation**: Confirms model focuses on biologically relevant features rather than background artifacts
- **Scientific Value**: Provides insights into visual cues important for marine species identification

## Results

### 🏆 Model Performance
The project successfully implements a ResNet50-based marine species classifier with comprehensive XAI analysis. The model was trained using transfer learning with the following configuration:

- **Architecture**: ResNet50 (pre-trained on ImageNet)
- **Training Strategy**: Fine-tuning with frozen backbone layers
- **Image Size**: 224x224 pixels
- **Batch Size**: 32
- **Optimizer**: Adam with learning rate 0.001
- **Training Epochs**: 25

### 📊 **Achieved Results**
- **🎯 Final Test Accuracy: 85.41%** - Excellent performance for 9-class marine species classification
- **🏋️ Training**: Successfully completed 25 epochs with validation monitoring
- **💾 Model Saved**: Best performing model checkpoint saved as `resnet50_best_model.pth`

### 🖼️ **Generated Output Images**

The project successfully generates multiple types of visualizations:

#### 🔍 **XAI Visualizations** (Grad-CAM Heatmaps):

<div align="center">

| Species | Grad-CAM Visualization | File Name |
|---------|----------------------|-----------|
| 🐟 **Fish** | ![Fish Grad-CAM](outputs/visualizations/gradcam_Fish_93a6d5a722daadd3.png) | `gradcam_Fish_93a6d5a722daadd3.png` |
| 🦞 **Lobster #1** | ![Lobster Grad-CAM 1](outputs/visualizations/gradcam_Lobster_1475e042f8d9255f.png) | `gradcam_Lobster_1475e042f8d9255f.png` |
| 🦞 **Lobster #2** | ![Lobster Grad-CAM 2](outputs/visualizations/gradcam_Lobster_418f44ae9827c21e.png) | `gradcam_Lobster_418f44ae9827c21e.png` |
| 🦪 **Oyster** | ![Oyster Grad-CAM](outputs/visualizations/gradcam_Oyster_9a4e16ebca5aa320.png) | `gradcam_Oyster_9a4e16ebca5aa320.png` |
| 🦑 **Squid** | ![Squid Grad-CAM](outputs/visualizations/gradcam_Squid_602cf8e3204d82e8.png) | `gradcam_Squid_602cf8e3204d82e8.png` |

</div>

#### 📈 **Performance Visualizations**:

<div align="center">

| Visualization Type | Image | Description |
|-------------------|-------|-------------|
| 📊 **Confusion Matrix** | ![Confusion Matrix](outputs/visualizations/test_confusion_matrix.png) | Shows classification performance across all 9 marine species |

</div>

> **Note**: These Grad-CAM heatmaps show the regions of the input images that the model focuses on when making predictions. Red/warm colors indicate high importance, while blue/cool colors indicate lower importance for the classification decision.

### Key Achievements

✅ **Successful Model Training**: Implemented end-to-end training pipeline with validation monitoring  
✅ **High Accuracy**: Achieved **85.41% test accuracy** on 9-class marine species classification  
✅ **XAI Integration**: Successfully generated Grad-CAM visualizations showing model decision patterns  
✅ **Multi-Class Classification**: Accurately classifies 9 marine species classes  
✅ **Explainable Predictions**: Provides visual explanations for each classification decision  

### XAI Analysis Results

The Explainable AI component successfully generates heatmap visualizations that reveal:

- **Feature Focus**: Model correctly identifies species-specific anatomical features
- **Decision Transparency**: Clear visual explanations for each prediction with **85.41% accuracy**
- **Quality Assurance**: Ability to detect if model focuses on relevant biological features vs. background artifacts

**Generated Visualizations** (saved in `outputs/visualizations/`):
- 🎨 **5 Grad-CAM heatmaps** for Fish, Lobster (2), Oyster, and Squid classifications
- 📊 **Confusion matrix visualization** for comprehensive performance analysis
- 🔍 **High-quality overlays** showing exactly where the model focuses for each prediction

### Performance Metrics Summary

| Metric | Value | Description |
|--------|-------|-------------|
| 🎯 **Test Accuracy** | **85.41%** | Overall classification accuracy |
| 🏷️ **Classes** | 9 | Marine species categories |
| 📦 **Model Size** | ResNet50 | Pre-trained backbone |
| 🔄 **Training Epochs** | 25 | Optimal training duration |
| 💾 **Model File** | `resnet50_best_model.pth` | Saved checkpoint |

### Technical Implementation Status

| Component | Status | Description |
|-----------|--------|-------------|
| Data Pipeline | ✅ Complete | Automated data loading with augmentation |
| Model Training | ✅ Complete | ResNet50 fine-tuning with validation |
| Model Evaluation | ✅ Complete | Comprehensive metrics and visualization |
| XAI Analysis | ✅ Complete | Grad-CAM explanations working successfully |
| Results Export | ✅ Complete | Model checkpoints and visualizations saved |

### Model Output Structure

```
📈 outputs/
├── 💾 models/
│   └── resnet50_best_model.pth     # Trained model checkpoint (85.41% accuracy)
├── 🎨 visualizations/
│   ├── gradcam_Fish_*.png          # Grad-CAM for Fish predictions
│   ├── gradcam_Lobster_*.png       # Grad-CAM for Lobster predictions (2 images)
│   ├── gradcam_Oyster_*.png        # Grad-CAM for Oyster predictions
│   ├── gradcam_Squid_*.png         # Grad-CAM for Squid predictions
│   └── test_confusion_matrix.png   # Performance visualization matrix
└── 📊 predictions/                  # Model prediction outputs
```

### 🎯 **Real Results Achieved:**
- ✅ **85.41% Test Accuracy** - Excellent performance for marine species classification
- ✅ **5 High-Quality XAI Visualizations** - Clear Grad-CAM heatmaps generated
- ✅ **Complete Pipeline Working** - From data loading to XAI analysis
- ✅ **Reproducible Results** - Saved model and configuration for future use

---

## 🎨 Visual Results Gallery

### 🔍 Explainable AI (Grad-CAM) Results

<div align="center">

**🐟 Fish Species Classification**
![Fish Grad-CAM](outputs/visualizations/gradcam_Fish_93a6d5a722daadd3.png)

**🦞 Lobster Species Classification - Sample 1**
![Lobster Grad-CAM 1](outputs/visualizations/gradcam_Lobster_1475e042f8d9255f.png)

**🦞 Lobster Species Classification - Sample 2**
![Lobster Grad-CAM 2](outputs/visualizations/gradcam_Lobster_418f44ae9827c21e.png)

**🦪 Oyster Species Classification**
![Oyster Grad-CAM](outputs/visualizations/gradcam_Oyster_9a4e16ebca5aa320.png)

**🦑 Squid Species Classification**
![Squid Grad-CAM](outputs/visualizations/gradcam_Squid_602cf8e3204d82e8.png)

**📊 Model Performance - Confusion Matrix**
![Confusion Matrix](outputs/visualizations/test_confusion_matrix.png)

</div>

### 🧠 What These Visualizations Show:

- **🔥 Red/Warm Areas**: Regions the model considers most important for classification
- **❄️ Blue/Cool Areas**: Regions with lower importance for the decision
- **🎯 Focus Patterns**: Model correctly identifies species-specific anatomical features
- **✅ Validation**: Confirms the model focuses on relevant biological features rather than background

---

## Future Enhancements

### 🔬 Advanced XAI Techniques
- 🧪 **Explore More Methods**: Implement and compare other XAI methods like LIME, SHAP, Integrated Gradients, or DeepLift for richer insights
- 📊 **Comparative Analysis**: Side-by-side comparison of different explanation methods

### ⚡ Performance & Deployment
- 🚀 **Real-time Inference**: Optimize the model for faster inference to enable potential real-time applications
- 🌐 **Web Deployment**: Create a simple web application (e.g., using Flask/Streamlit) to demonstrate the model and XAI explanations interactively
- 📱 **Mobile App**: Develop a mobile application for field marine biologists

### 📈 Data & Model Improvements
- 📊 **Larger Dataset**: Train on a larger and more varied dataset of underwater species
- 🎥 **Video Data**: Potentially include video data for temporal analysis
- 🌊 **Underwater Enhancement**: Integrate pre-processing steps specifically designed for challenging underwater images (dehazing, color correction)

### 🛡️ Robustness & Security
- 🔒 **Adversarial Robustness**: Investigate the model's robustness to adversarial attacks
- 🔍 **XAI for Security**: How XAI can help in understanding vulnerabilities
- 🧪 **Stress Testing**: Test model performance under various underwater conditions

## Contributing

🤝 **Contributions are welcome!** If you have suggestions for improvements, bug fixes, or new features, please feel free to:

- 🐛 **Report Issues**: Open an issue for bugs or feature requests
- 🔀 **Submit Pull Requests**: Fork the repo and submit your improvements
- 💡 **Share Ideas**: Contribute to discussions and suggest enhancements
- 📚 **Improve Documentation**: Help make the documentation even better

### How to Contribute:
1. 🍴 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. 📤 Push to the branch (`git push origin feature/AmazingFeature`)
5. 🔀 Open a Pull Request

## License

📄 This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## Contact

📬 **For any questions or collaborations, feel free to reach out:**

- 👤 **Name**: Atharva Ralegankar
- 📧 **Email**: [ralegankaratharva@gmail.com](mailto:ralegankaratharva@gmail.com)
- 💼 **LinkedIn**: [atharvaralegankar](https://www.linkedin.com/in/atharvaralegankar/)
- 🐙 **GitHub**: [atharva262005](https://github.com/atharva262005)

---

<div align="center">

### ⭐ If you found this project helpful, please give it a star! ⭐

**Made with ❤️ for marine conservation and AI research**

</div>