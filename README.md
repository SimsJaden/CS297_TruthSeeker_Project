# CS297_TruthSeeker_Project
# Truth-Seeker Machine Learning Project

## Author
**Jaden Sims**  
Boise State College of Engineering, Boise, ID 83716 USA  

---

## Abstract
This project explores optimizing machine learning models for text classification using the Truth-Seeker dataset. The dataset includes textual data labeled for binary classification (true/false) and multi-class classification (three and five labels). A pre-trained BERT-based architecture was fine-tuned on the data using advanced preprocessing and optimization techniques. Efforts focused on reducing training time via hyperparameter tuning while maintaining acceptable accuracy levels. The final models achieved classification accuracies of approximately 79% for binary classification and 49% for 4-class classification, with significantly reduced runtime.

---

## Introduction
Machine learning plays a critical role in solving classification challenges, from misinformation detection to sentiment analysis. This project uses the Truth-Seeker dataset to tackle binary and multi-class classification tasks while emphasizing runtime efficiency and avoiding data leakage. Fine-tuning a pre-trained BERT architecture, optimizing hyperparameters, and employing tokenization techniques formed the foundation of this project.

---

## Model Optimization Process
The following optimizations were implemented:
1. **Learning Rate**: Adjusted to `5e-5`, reducing instability and runtime.
2. **Batch Size**: Set to 8 to improve GPU memory usage.
3. **Epochs**: Reduced to 1, cutting overall runtime without significant accuracy loss.
4. **Mixed Precision Training**: Enabled fp16 calculations for faster computation.

Data preprocessing ensured no overlap between train-test splits and limited tokenized sequences to 256 tokens.

---

## Challenges
1. **Technical Setup**: Managing dependencies across Python, PyTorch, and the Transformers library.
2. **Performance**: Achieving high accuracy for binary classification but struggling with multi-class tasks, especially 5-class classification.
3. **Runtime**: Reducing training time from 4 hours to under 1 hour while maintaining performance.

---

## Results
- **Binary Classification**: Achieved 79.68% accuracy with well-balanced precision and recall.
- **Multi-Class Classification (5 Labels)**: Accuracy of ~49%, reflecting challenges with closely related labels.
- Additional experiments with SMOTE and ensemble methods showed potential for improvement.

---

## Future Improvements
1. Addressing class imbalance using advanced techniques like SMOTE.
2. Experimenting with ensemble methods or alternative architectures.
3. Exploring learning rate schedulers and other optimization techniques.
4. Leveraging distributed computing to speed up experiments.

---

## Conclusion
The project successfully fine-tuned a BERT model for binary and multi-class classification, achieving strong results for binary tasks. However, multi-class classification accuracy highlighted areas for further improvement. Future work will focus on addressing class imbalance, enhancing model architecture, and optimizing training pipelines.
