#WEEK 3 SPECIALIZATION (edited after deadline because of the misprint of certain numbers)
The RoBERTa-based Transformer model for Named Entity Recognition was fine-tuned over three epochs, showing strong improvements in performance. During the first epoch, the model achieved a training loss of 0.1732 and a validation loss of 0.0484. The corresponding F1 score was 0.9278, indicating that even in the early stages of fine-tuning, the pre-trained Transformer architecture was already extracting useful contextual representations for named entity labeling.

In the second epoch, the training loss dropped sharply to 0.0319, while the validation loss decreased to 0.0329. The F1 score improved significantly to 0.9538. This suggests that the Transformer’s attention mechanisms and contextual embeddings were adapting well to the CoNLL-2003 dataset, allowing it to better distinguish between entity boundaries and types.

By the third epoch, the training loss further decreased to 0.0168, and the F1 score climbed to an impressive 0.9607. However, the validation loss, while showing a slight further decrease to 0.0303, appears to be plateauing on the loss curve. This divergence, where the training loss continues to drop significantly while the validation loss shows only marginal gains and flattens, suggests the onset of overfitting. The model is becoming increasingly specialized to the training data, potentially at the expense of its generalization ability to unseen examples.

Overall, the Transformer’s architecture proved highly effective for the NER task, achieving near state-of-the-art results. For future iterations, strategies like early stopping based on validation loss would be crucial to prevent further overfitting and ensure the deployment of the most generalizable model.



#WEEK 2
1. FashionMNIST

For this project, I implemented a convolutional neural network (CNN) using PyTorch to classify images from the FashionMNIST dataset — a collection of 28×28 grayscale images across 10 clothing categories. PyTorch was chosen for its dynamic computation graph and flexibility, making it ideal for developing and debugging deep learning models at a conceptual level, which aligns well with my academic setting and goals.
The model architecture consists of two convolutional layers with Batch Normalization, ReLU activations, and MaxPooling, followed by a fully connected classification head with Dropout regularization to reduce overfitting. I used the Adam optimizer for efficient gradient updates, and a StepLR scheduler to reduce the learning rate halfway through training for better convergence. Data was normalized to mean 0.5 and standard deviation 0.5 to stabilize learning.
The model was trained for 10 epochs on a batch size of 64, and achieved a test accuracy of over 92%, which is a strong result for FashionMNIST. The loss curve plotted across epochs demonstrated steady and consistent convergence, indicating effective learning without signs of overfitting or vanishing gradients.
To support qualitative understanding, I visualized sample predictions alongside their ground truth labels. Most predictions were accurate, and even the incorrect ones (e.g., a coat being classified as a shirt) were justifiable given visual similarity — reflecting the model's real-world reasoning limitations. I also included a confusion matrix and a detailed classification report to break down the model's performance across all categories, with especially strong precision and recall for classes like trousers, bags, and sneakers.
Finally, I saved both the model and visual artifacts (loss curve, confusion matrix, and predictions), ensuring reproducibility and professional presentation. 

<img width="451" alt="fashionoutput" src="https://github.com/user-attachments/assets/380950ec-e7a3-4006-a3cc-425de6a0e821" />
<img width="911" alt="fashionreport" src="https://github.com/user-attachments/assets/9f895361-35a4-470f-8ce6-c3e1332188e6" />

confusion matrix image: ![confusion_matrix](https://github.com/user-attachments/assets/936ea271-406d-4955-88d1-d96177dbf63b)

loss curve image: ![loss_curve](https://github.com/user-attachments/assets/a7e024cc-b7bd-4678-a8ab-02aae188bd40)

sample predictions image: ![sample_predictions](https://github.com/user-attachments/assets/310082b7-abf2-4fc4-bf48-3d153103be05)


2. ImageNet
To solve the image classification problem effectively, I chose a transfer learning approach using the pretrained ResNet18 model from PyTorch. ResNet18 strikes a strong balance between accuracy and speed—it is lightweight enough for quick fine-tuning and inference, yet powerful due to its residual architecture. Since training a deep model from scratch requires extensive data and compute resources, leveraging a pretrained model with fine-tuning allowed me to achieve good performance on a limited custom dataset.
My strategy was to freeze all layers of ResNet18 initially to retain the generalized image features learned from ImageNet. I then unfroze only the last residual block (layer4), enabling the model to learn domain-specific patterns in my custom dataset without disrupting earlier layers. This selective fine-tuning approach prevents overfitting and reduces the training time, while still giving the model flexibility to adapt to the target classes—car, dog, and cat.
To customize the classifier, I replaced the final fully connected layer with a new head consisting of a linear layer, ReLU activation, dropout, and a final linear layer mapping to 3 output classes. This lightweight architecture added non-linearity and regularization while being efficient for small datasets. The model was trained using the cross-entropy loss function and Adam optimizer, which provides fast and adaptive learning. I split the dataset into an 80/20 ratio for training and validation to track overfitting and generalization.
During inference, I built a separate dataset class to handle image loading and transformation. The model then performed predictions using softmax probabilities, and I extracted the top predicted classes. For user-friendly output, I displayed the top predictions alongside probability scores and plotted both the input image and prediction bar chart using matplotlib. Additionally, I generated a CSV file logging each image’s prediction, which is useful for batch processing and verification.

<img width="455" alt="imagenet training output" src="https://github.com/user-attachments/assets/de56c98b-6290-420e-86a4-b5dd7aff0784" />

<img width="556" alt="imagenet input" src="https://github.com/user-attachments/assets/fc22ef88-0079-4e01-98d3-2328bc3f1a18" />

<img width="581" alt="imagenet graph" src="https://github.com/user-attachments/assets/4b33799c-0404-4ff2-8018-99d5e50a581d" />


3.Face Expression Recognition Dataset
In this project, we tackled the task of facial expression recognition using deep learning. The goal was to classify images of human faces into different emotional categories such as happy, sad, angry, and others. This is a complex problem in computer vision due to the subtle and sometimes ambiguous nature of facial expressions, as well as variations in lighting, pose, and individual facial features.
To build our solution, we selected PyTorch as the deep learning framework. PyTorch was chosen for its clean, flexible API and strong support for transfer learning through the torchvision library. Its ease of use, debugging capabilities, and large community made it ideal for iterative experimentation and fine-tuning, especially for image-based tasks like this one.
Instead of training a convolutional neural network from scratch — which would require a vast amount of data and computational resources — we adopted a transfer learning approach using the ResNet-18 architecture pretrained on ImageNet. ResNet-18 is a well-established CNN known for its residual connections and efficiency. Because it has already learned to extract low-level and mid-level features from millions of images, we were able to repurpose its weights for our specific task with only minor adjustments. We replaced the final fully connected layer of ResNet-18 with a new layer suited to the number of expression classes in our dataset.
To ensure effective training without overfitting, we froze the early layers of the network and fine-tuned the deeper layers, specifically layer4 and the final classification head. This strategy balances efficiency and adaptability, allowing the network to retain its general visual understanding while adapting to the nuances of facial emotion classification.
We also applied data augmentation techniques such as random horizontal flips, rotations, and color jittering. These augmentations helped the model generalize better by exposing it to variations that resemble real-world scenarios. Additionally, we used a learning rate scheduler to gradually reduce the learning rate during training, which improves convergence and helps fine-tune the model’s performance.
For evaluation, we tracked the training loss and computed validation accuracy across epochs. We further analyzed the results using a confusion matrix and a classification report to better understand class-wise performance, which is particularly important in imbalanced or subtle classification tasks like facial expression recognition.
Overall, this approach led to a solid validation accuracy of over 64%, which is competitive for a small model trained on a limited dataset in just a few epochs. With further tuning — including extended training, early stopping, and hyperparameter optimization — this pipeline can be extended to achieve even higher performance. This project demonstrates how transfer learning can be effectively applied to solve complex visual classification problems with efficiency and accuracy.

<img width="452" alt="outputfacecm" src="https://github.com/user-attachments/assets/507851b8-a385-4d17-8def-bd488735437b" />

<img width="940" alt="outputface" src="https://github.com/user-attachments/assets/60874e6a-f9d4-4fbc-8a80-1f2813b2a826" />

4. DeepWeeds
For this project, we tackled the DeepWeeds dataset—a real-world image classification challenge—using transfer learning with a pretrained ResNet-18 model in PyTorch. The goal was to develop a high-accuracy deep learning pipeline capable of identifying weed species from image data. We leveraged a well-established convolutional architecture (ResNet-18) pretrained on ImageNet to extract rich features while training only the classification head specific to our dataset. This approach drastically reduced training time and avoided overfitting on a relatively smaller dataset.

The dataset was split into structured folders for training, validation, and testing using the provided CSV label files. Images were resized to 224x224 to match ResNet’s input dimension requirements. For the training set, we applied random horizontal flips and rotations to augment the data and improve generalization. Standard normalization using ImageNet’s mean and standard deviation was applied to align the input distribution with what the pretrained model expects.

We froze all layers in ResNet-18 except the fully connected (fc) classification head. The fc layer was replaced with a custom head: a linear layer with 256 neurons, ReLU activation, dropout for regularization, and a final layer mapping to the number of weed classes. Only this head was trained initially. To push validation accuracy higher, we fine-tuned the model: unfreezing the later convolutional blocks allowed the network to better adapt to domain-specific weed features.

Training was conducted for 15 epochs with the Adam optimizer and a learning rate of 1e-4. The model progressively reduced training loss from 1.0470 to 0.2767, indicating effective learning and optimization. The learning curve remained smooth and did not show overfitting, thanks to augmentation and dropout regularization. The final validation accuracy achieved was 85.60%, meeting our target.

The confusion matrix revealed excellent classification performance on certain classes (e.g., classes 2, 3, and 8 with high true positives), while a few classes (like 0 and 4) had more misclassifications—indicating a need for either more data or better class-specific augmentation. Misclassifications mostly occurred between visually similar species, which is common in fine-grained recognition tasks.

By combining transfer learning, fine-tuning, and strong data augmentations, we achieved robust performance on the DeepWeeds dataset. The approach is scalable and can be improved further by experimenting with deeper models (e.g., ResNet-50), adjusting learning rates per layer, or applying test-time augmentation. Additionally, integrating explainability tools like Grad-CAM could help visualize which parts of the image drive predictions, aiding trust and model refinement.

<img width="306" alt="outputofdeepweeds" src="https://github.com/user-attachments/assets/cd74fa15-c10f-45e6-a0f7-6820e738e745" />

<img width="940" alt="image" src="https://github.com/user-attachments/assets/44c0c26a-80d1-466f-951e-8f1508e7834b" />



#WEEK 1
TASK 1 NOTES

Source: UCI Heart Disease Dataset (Cleveland) https://archive.ics.uci.edu/dataset/45/heart+disease


This project explored binary classification of heart disease using the Cleveland dataset, where the target variable ‘num’ was binarized to represent presence (1) or absence (0) of disease. After handling missing values using median imputation and scaling features, I implemented and compared four models: Logistic Regression, Random Forest, Support Vector Classifier (SVC), and XGBoost.

Among these, Random Forest emerged as the top performer, achieving an accuracy of 88.5% and a ROC AUC of 0.959. Its ensemble structure allowed it to capture non-linear interactions and reduce overfitting through aggregation. Interestingly, Logistic Regression achieved a similarly high ROC AUC (0.958) despite slightly lower accuracy, indicating strong discrimination but potentially miscalibrated thresholding.

SVC matched Random Forest in accuracy but had a marginally lower AUC (0.944), suggesting robust decision boundaries but less reliable probability estimates. XGBoost, while typically high-performing, achieved the lowest AUC (0.927)—likely due to sensitivity to hyperparameters and the limited sample size (~300 instances).

Feature importance analysis highlighted ‘thalach’, ‘oldpeak’, and ‘ca’ as key predictors, aligning with clinical intuition around heart stress test indicators. The dataset's modest imbalance (~55% class 0 vs 45% class 1) justified using ROC AUC over raw accuracy as a comparative metric. {Edited to include the reasons that a model might not be efficient enough, post deadline}



TASK 2 NOTES

I worked with the “Household Power Consumption” dataset, which contains time-series data on household electricity usage. The original dataset had several columns and some missing values. To ensure efficient processing and focus on meaningful patterns, I first cleaned the data by dropping missing values and selecting only four core features: Global Active Power, Global Reactive Power, Voltage, and Global Intensity. These were converted to numeric types, and only valid rows were retained. After this, I randomly sampled 50,000 rows from the dataset to reduce computational load while maintaining statistical integrity.

Next, I separated the features from the target variable, which was Global Active Power. The features were scaled using standardization to ensure all values contributed equally to model training. The data was then split into training and test sets using an 80-20 split.

I implemented and evaluated three different regression models: Linear Regression, Decision Tree Regressor, and Random Forest Regressor. Linear Regression achieved an RMSE of 0.042 and an R² score of 0.998, indicating extremely high accuracy and minimal prediction error. The Decision Tree model produced a slightly higher RMSE of 0.055 and an R² of 0.997, still strong but marginally less accurate. Random Forest, which is an ensemble model based on multiple decision trees, performed nearly as well as Linear Regression, with an RMSE of 0.047 and R² of 0.998.

From these results, the Linear Regression and Random Forest models performed the best. Linear Regression was the most efficient and surprisingly accurate, suggesting that the relationship between the features and the target is largely linear. Random Forest also performed very well, benefiting from ensemble averaging and reduced overfitting compared to a single tree. The Decision Tree, while still good, was the weakest of the three, possibly due to limited depth and sensitivity to small variations in the data.

Lastly, I applied KMeans clustering on the same dataset using three clusters. The dataset was standardized before clustering. The resulting clusters were visualized using a scatterplot of Voltage vs Global Intensity, colored by cluster label. This allowed us to explore patterns in the electricity usage without a supervised target.

Overall, the workflow involved structured preprocessing, applying multiple ML models, performance evaluation using RMSE and R² metrics, and an unsupervised clustering step. Among the regression models, Linear Regression was the top performer in this case, suggesting a strong linear relationship in the data. The analysis and comparison of different models helped understand how model complexity affects performance on a real-world dataset.


