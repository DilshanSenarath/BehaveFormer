# BehaveFormer: A Framework with Spatio-Temporal Dual Attention Transformers for IMU enhanced Keystroke Dynamics
Welcome to the BehaveFormer repository! This repository presents an innovative framework proposed in our research paper [1] that combines keystroke dynamics and Inertial Measurement Unit (IMU) data to create a reliable and accurate Continuous Authentication (CA) system based on behavioural biometrics. Our approach introduces the Spatio-Temporal Dual Attention Transformers (STDAT), a novel transformer architecture that extracts highly discriminative features from keystroke dynamics. By incorporating IMU data, BehaveFormer significantly improves the discriminative power of keystroke dynamics authentication. We invite you to explore this repository and leverage the power of BehaveFormer to enhance the security and usability of CA systems.

# Abstract
Continuous Authentication using behavioural biometrics is a type of biometric identification that recognizes individuals based on their unique behavioural characteristics, like their typing style. However, the existing systems that use keystroke or touch stroke data have limited accuracy and reliability. To improve this, smartphones' Inertial Measurement Unit sensors, which include accelerometers, gyroscopes, and magnetometers, can be used to gather data on users' behavioural patterns, such as how they hold their phones. Combining this IMU data with keystroke data can enhance the accuracy of behavioural biometrics-based CA. This paper [1] proposes BehaveFormer (See Fig. 01), a new framework that employs keystroke and IMU data to create a reliable and accurate behavioural biometric CA system. It includes two Spatio-Temporal Dual Attention Transformer (STDAT) (See Fig. 02), a novel transformer we introduce to extract more discriminative features from keystroke dynamics. Experimental results on three publicly available datasets (Aalto DB, HMOG DB, and HuMIdb) demonstrate that BehaveFormer outperforms the state-of-the-art behavioural biometric-based CA systems. For instance, on the HuMIdb dataset, BehaveFormer achieved an EER of 2.95\%. Additionally, the proposed STDAT has been shown to improve the BehaveFormer system even when only keystroke data is used. For example, on the Aalto DB dataset, BehaveFormer achieved an EER of 1.80\%. These results demonstrate the effectiveness of the proposed STDAT and the incorporation of IMU data for behavioural biometric authentication.

*Fig. 01: BehaveFormer Model Diagram*
![Binary image](images/behaveformer.png "BehaveFormer")

*Fig. 02: Spatio-Temporal Dual Attention Transformer (STDAT) Diagram*
![Binary image](images/stdat.png "STDAT")

# Datasets
|    Dataset   | Subjects | Sessions | Actions                                          | Modalities                                                                             | Download                                                                                                                                                                                           |
|:------------:|:--------:|:--------:|--------------------------------------------------|----------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Aalto DB [2] | ~260 000 |    15    | Typing                                           | Keystroke                                                                              | <a href="https://userinterfaces.aalto.fi/typing37k/" style="padding: 10px 20px; background-color: #4CAF50; color: white; text-decoration: none; border-radius: 4px; cursor: pointer;">Aalto DB</a> |
|  HMOG DB [3] |    100   |    24    | Reading, Typing, Map navigation                  | Keystroke, Raw touch event, IMU                                                        | <a href="https://hmog-dataset.github.io/hmog/" style="padding: 10px 20px; background-color: #4CAF50; color: white; text-decoration: none; border-radius: 4px; cursor: pointer;">HMOG DB</a>        |
|  Humidb [4]  |    599   |    1-5   | Typing, Swipe, Tap, Hand gesture, Finger writing | Keystroke, Touch, IMU, Light, GPS, WiFi, Bluetooth, Orientation, Proximity, Microphone | <a href="https://github.com/BiDAlab/HuMIdb" style="padding: 10px 20px; background-color: #4CAF50; color: white; text-decoration: none; border-radius: 4px; cursor: pointer;">Humidb</a>            |

# Preprocessing and Feature Creation

# Training

# Evaluation

# References
[1] - Our Paper

[2] - K. Palin, A. M. Feit, S. Kim, P. O. Kristensson, and A. Oulasvirta. How do people type on mobile devices? observations from a study with 37,000 volunteers. In Proceedings of the 21st International Conference on Human-Computer Interaction with Mobile Devices and Services, pages 1–12, Taipei, Taiwan, 2019.

[3] - Z. Sitov ́a, J. ˇSedˇenka, Q. Yang, G. Peng, G. Zhou, P. Gasti, and K. S. Balagani. Hmog: New behavioral biometric features for continuous authentication of smartphone users. IEEE Transactions on Information Forensics and Security, 11(5):877–892, 2016.

[4] - A. Acien, A. Morales, J. Fierrez, R. Vera-Rodriguez, and O. Delgado-Mohatar. Becaptcha: Behavioral bot detection using touchscreen and mobile sensors benchmarked on humidb. Engineering Applications of Artificial Intelligence, 98:104058, 2021.