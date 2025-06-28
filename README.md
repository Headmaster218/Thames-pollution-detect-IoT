# Intelligent Waterborne Fecal Contamination Monitoring System

## AquaSense

This project presents a **low-cost, deployable, AI-integrated platform** for monitoring fecal contamination in natural waters. By using **inorganic water quality parameters** (e.g., DO, PH, turbidity, conductivity) as surrogate inputs, the system predicts **E. coli concentration** and collects physical water samples for future validation. It integrates **real-time sensing, wireless transmission, machine learning inference, data visualization, and automated water sampling**, forming a complete perception–prediction–collection loop.

## 🎥 Video preview

[![Press Here~](./Reports/AquaSense.png)](./Reports/AquaSense.mp4)


---

## 📦 Project Summary

- **Title:** Integrated AI-driven Monitoring and Sampling System for Fecal Contamination in Urban Waters  
- **Core Idea:** Replace expensive, time-delayed E. coli testing methods with an inorganic-indicator-based prediction model
- **Application Scenario:** Urban rivers (e.g., Thames), lakes, and distributed low-resource field sites  
- **Lead Developer:** [Your Name] (system architect, embedded design, ML modeling)  
- **Collaborators:** [Names or initials if applicable]

---

## 🧠 System Architecture

```
       +------------------+      LoRa        +-------------------+
       | Floating Buoy    |----------------->| LoRa Gateway      |
       | (Sensors + MCU)  |                  | (MQTT Broker)     |
       +--------+---------+                  +-------------------+
                |                                        |
                | LoRa                                   v MQTT
      +---------v---------+                       +---------------+
      | Autonomous Sampler|                       | Cloud Server  |
      | (RTC + Pump +     |<--------------------->| Data Storage  |
      |  Bottle Rotation) |                       +------+--------+
      +-------------------+                              |
                                                         v
                                 +-------------------------------+
                                 |   Web Dashboard & Mobile App  |
                                 |   (Real-time + Historical)    |
                                 +-------------------------------+
```
---

## 🧩 System Modules

### 1. 🧪 Sensor Unit (Floating Buoy)
- Turbidity, Dissolved Oxygen, PH, conductivity sensors
- Arduino-based circuit with solar power
- Periodic sampling + LoRa transmission

### 2. 🔄 Autonomous Sampling Unit 
- Peristaltic pump + rotating vial mechanism
- RTC-controlled periodic water collection
- Time-synced with sensor data via LoRa

### 3. 🌐 Server & Visualization
- LoRa gateway + MQTT + Python backend
- Database (SQLite)
- Web frontend with charts and map
- Mobile app with charts and map

### 4. 🧠 Machine Learning Module
- Trained neural network (FCN) on public water data (~23GB → 1MB preprocessed)
- Predicts log-scaled E. coli concentration
- RMSE ≈ 0.83 on test set
- Optional model variants: XGBoost, RF for comparison

---

## ⚙️ Deployment Features

- Fully solar-powered, suitable for field deployment
- Wireless transmission via LoRa (long range, low power)
- Real-time + historical data access
- Expandable to swarm systems or RL-based control

---

## 🚀 Key Contributions

- ✅ First attempt to estimate fecal contamination using only inorganic indicators  
- ✅ End-to-end system: sensing, prediction, visualization, sampling  
- ✅ Open-source, low-cost, and designed for global scalability  
- ✅ Modular and extensible: supports AI upgrades, edge deployment, or swarm coordination  

---

## 📌 Future Directions

- Integrate control strategies (e.g., RL-based adaptive sampling)  
- Train with real E. coli lab data using automated sampler  
- Edge deployment of ML model on MCU (TinyML)  
- Multi-node deployment and anomaly correlation  

---

## 📁 Folder Structure

```
Root/
├── 3D_Models/        # CAD models
├── Arduino_Code/     # Embedded firmware
├── Backend_code/     # Backend services, APIs
├── MLcode_Tommy_try/ # Experimental ML models by "Tommy"
├── ML_Main/          # Main ML pipeline: data, preprocess, training, model
├── Reports/          # Project documentation, research papers, final reports
└── Visualization/    # Frontend interfaces for data visualization and monitoring
    ├── APP/          # Mobile app interface "BlueGuardian"
    └── Web/          # Web dashboard 
```
---
## Author
Zhuohang Wu @
zhuohang2024@163.com

---

## 📬 Contact & Acknowledgements

For technical inquiries, collaboration requests, or feedback, feel free to contact:

**📌 Project Lead:**  
Zhuohang Wu (John) – zhuohang2024@163.com

**👥 Team Members:**  
Chenghao Xin, Xueyin Fan

**🧑‍🏫 Supervisors:**  
- Dr. Akin Delibasi – Department of Computer Science, UCL  
- Dr. Valerio Modugno – Department of Computer Science, UCL  
- Dr. Izzy Bishop – Department of Genetics, Evolution & Environment, UCL

---

