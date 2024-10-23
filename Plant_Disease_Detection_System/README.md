# Plant-disease-detection

## About the Project
An application that for farmers to detect the type of plant or crops, detect any kind of diseases in them. The app sends the image of the plant to the server where it is analysed using CNN classifier model. Once detected, the disease and its solutions are displayed to the user

## Model

Trained to identify 5 classes for **Disease Detection** and 24 classes for **Disease Classification**. 
Dataset can be downloaded form [kaggle](https://www.kaggle.com/abdallahalidev/plantvillage-dataset)

           - Disease Classification Classes

                       - Apple___Apple_scab
                       - Apple___Black_rot
			   - Apple___Cedar_apple_rust
			   - Apple___healthy
			   - Blueberry___healthy
			   - Cherry___healthy
			   - Cherry___Powdery_mildew
			   - Grape___Black_rot
			   - Grape___Esca_Black_Measles
			   - Grape___healthy
			   - Grape___Leaf_blight_Isariopsis_Leaf_Spot
			   - Orange___Haunglongbing
			   - Peach___Bacterial_spot
			   - Peach___healthy
			   - Pepper,_bell___Bacterial_spot
			   - Pepper,_bell___healthy
			   - Potato___Early_blight
			   - Potato___healthy
			   - Raspberry___healthy
			   - Soybean___healthy
			   - Squash___Powdery_mildew
			   - Strawberry___healthy
			   - Strawberry___Leaf_scorch
			
            - Disease Detection Classes
            
			   - Cherry___healthy
			   - Cherry___Powdery_mildew
			   - Grape___Black_rot
			   - Grape___Esca_Black_Measles
			   - Grape___healthy
			   - Grape___Leaf_blight_Isariopsis_Leaf_Spot 
---
## Cloning the project  
* Run command `git clone "https://github.com/Saideepthi123/Plant-disease-detection.git"` and change into the project folder
* Create a virtual environment `env` in the repository (use virtualenv, etc)
*  Activate virtual environment
* Install the requirements


To create virtual environment and install requirements run following commands
```shell script
virtualenv env
```

To activate the environment use following commands:
Window: 
```shell script
.\env\Scripts\activate
```
Ubuntu/Linux
```shell script
source env/bin/activate
```
pip install -r requirements.txt

Command to run the app
---
 - streamlit run app.py

## Demo

- About

	![image](https://user-images.githubusercontent.com/52497119/118315341-0a7e7d80-b513-11eb-8565-24da0c206fdb.png)
	
- Disease Predection

	![image](https://user-images.githubusercontent.com/52497119/118315208-da36df00-b512-11eb-8b3a-4982fe2b3935.png)
	
	- Image Upload
	
		![3](https://user-images.githubusercontent.com/52497119/118315820-a01a0d00-b513-11eb-9a49-69176e64ed42.PNG)
		
	- Image Detected
	  	 
	        
	![image](https://user-images.githubusercontent.com/52497119/119301851-c01e9e80-bc80-11eb-9e86-c23947307072.png)

		
- Disease Classification

	- Image Upload
	
		![5](https://user-images.githubusercontent.com/52497119/118316025-e96a5c80-b513-11eb-8735-866427410077.PNG)
		
	- Image Classified
	
		![6](https://user-images.githubusercontent.com/52497119/118316149-1585dd80-b514-11eb-8c4b-8c9627d44e93.PNG)

- Treatement Page

	![7](https://user-images.githubusercontent.com/52497119/118316232-33534280-b514-11eb-8a71-3922c7e6267e.PNG)
	
## Required Libraries
- opencv-contrib-python-headless
- tensorflow-cpu
- streamlit
- numpy
- pandas
- pillow
- keras
- matplotlib
			
