from flask import Flask, render_template, request
from flask import Flask, render_template, request, redirect, url_for, flash, session
app = Flask(__name__)

app.secret_key = 'your_secret_key' 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix




from flask import Flask, render_template, request, redirect, url_for, session, flash
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from decimal import Decimal


from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
import pandas as pd
# Encode categorical columns
label_encoder_origin = LabelEncoder()
label_encoder_dest = LabelEncoder()
import lightgbm as lgb
perfect_df=pd.read_csv('filtered_file.csv')
perfect_df = perfect_df[['MONTH','DAY_OF_MONTH','DAY_OF_WEEK','CRS_DEP_TIME','DEP_TIME','ORIGIN','DEST','ARR_DEL15']] 
perfect_df['ORIGIN'] = label_encoder_origin.fit_transform(perfect_df['ORIGIN'])
perfect_df['DEST'] = label_encoder_dest.fit_transform(perfect_df['DEST'])



print("Origin Encoding:")
for category, code in zip(label_encoder_origin.classes_, range(len(label_encoder_origin.classes_))):
    print(f"{category} -> {code}")



print("Destination Encoding:")
for category, code in zip(label_encoder_dest.classes_, range(len(label_encoder_dest.classes_))):
    print(f"{category} -> {code}")

# Check class distribution
print("Class distribution before undersampling:")
print(perfect_df['ARR_DEL15'].value_counts())

# Separate majority and minority classes
majority_class = perfect_df[perfect_df['ARR_DEL15'] == 0]
minority_class = perfect_df[perfect_df['ARR_DEL15'] == 1]

# Undersample majority class
majority_class_undersampled = resample(
    majority_class,
    replace=False,
    n_samples=len(minority_class),
    random_state=42
)

# Combine classes
undersampled_df = pd.concat([majority_class_undersampled, minority_class])
undersampled_df = undersampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Verify class balance
print("Class distribution after undersampling:")
print(undersampled_df['ARR_DEL15'].value_counts())

# Split into features and target
X = undersampled_df.drop('ARR_DEL15', axis=1)
# print("lenghth of the parateters")
# print(len(X))
y = undersampled_df['ARR_DEL15']

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(X_train)




# Train the model
from lightgbm import LGBMClassifier
lgb_model = LGBMClassifier(random_state=42, class_weight='balanced')
lgb_model.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
y_pred = lgb_model.predict(X_test)
print(f'Accuracy lgb: {accuracy_score(y_test, y_pred):.2f}')
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
# import seaborn as sns
# import matplotlib.pyplot as plt
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=lgb_model.classes_, yticklabels=lgb_model.classes_)
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.title('Confusion Matrix')
# plt.show()


@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'admin':
            session['logged_in'] = True
            return redirect(url_for('prediction'))
        else:
            flash('Invalid username or password!', 'error')
    return render_template('login.html', title='Login')


@app.route('/home')
def home():
    return render_template('home.html', title='Home')
@app.route('/ori_tabel')
def ori_tabel():
    # Assuming you have a CSV file or a pandas DataFrame with your dataset
    # Load the dataset into a pandas DataFrame (replace with your actual dataset path)
    df = pd.read_csv('flightdata.csv')  # Adjust this to the path of your dataset

    # Convert the DataFrame to an HTML table
    table_html = df.to_html(classes='data table table-bordered table-striped', index=False)

    return render_template('ori_tabel.html', table_html=table_html)


@app.route('/fil_tabel')
def fil_tabel():
    # Assuming you have a CSV file or a pandas DataFrame with your dataset
    # Load the dataset into a pandas DataFrame (replace with your actual dataset path)
    df = pd.read_csv('filtered_file.csv')  # Adjust this to the path of your dataset

    # Convert the DataFrame to an HTML table
    table_html = df.to_html(classes='data table table-bordered table-striped', index=False)

    return render_template('fil_tabel.html', table_html=table_html)


@app.route('/analysis')
def analysis():
    # This will render the gallery page with images and explanations
    return render_template('analysis.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        # Extract 12 fields and convert to integers
        a = int(request.form['field1'])
        b = int(request.form['field2'])
        c = int(request.form['field3'])
        d = int(request.form['field4'])
        e = int(request.form['field5'])
        f = int(request.form['field6'])
        g = int(request.form['field7'])
  
     
     
 

       
        # Process the data
        data = {"a": a, "b": b, "c": c, "d": d, "e": e, "f": f, 
                "g": g}
        
        return f"Received Data: {data}"
    
    return render_template('prediction.html', title='Prediction')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract values from the form, using .get() to handle missing values
        a = Decimal(request.form.get('a', 0))
        b = Decimal(request.form.get('b', 0))
        c = Decimal(request.form.get('c', 0))
        d = Decimal(request.form.get('d', 0))
        e = Decimal(request.form.get('e', 0))
        f = Decimal(request.form.get('f', 0))
        g = Decimal(request.form.get('g', 0))
     
    

        # Perform prediction logic (here it is a simple sum for demo purposes)
        # result = a + b + c + d + e + f + g + h + i + j + k + l

        new_features = [[a,b,c,d,e,f,g]]


        # Make a prediction
        target_pred = lgb_model.predict(new_features)[0]  

        # Output the prediction (0 for "Normal" and 1 for "DoS")
        predicted_label = 'No Delay' if target_pred == 0 else 'The Flight will be Delayed'
        print(f"Predicted label for the input features: {predicted_label}")

        # Render result in the prediction_result page
        return render_template('prediction_result.html', result=predicted_label,a=a,b=b,c=c,d=d,e=e,f=f,g=g)

    except ValueError:
        # If there's an error in conversion, show an error message
        return render_template('prediction_result.html', result="Error: All inputs must be valid integers!")
    

@app.route('/logout')
def logout():
    session.clear()  # Clears all session data
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('login'))
if __name__ == '__main__':
    app.run(debug=False)
