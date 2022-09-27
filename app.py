from flask import Flask,render_template,request
import joblib
import pandas as pd


app = Flask(__name__)




@app.route('/')
@app.route('/main')
def main():
	return render_template('main.html')

@app.route('/predict',methods=['POST'])
def predict():
	int_features =[x for x in request.form.values()]
	print(int_features)
	ones = joblib.load("one_hot.joblib")
	mod = joblib.load("model.pkl")
	print("Hello")
	print(int_features)
	check=[int_features]
	check=pd.DataFrame(check,columns=['brand', 'model_name', 'fuel_type', 'city', 'year',
       'distance_travelled(kms)', 'car_age', 'new and less used',
       'inv_car_price', 'inv_car_dist', 'inv_car_age', 'std_invprice',
       'std_invdistance_travelled', 'std_invrank', 'best_buy1'])
	liss=ones.transform(check.iloc[:,:4])
	coll=ones.get_feature_names_out()
	dd = pd.DataFrame(liss,columns=coll)
	dd1=check.iloc[:,4:]
	fin = pd.concat([dd,dd1],axis=1)
	res= mod.predict(fin)
	print("$$$$$$$$$$$$$$$")
	print(res)




	
	
	# # x=pd.get_dummies(int_features,drop_first=True)
	# print(x)
	# final_features = [np.array(x).ravel()]
	# print(final_features)
	# prediction = model.predict(final_features)
	# print("Hello2")
	return render_template('main.html',prediction_text="Your Car Estimated Cost is INR: ₹{}".format(res))



if __name__ =="__main__":
	app.debug=True
	app.run('127.0.0.4',port=7000)