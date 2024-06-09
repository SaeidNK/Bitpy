import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score

# Provided data
data = {
    "Date": ["2024-02-27", "2024-02-28", "2024-02-29", "2024-03-01", "2024-03-02", "2024-03-03", "2024-03-04", "2024-03-05", "2024-03-06", "2024-03-07", "2024-03-08", "2024-03-09", "2024-03-10", "2024-03-11", "2024-03-12", "2024-03-13", "2024-03-14", "2024-03-15", "2024-03-16", "2024-03-17", "2024-03-18", "2024-03-19", "2024-03-20", "2024-03-21", "2024-03-22", "2024-03-23", "2024-03-24", "2024-03-25", "2024-03-26", "2024-03-27", "2024-03-28", "2024-03-29", "2024-03-30", "2024-03-31", "2024-04-01", "2024-04-02", "2024-04-03", "2024-04-04", "2024-04-05", "2024-04-06", "2024-04-07", "2024-04-08", "2024-04-09", "2024-04-10", "2024-04-11", "2024-04-12", "2024-04-13", "2024-04-14", "2024-04-15", "2024-04-16", "2024-04-17", "2024-04-18", "2024-04-19", "2024-04-20", "2024-04-21", "2024-04-22", "2024-04-23", "2024-04-24", "2024-04-25", "2024-04-26", "2024-04-27", "2024-04-28", "2024-04-29", "2024-04-30", "2024-05-01", "2024-05-02", "2024-05-03", "2024-05-04", "2024-05-05", "2024-05-06", "2024-05-07", "2024-05-08", "2024-05-09", "2024-05-10", "2024-05-11", "2024-05-12", "2024-05-13", "2024-05-14", "2024-05-15", "2024-05-16", "2024-05-17", "2024-05-18", "2024-05-19", "2024-05-20", "2024-05-21", "2024-05-22", "2024-05-23", "2024-05-24", "2024-05-25", "2024-05-26", "2024-05-27", "2024-05-28", "2024-05-29", "2024-05-30", "2024-05-31", "2024-06-01", "2024-06-02", "2024-06-03", "2024-06-04", "2024-06-05"],
    "Actual Price": [57085.37109375, 62504.7890625, 61198.3828125, 62440.6328125, 62029.84765625, 63167.37109375, 68330.4140625, 63801.19921875, 66106.8046875, 66925.484375, 68300.09375, 68498.8828125, 69019.7890625, 72123.90625, 71481.2890625, 73083.5, 71396.59375, 69403.7734375, 65315.1171875, 68390.625, 67548.59375, 61912.7734375, 67913.671875, 65491.390625, 63778.76171875, 64062.203125, 67234.171875, 69958.8125, 69987.8359375, 69455.34375, 70744.953125, 69892.828125, 69645.3046875, 71333.6484375, 69702.1484375, 65446.97265625, 65980.8125, 68508.84375, 67837.640625, 68896.109375, 69362.5546875, 71631.359375, 69139.015625, 70587.8828125, 70060.609375, 67195.8671875, 63821.47265625, 65738.7265625, 63426.2109375, 63811.86328125, 61276.69140625, 63512.75390625, 63843.5703125, 64994.44140625, 64926.64453125, 66837.6796875, 66407.2734375, 64276.8984375, 64481.70703125, 63755.3203125, 63419.140625, 63113.23046875, 63841.12109375, 60636.85546875, 58254.01171875, 59123.43359375, 62889.8359375, 63891.47265625, 64031.1328125, 63161.94921875, 62334.81640625, 61187.94140625, 63049.9609375, 60792.77734375, 60793.7109375, 61448.39453125, 62901.44921875, 61552.7890625, 66267.4921875, 65231.58203125, 67051.875, 66940.8046875, 66278.3671875, 71448.1953125, 70136.53125, 69122.3359375, 67929.5625, 68526.1015625, 69265.9453125, 68518.09375, 69394.5546875, 68296.21875, 67578.09375, 68364.9921875, 67491.4140625, 67706.9375, 67751.6015625, 68804.78125, 70567.765625, 70849.0234375],
    "GB Predictions": [56728.4619691088, 62411.998326454865, 61535.44201785345, 62447.8971730565, 61915.09314928781, 62920.63261923348, 67291.02706188653, 63500.33179888375, 65927.18149535474, 66557.61451320913, 67044.80559657532, 67041.97667359623, 67047.61688042564, 67092.80243825627, 67096.32898933961, 67101.64710654716, 67078.3504641316, 67079.0674900312, 65717.26155553853, 67388.12933116389, 67077.30672170225, 62340.77247538279, 67368.07195559057, 65717.04680583507, 63610.40608035267, 64045.131473588284, 66558.65284732098, 67077.30672170225, 67384.8346731519, 67104.86236697617, 67073.53677372873, 67076.49511750079, 67076.49511750079, 67075.22618228794, 67064.22578566357, 65404.96138656258, 65928.67075905502, 67066.63375708848, 67070.16030817182, 67073.89770063873, 67072.27396391789, 67069.20055138478, 67072.27396391789, 67069.20055138478, 67070.00943024515, 66572.93723271429, 63882.90912120334, 66233.28691300242, 63885.55373837723, 63578.17254878961, 61366.72773690025, 63588.175000923875, 63576.54881206879, 64943.49142137774, 64929.04807361681, 66565.64820293307, 65935.66611466711, 64357.284227961754, 64421.52705411721, 63585.28832286804, 63579.369650847744, 63064.72789938705, 63527.986929516825, 60759.08569293114, 58262.09488036315, 58939.51026571299, 62919.04755766845, 64041.426862855675, 64356.083872998, 63090.54713389851, 62206.00171762073, 61402.901298881625, 63082.58955753261, 60968.73586275991, 60964.90146207925, 61455.29413164613, 62924.68776449787, 61585.17977134382, 65953.54634161494, 65400.72890421852, 66557.71502298891, 66567.44273295964, 65910.51191984028, 67072.33813256699, 67072.33813256699, 67060.51826029754, 67034.74325340209, 67072.27396391789, 67072.27396391789, 67044.44022148255, 67056.84747906179, 67023.18421570171, 67056.84747906179, 67000.50473091274, 67000.50473091274, 66946.62025184746, 67020.91968202897, 67042.17568780981, 67070.00943024515, 67071.63316696598]
}

df = pd.DataFrame(data)

# Compute error metrics
mse = mean_squared_error(df["Actual Price"], df["GB Predictions"])
mae = mean_absolute_error(df["Actual Price"], df["GB Predictions"])
r2 = r2_score(df["Actual Price"], df["GB Predictions"])

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R^2 Score: {r2}")

# Detailed analysis
diff = df["Actual Price"] - df["GB Predictions"]
df["Difference"] = diff
print(df.describe())
