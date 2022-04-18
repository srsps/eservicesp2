
#The Iterative method used

# X_19['Pow-1'] = ''
# X_19['Pow-1'][0] = 81.95498
# Y_19['PowRF'] = ''
# Y_19['PowNN'] = ''
# Y_19['PowXGB'] = ''
# # RF Model
# end = len(Y_19['PowRF'])
# for i in range(0, end):
#     Y_19['PowRF'][i] = RF_model.predict(X_19.iloc[i, :].to_numpy().reshape(1, -1))
#     X_19['Pow-1'][i + 1] = Y_19['PowRF'][i]



# y_pred_RF19=np.vstack(Y_19['PowRF'].values)

# y_19=Y_19['Power_kW'].values

# #NN Model

# for i in range(0, end):
#     Y_19['PowNN'][i] = NN_model.predict(X_19.iloc[i, :].to_numpy().reshape(1, -1))
#     X_19['Pow-1'][i + 1] = Y_19['PowNN'][i]
    
# y_pred_NN19=np.vstack(Y_19['PowNN'].values)


# #XGB_model

# for i in range(0, end):
#     Y_19['PowXGB'][i] = NN_model.predict(X_19.iloc[i, :].to_numpy().reshape(1, -1))
#     X_19['Pow-1'][i + 1] = Y_19['PowXGB'][i]
    
# y_pred_XGB19=np.vstack(Y_19['PowXGB'].values)


# process = RobustScaler()
# X_tr19 = process.fit(X_19).transform(X_19)

# for i in range(0, end):
#     Y_19['PowGB'][i] = NN_model.predict(X_19.iloc[i, :].to_numpy().reshape(1, -1))
#     X_19['Pow-1'][i + 1] = Y_19['PowGB'][i]
    
# y_pred_GB19=np.vstack(Y_19['PowGB'].values)